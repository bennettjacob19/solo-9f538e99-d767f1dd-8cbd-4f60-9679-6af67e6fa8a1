"""Persistent state for the backup service.

Two SQLite databases are involved:

``source`` (default ``data/jobs.db``)
    The sample/operational job database that is backed up.  Its schema lives in
    :mod:`nybackup.source`.

``state`` (default ``data/state.db``)
    Book-keeping owned by the web app: configuration, the virtual clock offset
    and one row per backup *plan* keyed by its New York plan date.

The state database is never included in a backup.  It is opened with
``check_same_thread=False`` and guarded by a re-entrant lock, because both the
scheduler thread and request threads use it.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import date, datetime, timezone
from typing import Any, Iterable, Optional

DEFAULT_CONFIG = {
    "backup_time": "02:30",
    "timezone": "America/New_York",
    "clock_offset_seconds": 0.0,
    # How many of the newest *unprotected* downloadable backups to keep.
    # 0 means "unlimited" (the default); a limit is always >= 1.
    "retention_count": 0,
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS config (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS plans (
    plan_date           TEXT PRIMARY KEY,
    status              TEXT NOT NULL,       -- pending|running|complete|failed|skipped|cleaned
    due_utc             TEXT NOT NULL,
    configured_time     TEXT NOT NULL,       -- HH:MM in effect for this plan
    started_utc         TEXT,                -- virtual clock
    started_real_utc    TEXT,                -- wall clock
    completed_utc       TEXT,                -- virtual clock
    completed_real_utc  TEXT,                -- wall clock (file mtime)
    filename            TEXT,
    size_bytes          INTEGER,
    sha256              TEXT,
    partial_path        TEXT,
    shifted             INTEGER NOT NULL DEFAULT 0,
    ambiguous           INTEGER NOT NULL DEFAULT 0,
    error               TEXT,
    attempts            INTEGER NOT NULL DEFAULT 0,
    protected           INTEGER NOT NULL DEFAULT 0,  -- protected backups ignore the retention quota
    cleaned_utc         TEXT,                -- virtual clock time of the retention cleanup
    cleanup_error       TEXT,                -- last failed cleanup attempt (retried later)
    updated_utc         TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc        TEXT NOT NULL,             -- virtual clock
    real_ts_utc   TEXT NOT NULL,             -- wall clock
    level         TEXT NOT NULL,
    message       TEXT NOT NULL
);
"""

# Columns added after the first release: existing state databases are upgraded
# in place with ALTER TABLE (SQLite allows constant defaults there).
PLAN_COLUMN_MIGRATIONS = {
    "protected": "ALTER TABLE plans ADD COLUMN protected INTEGER NOT NULL DEFAULT 0",
    "cleaned_utc": "ALTER TABLE plans ADD COLUMN cleaned_utc TEXT",
    "cleanup_error": "ALTER TABLE plans ADD COLUMN cleanup_error TEXT",
}

def iso_utc(dt: Optional[datetime]) -> Optional[str]:
    return None if dt is None else dt.astimezone(timezone.utc).isoformat(timespec="seconds")


def parse_utc(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


class _Transaction:
    def __init__(self, store: "Store"):
        self.store = store
        self.conn: Optional[sqlite3.Connection] = None

    def __enter__(self) -> sqlite3.Connection:
        self.store._lock.acquire()
        try:
            self.conn = self.store._conn
            self.conn.execute("BEGIN IMMEDIATE")
        except BaseException:
            self.store._lock.release()
            raise
        return self.conn

    def __exit__(self, exc_type, exc, tb) -> bool:
        assert self.conn is not None
        try:
            if exc_type is None:
                self.conn.execute("COMMIT")
            else:
                self.conn.execute("ROLLBACK")
        finally:
            self.store._lock.release()
        return False


class Store:
    def __init__(self, state_path: str):
        self.path = os.path.abspath(state_path)
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            self.path, timeout=30, isolation_level=None, check_same_thread=False
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA busy_timeout=30000")
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._migrate()

    def _migrate(self) -> None:
        existing = {
            row[1] for row in self._conn.execute("PRAGMA table_info(plans)")
        }
        for column, ddl in PLAN_COLUMN_MIGRATIONS.items():
            if column not in existing:
                self._conn.execute(ddl)

    # -- plumbing ---------------------------------------------------------
    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def transaction(self) -> _Transaction:
        return _Transaction(self)

    def _fetchall(self, sql: str, params: Iterable[Any] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return list(self._conn.execute(sql, tuple(params)))

    def _fetchone(self, sql: str, params: Iterable[Any] = ()) -> Optional[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, tuple(params)).fetchone()

    # -- log --------------------------------------------------------------
    def log(
        self,
        level: str,
        message: str,
        *,
        virtual_now: Optional[datetime] = None,
        real_now: Optional[datetime] = None,
    ) -> None:
        virtual_now = virtual_now or datetime.now(timezone.utc)
        real_now = real_now or datetime.now(timezone.utc)
        with self.transaction() as c:
            c.execute(
                "INSERT INTO events(ts_utc, real_ts_utc, level, message) VALUES(?,?,?,?)",
                (iso_utc(virtual_now), iso_utc(real_now), level, message),
            )

    def recent_events(self, limit: int = 40) -> list[sqlite3.Row]:
        return self._fetchall("SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,))

    # -- configuration ----------------------------------------------------
    def config(self) -> dict[str, Any]:
        cfg = dict(DEFAULT_CONFIG)
        for row in self._fetchall("SELECT key, value FROM config"):
            try:
                cfg[row["key"]] = json.loads(row["value"])
            except json.JSONDecodeError:
                cfg[row["key"]] = row["value"]
        return cfg

    def set_config(self, updates: dict[str, Any]) -> dict[str, Any]:
        with self.transaction() as c:
            for key, value in updates.items():
                c.execute(
                    "INSERT INTO config(key, value) VALUES(?,?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (key, json.dumps(value)),
                )
        return self.config()

    # -- plans ------------------------------------------------------------
    def get_plan(self, plan_date: date) -> Optional[sqlite3.Row]:
        return self._fetchone(
            "SELECT * FROM plans WHERE plan_date=?", (plan_date.isoformat(),)
        )

    def all_plans(self) -> list[sqlite3.Row]:
        return self._fetchall("SELECT * FROM plans ORDER BY plan_date")

    def plans_for_dates(self, dates: Iterable[date]) -> dict[str, sqlite3.Row]:
        wanted = [d.isoformat() for d in dates]
        if not wanted:
            return {}
        placeholders = ",".join("?" for _ in wanted)
        rows = self._fetchall(
            f"SELECT * FROM plans WHERE plan_date IN ({placeholders})", wanted
        )
        return {row["plan_date"]: row for row in rows}

    def upsert_plan(self, plan_date: date, *, _real_now: Optional[datetime] = None, **fields: Any) -> None:
        real_now = _real_now or datetime.now(timezone.utc)
        fields.setdefault("updated_utc", iso_utc(real_now))
        if self.get_plan(plan_date) is None:
            fields.setdefault("status", "pending")
            fields.setdefault("configured_time", "00:00")
            fields.setdefault("due_utc", iso_utc(real_now))
            cols = ["plan_date"] + list(fields)
            values = [plan_date.isoformat()] + [fields[k] for k in fields]
            with self.transaction() as c:
                c.execute(
                    "INSERT INTO plans(%s) VALUES(%s)"
                    % (",".join(cols), ",".join("?" for _ in cols)),
                    values,
                )
        else:
            assignments = ", ".join(f"{k}=?" for k in fields)
            with self.transaction() as c:
                c.execute(
                    f"UPDATE plans SET {assignments} WHERE plan_date=?",
                    list(fields.values()) + [plan_date.isoformat()],
                )

    def delete_plan(self, plan_date: date) -> None:
        with self.transaction() as c:
            c.execute("DELETE FROM plans WHERE plan_date=?", (plan_date.isoformat(),))

    def clear_plans(self) -> None:
        with self.transaction() as c:
            c.execute("DELETE FROM plans")

    def plans_with_status(self, *statuses: str) -> list[sqlite3.Row]:
        placeholders = ",".join("?" for _ in statuses)
        return self._fetchall(
            f"SELECT * FROM plans WHERE status IN ({placeholders}) ORDER BY plan_date",
            list(statuses),
        )

    def begin_attempt(
        self,
        plan_date: date,
        *,
        due_utc: datetime,
        configured_time: str,
        shifted: bool,
        ambiguous: bool,
        partial_path: str,
        virtual_now: datetime,
        real_now: datetime,
    ) -> int:
        """Mark the plan as running and return the incremented attempt count."""
        with self.transaction() as c:
            row = c.execute(
                "SELECT attempts FROM plans WHERE plan_date=?", (plan_date.isoformat(),)
            ).fetchone()
            attempts = (row["attempts"] if row else 0) + 1
            c.execute(
                """
                INSERT INTO plans(plan_date, status, due_utc, configured_time,
                                  started_utc, started_real_utc, partial_path,
                                  shifted, ambiguous, attempts, updated_utc)
                VALUES(?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(plan_date) DO UPDATE SET
                    status='running',
                    due_utc=excluded.due_utc,
                    configured_time=excluded.configured_time,
                    started_utc=excluded.started_utc,
                    started_real_utc=excluded.started_real_utc,
                    completed_utc=NULL,
                    completed_real_utc=NULL,
                    filename=NULL,
                    size_bytes=NULL,
                    sha256=NULL,
                    partial_path=excluded.partial_path,
                    shifted=excluded.shifted,
                    ambiguous=excluded.ambiguous,
                    error=NULL,
                    attempts=excluded.attempts,
                    updated_utc=excluded.updated_utc
                """,
                (
                    plan_date.isoformat(),
                    "running",
                    iso_utc(due_utc),
                    configured_time,
                    iso_utc(virtual_now),
                    iso_utc(real_now),
                    partial_path,
                    int(shifted),
                    int(ambiguous),
                    attempts,
                    iso_utc(real_now),
                ),
            )
        return attempts

    def finish_attempt(
        self,
        plan_date: date,
        *,
        status: str,
        virtual_now: datetime,
        real_now: datetime,
        filename: Optional[str] = None,
        size_bytes: Optional[int] = None,
        sha256: Optional[str] = None,
        completed_virtual: Optional[datetime] = None,
        completed_real: Optional[datetime] = None,
        error: Optional[str] = None,
    ) -> None:
        """Record the outcome of an attempt.

        ``completed_virtual`` is the drill-clock time shown on the page;
        ``completed_real`` is when the bytes were actually produced (the file's
        mtime), which survives restarts and clock changes.
        """
        complete = status == "complete"
        with self.transaction() as c:
            c.execute(
                """
                UPDATE plans SET status=?, completed_utc=?, completed_real_utc=?,
                                 filename=?, size_bytes=?, sha256=?,
                                 partial_path=NULL, error=?, updated_utc=?
                WHERE plan_date=?
                """,
                (
                    status,
                    iso_utc(completed_virtual or real_now) if complete else None,
                    iso_utc(completed_real or real_now) if complete else None,
                    filename if complete else None,
                    size_bytes if complete else None,
                    sha256 if complete else None,
                    error,
                    iso_utc(real_now),
                    plan_date.isoformat(),
                ),
            )

    def mark_skipped(self, plan_date: date, reason: str, *, real_now: datetime) -> None:
        with self.transaction() as c:
            c.execute(
                """
                UPDATE plans SET status='skipped', error=?, partial_path=NULL,
                                 updated_utc=?
                WHERE plan_date=?
                """,
                (reason, iso_utc(real_now), plan_date.isoformat()),
            )

    # -- retention / protection --------------------------------------------
    def set_protected(self, plan_date: date, protected: bool, *, real_now: datetime) -> bool:
        """Toggle the protection flag of a completed backup.

        Returns ``False`` when the plan has no completed backup (nothing to
        protect) — e.g. it was cleaned by the retention rule in the meantime.
        """
        with self.transaction() as c:
            cur = c.execute(
                "UPDATE plans SET protected=?, updated_utc=? "
                "WHERE plan_date=? AND status='complete'",
                (int(protected), iso_utc(real_now), plan_date.isoformat()),
            )
            return cur.rowcount > 0

    def mark_cleaned(
        self,
        plan_date: date,
        *,
        virtual_now: datetime,
        real_now: datetime,
    ) -> None:
        """Record that the plan's backup file was removed by the retention rule.

        ``cleaned`` is terminal: the day is never re-backed-up, even if the
        drill clock moves back onto it.  The filename/size are kept so the
        page can still show what used to be there.
        """
        with self.transaction() as c:
            c.execute(
                """
                UPDATE plans SET status='cleaned', cleaned_utc=?, cleanup_error=NULL,
                                 partial_path=NULL, updated_utc=?
                WHERE plan_date=?
                """,
                (iso_utc(virtual_now), iso_utc(real_now), plan_date.isoformat()),
            )

    def set_cleanup_error(
        self,
        plan_date: date,
        error: Optional[str],
        *,
        real_now: datetime,
    ) -> None:
        with self.transaction() as c:
            c.execute(
                "UPDATE plans SET cleanup_error=?, updated_utc=? WHERE plan_date=?",
                (error, iso_utc(real_now), plan_date.isoformat()),
            )

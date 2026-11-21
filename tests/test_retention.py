"""Retention + protection tests: quota cleanup, protect races, crash recovery."""

from __future__ import annotations

import os
import threading
from datetime import date, datetime, timedelta, timezone

import pytest

from nybackup import backup, source
from nybackup.service import Service
from nybackup.store import Store


class Clock:
    def __init__(self, start: datetime):
        self.value = start
        self._lock = threading.Lock()

    def __call__(self) -> datetime:
        with self._lock:
            return self.value

    def set(self, value: datetime) -> None:
        with self._lock:
            self.value = value


def utc(*args) -> datetime:
    return datetime(*args, tzinfo=timezone.utc)


@pytest.fixture()
def env(tmp_path):
    src = str(tmp_path / "jobs.db")
    state = str(tmp_path / "state.db")
    out = str(tmp_path / "backups")
    os.makedirs(out, exist_ok=True)
    source.ensure_schema(src)
    source.seed(src, 5)
    clock = Clock(utc(2026, 11, 20, 8, 0))
    store = Store(state)
    service = Service(store, src, out, tick_seconds=0.01, clock=clock)
    yield {
        "src": src,
        "state": state,
        "out": out,
        "clock": clock,
        "store": store,
        "service": service,
    }
    service.stop()
    store.close()


def make_backups(env, days: list[date]) -> None:
    """Run one scheduled backup per day by advancing the drill clock."""
    clock, svc = env["clock"], env["service"]
    for day in days:
        # 08:00 UTC is 03:00/04:00 in New York: past the default 02:30 slot.
        clock.set(datetime(day.year, day.month, day.day, 8, 0, tzinfo=timezone.utc))
        svc.tick()
        row = env["store"].get_plan(day)
        assert row is not None and row["status"] == "complete", dict(row)


def files_on_disk(env) -> list[str]:
    return sorted(backup.list_backup_files(env["out"]).keys())


def test_retention_keeps_newest_and_cleans_excess(env):
    store, svc = env["store"], env["service"]
    days = [date(2026, 11, 20), date(2026, 11, 21), date(2026, 11, 22)]
    make_backups(env, days)
    assert files_on_disk(env) == [d.isoformat() for d in days]

    store.set_config({"retention_count": 2})
    result = svc.enforce_retention(trigger="settings")

    assert result["cleaned"] == ["2026-11-20"]
    assert files_on_disk(env) == ["2026-11-21", "2026-11-22"]
    old = store.get_plan(date(2026, 11, 20))
    assert old["status"] == "cleaned"
    assert old["cleaned_utc"]
    assert old["filename"] == "jobs-2026-11-20.sqlite"  # kept for display
    kept = store.get_plan(date(2026, 11, 21))
    assert kept["status"] == "complete"


def test_protected_backups_do_not_count_toward_quota(env):
    store, svc = env["store"], env["service"]
    days = [date(2026, 11, 20), date(2026, 11, 21), date(2026, 11, 22)]
    make_backups(env, days)

    # Protect the oldest, then keep only 1 normal backup.
    assert svc.set_protection(date(2026, 11, 20), True)["ok"]
    store.set_config({"retention_count": 1})
    svc.enforce_retention(trigger="settings")

    # 11-20 survives (protected, outside the quota); 11-22 is the one kept
    # normal backup; 11-21 is the excess and gets cleaned.
    assert files_on_disk(env) == ["2026-11-20", "2026-11-22"]
    assert store.get_plan(date(2026, 11, 20))["protected"] == 1
    assert store.get_plan(date(2026, 11, 21))["status"] == "cleaned"


def test_unprotect_triggers_cleanup(env):
    store, svc = env["store"], env["service"]
    days = [date(2026, 11, 20), date(2026, 11, 21)]
    make_backups(env, days)
    store.set_config({"retention_count": 1})
    assert svc.set_protection(date(2026, 11, 20), True)["ok"]
    svc.enforce_retention(trigger="protect")
    # Both survive: 11-20 protected, 11-21 the single normal slot.
    assert files_on_disk(env) == ["2026-11-20", "2026-11-21"]

    # Unprotecting makes 11-20 the excess normal backup -> cleaned.
    assert svc.set_protection(date(2026, 11, 20), False)["ok"]
    svc.enforce_retention(trigger="protect")
    assert files_on_disk(env) == ["2026-11-21"]
    assert store.get_plan(date(2026, 11, 20))["status"] == "cleaned"


def test_protect_saved_first_keeps_file(env):
    """Cleanup selected a day, but the protect was saved first: file stays."""
    store, svc = env["store"], env["service"]
    days = [date(2026, 11, 20), date(2026, 11, 21)]
    make_backups(env, days)
    store.set_config({"retention_count": 1})
    path = backup.final_path(env["out"], date(2026, 11, 20))

    # The protect is committed before the cleanup gets to re-check the row.
    assert svc.set_protection(date(2026, 11, 20), True)["ok"]
    outcome = svc._cleanup_one(date(2026, 11, 20), virtual_now=svc.now())

    assert outcome["action"] == "kept"
    assert os.path.exists(path)
    row = store.get_plan(date(2026, 11, 20))
    assert row["status"] == "complete" and row["protected"] == 1


def test_protect_on_cleaned_plan_is_refused(env):
    store, svc = env["store"], env["service"]
    make_backups(env, [date(2026, 11, 20), date(2026, 11, 21)])
    store.set_config({"retention_count": 1})
    svc.enforce_retention(trigger="settings")
    assert store.get_plan(date(2026, 11, 20))["status"] == "cleaned"

    result = svc.set_protection(date(2026, 11, 20), True)
    assert not result["ok"]
    assert "已按保留规则清理" in result["message"]


def test_failed_cleanup_stays_downloadable_and_retries(env, monkeypatch):
    store, svc = env["store"], env["service"]
    make_backups(env, [date(2026, 11, 20), date(2026, 11, 21)])
    store.set_config({"retention_count": 1})
    path = backup.final_path(env["out"], date(2026, 11, 20))

    real_remove = os.remove

    def failing_remove(target, *args, **kwargs):
        if os.path.abspath(target) == os.path.abspath(path):
            raise OSError("只读文件系统")
        return real_remove(target, *args, **kwargs)

    monkeypatch.setattr(os, "remove", failing_remove)
    result = svc.enforce_retention(trigger="tick")
    assert result["failed"] == ["2026-11-20"]

    # Still complete and downloadable; the reason is recorded for the page.
    row = store.get_plan(date(2026, 11, 20))
    assert row["status"] == "complete"
    assert "只读文件系统" in row["cleanup_error"]
    assert os.path.exists(path)

    # A repeated failure with the same cause does not spam the event log.
    events_before = len(store.recent_events(1000))
    svc.enforce_retention(trigger="tick")
    assert len(store.recent_events(1000)) == events_before

    # Once the cause is gone, a later pass retries and cleans it.
    monkeypatch.setattr(os, "remove", real_remove)
    svc.enforce_retention(trigger="tick")
    assert store.get_plan(date(2026, 11, 20))["status"] == "cleaned"
    assert not os.path.exists(path)


def test_failed_cleanup_cleared_when_rules_change(env, monkeypatch):
    """A stale failure record disappears when the latest rules keep the file."""
    store, svc = env["store"], env["service"]
    make_backups(env, [date(2026, 11, 20), date(2026, 11, 21)])
    store.set_config({"retention_count": 1})
    path = backup.final_path(env["out"], date(2026, 11, 20))

    def failing_remove(target, *args, **kwargs):
        if os.path.abspath(target) == os.path.abspath(path):
            raise OSError("磁盘忙")
        return None

    monkeypatch.setattr(os, "remove", failing_remove)
    svc.enforce_retention(trigger="tick")
    assert store.get_plan(date(2026, 11, 20))["cleanup_error"]

    # The limit is raised before the retry: the day is no longer excess, the
    # file is kept and the failure record is cleared.
    store.set_config({"retention_count": 0})  # unlimited
    svc.enforce_retention(trigger="settings")
    row = store.get_plan(date(2026, 11, 20))
    assert row["status"] == "complete"
    assert row["cleanup_error"] is None
    assert os.path.exists(path)


def test_crash_after_delete_before_record_shows_cleaned_on_restart(env):
    """File deleted, outcome never recorded, process exits: restart -> cleaned."""
    store, svc = env["store"], env["service"]
    make_backups(env, [date(2026, 11, 20)])
    path = backup.final_path(env["out"], date(2026, 11, 20))
    assert store.get_plan(date(2026, 11, 20))["status"] == "complete"

    # Simulate the crash window: the file is gone but the plan still says
    # "complete" (the cleanup outcome was never written).
    os.remove(path)
    store.close()

    store2 = Store(env["state"])
    svc2 = Service(store2, env["src"], env["out"], tick_seconds=0.01, clock=env["clock"])
    try:
        svc2.recover()
        row = store2.get_plan(date(2026, 11, 20))
        assert row["status"] == "cleaned"
        assert row["cleaned_utc"]
        # And it stays that way after the scheduler runs.
        svc2.tick()
        assert store2.get_plan(date(2026, 11, 20))["status"] == "cleaned"
        assert files_on_disk(env) == []
    finally:
        store2.close()


def test_cleaned_day_is_never_recreated(env):
    """Neither catch-up nor a manual run recreates a cleaned day, even when
    the drill clock is moved back onto it."""
    store, svc, clock = env["store"], env["service"], env["clock"]
    make_backups(env, [date(2026, 11, 20), date(2026, 11, 21)])
    store.set_config({"retention_count": 1})
    svc.enforce_retention(trigger="settings")
    assert store.get_plan(date(2026, 11, 20))["status"] == "cleaned"
    assert files_on_disk(env) == ["2026-11-21"]

    # Move the drill clock back onto the cleaned day and let the scheduler run.
    clock.set(utc(2026, 11, 20, 8, 0))
    svc.tick()
    assert store.get_plan(date(2026, 11, 20))["status"] == "cleaned"
    assert not os.path.exists(backup.final_path(env["out"], date(2026, 11, 20)))

    # A manual run is refused as well.
    result = svc.run_plan_now(date(2026, 11, 20))
    assert result["action"] == "none"
    assert "不再重建" in result["message"]
    assert not os.path.exists(backup.final_path(env["out"], date(2026, 11, 20)))

    # Moving forward again, the next day still backs up normally.
    clock.set(utc(2026, 11, 22, 8, 0))
    svc.tick()
    assert store.get_plan(date(2026, 11, 22))["status"] == "complete"


def test_manual_backup_triggers_retention(env):
    """Manual backups follow the same retention rule as scheduled ones."""
    store, svc, clock = env["store"], env["service"], env["clock"]
    store.set_config({"retention_count": 1})
    # Two manual backups on two plan dates.
    clock.set(utc(2026, 11, 20, 8, 0))
    assert svc.run_plan_now(date(2026, 11, 20))["action"] == "complete"
    clock.set(utc(2026, 11, 21, 8, 0))
    assert svc.run_plan_now(date(2026, 11, 21))["action"] == "complete"

    # The manual completion enforced the quota immediately.
    assert files_on_disk(env) == ["2026-11-21"]
    assert store.get_plan(date(2026, 11, 20))["status"] == "cleaned"


def test_scheduled_backup_triggers_retention_via_tick(env):
    store, svc, clock = env["store"], env["service"], env["clock"]
    store.set_config({"retention_count": 2})
    make_backups(env, [date(2026, 11, 20), date(2026, 11, 21)])
    assert files_on_disk(env) == ["2026-11-20", "2026-11-21"]
    # The third scheduled backup pushes the oldest over the quota; the same
    # tick that completes it also cleans the excess.
    clock.set(utc(2026, 11, 22, 8, 0))
    svc.tick()
    assert files_on_disk(env) == ["2026-11-21", "2026-11-22"]
    assert store.get_plan(date(2026, 11, 20))["status"] == "cleaned"


def test_unlimited_retention_cleans_nothing(env):
    store, svc = env["store"], env["service"]
    make_backups(env, [date(2026, 11, 20), date(2026, 11, 21), date(2026, 11, 22)])
    svc.enforce_retention(trigger="tick")
    assert files_on_disk(env) == ["2026-11-20", "2026-11-21", "2026-11-22"]
    for day in ("2026-11-20", "2026-11-21", "2026-11-22"):
        assert store.get_plan(date.fromisoformat(day))["status"] == "complete"


def test_cleaned_plan_survives_restart_and_is_not_adopted_back(env):
    """A cleaned day stays cleaned across restarts; a reappearing file is
    left on disk but not adopted."""
    store, svc = env["store"], env["service"]
    make_backups(env, [date(2026, 11, 20), date(2026, 11, 21)])
    store.set_config({"retention_count": 1})
    svc.enforce_retention(trigger="settings")
    assert store.get_plan(date(2026, 11, 20))["status"] == "cleaned"
    store.close()

    store2 = Store(env["state"])
    svc2 = Service(store2, env["src"], env["out"], tick_seconds=0.01, clock=env["clock"])
    try:
        svc2.recover()
        svc2.tick()
        assert store2.get_plan(date(2026, 11, 20))["status"] == "cleaned"
        assert files_on_disk(env) == ["2026-11-21"]
    finally:
        store2.close()


def test_protect_during_delete_window_is_serialised(env, monkeypatch):
    """A protect that lands *while* the cleanup is deleting must not lose the
    file: the state-store lock serialises the two, so either the protect is
    saved first (file kept) or the cleanup commits first (protect refused).
    """
    import time

    store, svc = env["store"], env["service"]
    target = date(2026, 11, 20)
    make_backups(env, [target, date(2026, 11, 21)])
    store.set_config({"retention_count": 1})
    path = backup.final_path(env["out"], target)

    real_remove = os.remove

    def slow_remove(p, *args, **kwargs):
        if os.path.abspath(p) == os.path.abspath(path):
            time.sleep(0.1)  # widen the delete window inside the transaction
        return real_remove(p, *args, **kwargs)

    monkeypatch.setattr(os, "remove", slow_remove)

    results: dict = {}

    def do_cleanup():
        results["cleanup"] = svc._cleanup_one(target, virtual_now=svc.now())

    def do_protect():
        time.sleep(0.02)  # arrive while the cleanup holds the delete window
        results["protect"] = svc.set_protection(target, True)

    cleaner = threading.Thread(target=do_cleanup)
    protector = threading.Thread(target=do_protect)
    cleaner.start()
    protector.start()
    cleaner.join()
    protector.join()

    row = store.get_plan(target)
    if results["protect"]["ok"]:
        # The protect was saved first: the original file must have been kept.
        assert results["cleanup"]["action"] == "kept"
        assert row["status"] == "complete" and row["protected"] == 1
        assert os.path.exists(path)
    else:
        # The cleanup committed first: the protect is refused, the file is
        # gone and the day is cleaned — never "protected but file deleted".
        assert results["cleanup"]["action"] == "cleaned"
        assert row["status"] == "cleaned"
        assert not os.path.exists(path)

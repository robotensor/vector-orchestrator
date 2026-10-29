"""Room on the run root: finished duels cleared, oldest first, once too little is free."""

from __future__ import annotations

import json
import os
import time

import pytest

from vector_orchestrator.cli import main
from vector_orchestrator.duel import storage
from vector_orchestrator.duel.storage import Reaper, clear, finished, reap

KB = 1024


def run_dir(root, name, *, outcome=True, at=None, bulk=3):
    """A duel's run directory: a request, a side's run holding `bulk` KB and a prompt of one,
    and, unless told otherwise, an outcome written at `at`."""
    path = root / name
    (path / "king" / "u-00").mkdir(parents=True)
    (path / "prompts").mkdir()
    (path / "king" / "u-00" / "clip.mp4").write_bytes(b"x" * KB * bulk)
    (path / "prompts" / "manifest.jsonl").write_bytes(b"y" * KB)
    (path / "request.json").write_text("{}")
    if outcome:
        (path / "outcome.json").write_text(json.dumps({"status": "published"}))
        if at is not None:
            os.utime(path / "outcome.json", (at, at))
    return path


@pytest.fixture
def kb_is_gb(monkeypatch):
    """A GB of a kilobyte, so that a few files make a threshold."""
    monkeypatch.setattr(storage, "GB", KB)


@pytest.fixture
def free(tmp_path, monkeypatch):
    """The free space as the reaper reads it: `base` KB plus whatever was cleared under tmp_path,
    as the notes say."""
    base = {"kb": 10}

    def free_bytes(path):
        notes = (json.loads(p.read_text()) for p in tmp_path.glob("*/cleared.json"))
        return base["kb"] * KB + sum(int(n["freed_bytes"]) for n in notes)

    monkeypatch.setattr(storage, "free_bytes", free_bytes)
    return base


def test_finished_duels_are_listed_oldest_first_and_only_them(tmp_path):
    old = run_dir(tmp_path, "a" * 16, at=1_000)
    new = run_dir(tmp_path, "b" * 16, at=2_000)
    run_dir(tmp_path, "c" * 16, outcome=False)  # may resume: not finished
    aside = run_dir(tmp_path, "d" * 16 + ".stale-1", outcome=False)
    os.utime(aside, (1_500, 1_500))
    (tmp_path / "notes.txt").write_text("")
    (tmp_path / "elsewhere").mkdir()
    (tmp_path / ("e" * 16)).mkdir()  # decided, nothing left to clear
    (tmp_path / ("e" * 16) / "outcome.json").write_text("{}")
    listed = [(f.path, f.how) for f in finished(tmp_path)]
    assert listed == [(old, "decided"), (aside, "aside"), (new, "decided")]
    assert finished(tmp_path / "missing") == []


def test_clearing_removes_the_directories_and_keeps_the_files(tmp_path):
    path = run_dir(tmp_path, "a" * 16, bulk=5)
    freed = clear(path)
    assert freed == 6 * KB
    assert sorted(p.name for p in path.iterdir()) == [
        "cleared.json",
        "outcome.json",
        "request.json",
    ]
    note = json.loads((path / "cleared.json").read_text())
    assert note["freed_bytes"] == freed and sorted(note["removed"]) == ["king", "prompts"]
    assert note["at"].endswith("Z")
    assert finished(tmp_path) == [], "not a candidate again"


def test_reaping_clears_the_oldest_until_enough_is_free(tmp_path, kb_is_gb, free):
    a = run_dir(tmp_path, "a" * 16, at=1)  # 4 KB each
    b = run_dir(tmp_path, "b" * 16, at=2)
    c = run_dir(tmp_path, "c" * 16, at=3)
    assert reap(tmp_path, 10) == [], "10 KB free already"
    cleared = reap(tmp_path, 17)  # 10 + 4 + 4 = 18 >= 17: two go
    assert [(f.path, held) for f, held in cleared] == [(a, 4 * KB), (b, 4 * KB)]
    assert (a / "cleared.json").exists() and (b / "cleared.json").exists()
    assert not (c / "cleared.json").exists() and (c / "king").is_dir()
    (last,) = finished(tmp_path)
    assert reap(tmp_path, 100) == [(last, 4 * KB)], "then the last; still short"
    assert reap(tmp_path, 100) == [], "nothing finished is left"


def test_reaping_leaves_the_duel_running_alone_and_a_dry_run_clears_nothing(
    tmp_path, kb_is_gb, free
):
    a = run_dir(tmp_path, "a" * 16, at=1)
    b = run_dir(tmp_path, "b" * 16, at=2)
    c = run_dir(tmp_path, "c" * 16, at=3)
    would = reap(tmp_path, 17, dry_run=True)
    assert [f.path for f, _ in would] == [a, b] and not list(tmp_path.glob("*/cleared.json"))
    cleared = reap(tmp_path, 17, keep=[a])
    assert [f.path for f, _ in cleared] == [b, c]
    assert (a / "king").is_dir(), "the one kept"


def test_the_threshold_comes_from_the_environment(monkeypatch):
    monkeypatch.delenv(storage.FREE_ENV, raising=False)
    assert storage.free_gb_from_env() == storage.DEFAULT_FREE_GB == 300.0
    monkeypatch.setenv(storage.FREE_ENV, "250")
    assert storage.free_gb_from_env() == 250.0
    monkeypatch.setenv(storage.FREE_ENV, "0")
    assert storage.free_gb_from_env() == 0.0, "never"
    monkeypatch.setenv(storage.FREE_ENV, "lots")
    assert storage.free_gb_from_env() == 300.0, "not a number: the default"


def test_the_free_space_is_read_where_the_run_root_will_be(tmp_path):
    assert storage.free_bytes(tmp_path / "not" / "yet") == storage.free_bytes(tmp_path) > 0


def test_the_reaper_clears_on_its_own_and_when_poked(tmp_path, kb_is_gb, free):
    a = run_dir(tmp_path, "a" * 16, at=1)
    reaper = Reaper(tmp_path, 12, poll_s=60.0, keep=lambda: [])  # 10 KB free: 12 wanted
    reaper.start()
    try:
        deadline = time.monotonic() + 10
        while not (a / "cleared.json").exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert (a / "cleared.json").exists(), "the first pass, on start"
        b = run_dir(tmp_path, "b" * 16, at=2)  # 14 KB free now: enough, until the bar is raised
        reaper.free_gb = 16
        reaper.poke()
        deadline = time.monotonic() + 10
        while not (b / "cleared.json").exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert (b / "cleared.json").exists(), "poked"
    finally:
        reaper.stop()
    assert not reaper._thread.is_alive()


def test_reap_by_hand(tmp_path, kb_is_gb, free, capsys):
    a = run_dir(tmp_path, "a" * 16, at=1)
    assert main(["reap", "--run-dir", str(tmp_path), "--free-gb", "12", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert f"would clear {a.name} (decided): 4.00 GB" in out and "14.0 GB after" in out
    assert (a / "king").is_dir()
    assert main(["reap", "--run-dir", str(tmp_path), "--free-gb", "12"]) == 0
    out = capsys.readouterr().out
    assert f"cleared {a.name} (decided): 4.00 GB" in out and "keeping 12 GB free" in out
    assert not (a / "king").exists()

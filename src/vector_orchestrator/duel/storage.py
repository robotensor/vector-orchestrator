"""Room on the run root, kept: what finished duels left behind is cleared, oldest first, once
less than `free_gb` GB is free on its filesystem.

A duel keeps everything it ran under `<run_root>/<event_id[:16]>/` (`orchestrate`): the
submissions it fetched, every prompt, each unit's run logs and clips - gigabytes a duel - and
nothing took them away. So an orchestrator keeps a `Reaper` beside its duels: every `POLL_S`
seconds, and whenever a duel ends, it reads the free space on the run root's filesystem and, while
that is under the threshold, clears the oldest finished duel, then the next.

- **Finished** is a duel with an `outcome.json` (published, refused or void: it reads as decided,
  `Orchestrator.outcome`), or a run moved aside as `<dir>.<why>-<n>` (never resumed). The duel
  that is running, and a duel that may resume (no outcome yet), are never touched. Oldest is by
  when the outcome was written; for a run moved aside, by the directory's own time.
- **Cleared** is the duel's directories gone (`<side>/`, `prompts/`, `slots/`: all the bulk) and
  its files kept (`request.json`, `outcome.json`, `forfeit.json`), so that the duel still reads
  as decided and is not run again, with `cleared.json` saying when and how much went.

The threshold is `$VECTOR_FREE_GB` (`DEFAULT_FREE_GB` unless set; 0 for never), in GB of 1024^3
bytes. The store - the published events and their clips - is the record, mirrored, and is not
touched. `vector-orchestrator reap` does one pass by hand.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import threading
from collections.abc import Callable, Collection
from dataclasses import dataclass
from pathlib import Path

from ..store.records import now_iso
from ..store.writer import atomic_write_json

log = logging.getLogger(__name__)

GB = 1024**3
#: GB to keep free on the run root's filesystem, unless `$VECTOR_FREE_GB` says.
DEFAULT_FREE_GB = 100.0
FREE_ENV = "VECTOR_FREE_GB"
#: How often the reaper reads the free space, besides when a duel ends.
POLL_S = 600.0
OUTCOME_FILE = "outcome.json"
CLEARED_FILE = "cleared.json"
#: A duel's run directory is its event id's first 16 hex digits; moved aside, `<dir>.<why>-<n>`.
_RUN_DIR = re.compile(r"[0-9a-f]{16}")
_ASIDE = re.compile(r"[0-9a-f]{16}\.[a-z-]+-\d+")


def free_gb_from_env() -> float:
    """`$VECTOR_FREE_GB`, else `DEFAULT_FREE_GB`. A value that is not a number is the default,
    logged."""
    raw = os.environ.get(FREE_ENV)
    if raw is None or not raw.strip():
        return DEFAULT_FREE_GB
    try:
        return max(0.0, float(raw))
    except ValueError:
        log.warning(
            "%s=%r is not a number of GB; %s GB are kept free", FREE_ENV, raw, DEFAULT_FREE_GB
        )
        return DEFAULT_FREE_GB


def free_bytes(path: str | os.PathLike[str]) -> int:
    """The free space on the filesystem holding `path` - its nearest existing ancestor, before it
    exists."""
    probe = Path(path).resolve()
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    return shutil.disk_usage(probe).free


@dataclass(frozen=True)
class Finished:
    path: Path
    #: When it finished: the outcome's write time, or the directory's for a run moved aside.
    at: float
    #: `decided` (an outcome) or `aside` (moved aside, never resumed).
    how: str


def _bulk(path: Path) -> list[Path]:
    """The directories under a run directory: what clearing removes."""
    return sorted(p for p in path.iterdir() if p.is_dir() and not p.is_symlink())


def _size(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.lstat(os.path.join(root, name)).st_size
            except OSError:
                pass
    return total


def finished(run_root: str | os.PathLike[str]) -> list[Finished]:
    """The run root's finished duels that still hold something to clear, oldest first."""
    try:
        entries = list(os.scandir(run_root))
    except OSError:
        return []
    out = []
    for entry in entries:
        if not entry.is_dir(follow_symlinks=False):
            continue
        path = Path(entry.path)
        if _RUN_DIR.fullmatch(entry.name):
            try:
                at = (path / OUTCOME_FILE).stat().st_mtime
            except OSError:
                continue  # not decided: it may resume
            how = "decided"
        elif _ASIDE.fullmatch(entry.name):
            at, how = entry.stat().st_mtime, "aside"
        else:
            continue
        if (path / CLEARED_FILE).exists() or not _bulk(path):
            continue
        out.append(Finished(path, at, how))
    return sorted(out, key=lambda f: (f.at, f.path.name))


def clear(path: str | os.PathLike[str]) -> int:
    """A finished duel's directories removed and its files kept, with `cleared.json` saying when
    and how much went. The bytes freed."""
    path = Path(path)
    freed, removed = 0, []
    for child in _bulk(path):
        freed += _size(child)
        shutil.rmtree(child, ignore_errors=True)
        removed.append(child.name)
    atomic_write_json(
        path / CLEARED_FILE, {"at": now_iso(), "freed_bytes": freed, "removed": removed}
    )
    return freed


def reap(
    run_root: str | os.PathLike[str],
    free_gb: float,
    *,
    keep: Collection[str | os.PathLike[str]] = (),
    dry_run: bool = False,
) -> list[tuple[Finished, int]]:
    """Finished duels cleared, oldest first, until `free_gb` GB are free on the run root's
    filesystem - none when they already are - but for `keep` (the duel running). Each with the
    bytes it held. With `dry_run`, what would be cleared, and nothing is."""
    need = float(free_gb) * GB
    kept = {Path(p).resolve() for p in keep}
    free = free_bytes(run_root)
    cleared: list[tuple[Finished, int]] = []
    if free >= need:
        return cleared
    log.info(
        "the run root %s has %.1f GB free, under the %.1f GB kept free",
        run_root,
        free / GB,
        need / GB,
    )
    for duel in finished(run_root):
        if free >= need:
            break
        if duel.path.resolve() in kept:
            continue
        if dry_run:
            held = sum(_size(p) for p in _bulk(duel.path))
            free += held
        else:
            held = clear(duel.path)
            free = free_bytes(run_root)
            log.info(
                "cleared %s (%s): %.2f GB; %.1f GB free now",
                duel.path.name, duel.how, held / GB, free / GB,
            )  # fmt: skip
        cleared.append((duel, held))
    if free < need:
        log.warning(
            "the run root %s has %.1f GB free, still under %.1f GB: nothing finished is left to "
            "clear",
            run_root, free / GB, need / GB,
        )  # fmt: skip
    return cleared


class Reaper:
    """A thread that keeps `free_gb` GB free on `run_root`: `reap` every `poll_s` seconds and
    whenever poked (a duel ended), but for the directories `keep()` names (the duel running).
    What goes wrong is logged, never raised: a duel is not stopped by its housekeeping."""

    def __init__(
        self,
        run_root: str | os.PathLike[str],
        free_gb: float,
        *,
        keep: Callable[[], Collection[str | os.PathLike[str]]] = tuple,
        poll_s: float = POLL_S,
    ) -> None:
        self.run_root = Path(run_root)
        self.free_gb = float(free_gb)
        self.poll_s = float(poll_s)
        self._keep = keep
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="storage-reaper", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def poke(self) -> None:
        """A pass now, besides the next on the clock."""
        self._wake.set()

    def stop(self, timeout: float = 10.0) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread.is_alive():
            self._thread.join(timeout)

    def once(self) -> list[tuple[Finished, int]]:
        try:
            return reap(self.run_root, self.free_gb, keep=self._keep())
        except Exception:  # noqa: BLE001 - the next pass tries again
            log.exception("could not clear the run root %s", self.run_root)
            return []

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.once()
            self._wake.wait(self.poll_s)
            self._wake.clear()

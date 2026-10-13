"""The append-only result store, as it is mirrored to Hugging Face:

    manifest.json              {schema}: the layout's version; each event names its own spec
    head.json                  {seq, block, king}: a cache of the index
    queue.json                 the queue, as the dashboard shows it
    running.json               the duel running now, as far as it has got (`progress.py`)
    champions.json             the models the lane pays now, as the validator computes them
    index-NNNN.jsonl           one canonical JSON record per published event: a `genesis`, a
                               `duel`, a `vacate` (a king whose submission is gone, dethroned,
                               the crown back to its successor) or a `retire` (a champion whose
                               submission is gone, struck off; the crown does not move)
    events/<event_id>.json     a duel's units and scores
    media/<sha[:2]>/<sha>.mp4  clips
    validator/<lane>.json      what the validator driving the store remembers (its lane's state:
                               which hotkey committed what), so another host can take it up

Every write is atomic (tmp + rename) except the index append, a single O_APPEND write and fsync, so
a reader sees at most one torn final line. The index is the truth: a record's seq comes from its
last whole line, never from `head.json`.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import shutil
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from ..canon import canonical_json, sha256_file
from ..spec import Spec

SCHEMA = 6
LINES_PER_PART = 1000
VIDEO = "mp4"

LOCK_FILE = ".orchestrator.lock"

#: Serialises appends to the index. Separate from `store_lock`, the one-writer lock over a store.
INDEX_LOCK_FILE = ".index.lock"

log = logging.getLogger(__name__)


def king_after(record: dict[str, Any], previous: dict | None) -> dict | None:
    """Who holds the crown after `record`: a genesis crowns its entrant (its `king`), a duel keeps
    its king unless the challenger dethroned it, a vacate hands it to the successor its challenger
    slot names (none: the throne is empty), and anything else - a retire - leaves it where it was."""
    if record.get("kind") == "genesis":
        return record.get("king")
    if record.get("kind") == "duel":
        return record.get("challenger") if record.get("dethroned") else record.get("king")
    if record.get("kind") == "vacate":
        return record.get("challenger") or None
    return previous


@contextmanager
def store_lock(root: str | Path):
    """One writer per store. Publishing commands take this; a second writer fails fast."""
    path = Path(root) / LOCK_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(
                f"another orchestrator is publishing to {root} (holds {path})"
            ) from exc
        yield
    finally:
        os.close(fd)


def atomic_write_text(path: Path, text: str) -> None:
    """Write `path` in one step. The temporary name is unique, so two writers of the same file
    cannot truncate each other's half-written copy."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def atomic_write_json(path: Path, obj: Any, pretty: bool = True) -> None:
    text = json.dumps(obj, indent=2 if pretty else None, sort_keys=True) + "\n"
    atomic_write_text(path, text)


def read_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def index_part_name(part: int) -> str:
    return f"index-{part:04d}.jsonl"


def last_record(root: str | Path) -> dict[str, Any] | None:
    """The last whole line of the newest index part under `root`, as the index itself has it."""
    root = Path(root)
    parts, part = [], 0
    while (root / index_part_name(part)).exists():
        parts.append(root / index_part_name(part))
        part += 1
    for path in reversed(parts):
        for line in reversed(path.read_bytes().split(b"\n")):
            if not line:
                continue
            try:
                record = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, ValueError):
                continue
            if isinstance(record, dict) and isinstance(record.get("seq"), int):
                return record
    return None


class Store:
    def __init__(self, root: str | Path, spec: Spec):
        self.root = Path(root)
        self.spec = spec
        self.touched: set[str] = set()
        #: A duel's worker threads touch files concurrently, and one touched between a drain's
        #: copy and its clear would miss the push.
        self._touched_lock = threading.Lock()

    # ---------------------------------------------------------------- paths
    def index_part_path(self, part: int) -> Path:
        return self.root / index_part_name(part)

    def head_path(self) -> Path:
        return self.root / "head.json"

    def queue_path(self) -> Path:
        return self.root / "queue.json"

    def running_path(self) -> Path:
        return self.root / "running.json"

    def champions_path(self) -> Path:
        return self.root / "champions.json"

    def event_path(self, event_id: str) -> Path:
        return self.root / "events" / f"{event_id}.json"

    def media_path(self, sha: str) -> Path:
        return self.root / "media" / sha[:2] / f"{sha}.{VIDEO}"

    def validator_path(self, lane: str) -> Path:
        return self.root / "validator" / f"{lane}.json"

    def _touch(self, path: Path) -> None:
        with self._touched_lock:
            self.touched.add(str(path.relative_to(self.root)))

    def touch(self, path: Path) -> None:
        """Mark `path`, a file of this store, as one the next mirror push must carry again."""
        self._touch(Path(path))

    # ---------------------------------------------------------------- manifest
    def init(self) -> dict[str, Any]:
        manifest = {"schema": SCHEMA}
        atomic_write_json(self.root / "manifest.json", manifest)
        self._touch(self.root / "manifest.json")
        if not self.head_path().exists():
            self.write_head(seq=0, block=0, king=None)
        return manifest

    def manifest(self) -> dict[str, Any] | None:
        return read_json(self.root / "manifest.json")

    # ---------------------------------------------------------------- head / index
    def head(self) -> dict[str, Any] | None:
        return read_json(self.head_path())

    def write_head(self, *, seq: int, block: int, king: dict | None) -> None:
        atomic_write_json(self.head_path(), {"seq": seq, "block": block, "king": king})
        self._touch(self.head_path())

    @contextmanager
    def index_lock(self):
        """Held across reading the last seq, appending and rewriting the head, so two appends
        cannot take the same number."""
        path = self.root / INDEX_LOCK_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            os.close(fd)

    def index_parts(self) -> list[int]:
        parts, part = [], 0
        while self.index_part_path(part).exists():
            parts.append(part)
            part += 1
        return parts

    @staticmethod
    def _is_whole_record(line: bytes) -> bool:
        """A line is a whole record when it is a JSON object in exactly its canonical form: a
        record cut anywhere short of its closing brace is not."""
        try:
            text = line.decode("utf-8")
            record = json.loads(text)
            return isinstance(record, dict) and canonical_json(record) == text
        except (UnicodeDecodeError, ValueError, RecursionError):
            return False

    def repair_tail(self) -> None:
        """Make the newest index part end in a whole line again.

        A crash can leave a partial final line. If what is there is a whole canonical record that
        lost only its newline it is completed; anything else is dropped, because appending after it
        would splice the next record onto half of one.
        """
        parts = self.index_parts()
        if not parts:
            return
        path = self.index_part_path(parts[-1])
        data = path.read_bytes()
        if not data or data.endswith(b"\n"):
            return
        kept, _, tail = data.rpartition(b"\n")
        if self._is_whole_record(tail):
            log.warning("%s: completing a final line that lost its newline", path)
            with open(path, "ab") as fh:
                fh.write(b"\n")
                fh.flush()
                os.fsync(fh.fileno())
            return
        log.warning("%s: dropping %d bytes of a torn final line", path, len(tail))
        with open(path, "r+b") as fh:
            fh.truncate(len(kept) + 1 if kept else 0)
            fh.flush()
            os.fsync(fh.fileno())

    def last_record(self) -> dict[str, Any] | None:
        """The last whole line of the newest index part, as the index itself has it."""
        return last_record(self.root)

    def next_seq(self) -> int:
        """One past the index's last record. `head.json` is a cache and never decides this."""
        last = self.last_record()
        return int(last["seq"]) + 1 if last else 1

    def part_of(self, seq: int) -> int:
        return (seq - 1) // LINES_PER_PART

    def append(self, record: dict[str, Any]) -> int:
        """Assign the next seq, append the record to the index and advance the head. The event
        is written first (`write_event`)."""
        if not self.event_path(str(record["event_id"])).is_file():
            raise RuntimeError(f"write the event before its record: {record['event_id']}")
        record = dict(record)
        with self.index_lock():
            self.repair_tail()
            seq = self.next_seq()
            record["seq"] = seq
            line = canonical_json(record) + "\n"
            path = self.index_part_path(self.part_of(seq))
            path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
            try:
                os.write(fd, line.encode("utf-8"))
                os.fsync(fd)
            finally:
                os.close(fd)
            self._touch(path)
            head = self.head()
            king = king_after(record, head.get("king") if head else None)
            self.write_head(seq=seq, block=int(record["block"]), king=king)
        return seq

    def rebuild_head(self) -> dict[str, Any] | None:
        """Rewrite `head.json` from the index, for a head a kill between the index line and the
        head's rewrite left behind; None for an empty index (left as is)."""
        with self.index_lock():
            self.repair_tail()
            king: dict | None = None
            last: dict[str, Any] | None = None
            for record in self.iter_index():
                king = king_after(record, king)
                last = record
            if last is None:
                return None
            self.write_head(seq=int(last["seq"]), block=int(last["block"]), king=king)
        return self.head()

    def iter_index(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        part = 0
        while True:
            path = self.index_part_path(part)
            if not path.exists():
                break
            for raw in path.read_text(encoding="utf-8").split("\n"):
                if not raw.strip():
                    continue
                try:
                    out.append(json.loads(raw))
                except ValueError:
                    continue
            part += 1
        out.sort(key=lambda r: r.get("seq", 0))
        return out

    # ---------------------------------------------------------------- events / queue / boards / media
    def write_event(self, event: dict[str, Any]) -> Path:
        path = self.event_path(str(event["event_id"]))
        atomic_write_json(path, event)
        self._touch(path)
        return path

    def event(self, event_id: str) -> dict[str, Any] | None:
        return read_json(self.event_path(event_id))

    def write_queue(self, snapshot: dict[str, Any]) -> None:
        atomic_write_json(self.queue_path(), snapshot)
        self._touch(self.queue_path())

    def write_running(self, doc: dict[str, Any]) -> None:
        atomic_write_json(self.running_path(), doc)
        self._touch(self.running_path())

    def write_champions(self, doc: dict[str, Any]) -> None:
        atomic_write_json(self.champions_path(), doc)
        self._touch(self.champions_path())

    def write_validator(self, lane: str, doc: dict[str, Any]) -> None:
        """The driving validator's state for `lane`, kept with the store and mirrored with it."""
        atomic_write_json(self.validator_path(lane), doc)
        self._touch(self.validator_path(lane))

    def put_media(self, src: str | Path) -> str:
        src = Path(src)
        sha = sha256_file(src)
        dst = self.media_path(sha)
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_name(dst.name + ".tmp")
            shutil.copyfile(src, tmp)
            os.replace(tmp, dst)
        self._touch(dst)
        return sha

    def has_media(self, sha: str) -> bool:
        return self.media_path(sha).exists()

    def drain_touched(self) -> list[str]:
        with self._touched_lock:
            out = sorted(self.touched)
            self.touched.clear()
        return out

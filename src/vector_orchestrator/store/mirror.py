"""Mirror the local store to a Hugging Face dataset repo, one commit per publish.

The dashboard reads a published store from that repo (`VECTOR_STORE=owner/name`), so the mirror is
what makes a result public. `huggingface_hub` is imported only when a mirror is made.

**The mirror is the store's history.** A validator started on another host - a new machine, a
rebuilt disk - begins with an empty store, and its first push would write a one-record index and
a kingless head over the mirror's. Two rules keep that from happening:

- `restore` takes the store up where the mirror has it before anything is written: the manifest,
  the head and the index parts (and any of the caller's own files named to it). Never the events
  or the clips - nothing reads them to go on publishing - so it costs kilobytes, not the store.
- every push first reads the mirror's `head.json` and refuses when the mirror counts more records
  than this store (`MirrorAhead`), or when the store's own index says something else happened at
  the mirror's head (`StoreDiverged`): a store never writes over a history it does not share.
"""

from __future__ import annotations

import json
import logging
import re
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .writer import Store, atomic_write_text, index_part_name, king_after, last_record

log = logging.getLogger(__name__)

#: The repository's own files rather than the store's: the Hub writes `.gitattributes`, and the
#: dataset card is written by hand. A rebuilt store must not take either of them with it.
REPO_OWNED = frozenset({".gitattributes", "README.md"})


#: Exactly the store's layout (`store/writer.py`). Nothing else is uploaded: the mirror makes a
#: dataset repository public, and a root that is not quite a store - an operator's working
#: directory - must not publish whatever else is in it. Dotfiles (the writer's locks) and
#: half-written `.tmp` files are the store's own bookkeeping and are not part of what a reader
#: reads.
STORE_LAYOUT = re.compile(
    r"(?:manifest|head|queue|running|champions)\.json"
    r"|index-\d{4,}\.jsonl"
    r"|events/[0-9a-f]{64}\.json"
    r"|media/[0-9a-f]{2}/[0-9a-f]{64}\.mp4"
    r"|validator/[a-z0-9_-]+\.json"
)


class MirrorUnreachable(RuntimeError):
    """The mirror could not be read (no connection, a server error, no access): nothing may be
    decided from it."""


class PushRefused(RuntimeError):
    """The store does not share the mirror's history, so nothing of it is pushed."""


class MirrorAhead(PushRefused):
    """The mirror holds records this store does not; pushing would write over them."""


class StoreDiverged(PushRefused):
    """The store and the mirror disagree about a record both hold, or the mirror is not whole: a
    human decides which is right."""


def local_seq(root: Path) -> int:
    """How many records the store's own index holds."""
    record = last_record(root)
    return int(record["seq"]) if record else 0


def head_at(root: Path, seq: int) -> dict[str, Any] | None:
    """The `head.json` the store's own index gives after record `seq`; None when it has no such
    record."""
    root = Path(root)
    king: dict | None = None
    last: dict[str, Any] | None = None
    part = 0
    while (root / index_part_name(part)).exists():
        for line in (root / index_part_name(part)).read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if not isinstance(record, dict) or not isinstance(record.get("seq"), int):
                continue
            if record["seq"] > seq:
                break
            king = king_after(record, king)
            last = record
        part += 1
    if last is None or last["seq"] != seq:
        return None
    return {"seq": seq, "block": int(last["block"]), "king": king}


def compare(root: Path, head: dict[str, Any] | None, repo: str) -> None:
    """Refuse a store that does not share the mirror's history up to the mirror's head: one
    behind it (`MirrorAhead`), or one whose own index gives another head there (`StoreDiverged`)."""
    theirs = int((head or {}).get("seq") or 0)
    ours = local_seq(root)
    if theirs > ours:
        raise MirrorAhead(
            f"{repo} holds {theirs} records and {root} only {ours}: "
            "nothing is pushed until the store is restored from it"
        )
    if theirs and head_at(root, theirs) != {
        k: (head or {}).get(k) for k in ("seq", "block", "king")
    }:
        raise StoreDiverged(
            f"{root} and {repo} disagree about record #{theirs}: the two stores published "
            "different records"
        )


def store_files(root: Path) -> list[str]:
    """Every file of a store that belongs in the mirror, relative to its root."""
    root = Path(root)
    return sorted(
        str(p.relative_to(root))
        for p in root.rglob("*")
        if p.is_file() and STORE_LAYOUT.fullmatch(str(p.relative_to(root)))
    )


def check_is_store(root: Path) -> list[str]:
    """The store's files, or `ValueError` if `root` is not a store. Anything else it holds is
    logged and left behind."""
    root = Path(root)
    if not (root / "manifest.json").is_file():
        raise ValueError(f"{root} holds no manifest.json; it is not a store")
    files = store_files(root)
    known = set(files)
    beside = sorted(
        str(p.relative_to(root))
        for p in root.rglob("*")
        if p.is_file()
        and str(p.relative_to(root)) not in known
        and not any(part.startswith(".") for part in p.relative_to(root).parts)
    )
    if beside:
        log.warning("not mirroring %d file(s) that are not the store's: %s", len(beside), beside)
    return files


class Mirror:
    def __init__(self, root: Path, repo: str, *, token: str | None = None, private: bool = False):
        from huggingface_hub import HfApi

        self.root = Path(root)
        self.repo = repo
        self.api = HfApi(token=token)
        self.api.create_repo(repo, repo_type="dataset", exist_ok=True, private=private)

    def read(self, path: str) -> bytes | None:
        """One file of the repository as it is now; None when the repository has no such file.
        Anything else that stops it being read is `MirrorUnreachable`."""
        from huggingface_hub.errors import RemoteEntryNotFoundError

        with tempfile.TemporaryDirectory() as cache:
            try:
                got = self.api.hf_hub_download(
                    self.repo, path, repo_type="dataset", cache_dir=cache
                )
            except RemoteEntryNotFoundError:
                return None
            except Exception as exc:  # noqa: BLE001 - whatever it is, the mirror was not read
                raise MirrorUnreachable(f"{self.repo}: {path}: {exc}") from exc
            return Path(got).read_bytes()

    def head(self) -> dict[str, Any] | None:
        """The repository's `head.json`; None for a repository nothing was published to yet."""
        data = self.read("head.json")
        if data is None:
            return None
        try:
            head = json.loads(data)
        except ValueError as exc:
            raise StoreDiverged(f"{self.repo}: head.json is not JSON") from exc
        if not isinstance(head, dict):
            raise StoreDiverged(f"{self.repo}: head.json is not an object")
        return head

    def check_not_behind(self) -> None:
        """Refuse a push from a store that does not share the repository's history (`compare`)."""
        compare(self.root, self.head(), self.repo)

    def push(self, files: list[str], message: str = "publish") -> str | None:
        """Add exactly `files` (those that exist) in one commit: what one publish touched. Refused
        when the repository is ahead of the store (`check_not_behind`)."""
        from huggingface_hub import CommitOperationAdd

        ops = [
            CommitOperationAdd(path_in_repo=f, path_or_fileobj=str(self.root / f))
            for f in sorted(set(files))
            if (self.root / f).exists()
        ]
        if not ops:
            return None
        self.check_not_behind()
        info = self.api.create_commit(
            repo_id=self.repo, repo_type="dataset", operations=ops, commit_message=message
        )
        log.info("mirrored %d files to %s", len(ops), self.repo)
        return str(getattr(info, "oid", ""))

    def push_all(self, message: str = "full mirror") -> str | None:
        """Every file of the store's layout, in one commit. Only ever adds."""
        return self.push(check_is_store(self.root), message)

    def replace_all(self, message: str = "replace the store") -> str:
        """One commit that makes the repo exactly the local store: every file added, every path the
        store no longer has deleted.

        `push_all` only ever adds, which is right while a store grows and wrong when it is rebuilt:
        the previous layout's records and clips would linger beside the new ones under names nothing
        references. Deleting and adding in one commit means a reader never sees the two mixed, and
        never sees an empty store.
        """
        from huggingface_hub import CommitOperationAdd, CommitOperationDelete

        local = check_is_store(self.root)
        if not local:
            raise ValueError(f"{self.root} holds no files; refusing to empty {self.repo}")
        missing = missing_events(self.root)
        if missing:
            raise ValueError(
                f"{self.root} lacks {len(missing)} event(s) its index names (a store restored "
                f"from the mirror holds no events): refusing to delete them from {self.repo}"
            )
        self.check_not_behind()
        info = self.api.repo_info(self.repo, repo_type="dataset", files_metadata=False)
        remote = {s.rfilename for s in (info.siblings or [])}
        stale = sorted(remote - set(local) - REPO_OWNED)
        ops: list[Any] = [CommitOperationDelete(path_in_repo=f) for f in stale]
        ops += [
            CommitOperationAdd(path_in_repo=f, path_or_fileobj=str(self.root / f)) for f in local
        ]
        commit = self.api.create_commit(
            repo_id=self.repo, repo_type="dataset", operations=ops, commit_message=message
        )
        log.info("replaced %s: %d files, %d stale paths removed", self.repo, len(local), len(stale))
        return str(getattr(commit, "oid", ""))


def missing_events(root: Path) -> list[str]:
    """The events the store's index names that the store does not hold."""
    root = Path(root)
    missing = []
    part = 0
    while (root / index_part_name(part)).exists():
        for line in (root / index_part_name(part)).read_text(encoding="utf-8").splitlines():
            try:
                event_id = json.loads(line).get("event_id")
            except (ValueError, AttributeError):
                continue
            if event_id and not (root / "events" / f"{event_id}.json").is_file():
                missing.append(str(event_id))
        part += 1
    return missing


def restore(store: Store, mirror: Mirror, extra: Iterable[str] = ()) -> bool:
    """Take the store up where the mirror has it; whether anything was taken.

    When the mirror counts more records than the store, its manifest, head and index parts are
    written into the store - after checking that every record the store already holds is the
    mirror's too (`StoreDiverged` if not). A store level with the mirror or ahead of it keeps what
    it holds, once its index gives the mirror's head at the mirror's seq (`compare`). `extra`
    names the caller's own files in the store (`validator/<lane>.json`): each is taken when the
    index was, or when the store has none.

    Raises `MirrorUnreachable` when the mirror cannot be read: the caller decides whether what it
    holds is enough to go on with. Holds the index lock, so no record is appended meanwhile.
    """
    with store.index_lock():
        store.repair_tail()
        head = mirror.head()
        theirs = int((head or {}).get("seq") or 0)
        ours = local_seq(store.root)
        taken = False
        if theirs <= ours:
            compare(store.root, head, mirror.repo)
        else:
            parts: dict[int, bytes] = {}
            for part in range(store.part_of(theirs) + 1):
                data = mirror.read(index_part_name(part))
                if data is None:
                    raise StoreDiverged(
                        f"{mirror.repo} counts {theirs} records but has no {index_part_name(part)}"
                    )
                parts[part] = data
            for part in store.index_parts():
                if not parts.get(part, b"").startswith(store.index_part_path(part).read_bytes()):
                    raise StoreDiverged(
                        f"{store.index_part_path(part)} is not the start of {mirror.repo}'s: the "
                        "two stores published different records"
                    )
            manifest = mirror.read("manifest.json")
            head = mirror.read("head.json")
            if manifest is None or head is None:
                raise StoreDiverged(f"{mirror.repo} has an index but no manifest.json or head.json")
            atomic_write_text(store.root / "manifest.json", manifest.decode("utf-8"))
            for part, data in parts.items():
                atomic_write_text(store.index_part_path(part), data.decode("utf-8"))
            atomic_write_text(store.head_path(), head.decode("utf-8"))
            log.info(
                "restored %s from %s: %d records (was %d)", store.root, mirror.repo, theirs, ours
            )
            taken = True
        for name in extra:
            path = store.root / name
            if taken or not path.is_file():
                data = mirror.read(name)
                if data is not None:
                    atomic_write_text(path, data.decode("utf-8"))
        return taken


def mirror_store(
    root: str | Path,
    repo: str,
    *,
    message: str = "publish",
    prune: bool = False,
    files: list[str] | None = None,
    token: str | None = None,
) -> int:
    """Mirror `root` to `repo`; return how many files the commit covered.

    `files` (what one publish touched) is one commit of exactly those; without it the whole store
    is uploaded, and `prune` makes the repository exactly the store.
    """
    root = Path(root)
    check_is_store(root)
    m = Mirror(root, repo, token=token)
    if prune:
        m.replace_all(message)
        return len(store_files(root))
    if files is None:
        m.push_all(message)
        return len(store_files(root))
    m.push(files, message)
    return len(files)

"""Mirroring to a Hugging Face dataset, against a stand-in `HfApi`: no network in the pure tests."""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import huggingface_hub
import pytest
from huggingface_hub.errors import RemoteEntryNotFoundError

from vector_orchestrator.store import mirror
from vector_orchestrator.store.writer import Store, store_lock


class FakeApi:
    remote: set[str] = set()
    commits: list[tuple[str, list]] = []
    #: What the repository holds, by path, for `hf_hub_download`; and what it raises instead.
    files: dict[str, bytes] = {}
    fail: Exception | None = None
    reads: list[str] = []
    #: What `create_repo` raises, and how often it was asked.
    create_fail: Exception | None = None
    creates: int = 0

    def __init__(self, token=None):
        self.token = token

    def create_repo(self, repo, **kw):
        FakeApi.creates += 1
        if FakeApi.create_fail is not None:
            raise FakeApi.create_fail

    def repo_info(self, repo, **kw):
        return SimpleNamespace(siblings=[SimpleNamespace(rfilename=f) for f in FakeApi.remote])

    def create_commit(self, *, repo_id, repo_type, operations, commit_message):
        FakeApi.commits.append((commit_message, operations))
        return SimpleNamespace(oid="c0ffee")

    def hf_hub_download(self, repo_id, filename, *, repo_type, cache_dir):
        FakeApi.reads.append(filename)
        if FakeApi.fail is not None:
            raise FakeApi.fail
        if filename not in FakeApi.files:
            raise RemoteEntryNotFoundError(
                "404", response=SimpleNamespace(headers={}, status_code=404, request=None)
            )
        path = Path(cache_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(FakeApi.files[filename])
        return str(path)


@pytest.fixture
def api(monkeypatch):
    FakeApi.remote, FakeApi.commits, FakeApi.files, FakeApi.reads = set(), [], {}, []
    FakeApi.fail, FakeApi.create_fail, FakeApi.creates = None, None, 0
    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    return FakeApi


@pytest.fixture
def store(spec, tmp_path):
    s = Store(tmp_path / "store", spec)
    s.init()
    with store_lock(s.root):
        pass
    (s.root / "queue.json.tmp").write_text("{")
    return s


def test_mirror_lists_only_the_store_files(store):
    # Everything a wrong root might hold beside a store, all of it published if it were uploaded.
    (store.root / "runs").mkdir()
    (store.root / "runs" / "duel.log").write_text("a run log\n")
    (store.root / "notes.md").write_text("scratch")
    (store.root / "events").mkdir(parents=True)
    (
        store.root
        / "events"
        / "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee.json"
    ).write_text("{}")
    (store.root / "events" / "draft.txt").write_text("no")

    files = mirror.store_files(store.root)
    assert files == [
        "events/eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee.json",
        "head.json",
        "manifest.json",
    ]
    assert (store.root / ".orchestrator.lock").exists()


def test_a_root_that_is_not_a_store_is_refused(tmp_path, api):
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs" / "duel.log").write_text("a run log\n")
    with pytest.raises(ValueError, match="holds no manifest.json"):
        mirror.mirror_store(tmp_path, "org/store")
    assert api.commits == []


def test_prune_deletes_stale_paths_but_keeps_the_repository_own_files(store, api):
    api.remote = {"README.md", ".gitattributes", "manifest.json", "events/t/old.json"}
    assert mirror.mirror_store(store.root, "org/store", prune=True, message="rebuild") == 2
    ((message, ops),) = api.commits
    deleted = sorted(op.path_in_repo for op in ops if type(op).__name__ == "CommitOperationDelete")
    added = sorted(op.path_in_repo for op in ops if type(op).__name__ == "CommitOperationAdd")
    assert message == "rebuild"
    assert deleted == ["events/t/old.json"]
    assert added == ["head.json", "manifest.json"]


def test_a_full_mirror_uploads_the_store_and_nothing_beside_it(store, api):
    (store.root / "runs").mkdir()
    (store.root / "runs" / "duel.log").write_text("a run log\n")
    assert mirror.mirror_store(store.root, "org/store") == 2
    ((_, ops),) = api.commits
    assert sorted(op.path_in_repo for op in ops) == ["head.json", "manifest.json"]


def test_push_adds_exactly_what_a_publish_touched(store, api):
    assert mirror.mirror_store(store.root, "org/store", files=store.drain_touched()) == 2
    ((_, ops),) = api.commits
    assert sorted(op.path_in_repo for op in ops) == ["head.json", "manifest.json"]


def test_an_empty_directory_never_empties_the_repository(tmp_path, api):
    """--prune deletes every remote path the local store lacks; a wrong root must not reach it."""
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="holds no manifest.json"):
        mirror.mirror_store(empty, "org/store", prune=True)
    assert api.commits == []


def test_the_cli_mirrors_a_store(store, api, capsys):
    from vector_orchestrator.cli import main

    assert (
        main(["store", "mirror", "--root", str(store.root), "--repo", "org/store", "--prune"]) == 0
    )
    assert capsys.readouterr().out.strip() == "mirrored 2 files to org/store"
    assert len(api.commits) == 1


# -- a store behind its mirror is never pushed over it, and is taken up from it -----------------


def publish(store, n, *, start=1):
    """Append `n` records (with their events) to `store`, numbered from `start`."""
    for i in range(start, start + n):
        event_id = f"{i:064x}"
        store.write_event({"event_id": event_id})
        store.append({"event_id": event_id, "block": i, "kind": "duel", "dethroned": False})


def mirrored(store):
    """What a mirror of `store` holds: every file of its layout, by path."""
    return {f: (store.root / f).read_bytes() for f in mirror.store_files(store.root)}


@pytest.fixture
def published(spec, tmp_path):
    """A store of three records, as the host that ran them left it."""
    s = Store(tmp_path / "published", spec)
    s.init()
    publish(s, 3)
    (s.root / "validator").mkdir()
    (s.root / "validator" / "vector.json").write_text('{"entries": {}}\n')
    return s


def test_a_push_from_a_store_behind_the_mirror_is_refused(store, published, api):
    api.files = mirrored(published)
    with pytest.raises(mirror.MirrorAhead, match="holds 3 records"):
        mirror.mirror_store(store.root, "org/store", files=store.drain_touched())
    assert api.commits == []


def test_a_push_to_an_empty_mirror_or_one_not_ahead_goes_through(store, published, api):
    mirror.mirror_store(store.root, "org/store", files=["head.json"])
    api.files = mirrored(published)
    mirror.mirror_store(published.root, "org/store", files=["head.json"])
    assert len(api.commits) == 2


def test_a_prune_is_refused_when_the_mirror_is_ahead(store, published, api):
    api.files = mirrored(published)
    with pytest.raises(mirror.MirrorAhead):
        mirror.mirror_store(store.root, "org/store", prune=True)
    assert api.commits == []


def test_an_empty_store_is_restored_from_the_mirror_without_events_or_clips(
    spec, tmp_path, published, api
):
    api.files = mirrored(published)
    fresh = Store(tmp_path / "fresh", spec)
    m = mirror.Mirror(fresh.root, "org/store")

    assert mirror.restore(fresh, m, extra=["validator/vector.json"]) is True
    assert fresh.manifest() == published.manifest()
    assert fresh.head() == published.head()
    assert [r["seq"] for r in fresh.iter_index()] == [1, 2, 3]
    assert fresh.next_seq() == 4
    assert (fresh.root / "validator" / "vector.json").read_text() == '{"entries": {}}\n'
    assert not (fresh.root / "events").exists()
    assert not any(name.startswith(("events/", "media/")) for name in api.reads)


def test_a_restored_store_publishes_after_the_mirror_records(spec, tmp_path, published, api):
    api.files = mirrored(published)
    fresh = Store(tmp_path / "fresh", spec)
    mirror.restore(fresh, mirror.Mirror(fresh.root, "org/store"))
    publish(fresh, 1, start=4)
    assert fresh.head()["seq"] == 4
    lines = fresh.index_part_path(0).read_bytes()
    assert lines.startswith(published.index_part_path(0).read_bytes())


def test_a_store_behind_the_mirror_takes_up_the_rest(spec, tmp_path, published, api):
    behind = Store(tmp_path / "behind", spec)
    behind.init()
    publish(behind, 2)
    api.files = mirrored(published)
    assert mirror.restore(behind, mirror.Mirror(behind.root, "org/store")) is True
    assert behind.next_seq() == 4


def test_a_store_that_published_other_records_is_not_overwritten(spec, tmp_path, published, api):
    other = Store(tmp_path / "other", spec)
    other.init()
    other.write_event({"event_id": "f" * 64})
    other.append({"event_id": "f" * 64, "block": 9, "kind": "duel", "dethroned": False})
    before = other.index_part_path(0).read_bytes()
    api.files = mirrored(published)
    with pytest.raises(mirror.StoreDiverged, match="published different records"):
        mirror.restore(other, mirror.Mirror(other.root, "org/store"))
    assert other.index_part_path(0).read_bytes() == before


def test_a_store_level_with_the_mirror_reads_only_its_head(spec, published, api):
    api.files = mirrored(published)
    assert mirror.restore(published, mirror.Mirror(published.root, "org/store")) is False
    assert api.reads == ["head.json"]


def test_a_missing_file_of_its_own_is_taken_even_when_the_index_is_level(published, api):
    api.files = mirrored(published)
    (published.root / "validator" / "vector.json").unlink()
    m = mirror.Mirror(published.root, "org/store")
    assert mirror.restore(published, m, extra=["validator/vector.json"]) is False
    assert (published.root / "validator" / "vector.json").is_file()


def test_a_mirror_that_cannot_be_read_restores_nothing(spec, tmp_path, api):
    api.fail = OSError("no route to host")
    fresh = Store(tmp_path / "fresh", spec)
    with pytest.raises(mirror.MirrorUnreachable, match="no route to host"):
        mirror.restore(fresh, mirror.Mirror(fresh.root, "org/store"))
    assert fresh.manifest() is None


def test_a_mirror_whose_head_counts_records_it_lacks_is_not_taken(spec, tmp_path, published, api):
    api.files = {k: v for k, v in mirrored(published).items() if not k.startswith("index-")}
    fresh = Store(tmp_path / "fresh", spec)
    with pytest.raises(mirror.StoreDiverged, match="no index-0000.jsonl"):
        mirror.restore(fresh, mirror.Mirror(fresh.root, "org/store"))
    assert fresh.manifest() is None


def test_a_restored_store_is_never_pruned_into_the_mirror(spec, tmp_path, published, api):
    """It holds no events: a prune would delete every one of them from the mirror."""
    api.files = mirrored(published)
    fresh = Store(tmp_path / "fresh", spec)
    mirror.restore(fresh, mirror.Mirror(fresh.root, "org/store"))
    with pytest.raises(ValueError, match="lacks 3 event"):
        mirror.mirror_store(fresh.root, "org/store", prune=True)
    assert api.commits == []


def test_the_validator_state_is_part_of_the_mirrored_layout(published):
    assert "validator/vector.json" in mirror.store_files(published.root)


def test_a_refused_push_keeps_its_files_for_later_and_says_so(spec, store, published, api, caplog):
    from vector_orchestrator.duel.orchestrate import Orchestrator

    api.files = mirrored(published)
    orch = Orchestrator(
        spec, store, None, store.root / "runs", mirror=mirror.Mirror(store.root, "o/s")
    )
    with caplog.at_level(logging.ERROR):
        orch.push_touched()
    assert "mirror push refused" in caplog.text
    assert {"head.json", "manifest.json"} <= store.touched
    assert api.commits == []


def test_a_store_level_with_the_mirror_but_with_another_history_is_refused(spec, tmp_path, api):
    """Two hosts that each published a first record: the counts agree, the records do not."""
    ours = Store(tmp_path / "ours", spec)
    ours.init()
    ours.write_event({"event_id": "a" * 64})
    ours.append({"event_id": "a" * 64, "block": 1, "kind": "genesis", "king": {"key": "k1"}})
    theirs = Store(tmp_path / "theirs", spec)
    theirs.init()
    theirs.write_event({"event_id": "b" * 64})
    theirs.append({"event_id": "b" * 64, "block": 1, "kind": "genesis", "king": {"key": "k2"}})
    api.files = mirrored(theirs)

    with pytest.raises(mirror.StoreDiverged, match="disagree about record #1"):
        mirror.mirror_store(ours.root, "org/store", files=["queue.json", "head.json"])
    with pytest.raises(mirror.StoreDiverged):
        mirror.restore(ours, mirror.Mirror(ours.root, "org/store"))
    assert api.commits == []


def test_a_store_ahead_of_the_mirror_on_the_same_history_pushes(spec, published, tmp_path, api):
    behind = Store(tmp_path / "behind", spec)
    behind.init()
    publish(behind, 2)
    api.files = mirrored(behind)
    mirror.mirror_store(published.root, "org/store", files=["head.json"])
    assert mirror.restore(published, mirror.Mirror(published.root, "org/store")) is False
    assert len(api.commits) == 1


def test_a_mirror_whose_index_is_short_of_its_head_is_not_taken(spec, tmp_path, published, api):
    """A host killed between pushing its head and its index left the mirror so: not whole."""
    api.files = mirrored(published)
    lines = published.index_part_path(0).read_bytes().splitlines(keepends=True)
    api.files["index-0000.jsonl"] = b"".join(lines[:2])
    fresh = Store(tmp_path / "fresh", spec)
    with pytest.raises(mirror.StoreDiverged, match="holds 2: the mirror is not whole"):
        mirror.restore(fresh, mirror.Mirror(fresh.root, "org/store"))
    assert fresh.manifest() is None and fresh.index_parts() == []


def test_a_mirror_head_its_own_index_does_not_give_is_not_taken(spec, tmp_path, published, api):
    """Not a second read of the mirror's head, which a push between the reads could move on."""
    api.files = mirrored(published)
    api.files["head.json"] = b'{"seq": 3, "block": 3, "king": {"key": "stale"}}'
    fresh = Store(tmp_path / "fresh", spec)
    with pytest.raises(mirror.StoreDiverged, match="not what its own index gives"):
        mirror.restore(fresh, mirror.Mirror(fresh.root, "org/store"))
    assert fresh.manifest() is None and fresh.index_parts() == []
    api.files["head.json"] = (published.root / "head.json").read_bytes()
    mirror.restore(fresh, mirror.Mirror(fresh.root, "org/store"))
    assert fresh.head() == published.head()


def test_a_hub_down_when_the_mirror_is_made_is_tried_again_at_the_push(store, api):
    api.create_fail = OSError("no route to host")
    m = mirror.Mirror(store.root, "org/store")  # does not raise: a store may go on without it
    with pytest.raises(OSError):
        m.push(["head.json"])
    api.create_fail = None
    m.push(["head.json"])
    m.push(["head.json"])
    assert api.creates == 3 and len(api.commits) == 2


def test_the_cli_says_why_a_push_was_refused(store, published, api, capsys):
    from vector_orchestrator.cli import main

    api.files = mirrored(published)
    assert main(["store", "mirror", "--root", str(store.root), "--repo", "org/store"]) == 2
    assert "holds 3 records" in capsys.readouterr().err

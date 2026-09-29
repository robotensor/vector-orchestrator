"""`vector-orchestrator store init` and `queue add|list|remove`, through the console script."""

from __future__ import annotations

import json
import subprocess
import sysconfig
from pathlib import Path

import pytest

from store_helpers import TRACK, check_store
from submission_helpers import SHA_A, FakeHub
from vector_orchestrator.canon import Signer
from vector_orchestrator.cli import main
from vector_orchestrator.ids import SubmissionRef
from vector_orchestrator.queue import Queue

SHA_1, SHA_2 = "1" * 40, "2" * 40


@pytest.fixture
def hub(monkeypatch):
    """The Hub `queue add` asks, with org/policy at SHA_1 and org/other at SHA_2."""
    fake = FakeHub()
    fake.add("org/policy", SHA_1, {"policy.yaml": 80})
    fake.add("org/other", SHA_2, {"policy.yaml": 80})
    monkeypatch.setattr("huggingface_hub.HfApi", lambda: fake)
    return fake


def cli(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess:
    script = Path(sysconfig.get_path("scripts")) / "vector-orchestrator"
    return subprocess.run([str(script), *args], capture_output=True, text=True, cwd=cwd)


def test_store_init_writes_the_manifest_and_a_head_per_track(tmp_path):
    root = tmp_path / "store"
    init = cli("store", "init", str(root), cwd=tmp_path)
    assert init.returncode == 0, init.stderr
    manifest = json.loads(init.stdout)
    key = tmp_path / "keys" / "orchestrator.ed25519"
    assert "generated signing key" in init.stderr and key.stat().st_mode & 0o777 == 0o600
    assert manifest["validator_key"] == Signer.from_file(key).verify_key_hex
    assert (root / "tracks" / TRACK / "head.json").exists()


def test_init_refuses_to_resign_a_store_with_another_key(tmp_path):
    root = tmp_path / "store"
    assert cli("store", "init", str(root), "--key", str(tmp_path / "a")).returncode == 0
    again = cli("store", "init", str(root), "--key", str(tmp_path / "a"))
    assert again.returncode == 0, "re-initialising with the same key is idempotent"
    assert (
        cli("store", "init", str(tmp_path / "other"), "--key", str(tmp_path / "b")).returncode == 0
    )
    other = cli("store", "init", str(root), "--key", str(tmp_path / "b"))
    assert other.returncode == 1 and "would invalidate every record" in other.stderr


def test_init_on_a_store_whose_key_is_missing_generates_nothing(tmp_path):
    """A key generated before the store was looked at is left lying in keys/, where the next run
    picks it up as the store's key."""
    root = tmp_path / "store"
    assert cli("store", "init", str(root), "--key", str(tmp_path / "a")).returncode == 0
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    done = cli("store", "init", str(root), cwd=elsewhere)
    assert done.returncode == 1 and "generated signing key" not in done.stderr
    assert "cannot be re-initialised" in done.stderr
    assert not (elsewhere / "keys").exists()


def test_init_refuses_a_key_inside_the_store(tmp_path):
    root = tmp_path / "store"
    done = cli("store", "init", str(root), "--key", str(root / "keys" / "k"))
    assert done.returncode == 2 and "outside the store" in done.stderr
    assert not (root / "keys").exists()


def test_queue_add_list_remove_and_publish_the_snapshot(spec, hub, tmp_path, capsys):
    queue, root = tmp_path / "queue", tmp_path / "store"
    assert main(["store", "init", str(root), "--key", str(tmp_path / "k")]) == 0
    capsys.readouterr()
    base = ["queue", "--queue", str(queue), "--store", str(root)]

    assert main([*base, "add", "org/policy", "1" * 40, "--duel-size", "smoke"]) == 0
    added = capsys.readouterr().out
    key = SubmissionRef.make("org/policy", "1" * 40).key
    assert added.strip() == f"org/policy@{'1' * 40} key={key} track={TRACK} position=1"
    assert main([*base, "add", "org/other", "2" * 40]) == 0
    capsys.readouterr()

    assert main([*base, "list"]) == 0
    listing = capsys.readouterr().out.splitlines()
    assert listing[0].split()[:3] == ["1", key, f"org/policy@{'1' * 40}"]
    assert listing[-1] == "in_progress=- block=0"

    snapshot = json.loads((root / "tracks" / TRACK / "queue.json").read_text())
    assert [e["repo"] for e in snapshot["entries"]] == ["org/policy", "org/other"]

    assert main([*base, "remove", key]) == 0
    assert main([*base, "remove", key]) == 1
    capsys.readouterr()
    snapshot = json.loads((root / "tracks" / TRACK / "queue.json").read_text())
    assert [e["repo"] for e in snapshot["entries"]] == ["org/other"]
    assert check_store(root, spec).ok


def test_queue_add_waits_for_no_one_while_a_duel_holds_the_store(spec, hub, tmp_path, capsys):
    from vector_orchestrator.store.writer import store_lock

    queue, root = tmp_path / "queue", tmp_path / "store"
    assert main(["store", "init", str(root), "--key", str(tmp_path / "k")]) == 0
    capsys.readouterr()
    base = ["queue", "--queue", str(queue), "--store", str(root)]
    with store_lock(root):
        assert main([*base, "add", "org/policy", "1" * 40]) == 0
    out = capsys.readouterr()
    assert "it will publish the queue snapshot" in out.err
    assert not (root / "tracks" / TRACK / "queue.json").exists()
    assert main([*base, "list"]) == 0
    assert "org/policy" in capsys.readouterr().out, "the entry was still queued"


def test_queue_add_on_a_corrupt_queue_file_says_so(hub, tmp_path, capsys):
    queue = tmp_path / "queue"
    queue.mkdir()
    (queue / f"{TRACK}.json").write_text('{"entries": [], "block": 3')
    assert main(["queue", "--queue", str(queue), "add", "org/policy", "1" * 40]) == 2
    assert "is not a readable queue file" in capsys.readouterr().err


def test_queue_add_resolves_a_branch_or_tag_to_its_commit_once(tmp_path, capsys, monkeypatch):
    hub = FakeHub()
    hub.add("org/policy", SHA_A, {"policy.yaml": 80}, "main", "v1")
    monkeypatch.setattr("huggingface_hub.HfApi", lambda: hub)
    base = ["queue", "--queue", str(tmp_path / "queue")]
    assert main([*base, "add", "org/policy", "main"]) == 0
    out = capsys.readouterr()
    assert out.err.strip() == f"resolved org/policy@main to {SHA_A}"
    key = SubmissionRef.make("org/policy", SHA_A).key
    assert out.out.strip() == f"org/policy@{SHA_A} key={key} track={TRACK} position=1"
    (entry,) = Queue(tmp_path / "queue" / f"{TRACK}.json").entries()
    assert entry.revision == SHA_A, "the queue holds the commit, not the name"
    # The tag names the same commit: the same key, moved to the back, not queued twice.
    assert main([*base, "add", "org/policy", "v1"]) == 0
    assert [e.key for e in Queue(tmp_path / "queue" / f"{TRACK}.json").entries()] == [key]
    # A sha is confirmed on the Hub and queued as given, with nothing to say about resolving it.
    hub.add("org/other", "2" * 40, {"policy.yaml": 80})
    capsys.readouterr()
    assert main([*base, "add", "org/other", "2" * 40]) == 0
    assert capsys.readouterr().err == ""
    assert hub.calls == [("org/policy", "main"), ("org/policy", "v1"), ("org/other", "2" * 40)]


def test_queue_add_refuses_a_commit_only_a_pull_request_holds(tmp_path, capsys, monkeypatch):
    """Anyone on the Hub can open a pull request on a public repository, and the Hub serves its
    commit by sha: queued, it would be duelled and published as the owner's code."""
    hub = FakeHub()
    hub.add("org/policy", SHA_A, {"policy.yaml": 80}, "main")
    hub.add("org/policy", SHA_2, {"policy.yaml": 80}, pr=1, parent=SHA_A)
    monkeypatch.setattr("huggingface_hub.HfApi", lambda: hub)
    base = ["queue", "--queue", str(tmp_path / "queue")]
    assert main([*base, "add", "org/policy", SHA_2]) == 2
    assert main([*base, "add", "org/policy", "refs/pr/1"]) == 2
    err = capsys.readouterr().err
    assert f"org/policy@{SHA_2}: no branch or tag of the repository holds this commit" in err
    assert "org/policy@refs/pr/1: a ref under refs/, a pull request's included, is not" in err
    assert Queue(tmp_path / "queue" / f"{TRACK}.json").entries() == []
    assert main([*base, "add", "org/policy", SHA_A]) == 0


def test_queue_add_refuses_what_it_cannot_queue(tmp_path, capsys, monkeypatch):
    hub = FakeHub()
    hub.add("org/policy", SHA_A, {"policy.yaml": 80}, "main")
    monkeypatch.setattr("huggingface_hub.HfApi", lambda: hub)
    base = ["queue", "--queue", str(tmp_path / "queue")]
    assert main([*base, "add", "not a repo", "1" * 40]) == 2
    assert main([*base, "add", "org/policy", "1" * 40, "--duel-size", "enormous"]) == 2
    assert main(["queue", "--queue", str(tmp_path / "queue"), "--track", "video_only", "list"]) == 2
    # A name the Hub does not know, an abbreviation (another key for the same code) and a
    # repository that is not there are refused with the Hub's answer.
    assert main([*base, "add", "org/policy", "no-such-branch"]) == 2
    assert main([*base, "add", "org/policy", "a" * 7]) == 2
    assert main([*base, "add", "org/missing", "main"]) == 2
    # A sha is a commit of the repository or it is refused now, not when its duel comes.
    assert main([*base, "add", "org/policy", "0" * 40]) == 2
    err = capsys.readouterr().err
    assert "is not a Hugging Face repo id" in err and "is not one of smoke, light" in err
    assert "unknown track 'video_only'; the tracks are franka_1arm" in err
    assert err.count("revision not found: Invalid rev id: ") == 3
    assert f"org/policy@{'0' * 40}: revision not found" in err
    assert "org/missing@main: repository not found" in err and "Request ID" not in err
    assert hub.calls == [
        ("org/policy", "no-such-branch"),
        ("org/policy", "a" * 7),
        ("org/missing", "main"),
        ("org/policy", "0" * 40),
    ]
    assert (
        not list((tmp_path / "queue").glob("*.json"))
        or not Queue(tmp_path / "queue" / f"{TRACK}.json").entries()
    )

"""The policy runtime seam, through the subprocess runtime the pure suite duels with."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from duel_helpers import REPLAY, REPLAY_REF, ZERO, FakePolicyRuntime
from submission_helpers import write_policy_repo
from vector_orchestrator.duel.runtime import (
    POLICY,
    PolicyDied,
    RuntimeUnavailable,
    ServerPool,
    SubmissionRefused,
    SubprocessPolicyRuntime,
    mirror_log,
    tree_hash,
)


@pytest.fixture
def runtime(spec):
    return SubprocessPolicyRuntime(
        spec, {"org/replay": REPLAY, "org/zero": ZERO}, start_timeout_s=30
    )


def eventually(check, within_s: float = 5.0) -> bool:
    deadline = time.monotonic() + within_s
    while not check():
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.01)
    return True


def test_the_copy_of_a_policys_log_follows_its_end_and_never_a_link_or_a_pipe(tmp_path):
    live = tmp_path / "sockets" / "policy.log"
    live.parent.mkdir()
    live.write_text("listening on policy.sock\n")
    secret = tmp_path / "secret.txt"
    secret.write_text("the other side's results\n")
    copy = tmp_path / "unit" / "policy-tail.log"
    with mirror_log(live, copy, limit=64, interval_s=0.01) as mirrored:
        assert mirrored == copy and copy.read_text() == "listening on policy.sock\n"
        with open(live, "a") as fh:
            fh.write("x" * 100 + "\nact raised\n")
        assert eventually(lambda: copy.read_bytes().endswith(b"act raised\n"))
        assert len(copy.read_bytes()) == 64, "the copy is the log's end, not all of it"
        live.unlink()
        live.symlink_to(secret)
        time.sleep(0.1)
        assert copy.read_bytes().endswith(b"act raised\n"), "a link was followed"
        live.unlink()
        os.mkfifo(live)
        time.sleep(0.1)  # a pipe nobody writes to would block a reader that waited on it
        assert copy.read_bytes().endswith(b"act raised\n")
    assert not copy.exists(), "the copy outlived the unit"


def test_resolve_keeps_a_commit_and_pins_a_name_to_the_directory(runtime):
    assert runtime.resolve("org/replay", "a" * 40).revision == "a" * 40
    pinned = runtime.resolve("org/replay", "main")
    assert pinned.revision == tree_hash(REPLAY) and pinned.repo == "org/replay"
    with pytest.raises(RuntimeUnavailable, match="no local directory for org/other"):
        runtime.resolve("org/other", "main")


def test_prepare_checks_the_manifest_and_says_hello(runtime, tmp_path):
    ref = runtime.resolve("org/replay", "a" * 40)
    fetched = runtime.fetch(ref, workdir=tmp_path)
    assert fetched.commit == "a" * 40 and fetched.root == REPLAY.resolve()
    prepared = runtime.prepare(fetched, workdir=tmp_path / "check")
    assert prepared.commit == "a" * 40 and prepared.weights_sha256 is None
    assert "listening" in (tmp_path / "check" / "policy.log").read_text()


def test_a_manifest_naming_a_missing_class_is_refused_at_hello(spec, tmp_path):
    repo = write_policy_repo(tmp_path / "repo", policy="pkg.policy:Missing")
    runtime = SubprocessPolicyRuntime(spec, {"org/broken": repo}, start_timeout_s=30)
    fetched = runtime.fetch(runtime.resolve("org/broken", "b" * 40), workdir=tmp_path)
    with pytest.raises(SubmissionRefused) as refused:
        runtime.prepare(fetched, workdir=tmp_path / "check")
    assert refused.value.step == "hello" and "Missing" in refused.value.reason


def test_a_directory_without_a_manifest_is_refused(spec, tmp_path):
    (tmp_path / "empty").mkdir()
    runtime = SubprocessPolicyRuntime(spec, {"org/empty": tmp_path / "empty"})
    fetched = runtime.fetch(runtime.resolve("org/empty", "c" * 40), workdir=tmp_path)
    with pytest.raises(SubmissionRefused) as refused:
        runtime.prepare(fetched, workdir=tmp_path)
    assert refused.value.step == "manifest"


def test_each_serve_is_a_fresh_server_with_its_own_key_gone_after_its_unit(runtime, tmp_path):
    from vector_protocol.client import RemotePolicy

    prepared = runtime.prepare(
        runtime.fetch(runtime.resolve("org/zero", "d" * 40), workdir=tmp_path), workdir=tmp_path
    )
    seen = []
    for unit in ("fp-000", "fs-001"):
        with runtime.serve(prepared, workdir=tmp_path / unit) as served:
            key = bytes.fromhex(served.env[served.authkey_env])
            assert list(served.env) == [served.authkey_env] and len(key) == 32
            with RemotePolicy(served.address, key, timeout_s=10) as policy:
                assert policy.hello()["policy"] == "zero.policy:ZeroPolicy"
            assert served.died() is None, "a server that ended with its client did not die"
            seen.append((served.address, key))
        assert not Path(served.address).parent.exists(), "the socket directory outlived its unit"
        assert served.log_file == tmp_path / unit / "policy.log" and served.log_file.is_file()
    assert seen[0][0] != seen[1][0] and seen[0][1] != seen[1][1]


def test_a_local_policy_is_told_nothing_of_the_run_directory_and_its_log_lands_there(
    spec, tmp_path
):
    from vector_protocol.client import RemotePolicy

    seen = []

    class Watching(FakePolicyRuntime):
        def _started(self, process, served):
            seen.append(" ".join(process.args))

    runtime = Watching(spec)
    prepared = runtime.prepare(runtime.fetch(REPLAY_REF, workdir=tmp_path), workdir=tmp_path)
    unit_dir = tmp_path / "runs" / "challenger" / "fp-000"
    with runtime.serve(prepared, workdir=unit_dir) as served:
        key = bytes.fromhex(served.env[served.authkey_env])
        with RemotePolicy(served.address, key, timeout_s=10) as policy:
            policy.hello()
    assert len(seen) == 2 and all(str(tmp_path / "runs") not in argv for argv in seen), seen
    assert "challenger" not in seen[1], "the policy's command line names its side"
    assert served.log_file == unit_dir / "policy.log"
    assert "listening on" in served.log_file.read_text()


def test_a_server_killed_under_its_unit_is_reported_dead_by_its_own_doing(spec, tmp_path):
    runtime = FakePolicyRuntime(spec, kill_on_serve={0})
    prepared = runtime.prepare(runtime.fetch(REPLAY_REF, workdir=tmp_path), workdir=tmp_path)
    with runtime.serve(prepared, workdir=tmp_path / "fp-000") as served:
        end = served.died()
        assert end.cause == POLICY
        assert end.reason.startswith("the policy process was killed by signal 9")


def test_a_server_that_never_listens_is_a_dead_policy(runtime, tmp_path):
    prepared = runtime.prepare(
        runtime.fetch(runtime.resolve("org/zero", "e" * 40), workdir=tmp_path), workdir=tmp_path
    )
    runtime.python = "/bin/false"
    started = time.monotonic()
    with pytest.raises(PolicyDied, match=r"exited \(1\) before listening"):
        with runtime.serve(prepared, workdir=tmp_path / "u"):
            pass
    assert time.monotonic() - started < 10


def _prepared(runtime, repo, sha, tmp_path):
    return runtime.prepare(
        runtime.fetch(runtime.resolve(repo, sha), workdir=tmp_path), workdir=tmp_path / repo
    )


def _hello(served):
    from vector_protocol.client import RemotePolicy

    key = bytes.fromhex(served.env[served.authkey_env])
    with RemotePolicy(served.address, key, timeout_s=10) as policy:
        return policy.hello()


def test_a_pool_keeps_one_server_for_a_submissions_units_and_starts_another_for_another(
    runtime, tmp_path
):
    zero = _prepared(runtime, "org/zero", "d" * 40, tmp_path)
    replay = _prepared(runtime, "org/replay", "a" * 40, tmp_path)
    pool = ServerPool(tmp_path / "slot")
    pids = []
    for unit, prepared in (("u0", zero), ("u1", zero), ("u2", replay)):
        with runtime.serve(prepared, workdir=tmp_path / unit, pool=pool) as served:
            _hello(served)
            assert served.died() is None
            pids.append(served.pgid)
        assert pool.current is not None and pool.current.alive(), "the server outlived its unit"
    assert pids[0] == pids[1] != pids[2]
    first, second = ((tmp_path / u / "policy.log").read_text() for u in ("u0", "u1"))
    assert "listening on" in first and "listening on" not in second, "a unit got another's log"
    assert "hello from" in second
    kept = pool.current
    pool.close()
    assert not kept.alive() and pool.current is None
    assert not kept.sockets.exists()


def test_a_kept_server_that_died_is_started_again_for_the_next_unit(runtime, tmp_path):
    zero = _prepared(runtime, "org/zero", "d" * 40, tmp_path)
    pool = ServerPool(tmp_path / "slot")
    with runtime.serve(zero, workdir=tmp_path / "u0", pool=pool) as served:
        first = served.pgid
        os.killpg(first, 9)
        assert eventually(lambda: served.died() is not None)
        assert served.died().cause == POLICY
    assert pool.current is None
    with runtime.serve(zero, workdir=tmp_path / "u1", pool=pool) as served:
        assert served.pgid != first
        _hello(served)
    pool.close()


def test_a_server_is_put_on_the_card_it_is_given(runtime, tmp_path):
    zero = _prepared(runtime, "org/zero", "d" * 40, tmp_path)
    pool = ServerPool(tmp_path / "slot")
    for card in ("3", "5"):
        with runtime.serve(
            zero, workdir=tmp_path / card, env={"CUDA_VISIBLE_DEVICES": card}, pool=pool
        ) as served:
            environ = Path(f"/proc/{served.pgid}/environ").read_bytes().split(b"\0")
            assert f"CUDA_VISIBLE_DEVICES={card}".encode() in environ
    pool.close()

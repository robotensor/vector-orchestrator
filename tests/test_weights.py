"""Submissions as weights: chain-seeded units, the paired crown rule, the weights runtime and the
stall watchdog."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from vector_orchestrator.duel import score
from vector_orchestrator.duel.orchestrate import DuelRequest
from vector_orchestrator.duel.runtime import RuntimeUnavailable, SubmissionRefused
from vector_orchestrator.duel.weights_runtime import WeightsPolicyRuntime
from vector_orchestrator.ids import SubmissionRef
from vector_orchestrator.units import deal, seed_key

SHA = "a" * 40
HASH = "0x" + "b" * 64


def Check(**fields):
    return SimpleNamespace(**fields)


@pytest.fixture
def vector_spec(spec):
    return spec


def request(**seed):
    return DuelRequest(
        SubmissionRef.make("m/challenger", SHA),
        SubmissionRef.make("m/king", "c" * 40),
        block=3,
        **seed,
    )


def test_a_request_carries_its_seed_block_through_request_json(vector_spec):
    req = request(seed_block=123, seed_block_hash=HASH)
    doc = req.as_dict(vector_spec)
    assert doc["seed_block"] == 123 and doc["seed_block_hash"] == HASH
    assert DuelRequest.from_dict(doc) == req
    assert req.entropy == f"123:{HASH}"


def test_a_request_without_a_seed_block_has_no_entropy(vector_spec):
    doc = request().as_dict(vector_spec)
    assert "seed_block" not in doc and request().entropy is None
    assert DuelRequest.from_dict(doc) == request()


@pytest.mark.parametrize(
    "seed",
    [
        {"seed_block": 1},
        {"seed_block_hash": HASH},
        {"seed_block": 1, "seed_block_hash": "0x1234"},
        {"seed_block": -1, "seed_block_hash": HASH},
    ],
)
def test_a_half_or_malformed_seed_is_refused(seed):
    with pytest.raises(ValueError):
        request(**seed)


def test_entropy_moves_every_unit_and_the_same_entropy_deals_the_same_ones(spec):
    plain = deal(spec, "d", None, 2)
    seeded = deal(spec, "d", "7:0xab", 2)
    assert [u["unit_id"] for u in plain] == [u["unit_id"] for u in seeded]
    assert all(a["scene_seeds"] != b["scene_seeds"] for a, b in zip(plain, seeded, strict=True))
    assert deal(spec, "d", "7:0xab", 2) == seeded
    assert seed_key("d") == "d" and seed_key("d", "7:0xab") == "d|7:0xab"


def test_a_smaller_duels_units_are_the_first_of_a_larger_one_and_scenes_never_repeat(spec):
    small, large = deal(spec, "d", None, 2), deal(spec, "d", None, 10)
    assert [u for u in large if u["unit_id"].endswith(("-00", "-01"))] == small
    for unit in large:
        assert not set(unit["scene_seeds"]) & set(unit["demo_seeds"])
        assert min(unit["scene_seeds"] + unit["demo_seeds"]) >= 2_000_000
    assert len(large) == 10 * len(spec.tasks)


# -- the crown rule --------------------------------------------------------------------------


def test_the_sign_test():
    assert score.sign_test_p(0, 0) == 1.0
    assert score.sign_test_p(5, 0) == pytest.approx(1 / 32)
    assert score.sign_test_p(3, 3) == pytest.approx(42 / 64)


def units(pairs):
    """Rows of (challenger_success, king_success), all of one task."""
    return [
        {"task": "t", "void": False, "challenger_success": c, "king_success": k} for c, k in pairs
    ]


def test_a_clear_win_moves_the_crown():
    rows = units([(True, False)] * 8 + [(True, True)] * 2 + [(False, False)] * 6)
    v = score.verdict(rows, 3.0, ["t"], paired_alpha=0.05)
    assert v.dethroned and v.reason == "margin-met"
    assert v.paired_p_value == pytest.approx(1 / 256)


# -- the weights runtime ---------------------------------------------------------------------


def hub(files: dict[str, int]):
    info = SimpleNamespace(
        sha=SHA, siblings=[SimpleNamespace(rfilename=n, size=s) for n, s in files.items()]
    )
    return SimpleNamespace(repo_info=lambda *a, **k: info)


def runtime(tmp_path, vector_spec, files, checker=None, downloads=None):
    def download(repo, *, revision, repo_type, local_dir):
        (downloads if downloads is not None else []).append(repo)
        Path(local_dir, "model.safetensors").write_bytes(b"weights")

    return WeightsPolicyRuntime(
        vector_spec,
        python="python",
        cache_root=tmp_path / "cache",
        api=hub(files),
        download=download,
        checker=checker or (lambda path: Check(ok=True, errors=[], weights_sha256="f" * 64)),
        hello=False,
    )


def test_a_repository_holding_anything_but_weights_and_a_readme_is_refused(tmp_path, vector_spec):
    rt = runtime(tmp_path, vector_spec, {"model.safetensors": 7, "policy.py": 10})
    with pytest.raises(SubmissionRefused) as refused:
        rt.fetch(SubmissionRef.make("m/r", SHA), workdir=tmp_path / "w")
    assert refused.value.step == "fetch" and "policy.py" in refused.value.reason


def test_a_repository_without_its_weights_file_is_refused(tmp_path, vector_spec):
    rt = runtime(tmp_path, vector_spec, {"README.md": 5})
    with pytest.raises(SubmissionRefused, match="holds no model.safetensors"):
        rt.fetch(SubmissionRef.make("m/r", SHA), workdir=tmp_path / "w")


def test_only_the_weights_file_is_fetched_and_the_manifest_is_the_validators(tmp_path, vector_spec):
    downloads: list[str] = []
    rt = runtime(
        tmp_path, vector_spec, {"model.safetensors": 7, "README.md": 5}, downloads=downloads
    )
    fetched = rt.fetch(SubmissionRef.make("m/r", SHA), workdir=tmp_path / "w")
    assert downloads == ["m/r"] and sorted(p.name for p in fetched.root.iterdir()) == [
        "model.safetensors"
    ]
    prepared = rt.prepare(fetched, workdir=tmp_path / "w" / "check")
    manifest = json.loads(Path(prepared.handle).read_text())
    assert manifest["policy"] == "vector_runtime.policy:VectorPolicy"
    assert manifest["kwargs"]["weights"] == str((fetched.root / "model.safetensors").resolve())
    assert prepared.weights_sha256 == "f" * 64 and prepared.commit == SHA


def test_weights_that_fail_the_template_check_are_refused(tmp_path, vector_spec):
    rt = runtime(
        tmp_path,
        vector_spec,
        {"model.safetensors": 7},
        checker=lambda path: Check(ok=False, errors=["key x: wrong shape"], weights_sha256=None),
    )
    fetched = rt.fetch(SubmissionRef.make("m/r", SHA), workdir=tmp_path / "w")
    with pytest.raises(SubmissionRefused) as refused:
        rt.prepare(fetched, workdir=tmp_path / "w" / "check")
    assert refused.value.step == "check" and "wrong shape" in refused.value.reason


def test_an_installed_runtime_that_is_not_the_pinned_architecture_serves_nothing(
    tmp_path, vector_spec, write_spec
):
    doc = {**vector_spec.raw, "model": {**vector_spec.model, "tensors_sha256": "0" * 64}}
    other = write_spec(doc, name="other-architecture.json")
    rt = runtime(tmp_path, other, {"model.safetensors": 7})
    fetched = rt.fetch(SubmissionRef.make("m/r", SHA), workdir=tmp_path / "w")
    with pytest.raises(RuntimeUnavailable, match="the spec pins vector_v1.1"):
        rt.prepare(fetched, workdir=tmp_path / "w" / "check")


def test_the_manifest_path_is_absolute_even_for_a_relative_run_directory(
    tmp_path, vector_spec, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    rt = runtime(Path("."), vector_spec, {"model.safetensors": 7})
    fetched = rt.fetch(SubmissionRef.make("m/r", SHA), workdir=Path("w"))
    prepared = rt.prepare(fetched, workdir=Path("runs/check"))
    assert Path(prepared.handle).is_absolute() and Path(prepared.handle).is_file()


# -- the stall watchdog ----------------------------------------------------------------------


def test_a_process_that_sleeps_is_killed_as_stalled_and_a_busy_one_is_not(tmp_path, monkeypatch):
    import sys
    import time

    from vector_orchestrator import runner

    monkeypatch.setattr(runner, "STALL_POLL_S", 0.2)
    started = time.monotonic()
    hung = runner.run_argv(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        env={"PATH": "/usr/bin:/bin"},
        timeout_s=60,
        log_path=tmp_path / "hung.log",
        stall_s=2.0,
    )
    assert hung.stalled and not hung.timed_out and time.monotonic() - started < 20
    busy = runner.run_argv(
        [sys.executable, "-c", "import time\nend = time.time() + 4\nwhile time.time() < end: pass"],
        env={"PATH": "/usr/bin:/bin"},
        timeout_s=60,
        log_path=tmp_path / "busy.log",
        stall_s=2.0,
    )
    assert not busy.stalled and busy.returncode == 0


def test_the_contract_watches_for_hung_simulators(vector_spec):
    assert vector_spec.budgets["stall"] >= 60 and vector_spec.stall_retries >= 1


def test_a_benchmark_waiting_on_a_busy_policy_is_not_stalled(tmp_path, monkeypatch):
    import subprocess
    import sys

    from vector_orchestrator import runner

    monkeypatch.setattr(runner, "STALL_POLL_S", 0.2)
    busy = subprocess.Popen(
        [sys.executable, "-c", "import time\nend = time.time() + 6\nwhile time.time() < end: pass"],
        start_new_session=True,
    )
    try:
        waiting = runner.run_argv(
            [sys.executable, "-c", "import time; time.sleep(5)"],
            env={"PATH": "/usr/bin:/bin"},
            timeout_s=60,
            log_path=tmp_path / "wait.log",
            stall_s=2.0,
            watch_pgids=(busy.pid,),
        )
    finally:
        busy.kill()
        busy.wait()
    assert not waiting.stalled and waiting.returncode == 0

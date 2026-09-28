"""The RoboTwin driver: units dealt exactly as the plugin it replaced dealt them, command lines for
the checkout's `rt-bench`, results read and held to the harness source, and the checkout asked
through its own commands."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

from vector_orchestrator.benchmarks import robotwin as R
from vector_orchestrator.benchmarks.api import BENCHMARK_API_VERSION, BenchmarkRefused
from vector_orchestrator.benchmarks.units import seed_material

FIXTURES = Path(__file__).resolve().parent / "fixtures"
#: `rt-bench tasks list --json` of robotensor/RoboTwin-Vector when the derivation was pinned.
TABLE = R.Table.from_json(json.loads((FIXTURES / "robotwin-tasks.json").read_text()))


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


# -- units ----------------------------------------------------------------------------------------


def test_the_task_table_is_the_one_units_were_pinned_on():
    assert R.catalogue_sha256(TABLE) == R.CATALOGUE_SHA256


#: What the `robotensor-benchmark-robotwin` plugin's `derive_units(seed_material="pinned", ...)`
#: returned for the one-arm Franka track under units/2, pinned there on Python 3.10 and 3.12.
PINNED_FRANKA_SHA256 = "e7f7c2e9e39a3580da7e52581f8c7e4c382de384be1c7a1eb8229b4423fd37b0"
PINNED_FRANKA_TASKS = {
    "pick_and_place": ["place_empty_cup"] * 4,
    "stacking": ["stack_bowls_two"] * 4,
    "press_push": ["click_bell", "press_stapler", "click_bell", "press_stapler"],
}
#: The same for 48 units of vector_level1, taken from the plugin before it was removed.
PINNED_LEVEL1_SHA256 = "3fdbe6550ad54a7e913460d51fefeaad30578b6f9bc3116f5fb083303ce9ae72"


def test_the_franka_units_are_the_plugins():
    franka = {
        category: R.derive_units(
            TABLE, seed_material="pinned", count=4, suite=R.FRANKA_1ARM, category=category
        )
        for category in R.FRANKA_1ARM_CATEGORIES
    }
    assert {c: [u["task"] for u in units] for c, units in franka.items()} == PINNED_FRANKA_TASKS
    assert digest(franka) == PINNED_FRANKA_SHA256


def test_the_level1_units_are_the_plugins():
    units = R.derive_units(TABLE, seed_material="pinned", count=48, suite=R.VECTOR_LEVEL1)
    assert digest(units) == PINNED_LEVEL1_SHA256


#: The localnet rehearsal's light duel of 2026-09-23 (event 0c7c1d29...): its duel id and the chain
#: entropy it was fought under, and the digest of the 48 units its published record names.
REHEARSAL_DUEL_ID = "f958a150021bba09dfe67548b61317b0df7aabfdd571342cde64fcfa344e9332"
REHEARSAL_ENTROPY = "6750:0xe9a3eb93374b4153c2aa271552e55f67a3c12a64d31f03b09071e5e685823332"
REHEARSAL_UNITS_SHA256 = "68f982b4fa479516b5b31ffa8ba1189af3ec54ba6c472b1ee4250fb54540d713"


def test_a_published_duels_units_are_dealt_again_unchanged():
    material = seed_material(REHEARSAL_DUEL_ID, R.VECTOR_LEVEL1, REHEARSAL_ENTROPY)
    units = R.derive_units(TABLE, seed_material=material, count=48, suite=R.VECTOR_LEVEL1)
    shown = [{k: u[k] for k in ("task", "task_label", "instance_params")} for u in units]
    assert digest(shown) == REHEARSAL_UNITS_SHA256


def test_level1_units_carry_disjoint_demo_and_scene_seeds_outside_every_public_range():
    drawn = R.derive_units(
        TABLE, seed_material="duel|block|hash|skill", count=48, suite="vector_level1"
    )
    counts: dict[str, int] = {}
    for unit in drawn:
        params = unit["instance_params"]
        scene, demo = params["scene_seeds"], params["demo_seeds"]
        assert len(scene) == len(demo) == R.SCENE_SEED_CANDIDATES and not set(scene) & set(demo)
        assert all(R.PUBLIC_SEEDS_END <= s < R.SEED_BOUND for s in scene + demo)
        assert params["embodiment"] == "franka-panda" and params["scene_seed"] is None
        assert params["task_config"] == "level1" and params["level"] == 1
        counts[unit["task"]] = counts.get(unit["task"], 0) + 1
    assert set(counts.values()) == {3} and len(counts) == 16


def test_units_are_a_pure_function_of_the_seed_material_and_prefix_stable():
    a = R.derive_units(TABLE, seed_material="m", count=16, suite="vector_level1")
    b = R.derive_units(TABLE, seed_material="m", count=160, suite="vector_level1")
    assert a == b[:16]
    assert a != R.derive_units(TABLE, seed_material="m2", count=16, suite="vector_level1")


def test_another_task_table_deals_nothing():
    moved = dict(TABLE.tasks)
    moved["click_bell"] = {**moved["click_bell"], "category": "pick_and_place"}
    other = R.Table(tasks=moved, categories=TABLE.categories)
    with pytest.raises(ValueError, match="would deal other units"):
        R.derive_units(other, seed_material="m", count=1, suite="franka_1arm")


def test_a_malformed_table_is_refused():
    with pytest.raises(ValueError):
        R.Table.from_json({"tasks": {"x": {"category": "nope", "arms": "1"}}, "categories": {}})
    with pytest.raises(ValueError):
        R.Table.from_json([])


# -- command lines --------------------------------------------------------------------------------


class Fork(R.RoboTwinFork):
    """The driver with its checkout's answers given: the fixture table and a harness digest."""

    def __init__(self, root="/checkout", python="/env/bin/python"):
        super().__init__(root, python)
        self._table = TABLE
        self._harness = {"api": BENCHMARK_API_VERSION, "source_sha256": "a" * 64, "limits": {}}


def value_of(argv, flag):
    return argv[argv.index(flag) + 1]


def test_demo_make_records_on_demo_seeds_and_scores_on_scene_seeds(tmp_path, monkeypatch):
    monkeypatch.delenv(R.DENOISER_ENV, raising=False)
    unit = R.derive_units(TABLE, seed_material="m", count=1, suite="vector_level1")[0]
    argv = Fork().materialize_command(unit=unit, out_dir=str(tmp_path / "p"))
    params = unit["instance_params"]
    assert argv[:6] == ["/env/bin/python", "-m", "robotensor_bench.cli", "demo", "make", "--task"]
    assert value_of(argv, "--seed-candidates") == ",".join(map(str, params["demo_seeds"]))
    assert value_of(argv, "--eval-seed-candidates") == ",".join(map(str, params["scene_seeds"]))
    assert value_of(argv, "--task-config") == "level1"
    assert value_of(argv, "--embodiment") == "franka-panda"
    assert value_of(argv, "--save-freq") == "15" and "--legal-joints" in argv
    assert value_of(argv, "--expect-source-sha256") == "a" * 64
    assert value_of(argv, "--out") == str(tmp_path / "p") and "--denoiser" not in argv


def test_a_same_scene_unit_is_demonstrated_and_scored_on_its_scene_seeds(tmp_path):
    unit = R.derive_units(TABLE, seed_material="m", count=1, suite="franka_1arm")[0]
    argv = Fork().materialize_command(unit=unit, out_dir=str(tmp_path))
    assert value_of(argv, "--seed-candidates") == ",".join(
        map(str, unit["instance_params"]["scene_seeds"])
    )
    assert "--eval-seed-candidates" not in argv and "--legal-joints" not in argv
    assert value_of(argv, "--task-config") == "demo_clean"


def test_a_unit_that_cannot_be_made_is_refused_before_it_runs():
    unit = R.derive_units(TABLE, seed_material="m", count=1, suite="vector_level1")[0]
    unit["instance_params"]["demo_seeds"][0] = unit["instance_params"]["scene_seeds"][0]
    with pytest.raises(ValueError, match="share a seed"):
        Fork().materialize_command(unit=unit, out_dir="/tmp/x")
    with pytest.raises(ValueError, match="not a task of this benchmark"):
        Fork().materialize_command(unit={**unit, "task": "juggle"}, out_dir="/tmp/x")


def test_eval_run_plays_the_demonstration_directory_against_the_served_policy(
    tmp_path, monkeypatch
):
    monkeypatch.setenv(R.DENOISER_ENV, "none")
    unit = R.derive_units(TABLE, seed_material="m", count=1, suite="vector_level1")[0]
    argv = Fork().run_command(
        unit=unit,
        prompt=str(tmp_path / "prompts" / "vl-000" / "prompt.npz"),
        out_dir=str(tmp_path / "challenger" / "vl-000"),
        policy_address="relative/policy.sock",
        authkey_env="VECTOR_PROTOCOL_AUTHKEY",
        act_timeout_s=60,
        policy_budget_s=1170.0,
        unit_timeout_s=1800.0,
        policy_log=str(tmp_path / "policy.log"),
        a_key_a_later_orchestrator_adds=1,
    )
    assert argv[3:5] == ["eval", "run"]
    assert value_of(argv, "--demo") == str(tmp_path / "prompts" / "vl-000")
    assert value_of(argv, "--policy") == os.path.abspath("relative/policy.sock")
    assert value_of(argv, "--authkey-env") == "VECTOR_PROTOCOL_AUTHKEY"
    assert (value_of(argv, "--act-timeout-s"), value_of(argv, "--unit-timeout-s")) == (
        "60.0",
        "1800.0",
    )
    assert value_of(argv, "--policy-log") == str(tmp_path / "policy.log")
    assert value_of(argv, "--denoiser") == "none"
    assert value_of(argv, "--expect-source-sha256") == "a" * 64


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"act_timeout_s": 0}, "act_timeout_s must be a positive"),
        ({"unit_timeout_s": float("inf")}, "unit_timeout_s must be a positive"),
        ({"authkey_env": "KEY=abc"}, "authkey_env must name"),
        ({"prompt": "/p/other.npz"}, "prompt.npz"),
    ],
)
def test_eval_run_refuses_what_would_void_the_unit_once_it_ran(change, reason):
    unit = R.derive_units(TABLE, seed_material="m", count=1, suite="vector_level1")[0]
    kwargs = dict(
        unit=unit,
        prompt="/p/prompt.npz",
        out_dir="/o",
        policy_address="/tmp/p.sock",
        authkey_env="K",
    )
    extra = {k: v for k, v in change.items() if k not in kwargs}
    kwargs.update({k: v for k, v in change.items() if k in kwargs})
    with pytest.raises(ValueError, match=reason):
        Fork().run_command(**kwargs, **extra)


def test_a_host_port_address_is_kept_as_given():
    unit = R.derive_units(TABLE, seed_material="m", count=1, suite="franka_1arm")[0]
    argv = Fork().run_command(
        unit=unit,
        prompt="/p/prompt.npz",
        out_dir="/o",
        policy_address="127.0.0.1:7100",
        authkey_env="K",
    )
    assert value_of(argv, "--policy") == "127.0.0.1:7100"


def test_a_denoiser_the_harness_does_not_take_is_refused(monkeypatch):
    monkeypatch.setenv(R.DENOISER_ENV, "fast")
    unit = R.derive_units(TABLE, seed_material="m", count=1, suite="franka_1arm")[0]
    with pytest.raises(ValueError, match="ROBOTWIN_BENCH_DENOISER"):
        Fork().materialize_command(unit=unit, out_dir="/o")


# -- results --------------------------------------------------------------------------------------

MINE = "a" * 64


def result(tmp_path, doc):
    (tmp_path / "result.json").write_text(json.dumps(doc) if not isinstance(doc, str) else doc)
    return R.read_result(tmp_path, MINE)


def test_a_scored_success_and_a_failure_are_read_as_written(tmp_path):
    read = result(tmp_path, {"success": True, "void": False, "steps": 77.0, "source_sha256": MINE})
    assert (read["success"], read["void"], read["void_cause"], read["steps"]) == (
        True,
        False,
        None,
        77,
    )
    read = result(tmp_path, {"success": False, "void": False, "steps": 400, "source_sha256": MINE})
    assert read["success"] is False and read["error"] is None


@pytest.mark.parametrize("cause", ["harness", "policy"])
def test_a_void_result_keeps_its_cause_and_its_reason(tmp_path, cause):
    doc = {
        "success": None,
        "void": True,
        "void_cause": cause,
        "error": "why",
        "source_sha256": MINE,
    }
    read = result(tmp_path, doc)
    assert (read["void"], read["void_cause"], read["error"], read["steps"]) == (
        True,
        cause,
        "why",
        None,
    )


@pytest.mark.parametrize("source", ["0" * 64, None])
def test_a_result_other_harness_source_wrote_is_void_on_the_harness(tmp_path, source):
    read = result(tmp_path, {"success": True, "void": False, "steps": 3, "source_sha256": source})
    assert read["void"] is True and read["void_cause"] == "harness" and read["success"] is None
    assert "written by harness source" in read["error"]


@pytest.mark.parametrize(
    ("doc", "error"),
    [
        ("not json", "unreadable result.json"),
        ([], "result.json holds list"),
        ({"success": "yes", "void": False, "source_sha256": MINE}, "without a reason"),
        ({"success": True, "source_sha256": MINE}, "without a reason"),
    ],
)
def test_a_result_that_does_not_hold_up_is_void_on_the_harness(tmp_path, doc, error):
    read = result(tmp_path, doc)
    assert read["void"] is True and read["void_cause"] == "harness" and error in read["error"]


def test_a_missing_result_is_void_on_the_harness(tmp_path):
    read = R.read_result(tmp_path, MINE)
    assert read["void"] is True and "unreadable result.json" in read["error"]


# -- the checkout ---------------------------------------------------------------------------------

#: A checkout's `robotensor_bench.cli` standing in for the fork's: it answers `tasks list`, `config
#: check` and `demo verify` as `rt-bench` does, and records every command line it was run with.
FAKE_CLI = """
import hashlib, json, os, sys
from pathlib import Path

here = Path(__file__).resolve().parent
with open(here / "calls.jsonl", "a") as log:
    log.write(json.dumps({"argv": sys.argv[1:], "cwd": os.getcwd(), "pythonpath": os.environ.get("PYTHONPATH")}) + "\\n")
args = sys.argv[1:]
behaviour = (here / "behaviour").read_text().strip() if (here / "behaviour").exists() else "ok"
if args[:2] == ["tasks", "list"]:
    print((here / "tasks.json").read_text())
elif args[:2] == ["config", "check"]:
    if behaviour == "no-assets":
        print("rt-bench: the harness is unavailable: embodiment config missing", file=sys.stderr)
        sys.exit(3)
    api = 1 if behaviour == "old-api" else API
    print(json.dumps({"task_config": args[args.index("--task-config") + 1], "harness": {
        "api": api, "source_sha256": "b" * 64, "commit": "c0ffee",
        "limits": {"result_reserve_s": 30.0}}}))
elif args[:2] == ["demo", "verify"]:
    demo = Path(args[args.index("--demo") + 1])
    prompt = demo / "prompt.npz"
    sha = hashlib.sha256(prompt.read_bytes()).hexdigest() if prompt.exists() else None
    ok = prompt.exists() and "bad" not in demo.name
    print(json.dumps({"ok": ok, "sha256": sha, "problems": [] if ok else ["not the unit's"],
                      "task": args[args.index("--task") + 1], "scene_seed": 5, "embodiment": "franka-panda"}))
    sys.exit(0 if ok else 4)
else:
    sys.exit(2)
"""


@pytest.fixture
def checkout(tmp_path):
    root = tmp_path / "RoboTwin-Vector"
    package = root / "robotensor_bench"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "cli.py").write_text(FAKE_CLI.replace("API", str(BENCHMARK_API_VERSION)))
    (package / "tasks.json").write_text((FIXTURES / "robotwin-tasks.json").read_text())
    return root


def calls(root):
    path = root / "robotensor_bench" / "calls.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def test_found_from_the_hosts_environment(checkout):
    fork = R.RoboTwinFork.from_environment(
        {R.ROOT_ENV: str(checkout), R.PYTHON_ENV: sys.executable}
    )
    assert fork.workdir == str(checkout) and fork.python == sys.executable
    assert calls(checkout) == [], "finding a checkout runs nothing in it"
    legacy = R.RoboTwinFork.from_environment(
        {R.ROOT_ENV: str(checkout), R.LEGACY_PYTHON_ENV: sys.executable}
    )
    assert legacy.python == sys.executable


def test_preparing_asks_the_checkout_for_its_table_and_each_suites_config(checkout, tmp_path):
    fork = R.RoboTwinFork(checkout, sys.executable)
    fork.prepare(["vector_level1"])
    fork.prepare(["vector_level1"])  # asked once
    argvs = [c["argv"] for c in calls(checkout)]
    assert argvs == [
        ["tasks", "list", "--json"],
        ["config", "check", "--task-config", "level1", "--embodiment", "franka-panda"],
    ]
    # Every command runs from the checkout, with it on PYTHONPATH, wherever the orchestrator is.
    assert {(c["cwd"], c["pythonpath"]) for c in calls(checkout)} == {
        (str(checkout), str(checkout))
    }
    assert fork.source_sha256 == "b" * 64
    assert fork.info()["limits"] == {"result_reserve_s": 30.0}
    assert fork.info()["harness"]["commit"] == "c0ffee"
    assert fork.catalogue()["suites"]["vector_level1"] == list(R.SUITES["vector_level1"])


@pytest.mark.parametrize(
    ("behaviour", "reason"),
    [
        (
            "no-assets",
            "config check of level1 on franka-panda exited 3: rt-bench: the harness is unavailable",
        ),
        ("old-api", "speaks command-line contract 1; this orchestrator drives 2"),
    ],
)
def test_a_checkout_that_cannot_run_a_suite_is_refused(checkout, behaviour, reason):
    (checkout / "robotensor_bench" / "behaviour").write_text(behaviour)
    fork = R.RoboTwinFork(checkout, sys.executable)
    with pytest.raises(BenchmarkRefused, match=reason):
        fork.prepare(["vector_level1"])


def test_demo_verify_holds_the_prompt_to_its_units_flags(checkout, tmp_path):
    fork = R.RoboTwinFork(checkout, sys.executable)
    unit = R.derive_units(TABLE, seed_material="m", count=1, suite="vector_level1")[0]
    good = tmp_path / "prompts" / "vl-000"
    good.mkdir(parents=True)
    (good / "prompt.npz").write_bytes(b"a prompt")
    verdict = fork.verify_prompt(path=str(good / "prompt.npz"), unit=unit)
    assert verdict["ok"] is True
    assert verdict["sha256"] == hashlib.sha256(b"a prompt").hexdigest()
    argv = calls(checkout)[-1]["argv"]
    assert argv[:4] == ["demo", "verify", "--demo", str(good)]
    assert value_of(argv, "--seed-candidates") == ",".join(
        map(str, unit["instance_params"]["demo_seeds"])
    )
    assert value_of(argv, "--expect-source-sha256") == "b" * 64

    bad = tmp_path / "prompts" / "bad"
    bad.mkdir()
    (bad / "prompt.npz").write_bytes(b"another")
    refused = fork.verify_prompt(path=str(bad / "prompt.npz"), unit=unit)
    assert refused["ok"] is False and refused["problems"] == ["not the unit's"]


def test_a_checkout_whose_verify_fails_is_the_units_problem_not_a_verdict(checkout, tmp_path):
    fork = R.RoboTwinFork(checkout, sys.executable)
    fork._table, fork._harness = TABLE, {"api": 2, "source_sha256": "b" * 64}
    (checkout / "robotensor_bench" / "cli.py").write_text("import sys; sys.exit(3)\n")
    unit = R.derive_units(TABLE, seed_material="m", count=1, suite="vector_level1")[0]
    with pytest.raises(BenchmarkRefused, match="demo verify exited 3"):
        fork.verify_prompt(path=str(tmp_path / "prompt.npz"), unit=unit)


# -- the real checkout, when this host has one ------------------------------------------------------

REAL = os.environ.get(R.ROOT_ENV) or str(Path(__file__).resolve().parents[2] / "RoboTwin-Vector")


@pytest.mark.skipif(
    not (Path(REAL) / "robotensor_bench" / "cli.py").is_file(), reason="no RoboTwin-Vector checkout"
)
def test_the_real_checkouts_table_is_the_pinned_one():
    fork = R.RoboTwinFork(REAL, sys.executable)
    assert R.catalogue_sha256(fork.table()) == R.CATALOGUE_SHA256
    assert fork.table() == TABLE

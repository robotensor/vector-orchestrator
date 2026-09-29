"""The RoboTwin driver: command lines for the checkout's `rt-bench`, results read and held to the
harness source, and the checkout asked through its own commands."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

from vector_orchestrator import robotwin as R
from vector_orchestrator.spec import load_spec
from vector_orchestrator.units import deal

UNIT = deal(load_spec(), "d" * 64, None, 1)[0]


class Fork(R.RoboTwin):
    """The driver with its checkout's harness given."""

    def __init__(self, root="/checkout", python="/env/bin/python"):
        super().__init__(root, python, level=1)
        self._harness = {"api": R.API_VERSION, "source_sha256": "a" * 64, "limits": {}}


def value_of(argv, flag):
    return argv[argv.index(flag) + 1]


# -- command lines --------------------------------------------------------------------------------


def test_demo_make_records_on_demo_seeds_and_scores_on_scene_seeds(tmp_path, monkeypatch):
    monkeypatch.delenv(R.DENOISER_ENV, raising=False)
    argv = Fork().materialize_command(unit=UNIT, out_dir=str(tmp_path / "p"))
    assert argv[:7] == [
        "/env/bin/python",
        "-m",
        "robotensor_bench.cli",
        "demo",
        "make",
        "--task",
        UNIT["task"],
    ]
    assert value_of(argv, "--seed-candidates") == ",".join(map(str, UNIT["demo_seeds"]))
    assert value_of(argv, "--eval-seed-candidates") == ",".join(map(str, UNIT["scene_seeds"]))
    assert value_of(argv, "--task-config") == "level1"
    assert value_of(argv, "--embodiment") == "franka-panda"
    assert value_of(argv, "--save-freq") == "15" and "--legal-joints" in argv
    assert value_of(argv, "--expect-source-sha256") == "a" * 64
    assert value_of(argv, "--out") == str(tmp_path / "p") and "--denoiser" not in argv


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"demo_seeds": UNIT["scene_seeds"][:1]}, "share a seed"),
        ({"scene_seeds": []}, "not distinct seeds"),
        ({"scene_seeds": [1, 1]}, "not distinct seeds"),
        ({"demo_seeds": [2**32]}, "not distinct seeds"),
    ],
)
def test_a_unit_that_cannot_be_made_is_refused_before_it_runs(change, reason):
    with pytest.raises(ValueError, match=reason):
        Fork().materialize_command(unit={**UNIT, **change}, out_dir="/tmp/x")


def test_eval_run_plays_the_demonstration_directory_against_the_served_policy(
    tmp_path, monkeypatch
):
    monkeypatch.setenv(R.DENOISER_ENV, "none")
    argv = Fork().run_command(
        unit=UNIT,
        prompt=str(tmp_path / "prompts" / "u" / "prompt.npz"),
        out_dir=str(tmp_path / "challenger" / "u"),
        policy_address="relative/policy.sock",
        authkey_env="VECTOR_AUTHKEY",
        act_timeout_s=60,
        policy_budget_s=1170.0,
        unit_timeout_s=1800.0,
        policy_log=str(tmp_path / "policy.log"),
        step_ratio=2,
    )
    assert argv[3:5] == ["eval", "run"]
    assert value_of(argv, "--step-ratio") == "2.0"
    assert value_of(argv, "--demo") == str(tmp_path / "prompts" / "u")
    assert value_of(argv, "--policy") == os.path.abspath("relative/policy.sock")
    assert value_of(argv, "--authkey-env") == "VECTOR_AUTHKEY"
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
        ({"act_timeout_s": 0}, "act-timeout-s must be a positive"),
        ({"unit_timeout_s": float("inf")}, "unit-timeout-s must be a positive"),
        ({"authkey_env": "KEY=abc"}, "authkey_env must name"),
        ({"step_ratio": 0}, "--step-ratio must be a positive"),
    ],
)
def test_eval_run_refuses_what_would_void_the_unit_once_it_ran(change, reason):
    kwargs = dict(
        unit=UNIT,
        prompt="/p/prompt.npz",
        out_dir="/o",
        policy_address="/tmp/p.sock",
        authkey_env="K",
    )
    with pytest.raises(ValueError, match=reason):
        Fork().run_command(**{**kwargs, **change})


def test_a_host_port_address_is_kept_as_given():
    argv = Fork().run_command(
        unit=UNIT,
        prompt="/p/prompt.npz",
        out_dir="/o",
        policy_address="127.0.0.1:7100",
        authkey_env="K",
    )
    assert value_of(argv, "--policy") == "127.0.0.1:7100"


def test_a_denoiser_the_harness_does_not_take_is_refused(monkeypatch):
    monkeypatch.setenv(R.DENOISER_ENV, "fast")
    with pytest.raises(ValueError, match="ROBOTWIN_BENCH_DENOISER"):
        Fork().materialize_command(unit=UNIT, out_dir="/o")


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


TASKS = {"tasks": {task: {"category": "pick_and_place", "arms": "1"} for task in load_spec().tasks}}


@pytest.fixture
def checkout(tmp_path):
    root = tmp_path / "RoboTwin-Vector"
    package = root / "robotensor_bench"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "cli.py").write_text(FAKE_CLI.replace("API", str(R.API_VERSION)))
    (package / "tasks.json").write_text(json.dumps(TASKS))
    return root


def calls(root):
    path = root / "robotensor_bench" / "calls.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def test_found_from_the_hosts_environment(checkout, tmp_path):
    fork = R.RoboTwin.from_environment(1, {R.ROOT_ENV: str(checkout), R.PYTHON_ENV: sys.executable})
    assert fork.workdir == str(checkout) and fork.python == sys.executable
    assert calls(checkout) == [], "finding a checkout runs nothing in it"
    with pytest.raises(R.Unavailable, match="ROBOTWIN_BENCH_ROOT"):
        R.RoboTwin.from_environment(1, {})
    with pytest.raises(R.Unavailable, match="not a RoboTwin-Vector checkout"):
        R.RoboTwin.from_environment(1, {R.ROOT_ENV: str(tmp_path), R.PYTHON_ENV: sys.executable})


def test_preparing_asks_the_checkout_for_its_tasks_and_the_levels_config(checkout):
    fork = R.RoboTwin(checkout, sys.executable, 1)
    fork.prepare(load_spec().tasks)
    argvs = [c["argv"] for c in calls(checkout)]
    assert argvs == [
        ["tasks", "list", "--json"],
        ["config", "check", "--task-config", "level1", "--embodiment", "franka-panda"],
    ]
    # Every command runs from the checkout, with it on PYTHONPATH, wherever the orchestrator is.
    assert {(c["cwd"], c["pythonpath"]) for c in calls(checkout)} == {
        (str(checkout), str(checkout))
    }
    assert fork.info() == {"commit": "c0ffee", "source_sha256": "b" * 64}
    assert fork.result_reserve_s == 30.0
    with pytest.raises(R.Unavailable, match="has no task juggle"):
        fork.prepare(["juggle"])


@pytest.mark.parametrize(
    ("behaviour", "reason"),
    [
        ("no-assets", "config check exited 3: rt-bench: the harness is unavailable"),
        ("old-api", f"the harness speaks contract 1, not {R.API_VERSION}"),
    ],
)
def test_a_checkout_that_cannot_run_the_level_is_refused(checkout, behaviour, reason):
    (checkout / "robotensor_bench" / "behaviour").write_text(behaviour)
    with pytest.raises(R.Unavailable, match=reason):
        R.RoboTwin(checkout, sys.executable, 1).prepare(load_spec().tasks)


def test_demo_verify_holds_the_prompt_to_its_units_flags(checkout, tmp_path):
    fork = R.RoboTwin(checkout, sys.executable, 1)
    fork.prepare(load_spec().tasks)
    good = tmp_path / "prompts" / "u"
    good.mkdir(parents=True)
    (good / "prompt.npz").write_bytes(b"a prompt")
    verdict = fork.verify_prompt(path=str(good / "prompt.npz"), unit=UNIT)
    assert verdict["ok"] is True and verdict["sha256"] == hashlib.sha256(b"a prompt").hexdigest()
    argv = calls(checkout)[-1]["argv"]
    assert argv[:4] == ["demo", "verify", "--demo", str(good)]
    assert value_of(argv, "--seed-candidates") == ",".join(map(str, UNIT["demo_seeds"]))
    assert value_of(argv, "--expect-source-sha256") == "b" * 64

    bad = tmp_path / "prompts" / "bad"
    bad.mkdir()
    (bad / "prompt.npz").write_bytes(b"another")
    refused = fork.verify_prompt(path=str(bad / "prompt.npz"), unit=UNIT)
    assert refused["ok"] is False and refused["problems"] == ["not the unit's"]
    (checkout / "robotensor_bench" / "cli.py").write_text("import sys; sys.exit(3)\n")
    with pytest.raises(R.Unavailable, match="demo verify exited 3"):
        fork.verify_prompt(path=str(good / "prompt.npz"), unit=UNIT)


# -- the real checkout, when this host has one ------------------------------------------------------

REAL = os.environ.get(R.ROOT_ENV) or str(Path(__file__).resolve().parents[2] / "RoboTwin-Vector")


@pytest.mark.skipif(
    not (Path(REAL) / "robotensor_bench" / "cli.py").is_file(), reason="no RoboTwin-Vector checkout"
)
def test_the_real_checkout_has_every_task_of_the_spec():
    table = R.RoboTwin(REAL, sys.executable, 1).ask("tasks", "list", "--json")
    assert set(load_spec().tasks) <= set(table["tasks"])

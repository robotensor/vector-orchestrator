"""`vector-orchestrator benchmarks list|check`, against the fake benchmark registered as a driver."""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys

import pytest

from conftest import FAKE_SITE
from vector_orchestrator.benchmarks import drivers, robotwin
from vector_orchestrator.benchmarks.check import check_benchmark
from vector_orchestrator.cli import main


def register(monkeypatch, body: str, *, imports: str = "") -> None:
    """Register a subclass of the fake benchmark, `body` its class body, as the `fake` driver."""
    monkeypatch.syspath_prepend(str(FAKE_SITE))
    fake = importlib.import_module("vector_fake_benchmark")
    namespace: dict = {"Base": fake.FakeBenchmark}
    exec(f"{imports}class Variant(Base):\n{body}", namespace)
    variant = namespace["Variant"]()
    monkeypatch.setitem(drivers.DRIVERS, "fake", lambda pin: variant)


def test_the_orchestrators_command_line_checks_the_fake_benchmark_with_no_problems(fake_spec):
    done = subprocess.run(
        [sys.executable, "-m", "vector_fake_cli", "--spec", str(fake_spec.path)]
        + ["benchmarks", "check", "fake"],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": os.pathsep.join([str(FAKE_SITE), *sys.path])},
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert "problem:" not in done.stdout
    assert done.stdout.strip().splitlines()[-1] == "fake: ok"


def test_check_reports_the_fake_benchmark_clean(fake_installed, fake_spec):
    report = check_benchmark(fake_spec, "fake")
    assert report.problems == [] and report.notes == []
    assert report.info["embodiment"][:2] == ["franka-panda", "franka-panda"]


def test_list_names_every_declared_benchmark_and_where_it_runs(fake_installed, fake_spec, capsys):
    assert main(["--spec", str(fake_spec.path), "benchmarks", "list", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert [(r["name"], r["found"], r["workdir"], r["problems"]) for r in rows] == [
        ("fake", True, None, [])
    ]
    assert rows[0]["pin"]["module"] == "vector_fake_benchmark.command"


def test_check_refuses_another_api_version_with_the_reason(fake_spec, monkeypatch, capsys):
    register(monkeypatch, "    api_version = 1\n")
    assert main(["--spec", str(fake_spec.path), "benchmarks", "check", "fake"]) == 1
    out = capsys.readouterr().out
    assert "problem: api_version: speaks 1, this orchestrator speaks 2" in out
    assert out.strip().endswith("fake: REFUSED")


@pytest.mark.parametrize(
    "mutation, problem",
    [
        (
            "    def catalogue(self):\n        return {'suites': {}, 'categories': {}}\n",
            "franka_pick_and_place: suite 'franka_1arm' is not in its catalogue",
        ),
        (
            "    def derive_units(self, **kw):\n"
            "        import random\n"
            "        units = super().derive_units(**kw)\n"
            "        units[0]['instance_params']['scene_seed'] = random.random()\n"
            "        return units\n",
            "franka_pick_and_place: derive_units is not a pure function of its arguments",
        ),
        (
            "    def run_command(self, **kw):\n        return 'bench run'\n",
            "franka_pick_and_place: run_command: expected a non-empty list of strings",
        ),
        # A stub that returns nothing is as broken as one that raises.
        ("    def info(self):\n        pass\n", "info: returned NoneType, not a mapping"),
        ("    def catalogue(self):\n        pass\n", "catalogue: returned NoneType, not a mapping"),
        (
            "    def derive_units(self, **kw):\n        pass\n",
            "franka_pick_and_place: derive_units: returned NoneType, not a list",
        ),
        (
            "    def materialize_command(self, **kw):\n        pass\n",
            "franka_pick_and_place: materialize_command: expected a non-empty list of strings, got None",
        ),
        (
            "    def run_command(self, **kw):\n        pass\n",
            "franka_pick_and_place: run_command: expected a non-empty list of strings, got None",
        ),
    ],
)
def test_check_catches_a_benchmark_that_would_fail_a_duel(
    fake_spec, monkeypatch, mutation, problem
):
    register(monkeypatch, mutation)
    report = check_benchmark(fake_spec, "fake")
    assert any(p.startswith(problem) for p in report.problems), report.problems


def test_check_hands_run_command_an_address_in_the_wire_form(fake_spec, monkeypatch):
    """The policy wire listens on a Unix socket path or host:port (vector_policy.wire), so a
    benchmark that follows the contract and checks the address must pass `check`, not be failed by
    a URL."""
    register(
        monkeypatch,
        "    def run_command(self, *, policy_address, **kw):\n"
        "        host, _, port = policy_address.rpartition(':')\n"
        "        if not (os.path.isabs(policy_address) or (host and port.isdigit())):\n"
        "            raise ValueError(f'{policy_address!r} is not a socket path or host:port')\n"
        "        return super().run_command(policy_address=policy_address, **kw)\n",
        imports="import os\n",
    )
    assert check_benchmark(fake_spec, "fake").problems == []


def test_check_on_the_shipped_spec_says_where_robotwins_checkout_must_be_named(
    spec, monkeypatch, capsys
):
    monkeypatch.delenv(robotwin.ROOT_ENV, raising=False)
    assert main(["benchmarks", "check", "robotwin"]) == 1
    out = capsys.readouterr().out
    assert "problem: $ROBOTWIN_BENCH_ROOT is not set" in out


def test_python_dash_m_runs_the_same_cli(fake_spec):
    done = subprocess.run(
        [
            sys.executable,
            "-m",
            "vector_orchestrator",
            "--spec",
            str(fake_spec.path),
            "benchmarks",
            "list",
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(FAKE_SITE)},
    )
    assert done.returncode == 0 and done.stdout.startswith("fake"), done.stderr

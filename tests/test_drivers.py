"""Finding each benchmark the spec declares: a driver per benchmark, and its checkout on this host."""

import pytest

from vector_orchestrator.benchmarks import drivers, robotwin
from vector_orchestrator.benchmarks.api import BENCHMARK_API_VERSION, BenchmarkRefused


def test_the_fake_benchmark_is_found_through_its_driver(fake_installed, fake_spec):
    row = drivers.discover(fake_spec)["fake"]
    assert row.ok and row.benchmark is not None and row.benchmark.id == "fake"
    assert row.as_dict()["found"] is True and row.workdir is None
    assert drivers.load(fake_spec, "fake") is row.benchmark


def test_a_benchmark_no_driver_drives_is_refused(fake_spec):
    # The spec declares `fake`, but nothing registered a driver for it.
    row = drivers.discover(fake_spec)["fake"]
    assert not row.ok and "no driver: this orchestrator drives robotwin" in row.problems[0]
    with pytest.raises(BenchmarkRefused, match="fake: no driver"):
        drivers.load(fake_spec, "fake")


def test_a_benchmark_the_spec_does_not_declare_is_refused(fake_spec):
    with pytest.raises(BenchmarkRefused, match="robotwin: not declared"):
        drivers.load(fake_spec, "robotwin")


def test_robotwins_checkout_must_be_named_by_the_host(spec, monkeypatch):
    monkeypatch.delenv(robotwin.ROOT_ENV, raising=False)
    row = drivers.discover(spec)["robotwin"]
    assert not row.ok and "$ROBOTWIN_BENCH_ROOT is not set" in row.problems[0]


def test_a_directory_that_is_not_the_fork_is_refused(spec, monkeypatch, tmp_path):
    monkeypatch.setenv(robotwin.ROOT_ENV, str(tmp_path))
    row = drivers.discover(spec)["robotwin"]
    assert not row.ok and "is not a RoboTwin-Vector checkout" in row.problems[0]


def test_a_checkout_without_its_interpreter_is_refused(spec, monkeypatch, tmp_path):
    (tmp_path / "robotensor_bench").mkdir()
    (tmp_path / "robotensor_bench" / "cli.py").write_text("")
    monkeypatch.setenv(robotwin.ROOT_ENV, str(tmp_path))
    monkeypatch.setenv(robotwin.PYTHON_ENV, str(tmp_path / "no-python"))
    row = drivers.discover(spec)["robotwin"]
    assert not row.ok and "interpreter" in row.problems[0] and "no-python" in row.problems[0]


class Driven:
    """A benchmark that records what the drivers asked of it."""

    id = "fake"
    api_version = BENCHMARK_API_VERSION
    module = "vector_fake_benchmark.command"
    workdir = "/checkout"
    python = "/env/bin/python"

    def __init__(self, refuse=None):
        self.prepared = []
        self.refuse = refuse

    def prepare(self, suites):
        self.prepared.append(list(suites))
        if self.refuse:
            raise BenchmarkRefused(self.refuse)

    def info(self):
        return {}

    def catalogue(self):
        return {}

    def derive_units(self, *, seed_material, count, suite, category=None):
        return []

    def verify_prompt(self, *, path, unit):
        return {}

    def read_result(self, *, out_dir):
        return {}

    def materialize_command(self, *, unit, out_dir):
        return []

    def run_command(self, *, unit, prompt, out_dir, policy_address, authkey_env, **extra):
        return []


def test_preparing_asks_the_checkout_about_the_specs_suites(fake_spec, monkeypatch):
    driven = Driven()
    monkeypatch.setitem(drivers.DRIVERS, "fake", lambda pin: driven)
    row = drivers.discover(fake_spec, prepare=("fake",))["fake"]
    assert row.ok and driven.prepared == [["franka_1arm"]]
    assert (row.workdir, row.python) == ("/checkout", "/env/bin/python")
    # Found without preparing: a listing never runs the checkout's commands.
    driven.prepared.clear()
    drivers.discover(fake_spec)
    drivers.load(fake_spec, "fake")
    assert driven.prepared == []


def test_a_checkout_that_cannot_run_the_suites_is_refused(fake_spec, monkeypatch):
    monkeypatch.setitem(
        drivers.DRIVERS, "fake", lambda pin: Driven(refuse="config check of level1 exited 3")
    )
    row = drivers.discover(fake_spec, prepare=True)["fake"]
    assert not row.ok and row.problems == ["config check of level1 exited 3"]


def test_the_driver_must_run_the_module_the_spec_pins(fake_spec, monkeypatch):
    other = type("Other", (Driven,), {"module": "someone_else.cli"})
    monkeypatch.setitem(drivers.DRIVERS, "fake", lambda pin: other())
    row = drivers.discover(fake_spec)["fake"]
    assert not row.ok
    assert row.problems == [
        "the spec pins module 'vector_fake_benchmark.command'; the driver runs 'someone_else.cli'"
    ]


def test_a_driver_that_is_not_a_usable_benchmark_is_refused_with_the_reason(fake_spec, monkeypatch):
    broken = type("Broken", (Driven,), {"api_version": 1})
    monkeypatch.setitem(drivers.DRIVERS, "fake", lambda pin: broken())
    row = drivers.discover(fake_spec)["fake"]
    assert (
        f"api_version: speaks 1, this orchestrator speaks {BENCHMARK_API_VERSION}" in row.problems
    )


def test_a_driver_that_raises_is_reported_not_raised(fake_spec, monkeypatch):
    def explode(pin):
        raise OSError("disk gone")

    monkeypatch.setitem(drivers.DRIVERS, "fake", explode)
    row = drivers.discover(fake_spec)["fake"]
    assert row.problems == ["could not be found: OSError: disk gone"]

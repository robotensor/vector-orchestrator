from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

from vector_orchestrator.spec import load_spec, load_spec_file

#: The directory holding the fake benchmark's package and its stand-in simulator.
FAKE_SITE = Path(__file__).resolve().parent / "fake_benchmark"
#: Modules the fake brings; dropped after each test so none leaks into the next.
FAKE_MODULE_PREFIXES = ("vector_fake_benchmark", "vector_fake_simulator")
#: The small contract duel tests run: two tasks, two runs each, a short act timeout (a benchmark
#: reaching for a policy that died waits that long, and duel tests kill policies on purpose).
DUEL_TASKS = ["place_cube_plate", "stack_two_blocks"]


@pytest.fixture(scope="session")
def spec():
    """The shipped contract."""
    return load_spec()


@pytest.fixture
def spec_doc(spec):
    """A mutable copy of the shipped contract."""
    return json.loads(spec.path.read_text())


@pytest.fixture
def write_spec(tmp_path):
    """Save a (modified) contract and load it through the validator."""

    def write(doc, name="spec.json"):
        path = tmp_path / name
        path.write_text(json.dumps(doc, indent=1))
        return load_spec_file(path)

    return write


@pytest.fixture
def duel_spec(spec_doc, write_spec):
    doc = {**spec_doc, "tasks": list(DUEL_TASKS), "runs_per_task": 2}
    doc["budgets"] = {**doc["budgets"], "act": 2.0}
    return write_spec(doc, name="duel-spec.json")


@pytest.fixture(autouse=True)
def _forget_fake_modules():
    yield
    for name in [m for m in sys.modules if m.startswith(FAKE_MODULE_PREFIXES)]:
        del sys.modules[name]


@pytest.fixture
def fake_module(monkeypatch):
    """The fake benchmark's package, importable."""
    monkeypatch.syspath_prepend(str(FAKE_SITE))
    return importlib.import_module("vector_fake_benchmark")


@pytest.fixture
def fake_benchmark(fake_module):
    return fake_module.FakeBenchmark()

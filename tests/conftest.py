from __future__ import annotations

import copy
import importlib
import json
import os
import sys
from pathlib import Path

import pytest

from vector_orchestrator.benchmarks import drivers
from vector_orchestrator.spec import load_spec, load_spec_file

TRACK = "franka_1arm"

#: The directory holding the fake benchmark's package and its stand-in simulator: prepended to
#: `sys.path` and registered as the `fake` driver, it runs a duel the way RoboTwin's driver does.
FAKE_SITE = Path(__file__).resolve().parent / "fake_benchmark"
FAKE_PIN = {
    "repo": "robotensor/fake-benchmark",
    "branch": None,
    "module": "vector_fake_benchmark.command",
    "api_version": 2,
}
#: Modules the fake distributions bring; dropped after each test so none leaks into the next.
FAKE_MODULE_PREFIXES = ("vector_fake_benchmark", "vector_fake_simulator", "vector_variant_")


@pytest.fixture(scope="session")
def spec():
    return load_spec()


@pytest.fixture(scope="session")
def track():
    return TRACK


@pytest.fixture
def spec_doc(spec):
    """A mutable copy of the shipped contract, so a rule is tested against the real spec."""
    return json.loads(spec.path.read_text())


@pytest.fixture
def sandbox_spec(spec, tmp_path):
    """The contract, with the sandbox user this process can hand a directory to: root can give
    it to the spec's user, anyone else only to themselves. The limits are the spec's own."""
    if os.getuid() == 0:
        return spec
    doc = json.loads(spec.path.read_text())
    doc["submission"]["sandbox"]["user"] = f"{os.getuid()}:{os.getgid()}"
    (tmp_path / "sandbox-spec.json").write_text(json.dumps(doc))
    return load_spec_file(tmp_path / "sandbox-spec.json")


@pytest.fixture(autouse=True)
def shared_mounts(request, monkeypatch):
    """The pure suite mounts no filesystem: the shared directory's tmpfs is recorded here, as
    `("mount", directory, uid, gid)` and `("umount", directory)`, instead of mounted. A container
    test mounts it for real."""
    calls: list[tuple] = []
    if request.node.get_closest_marker("container") is None:
        monkeypatch.setattr(
            "vector_orchestrator.submissions.container.mount_shared_dir",
            lambda directory, uid, gid: calls.append(("mount", Path(directory), uid, gid)),
        )
        monkeypatch.setattr(
            "vector_orchestrator.submissions.container.unmount_shared_dir",
            lambda directory: calls.append(("umount", Path(directory))),
        )
    return calls


@pytest.fixture
def write_spec(tmp_path):
    """Save a (modified) contract and load it through the validator."""

    def write(doc, name="spec.json"):
        path = tmp_path / name
        path.write_text(json.dumps(doc, indent=1))
        return load_spec_file(path)

    return write


def fake_spec_doc(doc: dict, pin: dict | None = None) -> dict:
    """The shipped contract with every skill moved onto the fake benchmark, which is what a test
    without RoboTwin runs the franka_1arm track on."""
    doc = copy.deepcopy(doc)
    doc["benchmarks"] = {"fake": dict(FAKE_PIN if pin is None else pin)}
    for skill in doc["skills"].values():
        skill["benchmark"] = "fake"
    return doc


@pytest.fixture
def fake_spec(spec_doc, write_spec):
    return write_spec(fake_spec_doc(spec_doc), name="fake-spec.json")


@pytest.fixture
def duel_spec(spec_doc, write_spec):
    """`fake_spec` with a short `act_timeout_s`: a benchmark reaching for a policy that died waits
    that long before it gives up, and a duel test kills policies on purpose."""
    doc = fake_spec_doc(spec_doc)
    doc["budgets"]["act_timeout_s"] = 2.0
    return write_spec(doc, name="duel-spec.json")


@pytest.fixture(autouse=True)
def _forget_fake_modules():
    yield
    for name in [m for m in sys.modules if m.startswith(FAKE_MODULE_PREFIXES)]:
        del sys.modules[name]


@pytest.fixture
def fake_installed(monkeypatch):
    """The fake benchmark, on `sys.path` and registered as the `fake` driver."""
    monkeypatch.syspath_prepend(str(FAKE_SITE))
    monkeypatch.setitem(
        drivers.DRIVERS,
        "fake",
        lambda pin: importlib.import_module("vector_fake_benchmark").BENCHMARK,
    )
    return FAKE_SITE

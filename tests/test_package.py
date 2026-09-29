import importlib
import json
import os
import subprocess
import sys

from conftest import FAKE_SITE

# Simulator and model stacks a benchmark or a submission may use; none belongs in this process.
# `vector_fake_simulator` is the fake benchmark's stand-in for one.
FORBIDDEN = {"sapien", "mujoco", "robosuite", "torch", "curobo", "vector_fake_simulator"}


def test_the_package_imports_no_simulator():
    before = set(sys.modules)
    for name in (
        "vector_orchestrator",
        "vector_orchestrator.cli",
        "vector_orchestrator.duel.orchestrate",
    ):
        importlib.import_module(name)
    loaded = {name.split(".")[0] for name in set(sys.modules) - before}
    assert not loaded & FORBIDDEN


PROBE = """
import json, sys
import vector_fake_benchmark
from vector_orchestrator.spec import load_spec_file
from vector_orchestrator.units import deal

spec = load_spec_file(sys.argv[1])
fake = vector_fake_benchmark.FakeBenchmark()
fake.prepare(spec.tasks)
units = deal(spec, "d" * 64)
commands = [fake.materialize_command(unit=u, out_dir="/tmp/x") for u in units]
print(json.dumps({"units": len(units), "modules": sorted({m.split(".")[0] for m in sys.modules})}))
"""


def test_dealing_and_driving_the_benchmark_loads_no_simulator(duel_spec):
    done = subprocess.run(
        [sys.executable, "-c", PROBE, str(duel_spec.path)],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(FAKE_SITE)},
        check=True,
    )
    probe = json.loads(done.stdout)
    assert probe["units"] == 4 and "vector_fake_benchmark" in probe["modules"]
    assert not set(probe["modules"]) & FORBIDDEN

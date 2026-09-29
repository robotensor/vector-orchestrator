"""The contract: the shipped `spec.json` loads, and one it could not honour is refused."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest

from vector_orchestrator.spec import SpecError, load_spec_file, validate_spec

RUNTIME = (
    Path(__file__).resolve().parents[1] / "packages" / "vector-runtime" / "src" / "vector_runtime"
)


def test_the_shipped_spec_is_the_competition(spec):
    assert spec.version >= 1 and spec.level == 1
    assert len(spec.tasks) == 16 and len(set(spec.tasks)) == 16
    assert spec.runs_per_task == 10
    assert (spec.margin, spec.alpha, spec.max_void) == (3.0, 0.05, 0.2)
    assert spec.benchmark == {"repo": "robotensor/RoboTwin-Vector", "branch": "vector"}
    assert set(spec.raw) == {
        "version",
        "benchmark",
        "level",
        "tasks",
        "runs_per_task",
        "crown",
        "model",
        "budgets",
    }
    assert len(spec.fingerprint) == 64


def test_the_spec_pins_the_runtime_this_checkout_holds(spec):
    manifest = RUNTIME / f"{spec.model['architecture']}.json"
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == spec.model["tensors_sha256"]


@pytest.mark.parametrize(
    ("edit", "problem"),
    [
        (lambda d: d.update(version=0), "version"),
        (lambda d: d["benchmark"].update(repo="not a repo"), "benchmark.repo"),
        (lambda d: d.update(tasks=[]), "tasks"),
        (lambda d: d.update(tasks=["a", "a"]), "tasks"),
        (lambda d: d.update(tasks=["Bad Task"]), "tasks"),
        (lambda d: d.update(runs_per_task=0), "runs_per_task"),
        (lambda d: d["crown"].update(alpha=1.5), "crown.alpha"),
        (lambda d: d["crown"].update(max_void=2), "crown.max_void"),
        (lambda d: d["model"].update(tensors_sha256="x"), "model.tensors_sha256"),
        (lambda d: d["budgets"].pop("act"), "budgets.act"),
        (lambda d: d["budgets"].update(policy=4000), "budgets.policy < budgets.unit"),
        (lambda d: d["budgets"].update(stall_retries=-1), "budgets.stall_retries"),
    ],
)
def test_a_spec_that_cannot_be_honoured_is_refused(spec_doc, write_spec, edit, problem):
    doc = copy.deepcopy(spec_doc)
    edit(doc)
    assert any(e.startswith(problem) for e in validate_spec(doc)), validate_spec(doc)
    with pytest.raises(SpecError):
        write_spec(doc)


def test_the_shipped_spec_validates_clean(spec_doc):
    assert validate_spec(spec_doc) == []
    assert validate_spec([]) == ["a spec is a JSON object"]


def test_a_spec_file_loads_and_fingerprints_its_content(spec_doc, write_spec, tmp_path):
    a = write_spec(spec_doc, "a.json")
    b = write_spec({**spec_doc, "runs_per_task": 3}, "b.json")
    assert a.fingerprint != b.fingerprint and b.runs_per_task == 3
    assert load_spec_file(tmp_path / "a.json").fingerprint == a.fingerprint

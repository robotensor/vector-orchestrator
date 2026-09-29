"""The contract, `spec.json`: read once, validated, and read through `Spec` everywhere.

version        bumping it changes every duel id
benchmark      the RoboTwin-Vector checkout the units run on: repo and branch
level          the level benchmark's level the scenes are drawn at
tasks          the tasks, each scored on its own
runs_per_task  units per task and side in a duel
step_ratio     a rollout stops after this many times the expert's steps in its scene (or at the
               task's own step limit): lower it as models get faster
crown          margin (points), alpha (sign test), max_void (fraction of units)
model          the architecture submissions hold, and the digest of its tensor manifest
budgets        seconds: policy_start, act, materialize, unit, policy, side, duel, stall;
               and stall_retries
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from .canon import canonical_sha256

SPEC_ENV = "VECTOR_ORCHESTRATOR_SPEC"
TASK_RE = re.compile(r"[a-z][a-z0-9_]*")
REPO_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*")
HEX64_RE = re.compile(r"[0-9a-f]{64}")
BUDGETS = ("policy_start", "act", "materialize", "unit", "policy", "side", "duel", "stall")


class SpecError(ValueError):
    """A contract the orchestrator cannot honour."""


def _positive(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0


def validate_spec(doc: Any) -> list[str]:
    """Every problem with a contract, as short paths; empty when it can be honoured."""
    if not isinstance(doc, dict):
        return ["a spec is a JSON object"]
    errors: list[str] = []

    def need(path: str, ok: bool) -> None:
        if not ok:
            errors.append(path)

    need("version: int", isinstance(doc.get("version"), int) and _positive(doc.get("version")))
    bench = doc.get("benchmark") if isinstance(doc.get("benchmark"), dict) else {}
    need("benchmark.repo: owner/name", REPO_RE.fullmatch(str(bench.get("repo", ""))) is not None)
    need("benchmark.branch: str", isinstance(bench.get("branch"), str) and bool(bench["branch"]))
    need("level: int >= 1", isinstance(doc.get("level"), int) and _positive(doc.get("level")))
    tasks = doc.get("tasks")
    need(
        "tasks: distinct task names",
        isinstance(tasks, list)
        and bool(tasks)
        and all(isinstance(t, str) and TASK_RE.fullmatch(t) for t in tasks)
        and len(set(tasks)) == len(tasks),
    )
    runs = doc.get("runs_per_task")
    need("runs_per_task: int >= 1", isinstance(runs, int) and _positive(runs))
    need("step_ratio > 0", _positive(doc.get("step_ratio")))
    crown = doc.get("crown") if isinstance(doc.get("crown"), dict) else {}
    need("crown.margin in [0, 100]", _positive(crown.get("margin")) or crown.get("margin") == 0)
    need("crown.alpha in (0, 1)", _positive(crown.get("alpha")) and crown["alpha"] < 1)
    need(
        "crown.max_void in [0, 1]",
        isinstance(crown.get("max_void"), (int, float)) and 0 <= crown["max_void"] <= 1,
    )
    model = doc.get("model") if isinstance(doc.get("model"), dict) else {}
    need(
        "model.architecture: str",
        isinstance(model.get("architecture"), str) and bool(model["architecture"]),
    )
    need(
        "model.tensors_sha256: hex64",
        HEX64_RE.fullmatch(str(model.get("tensors_sha256", ""))) is not None,
    )
    budgets = doc.get("budgets") if isinstance(doc.get("budgets"), dict) else {}
    for key in BUDGETS:
        need(f"budgets.{key} > 0", _positive(budgets.get(key)))
    retries = budgets.get("stall_retries")
    need(
        "budgets.stall_retries: int >= 0",
        isinstance(retries, int) and not isinstance(retries, bool) and retries >= 0,
    )
    # What a policy's budget leaves of its unit is the benchmark's own time to write its result.
    if _positive(budgets.get("policy")) and _positive(budgets.get("unit")):
        need("budgets.policy < budgets.unit", budgets["policy"] < budgets["unit"])
    return errors


@dataclass(frozen=True)
class Spec:
    raw: dict[str, Any]
    path: Path
    fingerprint: str

    @property
    def version(self) -> int:
        return int(self.raw["version"])

    @property
    def benchmark(self) -> dict[str, str]:
        return dict(self.raw["benchmark"])

    @property
    def level(self) -> int:
        return int(self.raw["level"])

    @property
    def tasks(self) -> tuple[str, ...]:
        return tuple(self.raw["tasks"])

    @property
    def runs_per_task(self) -> int:
        return int(self.raw["runs_per_task"])

    @property
    def step_ratio(self) -> float:
        return float(self.raw["step_ratio"])

    @property
    def margin(self) -> float:
        """The crown's margin in percentage points."""
        return float(self.raw["crown"]["margin"])

    @property
    def alpha(self) -> float:
        return float(self.raw["crown"]["alpha"])

    @property
    def max_void(self) -> float:
        return float(self.raw["crown"]["max_void"])

    @property
    def model(self) -> dict[str, str]:
        return dict(self.raw["model"])

    @property
    def budgets(self) -> dict[str, float]:
        return {k: float(v) for k, v in self.raw["budgets"].items() if k != "stall_retries"}

    @property
    def stall_retries(self) -> int:
        return int(self.raw["budgets"]["stall_retries"])


def spec_path() -> Path:
    """`$VECTOR_ORCHESTRATOR_SPEC`, else the one packaged with the wheel, else the checkout's."""
    explicit = os.environ.get(SPEC_ENV)
    if explicit:
        return Path(explicit)
    packaged = Path(__file__).resolve().parent / "data" / "spec.json"
    if packaged.is_file():
        return packaged
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file() and (parent / "spec.json").is_file():
            return parent / "spec.json"
    raise FileNotFoundError(f"spec.json not found; set {SPEC_ENV}")


def load_spec_file(path: str | Path) -> Spec:
    p = Path(path)
    doc = json.loads(p.read_text(encoding="utf-8"))
    errors = validate_spec(doc)
    if errors:
        raise SpecError(f"{p}: invalid spec: " + ", ".join(errors))
    return Spec(raw=doc, path=p, fingerprint=canonical_sha256(doc))


@lru_cache(maxsize=4)
def _cached(path: str) -> Spec:
    return load_spec_file(path)


def load_spec(path: str | Path | None = None) -> Spec:
    return _cached(str(Path(path) if path else spec_path()))

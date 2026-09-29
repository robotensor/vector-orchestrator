"""A fake benchmark, standing where `robotwin.RoboTwin` stands: any task, no simulator.

The pure half below imports the standard library, and numpy inside `verify_prompt` to read a
prompt. The command half is `command.py`, a small script run as a subprocess, which imports
`vector_fake_simulator` - a stand-in for SAPIEN - so a test can check the orchestrator never loads it.

`run_command` reads the unit's `fake_behaviour` (default `policy`: drive the served policy) so a
test can make a unit fail, crash, hang or write nothing, its `fake_void_cause` for
`policy_then_void`, and its `fake_step_s`, how long the simulator takes over each step;
`materialize_command` reads its `fake_materialize` (default `succeed`) the same way.
`materialize` builds on the unit's first `scene_seeds` candidate (the second under
`fake_materialize: reject_first`) and says which in its result.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

COMMAND = str(Path(__file__).resolve().parent / "command.py")

#: What `run` keeps back from `--unit-timeout-s` to write its result, as RoboTwin's does.
RESULT_RESERVE_S = 1.0


def candidate_seeds(unit: Any) -> list[int]:
    """The scene seeds a unit's prompt may be built on."""
    return list(unit["scene_seeds"])


class FakeBenchmark:
    id = "fake"
    workdir = None
    result_reserve_s = RESULT_RESERVE_S

    def __init__(self) -> None:
        #: The task lists `prepare` was asked about.
        self.prepared: list[list[str]] = []

    def prepare(self, tasks: Any) -> None:
        self.prepared.append(list(tasks))

    def info(self) -> dict[str, Any]:
        return {"commit": "fake", "source_sha256": "fake"}

    def verify_prompt(self, *, path: str, unit: Any) -> dict[str, Any]:
        import io
        import zipfile

        import numpy as np

        problems = []
        try:
            data = Path(path).read_bytes()
            with np.load(io.BytesIO(data), allow_pickle=False) as arrays:
                doc = json.loads(bytes(arrays["meta"]).decode())
        except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
            return {"ok": False, "sha256": "", "problems": [f"unreadable prompt: {exc}"]}
        if not isinstance(doc, dict):
            return {"ok": False, "sha256": "", "problems": ["the prompt's meta is not an object"]}
        if doc.get("task") != unit["task"]:
            problems.append(f"task {doc.get('task')!r} is not the unit's {unit['task']!r}")
        candidates = candidate_seeds(unit)
        if doc.get("scene_seed") not in candidates:
            problems.append(f"scene_seed {doc.get('scene_seed')} is not the unit's {candidates}")
        return {
            "ok": not problems,
            "sha256": hashlib.sha256(data).hexdigest(),
            "problems": problems,
            "scene_seed": doc.get("scene_seed"),
        }

    def read_result(self, *, out_dir: str) -> dict[str, Any]:
        try:
            return json.loads((Path(out_dir) / "result.json").read_text())
        except (OSError, ValueError):
            return {"success": None, "void": True, "steps": None, "error": "unreadable result.json"}

    def materialize_command(self, *, unit: Any, out_dir: str) -> list[str]:
        argv = [
            sys.executable,
            COMMAND,
            "materialize",
            "--out",
            out_dir,
            "--task",
            str(unit["task"]),
        ]
        for seed in candidate_seeds(unit):
            argv += ["--scene-seed", str(seed)]
        return [*argv, "--behaviour", str(unit.get("fake_materialize", "succeed"))]

    def run_command(
        self,
        *,
        unit: Any,
        prompt: str,
        out_dir: str,
        policy_address: str,
        authkey_env: str,
        **extra: Any,
    ) -> list[str]:
        argv = [
            sys.executable,
            COMMAND,
            "run",
            "--prompt",
            prompt,
            "--out",
            out_dir,
            "--policy-address",
            policy_address,
            "--authkey-env",
            authkey_env,
            "--behaviour",
            str(unit.get("fake_behaviour", "policy")),
            "--act-timeout-s",
            str(float(extra.get("act_timeout_s", 30.0))),
            "--step-s",
            str(float(unit.get("fake_step_s", 0.0))),
            "--void-cause",
            str(unit.get("fake_void_cause", "")),
        ]
        for key, flag in (
            ("unit_timeout_s", "--unit-timeout-s"),
            ("policy_budget_s", "--policy-budget-s"),
        ):
            if extra.get(key) is not None:
                argv += [flag, repr(float(extra[key]))]
        if extra.get("policy_log"):
            argv += ["--policy-log", str(extra["policy_log"])]
        return argv


BENCHMARK = FakeBenchmark()

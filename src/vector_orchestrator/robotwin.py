"""RoboTwin-Vector, driven from its checkout: `python -m robotensor_bench.cli`, never imported.

The checkout (`$ROBOTWIN_BENCH_ROOT`) runs under the simulator environment's interpreter
(`$ROBOTWIN_BENCH_PYTHON`). Before a duel the orchestrator asks it for its task list and checks the
level's task config (`config check`), which also reports the harness: its command-line contract,
source digest and commit. Every command then names that digest (`--expect-source-sha256`), and
every result is held to it.

    demo make    a unit's demonstration: prompt.npz, demonstration.mp4, result.json
    demo verify  whether a prompt is the one its unit asked for, without a simulator
    eval run     a demonstration against a served policy: result.json, evaluation.mp4
"""

from __future__ import annotations

import json
import math
import numbers
import os
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

#: The command-line contract this orchestrator drives the harness by.
API_VERSION = 2
MODULE = "robotensor_bench.cli"
ROOT_ENV = "ROBOTWIN_BENCH_ROOT"
PYTHON_ENV = "ROBOTWIN_BENCH_PYTHON"
DEFAULT_PYTHON = "/root/miniforge3/envs/robotwin/bin/python"
#: The ray-tracing denoiser a host's cameras must render with, handed to each command.
DENOISER_ENV = "ROBOTWIN_BENCH_DENOISER"
EMBODIMENT = "franka-panda"
#: Control steps per demonstration frame.
SAVE_FREQ = 15
QUERY_TIMEOUT_S = 300.0

PROMPT_FILE = "prompt.npz"
DEMONSTRATION_CLIP = "demonstration.mp4"
EVALUATION_CLIP = "evaluation.mp4"
RESULT_FILE = "result.json"
#: Whose a void is, as the harness writes it.
VOID_CAUSES = ("harness", "policy")


class Unavailable(RuntimeError):
    """The checkout cannot be used: missing, not the fork, or a command of it failed."""


def unit_flags(unit: Mapping[str, Any], level: int) -> list[str]:
    """What a unit is, as `demo make` builds it and `demo verify` holds a prompt to: its task at the
    level, its demonstration candidates and its scene candidates, in order."""
    demo, scene = _seeds(unit.get("demo_seeds")), _seeds(unit.get("scene_seeds"))
    if set(demo) & set(scene):
        raise ValueError("a unit's demo_seeds and scene_seeds share a seed")
    return [
        *(
            "--task",
            str(unit["task"]),
            "--task-config",
            f"level{level}",
            "--embodiment",
            EMBODIMENT,
        ),
        *("--save-freq", str(SAVE_FREQ)),
        *(
            "--seed-candidates",
            ",".join(map(str, demo)),
            "--eval-seed-candidates",
            ",".join(map(str, scene)),
        ),
    ]


def _seeds(seeds: Any) -> list[int]:
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(isinstance(s, bool) or not isinstance(s, int) or not 0 <= s < 2**32 for s in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError(f"{seeds!r} are not distinct seeds in [0, 2**32)")
    return list(seeds)


def _seconds(key: str, value: Any) -> str:
    if (
        isinstance(value, bool)
        or not isinstance(value, numbers.Real)
        or not 0 < float(value) < math.inf
    ):
        raise ValueError(f"{key} must be a positive, finite number of seconds, not {value!r}")
    return repr(float(value))


def _address(address: str) -> str:
    """`host:port` as given; a Unix socket path made absolute."""
    host, sep, port = address.rpartition(":")
    if sep and host and port.isdigit() and "/" not in address:
        return address
    return os.path.abspath(address)


def denoiser_flags() -> list[str]:
    value = os.environ.get(DENOISER_ENV)
    if not value:
        return []
    if value not in ("oidn", "none"):
        raise ValueError(f"{DENOISER_ENV} must be oidn or none, not {value!r}")
    return ["--denoiser", value]


def read_result(out_dir: str | Path, source_sha256: str | None) -> dict[str, Any]:
    """The `result.json` a command wrote, made consistent: `success` a bool or None exactly when
    `void`, `steps` an int or None, `error` a reason whenever void, `void_cause` harness or policy.
    A result that is missing, unreadable or written by other harness source is a harness void."""
    path = Path(out_dir) / RESULT_FILE
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        doc, reason = None, f"unreadable {RESULT_FILE}: {type(exc).__name__}: {exc}"
    if not isinstance(doc, dict):
        reason = reason if doc is None else f"{RESULT_FILE} holds {type(doc).__name__}"
        return {
            "success": None,
            "void": True,
            "void_cause": "harness",
            "steps": None,
            "error": reason,
        }
    if source_sha256 is not None and doc.get("source_sha256") != source_sha256:
        return {
            **doc,
            "success": None,
            "void": True,
            "void_cause": "harness",
            "steps": None,
            "error": f"{RESULT_FILE} was written by harness source {doc.get('source_sha256')!r}, "
            f"not the {source_sha256} the duel checked",
        }
    success = doc.get("success")
    void = doc.get("void") is not False or not isinstance(success, bool)
    error = doc.get("error") if isinstance(doc.get("error"), str) and doc.get("error") else None
    steps = doc.get("steps")
    finite = (
        isinstance(steps, (int, float)) and not isinstance(steps, bool) and math.isfinite(steps)
    )
    cause = doc.get("void_cause") if doc.get("void_cause") in VOID_CAUSES else "harness"
    return {
        **doc,
        "success": None if void else success,
        "void": void,
        "void_cause": cause if void else None,
        "steps": int(steps) if finite and not void else None,
        "error": (error or "the benchmark wrote a void result without a reason") if void else error,
    }


class RoboTwin:
    """The checkout at `root`, run by `python`, for scenes of `level`."""

    id = "robotwin"

    def __init__(self, root: str | Path, python: str | None = None, level: int = 1) -> None:
        self.root = Path(root).resolve()
        self.python = python or DEFAULT_PYTHON
        self.level = int(level)
        self._harness: dict[str, Any] | None = None

    @classmethod
    def from_environment(cls, level: int, environ: Mapping[str, str] | None = None) -> RoboTwin:
        environ = os.environ if environ is None else environ
        root = environ.get(ROOT_ENV)
        if not root:
            raise Unavailable(f"${ROOT_ENV} is not set: set it to the RoboTwin-Vector checkout")
        fork = cls(root, environ.get(PYTHON_ENV), level)
        if not (fork.root / MODULE.replace(".", "/")).with_suffix(".py").is_file():
            raise Unavailable(f"{fork.root} is not a RoboTwin-Vector checkout: it has no {MODULE}")
        if not (os.path.isfile(fork.python) and os.access(fork.python, os.X_OK)):
            raise Unavailable(
                f"the simulator interpreter {fork.python} is missing: set ${PYTHON_ENV}"
            )
        return fork

    @property
    def workdir(self) -> str:
        return str(self.root)

    def argv(self, *arguments: str) -> list[str]:
        return [self.python, "-m", MODULE, *arguments]

    def ask(self, *arguments: str) -> Any:
        """Run one simulator-free command in the checkout and return the JSON it printed."""
        from .runner import benchmark_environment

        env = {**benchmark_environment(os.environ, ""), "PYTHONPATH": self.workdir}
        env.pop("", None)
        try:
            done = subprocess.run(
                self.argv(*arguments),
                cwd=self.workdir,
                env=env,
                capture_output=True,
                text=True,
                timeout=QUERY_TIMEOUT_S,
                stdin=subprocess.DEVNULL,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise Unavailable(f"{' '.join(arguments[:2])} could not run: {exc}") from exc
        if done.returncode not in (0, 4):
            raise Unavailable(
                f"{' '.join(arguments[:2])} exited {done.returncode}: {done.stderr.strip()[-400:]}"
            )
        lines = [line for line in done.stdout.splitlines() if line.strip()]
        for text in (done.stdout, lines[-1] if lines else ""):
            try:
                return json.loads(text)
            except ValueError:
                continue
        raise Unavailable(
            f"{' '.join(arguments[:2])} printed no JSON: {(done.stderr or done.stdout).strip()[-400:]}"
        )

    def prepare(self, tasks: Sequence[str]) -> None:
        """Before a duel: every task exists in the checkout, and the level's task config loads (its
        assets included) under a harness speaking `API_VERSION`."""
        table = self.ask("tasks", "list", "--json")
        known = table.get("tasks") if isinstance(table, Mapping) else None
        missing = [t for t in tasks if not isinstance(known, Mapping) or t not in known]
        if missing:
            raise Unavailable(f"the checkout has no task {', '.join(missing)}")
        report = self.ask(
            "config", "check", "--task-config", f"level{self.level}", "--embodiment", EMBODIMENT
        )
        harness = report.get("harness") if isinstance(report, Mapping) else None
        if not isinstance(harness, Mapping) or not harness.get("source_sha256"):
            raise Unavailable("config check reported no harness source digest")
        if harness.get("api") != API_VERSION:
            raise Unavailable(
                f"the harness speaks contract {harness.get('api')!r}, not {API_VERSION}"
            )
        if self._harness is not None and harness["source_sha256"] != self._harness["source_sha256"]:
            raise Unavailable("the checkout's harness source changed between two checks")
        self._harness = dict(harness)

    @property
    def harness(self) -> dict[str, Any]:
        if self._harness is None:
            raise Unavailable("the checkout was not prepared")
        return self._harness

    def info(self) -> dict[str, Any]:
        """What a duel publishes of the benchmark: the checkout's commit and harness digest."""
        return {
            "commit": self.harness.get("commit"),
            "source_sha256": self.harness["source_sha256"],
        }

    @property
    def result_reserve_s(self) -> float:
        """What `eval run` keeps back from its timeout to write its result."""
        value = (self.harness.get("limits") or {}).get("result_reserve_s")
        ok = (
            isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
        )
        return max(0.0, float(value)) if ok else 0.0

    def verify_prompt(self, *, path: str, unit: Mapping[str, Any]) -> dict[str, Any]:
        """`demo verify`: whether the prompt at `path` is the one `unit` asked for, and its sha256."""
        prompt = Path(path)
        verdict = self.ask(
            "demo", "verify", "--demo", str(prompt.parent.resolve()), *unit_flags(unit, self.level),
            "--expect-source-sha256", self.harness["source_sha256"],
        )  # fmt: skip
        if not isinstance(verdict, dict):
            raise Unavailable(f"demo verify printed {type(verdict).__name__}, not a verdict")
        return verdict

    def read_result(self, *, out_dir: str) -> dict[str, Any]:
        return read_result(out_dir, self.harness["source_sha256"])

    def materialize_command(self, *, unit: Mapping[str, Any], out_dir: str) -> list[str]:
        return self.argv(
            "demo", "make", *unit_flags(unit, self.level), "--legal-joints", *denoiser_flags(),
            "--expect-source-sha256", self.harness["source_sha256"], "--out", os.path.abspath(out_dir),
        )  # fmt: skip

    def run_command(
        self,
        *,
        unit: Mapping[str, Any],
        prompt: str,
        out_dir: str,
        policy_address: str,
        authkey_env: str,
        act_timeout_s: float | None = None,
        policy_budget_s: float | None = None,
        unit_timeout_s: float | None = None,
        policy_log: str | None = None,
    ) -> list[str]:
        """`eval run` of the demonstration at `prompt` against the policy at `policy_address`. The
        key travels as the name `authkey_env`, never as a value."""
        unit_flags(unit, self.level)
        if not authkey_env or "=" in authkey_env:
            raise ValueError(f"authkey_env must name an environment variable, not {authkey_env!r}")
        argv = ["eval", "run", "--demo", str(Path(prompt).parent.resolve())]
        argv += ["--policy", _address(policy_address), "--authkey-env", authkey_env]
        for flag, value in (
            ("--act-timeout-s", act_timeout_s),
            ("--policy-budget-s", policy_budget_s),
            ("--unit-timeout-s", unit_timeout_s),
        ):
            if value is not None:
                argv += [flag, _seconds(flag, value)]
        if policy_log:
            argv += ["--policy-log", os.path.abspath(policy_log)]
        return self.argv(
            *argv, *denoiser_flags(), "--expect-source-sha256", self.harness["source_sha256"],
            "--out", os.path.abspath(out_dir),
        )  # fmt: skip

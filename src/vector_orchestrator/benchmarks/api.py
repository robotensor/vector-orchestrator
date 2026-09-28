"""What a benchmark is to the orchestrator, and how the orchestrator checks it.

A benchmark is a fork of its simulator with the competition's harness inside it, run from its
checkout, never installed: RoboTwin-Vector runs `python -m robotensor_bench.cli ...` in its own
environment. Nothing of a benchmark is imported here. For each benchmark it knows, the orchestrator
holds a driver (`drivers.DRIVERS`) with the surface below: `Benchmark` is that surface, a
`Protocol` checked structurally by `validate_benchmark`, so a test can stand a fake in its place.

The surface splits in two, and the split is the reason the design works:

`PURE_METHODS` run in the orchestrator's process with **no simulator, no assets and no GPU**. The
orchestrator host, CI and a laptop all read a benchmark's catalogue, deal its units and verify a
prompt; none of them can build a scene. A driver answers them itself (dealing units, reading a
result file) or asks the checkout through a command that needs no simulator (`tasks list`,
`demo verify`).

`COMMAND_METHODS` return an **argv**, not a result. Everything that needs a simulator happens in a
subprocess the orchestrator launches, from the benchmark's `workdir` when it has one, so the
orchestrator never imports SAPIEN or MuJoCo. The files each command writes into its `out_dir` are
named below (`PROMPT_FILE` and friends).

A driver may also have `prepare(suites)`, which asks the checkout what a duel over those suites
needs before any unit is dealt, and refuses (`BenchmarkRefused`) a checkout that cannot run them.
`api_version` is the command-line contract a driver speaks with its checkout; the spec pins it.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

#: The contract this orchestrator drives a benchmark's checkout by: its command line's verbs, flags,
#: files and exit codes. 1 was the in-process plugin ABI of the installed distributions it replaced.
BENCHMARK_API_VERSION = 2


class BenchmarkRefused(RuntimeError):
    """A benchmark has no driver here, its checkout is missing or unusable, or it does not match
    its pin."""


#: Importable and callable with no simulator, no assets and no GPU.
PURE_METHODS = ("info", "catalogue", "derive_units", "verify_prompt", "read_result")

#: Return the argv of a subprocess that does need those things.
COMMAND_METHODS = ("materialize_command", "run_command")

METHODS = PURE_METHODS + COMMAND_METHODS

#: Keyword parameters each method must accept, so the orchestrator can call it by name.
REQUIRED_KEYWORDS: dict[str, tuple[str, ...]] = {
    "derive_units": ("seed_material", "count", "suite", "category"),
    "verify_prompt": ("path", "unit"),
    "read_result": ("out_dir",),
    "materialize_command": ("unit", "out_dir"),
    "run_command": ("unit", "prompt", "out_dir", "policy_address", "authkey_env"),
}

#: Methods that must also take `**extra`: the orchestrator forwards keywords a later duel adds.
VAR_KEYWORD_METHODS = ("run_command",)

#: Written by `materialize_command`'s subprocess: the prompt's named arrays (including the
#: privileged `meta`, which never reaches a policy) ...
PROMPT_FILE = "prompt.npz"
#: ... the demonstration clip the dashboard plays ...
DEMONSTRATION_CLIP = "demonstration.mp4"
#: ... and, by both commands, the result `read_result` reads.
RESULT_FILE = "result.json"
#: Written by `run_command`'s subprocess: the rollout clip.
EVALUATION_CLIP = "evaluation.mp4"

#: Keys every unit `derive_units` returns must carry, and the keys its `instance_params` must.
UNIT_KEYS = ("task", "task_label", "instance_params")
INSTANCE_PARAM_KEYS = ("scene_seed", "embodiment")


@runtime_checkable
class Benchmark(Protocol):
    """One benchmark, as the orchestrator sees it."""

    #: Stable identifier, matching `benchmarks.<id>` and `skills.<skill>.benchmark` in the spec.
    id: str
    #: The `BENCHMARK_API_VERSION` this driver speaks.
    api_version: int

    # -- pure ---------------------------------------------------------------------------

    def info(self) -> dict[str, Any]:
        """Identity and shape: id, api_version, embodiment, action space, cameras and the commits
        it is pinned to. Recorded on every duel it runs."""

    def catalogue(self) -> dict[str, Any]:
        """What can be drawn from: `{"suites": {suite: [task, ...]}, "categories": {category:
        label}, "tasks": {task: {"category": ...}}}` and whatever else the benchmark publishes.
        No demonstrations - a catalogue is the menu, not the meal."""

    def derive_units(
        self, *, seed_material: str, count: int, suite: str, category: str | None = None
    ) -> list[dict[str, Any]]:
        """`count` units from `suite` (restricted to `category` when given), a pure function of
        its arguments.

        What a unit of a benchmark is - a scene seed, an embodiment - is its driver's to say. It
        must be reproducible by a third party holding the published record, so derive from a hash
        of `seed_material`, never from a global RNG whose stream depends on a library version.
        Each unit carries at least `UNIT_KEYS`, and its `instance_params` at least
        `INSTANCE_PARAM_KEYS`.
        """

    def verify_prompt(self, *, path: str, unit: Mapping[str, Any]) -> dict[str, Any]:
        """Check a materialized prompt is the one `unit` asked for, without a simulator.

        Returns at least `{"ok": bool, "sha256": str, "problems": [str, ...]}`, where `sha256` is
        the sha256 of the file's bytes: a duel publishes it as the unit's `prompt_sha256` and voids
        the unit when the two disagree. This is what lets anyone holding the published prompt
        confirm it, so it must read the file rather than trust a manifest.
        """

    def read_result(self, *, out_dir: str) -> dict[str, Any]:
        """One command's result, as its subprocess wrote it in `out_dir`.

        Returns at least `{"success": bool | None, "void": bool, "steps": int | None,
        "error": str | None}`. `success` is None exactly when the unit is void.
        """

    # -- commands -----------------------------------------------------------------------

    def materialize_command(self, *, unit: Mapping[str, Any], out_dir: str) -> Sequence[str]:
        """The argv that writes `PROMPT_FILE`, `DEMONSTRATION_CLIP` and `RESULT_FILE` for one
        unit into `out_dir`. The orchestrator gives it a directory of its own, never a side's."""

    def run_command(
        self,
        *,
        unit: Mapping[str, Any],
        prompt: str,
        out_dir: str,
        policy_address: str,
        authkey_env: str,
        **extra: Any,
    ) -> Sequence[str]:
        """The argv that runs one unit from the prompt at `prompt` against a policy already served
        at `policy_address`, writing `RESULT_FILE` and `EVALUATION_CLIP` into `out_dir`.

        `policy_address` is in the policy wire's form: a Unix socket path, or `host:port`.
        `authkey_env` is the *name* of the environment variable holding the policy's authkey (hex).
        The key itself never appears on a command line, where any process on the host can read it.
        """


def validate_benchmark(obj: Any) -> list[str]:
    """Everything wrong with `obj` as a benchmark, in the order a reader would find it.

    Structural, not nominal, so a test's fake stands in for a driver without subclassing it.
    Returns an empty list when the benchmark is usable.
    """
    errors: list[str] = []

    identifier = getattr(obj, "id", None)
    if not isinstance(identifier, str) or not identifier:
        errors.append("id: expected a non-empty string")

    version = getattr(obj, "api_version", None)
    if not isinstance(version, int) or isinstance(version, bool):
        errors.append("api_version: expected an integer")
    elif version != BENCHMARK_API_VERSION:
        errors.append(
            f"api_version: speaks {version}, this orchestrator speaks {BENCHMARK_API_VERSION}"
        )

    for name in METHODS:
        method = getattr(obj, name, None)
        if method is None:
            errors.append(f"{name}: missing")
            continue
        if not callable(method):
            errors.append(f"{name}: not callable")
            continue
        errors.extend(_keyword_errors(name, method))

    return errors


def _keyword_errors(name: str, method: Any) -> list[str]:
    """The keyword arguments the orchestrator passes, which `method` must accept by name."""
    required = REQUIRED_KEYWORDS.get(name, ())
    if not required:
        return []
    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):  # a builtin or a C callable: take it on trust
        return []
    parameters = signature.parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return []
    accepted = {
        n
        for n, p in parameters.items()
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    errors = [f"{name}: does not accept {n}" for n in required if n not in accepted]
    if name in VAR_KEYWORD_METHODS:
        errors.append(f"{name}: does not accept **extra")
    return errors

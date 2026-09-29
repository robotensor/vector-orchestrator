"""Every unit's prompt, produced once per duel, before either side runs.

A demonstration cannot be published ahead of time: it shows how to solve its task. So the
orchestrator has the benchmark produce each unit's prompt here, on this host, before either side starts, and publishes
the prompts' sha256 with the event. Both sides then run from those same files, which is the whole
of what makes a duel fair on RoboTwin: an expert's trajectory is not reproducible from its seed
(CuRobo's IK seed generators live as long as the environment, and planning retries stop on a wall
clock), so a prompt produced per side would differ per side.

With the duel's `Slots`, each unit's demonstration runs in a slot on one card, and the prompts go
into a live `Materialized` the sides read as they land, so a unit is played while later ones are
still being made. For each unit, and resumably:

1. `materialize_command(unit, out_dir)` runs as a subprocess with the allow-listed environment and
   `budgets.materialize`, in `<root>/<unit_id>/`.
2. Its `result.json` must say the demonstration succeeded, and `prompt.npz` must exist.
3. `verify_prompt(path, unit)` must say ok, and hash the file to what this module hashes it to:
   the published `prompt_sha256` is the sha256 of the file's bytes, which is what a third party
   holding the file checks. A `prompt_sha256` the result names must be that hash too.
4. The scene seed the prompt was built on is kept (`Prompt.scene_seed`): the result's
   `scene_seed`, which must be the one `verify_prompt` read from the file when it names one. A
   benchmark may choose the scene only here - RoboTwin's expert tries a unit's candidate seeds in
   order and keeps the first it solves - so the duel publishes it as the unit's
   `instance_params.scene_seed`, the scene both sides played.

A unit whose materialization fails or is rejected is **void for both sides**, with the reason; it
is not retried, since a second try would be a different demonstration than the one already
recorded. Each unit's prompt is appended to `<root>/manifest.jsonl` as it is done, so a restarted
duel re-uses it rather than producing a new one, after checking the file still has its hash.

Prompts stay in the run directory. The benchmark subprocess reads them from there, and the
prompt's privileged `meta` never crosses the policy socket.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..canon import sha256_file
from ..robotwin import DEMONSTRATION_CLIP, PROMPT_FILE, RESULT_FILE
from ..runner import (
    benchmark_environment,
    in_workdir,
    out_of_memory,
    outcome_from,
    run_argv,
    workdir_of,
)
from .orphans import Ledger, reap_ledger

log = logging.getLogger(__name__)

MANIFEST_FILE = "manifest.jsonl"
#: The materialize command's output, in its unit's directory. A run log, never published.
LOG_FILE = "materialize.log"


def scene_seed_of(value: Any) -> int | None:
    """`value` as a scene seed: an integer, never a bool; None for anything else."""
    return value if isinstance(value, int) and not isinstance(value, bool) else None


@dataclass
class Prompt:
    """One unit's prompt on this host, or why it has none."""

    unit_id: str
    path: Path | None = None
    sha256: str | None = None
    demo_clip: Path | None = None
    demo_sha256: str | None = None
    void: bool = False
    error: str | None = None
    wall_s: float = 0.0
    #: The scene seed the benchmark built the prompt on, when it says.
    scene_seed: int | None = None

    def changed(self) -> str | None:
        """Why the file is no longer the prompt that was recorded, or None while it is."""
        if self.void or self.path is None:
            return None
        try:
            found = sha256_file(self.path)
        except OSError as exc:
            return f"the prompt file cannot be read any more: {exc}"
        if found != self.sha256:
            return f"the prompt file changed after it was materialized ({found[:12]}...)"
        return None

    def to_line(self, root: Path) -> dict[str, Any]:
        def rel(path: Path | None) -> str | None:
            return None if path is None else str(path.relative_to(root))

        return {
            "unit_id": self.unit_id,
            "prompt": rel(self.path),
            "sha256": self.sha256,
            "demo_clip": rel(self.demo_clip),
            "demo_sha256": self.demo_sha256,
            "void": self.void,
            "error": self.error,
            "wall_s": self.wall_s,
            "scene_seed": self.scene_seed,
        }

    @classmethod
    def from_line(cls, root: Path, doc: Mapping[str, Any]) -> Prompt:
        def path(value: Any) -> Path | None:
            return root / str(value) if value else None

        return cls(
            unit_id=str(doc["unit_id"]),
            path=path(doc.get("prompt")),
            sha256=doc.get("sha256"),
            demo_clip=path(doc.get("demo_clip")),
            demo_sha256=doc.get("demo_sha256"),
            void=bool(doc.get("void")),
            error=doc.get("error"),
            wall_s=float(doc.get("wall_s") or 0.0),
            scene_seed=scene_seed_of(doc.get("scene_seed")),
        )


@dataclass
class Materialized:
    """A duel's prompts. Made live (`done=False`), it fills as they land and `ready` hands them to
    the sides; `finish` says no more will come, `failed` that materializing stopped short."""

    root: Path
    prompts: dict[str, Prompt]
    done: bool = True
    failed: bool = False
    _cond: threading.Condition = field(
        default_factory=threading.Condition, repr=False, compare=False
    )

    @property
    def void(self) -> int:
        return sum(1 for p in self.prompts.values() if p.void)

    def put(self, prompt: Prompt) -> None:
        with self._cond:
            self.prompts[prompt.unit_id] = prompt
            self._cond.notify_all()

    def finish(self, *, failed: bool = False) -> None:
        with self._cond:
            self.done, self.failed = True, self.failed or failed
            self._cond.notify_all()

    def ready(self, unit_ids: Iterable[str]) -> Iterator[str]:
        """`unit_ids` as their prompts land, in landing order; once done, the ones that never got
        one - unless materializing failed, when nothing more is handed out."""
        left = list(unit_ids)
        while left:
            with self._cond:
                while not self.done and not any(u in self.prompts for u in left):
                    self._cond.wait()
                if self.failed:
                    return
                landed = [u for u in left if u in self.prompts]
                if not landed and self.done:
                    landed = list(left)
            for unit_id in landed:
                left.remove(unit_id)
                yield unit_id


def read_manifest(root: Path) -> dict[str, Prompt]:
    """The prompts already produced under `root`, the last line for a unit winning."""
    out: dict[str, Prompt] = {}
    try:
        text = (root / MANIFEST_FILE).read_text(encoding="utf-8")
    except FileNotFoundError:
        return out
    for line in text.splitlines():
        try:
            doc = json.loads(line)
            prompt = Prompt.from_line(root, doc)
        except (ValueError, KeyError, TypeError):
            continue  # a torn final line from a crash: that unit is materialized again
        out[prompt.unit_id] = prompt
    return out


def _append(root: Path, prompt: Prompt) -> None:
    with open(root / MANIFEST_FILE, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(prompt.to_line(root), sort_keys=True) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def materialize_units(
    spec: Any,
    units: Iterable[Mapping[str, Any]],
    root: Path,
    *,
    benchmark: Any,
    env: Mapping[str, str] | None = None,
    timeout_s: float | None = None,
    deadline: float | None = None,
    stop: Callable[[], bool] | None = None,
    on_prompt: Callable[[Mapping[str, Any], Prompt], None] | None = None,
    workers: int = 1,
    stall_s: float | None = None,
    stall_retries: int = 0,
    slots: Any = None,
    into: Materialized | None = None,
    abort: threading.Event | None = None,
) -> Materialized:
    """Every unit's prompt under `root`, re-using what an earlier run of this duel produced.

    With `slots` (`slots.Slots`) each unit is made in a slot, on its card; `into` is a live
    `Materialized` to fill as prompts land, finished on the way out. `abort` set stops it at once:
    a unit cut short by it is not recorded, so a resumed duel makes it again.

    `deadline` is the duel's, as a `time.monotonic()`: no unit's materialization runs past it, and
    a unit not started by then has no prompt. `stop`, asked before each unit, ends the loop early
    when it says so (the duel is already certain to be void); the units after it have no entry.
    With `workers` above 1, that many units are materialized at once (each is a subprocess of its
    own); a unit's prompt is recorded as it finishes, so the manifest's order is the finishing
    order, and `stop` and the deadline are asked as each unit starts."""
    root.mkdir(parents=True, exist_ok=True)
    budget = float(spec.budgets["materialize"]) if timeout_s is None else timeout_s
    environ = dict(benchmark_environment(os.environ, "") if env is None else env)
    done = read_manifest(root)
    result = into if into is not None else Materialized(root=root, prompts={})
    try:
        if slots is None and workers > 1:
            from .slots import Slots

            slots = Slots(workers, [None], root / "slots")
        if slots is not None:
            _materialize_pooled(
                units,
                root,
                done=done,
                result=result,
                benchmark=benchmark,
                env=environ,
                budget=budget,
                deadline=deadline,
                stop=stop,
                on_prompt=on_prompt,
                slots=slots,
                stall_s=stall_s,
                stall_retries=stall_retries,
                abort=abort,
            )
        else:
            _materialize_in_order(
                units,
                root,
                done=done,
                result=result,
                benchmark=benchmark,
                env=environ,
                budget=budget,
                deadline=deadline,
                stop=stop,
                on_prompt=on_prompt,
                stall_s=stall_s,
                stall_retries=stall_retries,
            )
    except BaseException:
        result.finish(failed=True)
        raise
    result.finish(failed=abort is not None and abort.is_set())
    return result


def _materialize_in_order(
    units: Iterable[Mapping[str, Any]],
    root: Path,
    *,
    done: Mapping[str, Prompt],
    result: Materialized,
    benchmark: Any,
    env: Mapping[str, str],
    budget: float,
    deadline: float | None,
    stop: Callable[[], bool] | None,
    on_prompt: Callable[[Mapping[str, Any], Prompt], None] | None,
    stall_s: float | None,
    stall_retries: int,
) -> None:
    for unit in units:
        if stop is not None and stop():
            break
        unit_id = str(unit["unit_id"])
        prompt = done.get(unit_id)
        if prompt is not None:
            reason = prompt.changed()
            if reason is not None:
                prompt = Prompt(unit_id=unit_id, void=True, error=f"materialize: {reason}")
                _append(root, prompt)
        elif deadline is not None and time.monotonic() >= deadline:
            prompt = Prompt(
                unit_id=unit_id,
                void=True,
                error="materialize: the duel ran out of its wall-clock budget",
            )
            _append(root, prompt)
        else:
            left = budget if deadline is None else min(budget, deadline - time.monotonic())
            prompt = materialize_unit(
                benchmark,
                unit,
                root / unit_id,
                env=env,
                timeout_s=left,
                stall_s=stall_s,
                stall_retries=stall_retries,
            )
            _append(root, prompt)
        if prompt.void:
            log.warning("unit %s has no prompt: %s", unit_id, prompt.error)
        result.put(prompt)
        if on_prompt is not None:
            on_prompt(unit, prompt)


def _materialize_pooled(
    units: Iterable[Mapping[str, Any]],
    root: Path,
    *,
    done: Mapping[str, Prompt],
    result: Materialized,
    benchmark: Any,
    env: Mapping[str, str],
    budget: float,
    deadline: float | None,
    stop: Callable[[], bool] | None,
    on_prompt: Callable[[Mapping[str, Any], Prompt], None] | None,
    slots: Any,
    stall_s: float | None = None,
    stall_retries: int = 0,
    abort: threading.Event | None = None,
) -> None:
    """`materialize_units` with a unit in each of `slots` at a time. A unit's prompt is recorded
    (the manifest line, `on_prompt`) by the worker that made it, one at a time."""
    from concurrent.futures import ThreadPoolExecutor

    from .slots import MAKE

    lock = threading.Lock()

    def finish(unit: Mapping[str, Any], prompt: Prompt, append: bool) -> None:
        with lock:
            if append:
                _append(root, prompt)
            if prompt.void:
                log.warning("unit %s has no prompt: %s", prompt.unit_id, prompt.error)
            result.put(prompt)
            if on_prompt is not None:
                on_prompt(unit, prompt)

    def aborted() -> bool:
        return abort is not None and abort.is_set()

    def job(unit: Mapping[str, Any]) -> None:
        try:
            make(unit)
        except BaseException:
            if abort is not None:
                abort.set()
            result.finish(failed=True)
            raise

    def make(unit: Mapping[str, Any]) -> None:
        unit_id = str(unit["unit_id"])
        with slots.take(MAKE) as slot:
            if aborted() or (stop is not None and stop()):
                return
            if deadline is not None and time.monotonic() >= deadline:
                prompt = Prompt(
                    unit_id=unit_id,
                    void=True,
                    error="materialize: the duel ran out of its wall-clock budget",
                )
            else:
                left = budget if deadline is None else min(budget, deadline - time.monotonic())
                prompt = materialize_unit(
                    benchmark,
                    unit,
                    root / unit_id,
                    env={**env, **slot.env},
                    timeout_s=left,
                    stall_s=stall_s,
                    stall_retries=stall_retries,
                )
        if not aborted():
            finish(unit, prompt, append=True)

    with ThreadPoolExecutor(max_workers=len(slots), thread_name_prefix="materialize") as pool:
        futures = []
        for unit in units:
            unit_id = str(unit["unit_id"])
            prompt = done.get(unit_id)
            if prompt is not None:
                reason = prompt.changed()
                if reason is not None:
                    prompt = Prompt(unit_id=unit_id, void=True, error=f"materialize: {reason}")
                    finish(unit, prompt, append=True)
                else:
                    finish(unit, prompt, append=False)
                continue
            futures.append(pool.submit(job, unit))
        for future in futures:
            future.result()


def _ran_out_of_memory(out_dir: Path, done: Any) -> bool:
    """Whether a materialize command failed because the GPU ran out of memory: said in its log's
    end or its result's error."""
    error = None
    try:
        doc = json.loads((out_dir / RESULT_FILE).read_text(encoding="utf-8"))
        error = doc.get("error") if isinstance(doc, dict) and doc.get("ok") is not True else None
    except (OSError, ValueError):
        pass
    failed = done.returncode not in (0, None) or error is not None
    return failed and out_of_memory(done.log_tail, error)


def materialize_unit(
    benchmark: Any,
    unit: Mapping[str, Any],
    out_dir: Path,
    *,
    env: Mapping[str, str],
    timeout_s: float,
    stall_s: float | None = None,
    stall_retries: int = 0,
) -> Prompt:
    """One unit's prompt, produced and verified. Every failure is a void prompt, never a raise.
    A run that stalls (`run_argv`'s watchdog) is started again, `stall_retries` times, and one that
    ran out of GPU memory once, within the same `timeout_s`."""
    unit_id = str(unit["unit_id"])
    name = str(getattr(benchmark, "id", "benchmark"))
    started = time.monotonic()

    def void(reason: str) -> Prompt:
        return Prompt(
            unit_id=unit_id,
            void=True,
            error=f"{name}: materialize: {reason}",
            wall_s=round(time.monotonic() - started, 3),
        )

    try:
        argv = [str(a) for a in benchmark.materialize_command(unit=unit, out_dir=str(out_dir))]
    except Exception as exc:  # noqa: BLE001 - the benchmark's failure is this unit's
        return void(f"materialize_command failed: {type(exc).__name__}: {exc}")
    if not argv:
        return void("materialize_command returned an empty argv")
    out_dir.mkdir(parents=True, exist_ok=True)
    reap_ledger(out_dir)  # an expert a killed orchestrator left running must not write here
    for stale in (PROMPT_FILE, DEMONSTRATION_CLIP, RESULT_FILE):
        try:
            (out_dir / stale).unlink(missing_ok=True)
        except OSError as exc:
            return void(f"could not clear a stale {stale}: {exc}")

    attempt = 0
    retried_memory = False
    while True:
        left = timeout_s - (time.monotonic() - started)
        done = run_argv(
            argv,
            env=in_workdir(env, workdir_of(benchmark)),
            timeout_s=max(left, 0.0),
            log_path=out_dir / LOG_FILE,
            ledger=Ledger(out_dir),
            stall_s=stall_s,
            cwd=workdir_of(benchmark),
        )
        if (
            not retried_memory
            and not done.stalled
            and _ran_out_of_memory(out_dir, done)
            and time.monotonic() - started < timeout_s
        ):
            retried_memory = True
            log.warning("materializing %s ran out of GPU memory; starting it again", unit_id)
            for stale in (PROMPT_FILE, DEMONSTRATION_CLIP, RESULT_FILE):
                (out_dir / stale).unlink(missing_ok=True)
            continue
        if not done.stalled or attempt >= stall_retries:
            break
        attempt += 1
        log.warning(
            "materializing %s stalled; starting it again (%d of %d)",
            unit_id,
            attempt,
            stall_retries,
        )
        for stale in (PROMPT_FILE, DEMONSTRATION_CLIP, RESULT_FILE):
            (out_dir / stale).unlink(missing_ok=True)
    if done.stalled:
        return void(f"stalled {attempt + 1} time(s): the simulator hung")
    if done.start_error is not None:
        return void(f"could not start: {done.start_error}")
    if done.timed_out:
        return void(f"exceeded its {timeout_s:g}s budget")
    if done.returncode != 0:
        return void(f"exited {done.returncode}: {done.log_tail}")
    if not (out_dir / RESULT_FILE).is_file():
        return void(f"exited 0 but wrote no {RESULT_FILE}: {done.log_tail}")
    try:
        outcome = outcome_from(benchmark.read_result(out_dir=str(out_dir)))
    except Exception as exc:  # noqa: BLE001
        return void(f"its result could not be read: {type(exc).__name__}: {exc}")
    if outcome.void or not outcome.success:
        return void(f"the demonstration did not succeed: {outcome.error or 'no reason given'}")

    prompt = out_dir / PROMPT_FILE
    if not prompt.is_file():
        return void(f"wrote no {PROMPT_FILE}")
    sha = sha256_file(prompt)
    try:
        verdict = benchmark.verify_prompt(path=str(prompt), unit=unit)
    except Exception as exc:  # noqa: BLE001
        return void(f"verify_prompt failed: {type(exc).__name__}: {exc}")
    if not isinstance(verdict, Mapping) or not verdict.get("ok"):
        problems = verdict.get("problems") if isinstance(verdict, Mapping) else None
        return void(
            "the prompt is not the one the unit asked for: "
            + "; ".join(str(p) for p in (problems or ["no reason given"]))
        )
    if verdict.get("sha256") != sha:
        return void(
            f"verify_prompt hashed the prompt to {str(verdict.get('sha256'))[:12]}..., "
            f"not the sha256 of its bytes {sha[:12]}..."
        )
    written = outcome.extra.get("prompt_sha256")
    if written is not None and written != sha:
        return void(
            f"its result says it wrote a prompt hashing to {str(written)[:12]}..., not the "
            f"sha256 of the file's bytes {sha[:12]}..."
        )
    reported, read = (
        scene_seed_of(outcome.extra.get("scene_seed")),
        scene_seed_of(verdict.get("scene_seed")),
    )
    if reported is not None and read is not None and reported != read:
        return void(
            f"its result says the prompt was built on scene seed {reported}, but the prompt's "
            f"own is {read}"
        )
    # Both sides read this file; neither benchmark run may change it in between.
    os.chmod(prompt, 0o444)
    clip = out_dir / DEMONSTRATION_CLIP
    has_clip = clip.is_file() and clip.stat().st_size > 0
    return Prompt(
        unit_id=unit_id,
        path=prompt,
        sha256=sha,
        demo_clip=clip if has_clip else None,
        demo_sha256=sha256_file(clip) if has_clip else None,
        wall_s=round(time.monotonic() - started, 3),
        scene_seed=reported if reported is not None else read,
    )

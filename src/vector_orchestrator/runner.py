"""Running the benchmark's commands as subprocesses and reading back what they wrote.

- **Nothing here imports the simulator**: everything needing it happens in the subprocess.
- **A unit that goes wrong is void, not fatal**: a command that cannot start, crashes, times out,
  stalls or writes an unreadable result gives a void unit with the reason on it.
- **The benchmark says what happened; the orchestrator decides what it means**: the result's
  fields are read with defaults, and what else it reported is kept in `Outcome.extra`.
"""

from __future__ import annotations

import logging
import math
import os
import signal
import subprocess
import threading
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .robotwin import EVALUATION_CLIP, RESULT_FILE

log = logging.getLogger(__name__)

#: What a result file is read for; the rest is the benchmark's own business, kept in `extra`.
RESULT_FIELDS = ("success", "void", "steps", "error")

#: The subprocess's stdout and stderr, in its `out_dir`. Never published: a run log.
LOG_FILE = "benchmark.log"

#: How much of the log a void unit's reason carries.
TAIL_CHARS = 400

#: What a benchmark subprocess inherits from the orchestrator when no environment is given: the
#: locale, the paths an interpreter needs, what a GPU simulator reads to find its device and
#: display, and the RoboTwin benchmark's own settings - the simulator environment's interpreter and
#: the denoiser a host's cameras must render with - neither of which is a secret. Never the
#: credentials that publish results (HF_TOKEN, the live token): the benchmark parses a hostile
#: policy's replies, and a bug there must not hand out write access to the record.
ENV_ALLOW = frozenset(
    {
        "ROBOTWIN_BENCH_PYTHON",
        "ROBOTWIN_BENCH_DENOISER",
        "PATH",
        "HOME",
        "USER",
        "LOGNAME",
        "SHELL",
        "TERM",
        "TZ",
        "TMPDIR",
        "LANG",
        "LANGUAGE",
        "VIRTUAL_ENV",
        "CONDA_PREFIX",
        "LD_LIBRARY_PATH",
        "DISPLAY",
        "XAUTHORITY",
        "XDG_RUNTIME_DIR",
        "CUDA_HOME",
        "CUDA_VISIBLE_DEVICES",
        "VK_ICD_FILENAMES",
        "MUJOCO_GL",
        "PYOPENGL_PLATFORM",
        "EGL_PLATFORM",
    }
)
ENV_ALLOW_PREFIXES = ("LC_", "NVIDIA_", "__EGL_", "__GLX_")


def benchmark_environment(
    source: Mapping[str, str], authkey_env: str, keep: Iterable[str] = ()
) -> dict[str, str]:
    """The allow-listed part of `source`, plus the authkey variable and any names in `keep`."""
    names = ENV_ALLOW | {authkey_env, *keep}
    return {k: v for k, v in source.items() if k in names or k.startswith(ENV_ALLOW_PREFIXES)}


def workdir_of(benchmark: Any) -> str | None:
    """The checkout a benchmark's commands run from, or None to run them where the orchestrator
    runs: a fork run from its checkout (`python -m robotensor_bench.cli`) needs its root as the
    working directory and on `PYTHONPATH`."""
    workdir = getattr(benchmark, "workdir", None)
    return str(workdir) if workdir else None


def in_workdir(env: Mapping[str, str], workdir: str | None) -> dict[str, str]:
    """`env` with `workdir` first on `PYTHONPATH`, so `python -m` finds the checkout's package
    wherever the command is started from. The orchestrator's own `PYTHONPATH` is never passed on."""
    out = dict(env)
    if workdir:
        out["PYTHONPATH"] = workdir
    return out


@dataclass
class Outcome:
    """One unit's result, with the defaults in one place: a benchmark that reports only what the
    ABI requires still produces a complete record."""

    success: bool | None
    void: bool
    steps: int | None
    error: str | None
    wall_s: float = 0.0
    #: The rollout clip the command wrote, when it wrote a non-empty one.
    clip: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def outcome_from(result: Mapping[str, Any], *, wall_s: float = 0.0) -> Outcome:
    """Read the benchmark's `read_result` mapping.

    `success is None` exactly when the unit is void - the ABI says so, and the two disagreeing is
    the kind of thing that turns into a wrong score rather than an error, so it is reconciled here
    in favour of void.
    """
    void = bool(result.get("void"))
    success = result.get("success")
    if not isinstance(success, bool):
        void = True
    if void:
        success = None
    error = _str_or_none(result.get("error"))
    if void and error is None:
        error = "the benchmark reported the unit void without a reason"
    return Outcome(
        success=success,
        void=void,
        steps=_int_or_none(result.get("steps")),
        error=error,
        wall_s=wall_s,
        extra={k: v for k, v in result.items() if k not in RESULT_FIELDS},
    )


#: What says the GPU ran out of memory, in a benchmark's log, error or detail: on a card shared by
#: several units that is nobody's doing, so the unit is run once more before it counts.
OUT_OF_MEMORY = ("out of memory", "outofmemoryerror")


def out_of_memory(*texts: Any) -> bool:
    joined = " ".join(str(text) for text in texts if text).lower()
    return any(mark in joined for mark in OUT_OF_MEMORY)


def voided(reason: str, *, wall_s: float = 0.0) -> Outcome:
    return Outcome(success=None, void=True, steps=None, error=reason, wall_s=wall_s)


def _finite(value: Any) -> bool:
    """A JSON number that can be scored and published: `json.loads` accepts NaN and Infinity, which
    `int()` refuses and canonical JSON cannot encode."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _int_or_none(value: Any) -> int | None:
    return int(value) if _finite(value) else None


def _str_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


@dataclass
class Completed:
    """How a subprocess ended. Exactly one of `start_error`, `timed_out` or `returncode` says so."""

    returncode: int | None
    timed_out: bool
    start_error: str | None
    wall_s: float
    log_tail: str
    #: The process group used under `min_cpu_rate` cores over `stall_s` seconds and was killed:
    #: a hung simulator (SAPIEN's camera read can block for good, polling at a trickle of CPU).
    stalled: bool = False


#: How often a running subprocess's CPU time is sampled for the stall watchdog.
STALL_POLL_S = 2.0
#: Below this many cores over the whole stall window, a process group is hung, not slow. Measured
#: (RoboTwin-Vector docs/install.md, "Rendering"): hung camera reads used 0.004-0.07 cores, healthy
#: simulator work never under 0.25 over 5 s; a unit waiting on its remote policy (~65 ms an act
#: against a ~0.15 s step) stays far above this.
MIN_CPU_RATE = 0.1


def group_cpu_seconds(pgid: int | Iterable[int]) -> float:
    """User and system CPU seconds of every live process in group `pgid` - or in any of several
    groups - and of the children they reaped, read from /proc; 0.0 where /proc cannot say."""
    groups = {pgid} if isinstance(pgid, int) else set(pgid)
    tick = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100
    total = 0
    try:
        entries = os.listdir("/proc")
    except OSError:
        return 0.0
    for entry in entries:
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/stat", "rb") as fh:
                stat = fh.read().decode(errors="replace")
        except OSError:
            continue
        # The command name is in parentheses and may hold spaces: split after its closing one.
        fields = stat[stat.rfind(")") + 2 :].split()
        try:
            if int(fields[2]) not in groups:
                continue
            total += sum(int(v) for v in fields[11:15])
        except (IndexError, ValueError):
            continue
    return total / float(tick)


class Running:
    """The process groups `run_argv` has running, so a duel that stops can end them at once:
    units run in worker threads, which an interrupt of the main thread does not reach."""

    def __init__(self) -> None:
        self._groups: set[int] = set()
        self._lock = threading.Lock()
        self._stopping = False

    def add(self, pgid: int) -> None:
        with self._lock:
            self._groups.add(pgid)
            stopping = self._stopping
        if stopping:  # started after `stop`: it is not let run either
            _killpg(pgid)

    def discard(self, pgid: int) -> None:
        with self._lock:
            self._groups.discard(pgid)

    def stop(self) -> list[int]:
        """Kill every group running, and every one started until `resume`."""
        with self._lock:
            self._stopping = True
            groups = sorted(self._groups)
        for pgid in groups:
            _killpg(pgid)
        return groups

    def resume(self) -> None:
        with self._lock:
            self._stopping = False


def _killpg(pgid: int) -> None:
    try:
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


#: Every benchmark subprocess of this process that is running now.
RUNNING = Running()


def run_argv(
    argv: list[str],
    *,
    env: Mapping[str, str],
    timeout_s: float,
    log_path: Path,
    ledger: Any = None,
    stall_s: float | None = None,
    min_cpu_rate: float = MIN_CPU_RATE,
    watch_pgids: Iterable[int] = (),
    cwd: str | None = None,
) -> Completed:
    """Run `argv` to completion or to `timeout_s`, its output going to `log_path`.

    With `stall_s`, the process group's CPU time is sampled every `STALL_POLL_S`, and a group that
    used under `min_cpu_rate` cores over the last `stall_s` seconds is killed as `stalled`: a hung
    simulator would otherwise sleep out the whole of `timeout_s`. `watch_pgids` are other process
    groups whose CPU counts as progress too: the policy server a benchmark waits on.

    It runs in its own session, and whatever ends the unit - a clean exit, a crash, the timeout, or
    the orchestrator itself being interrupted - the whole process group is killed on the way out: a
    simulator that forked a renderer must not outlive its unit and hold the GPU for the next one.
    Only a kill that runs no `finally` escapes that; for it, `ledger` (anything with
    `started(pid)` and `ended(pid)`) is told of the group while it runs, so whoever starts next
    can find and end it. `cwd` is the directory it runs in: a benchmark's checkout (`workdir_of`).
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with open(log_path, "wb") as sink:
        try:
            proc = subprocess.Popen(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=sink,
                stderr=subprocess.STDOUT,
                env=dict(env),
                cwd=cwd,
                start_new_session=True,
            )
        except OSError as exc:
            return Completed(None, False, str(exc), time.monotonic() - started, "")
        returncode: int | None = None
        timed_out = stalled = False
        RUNNING.add(proc.pid)
        try:
            if ledger is not None:
                ledger.started(proc.pid)
            if stall_s is None:
                returncode = proc.wait(timeout=timeout_s)
            else:
                returncode, timed_out, stalled = _watch(
                    proc, started, timeout_s, float(stall_s), min_cpu_rate, tuple(watch_pgids)
                )
                if stalled:
                    log.warning(
                        "%s used under %g cores for %gs: killed as stalled",
                        " ".join(argv[:4]),
                        min_cpu_rate,
                        stall_s,
                    )
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            # The session is the benchmark's alone (start_new_session), so this reaches only what
            # it started - including children still running after the direct child exited.
            _kill_group(proc)
            RUNNING.discard(proc.pid)
            if ledger is not None:
                ledger.ended(proc.pid)
    return Completed(
        returncode, timed_out, None, time.monotonic() - started, _tail(log_path), stalled=stalled
    )


def _watch(
    proc: subprocess.Popen,
    started: float,
    timeout_s: float,
    stall_s: float,
    min_rate: float,
    watch_pgids: tuple[int, ...] = (),
) -> tuple[int | None, bool, bool]:
    """Wait for `proc`, sampling its group's CPU time: `(returncode, timed_out, stalled)`."""
    samples: list[tuple[float, float]] = []
    while True:
        try:
            return proc.wait(timeout=STALL_POLL_S), False, False
        except subprocess.TimeoutExpired:
            pass
        now = time.monotonic()
        if now - started >= timeout_s:
            return None, True, False
        samples.append((now, group_cpu_seconds((proc.pid, *watch_pgids))))
        # The oldest sample at least stall_s old is the window's start.
        while len(samples) > 1 and now - samples[1][0] >= stall_s:
            samples.pop(0)
        first_t, first_cpu = samples[0]
        if now - first_t >= stall_s and (samples[-1][1] - first_cpu) / (now - first_t) < min_rate:
            return None, False, True


def _kill_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        if proc.poll() is None:
            proc.kill()
    proc.wait()


def _tail(path: Path, limit: int = TAIL_CHARS) -> str:
    """The end of a log of any size: nothing bounds what a benchmark prints, so only the last few
    kilobytes are ever read."""
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            fh.seek(max(0, fh.tell() - 4 * limit))
            data = fh.read(4 * limit)
    except OSError:
        return ""
    return data.decode("utf-8", errors="replace").strip()[-limit:]


def run_unit(
    benchmark: Any,
    unit: Mapping[str, Any],
    *,
    prompt: str | None,
    out_dir: str | Path,
    policy_address: str,
    authkey_env: str,
    timeout_s: float,
    env: Mapping[str, str] | None = None,
    extra: Mapping[str, Any] | None = None,
    ledger: Any = None,
    stall_s: float | None = None,
    min_cpu_rate: float = MIN_CPU_RATE,
    watch_pgids: Iterable[int] = (),
) -> Outcome:
    """One unit against a served policy, start to finish. Every failure is a void outcome, never an
    exception: one bad unit must not lose the rest of the duel. `ledger` and `stall_s` are
    `run_argv`'s; a stalled unit is void with `extra["stalled"]`, so the caller may play it again.

    `env` is the subprocess's whole environment; without it the benchmark gets
    `benchmark_environment(os.environ, authkey_env)`, not everything the orchestrator holds.
    """
    name = str(getattr(benchmark, "id", "benchmark"))
    started = time.monotonic()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    workdir = workdir_of(benchmark)
    environ = in_workdir(
        benchmark_environment(os.environ, authkey_env) if env is None else env, workdir
    )

    def void(reason: str) -> Outcome:
        return voided(f"{name}: {reason}", wall_s=round(time.monotonic() - started, 3))

    if not prompt:
        return void("the unit carries no materialized prompt")
    key = environ.get(authkey_env)
    if not key:
        return void(f"no policy authkey in ${authkey_env} for the benchmark subprocess")

    try:
        built = benchmark.run_command(
            unit=unit,
            prompt=str(prompt),
            out_dir=str(out),
            policy_address=policy_address,
            authkey_env=authkey_env,
            **dict(extra or {}),
        )
        argv = [str(a) for a in built]
    except Exception as exc:  # noqa: BLE001 - the command's failure is this unit's, not the duel's
        return void(f"run_command failed: {type(exc).__name__}: {exc}")
    if not argv:
        return void("run_command returned an empty argv")
    # Any process on the host can read another's command line; the key travels by variable name.
    if any(key in arg for arg in argv):
        return void("run_command put the policy authkey on the command line; refusing to run it")
    # What an earlier command left here - a previous attempt, or a materialize command pointed at
    # the same directory - must not be read as this run's result or clip.
    for stale in (RESULT_FILE, EVALUATION_CLIP):
        try:
            (out / stale).unlink(missing_ok=True)
        except OSError as exc:
            return void(f"could not clear a stale {stale}: {exc}")

    done = run_argv(
        argv,
        env=environ,
        timeout_s=timeout_s,
        log_path=out / LOG_FILE,
        ledger=ledger,
        stall_s=stall_s,
        min_cpu_rate=min_cpu_rate,
        watch_pgids=watch_pgids,
        cwd=workdir,
    )
    if done.start_error is not None:
        return void(f"could not start the benchmark: {done.start_error}")
    if done.stalled:
        outcome = void(f"stalled: no progress for {stall_s:g}s (a hung simulator); killed")
        # A hang is nobody's doing: void on the harness whatever the policy's server did when its
        # client was killed under it.
        outcome.extra["stalled"] = True
        outcome.extra["void_cause"] = "harness"
        return outcome
    if done.timed_out:
        return void(f"unit exceeded its {timeout_s:g}s budget")
    if done.returncode != 0:
        return void(f"exited {done.returncode}: {done.log_tail}")
    if not (out / RESULT_FILE).is_file():
        return void(f"exited 0 but wrote no {RESULT_FILE}: {done.log_tail}")

    try:
        result = benchmark.read_result(out_dir=str(out))
    except Exception as exc:  # noqa: BLE001
        return void(f"read_result failed: {type(exc).__name__}: {exc}")
    if not isinstance(result, Mapping):
        return void(f"read_result returned {type(result).__name__}, not a mapping")

    try:
        outcome = outcome_from(result, wall_s=round(time.monotonic() - started, 3))
    except Exception as exc:  # noqa: BLE001 - whatever the mapping holds, it is this unit's problem
        return void(f"unusable result: {type(exc).__name__}: {exc}")
    if outcome.void and not outcome.error.startswith(f"{name}: "):  # type: ignore[union-attr]
        outcome.error = f"{name}: {outcome.error}"
    clip = out / EVALUATION_CLIP
    if clip.is_file() and clip.stat().st_size > 0:
        outcome.clip = str(clip)
    return outcome

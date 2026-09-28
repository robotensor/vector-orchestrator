"""The benchmarks this orchestrator drives, and where each one's checkout is on this host.

A benchmark is a fork run from its checkout, never installed or imported, so finding one is a
question of where, not what is installed: `spec.json` pins each benchmark by repository, branch and
command-line contract (`api_version`, `module`), and this host says where its checkout and
interpreter are (RoboTwin-Vector: `$ROBOTWIN_BENCH_ROOT` and `$ROBOTWIN_BENCH_PYTHON`). A benchmark
the orchestrator has no driver for is refused, whatever the spec says.

Nothing here raises for a listing: an operator asking what is wrong is told everything at once.
`load` is the one call that refuses, and it says why.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, field
from typing import Any

from . import robotwin
from .api import BenchmarkRefused, validate_benchmark

#: Every benchmark this orchestrator can drive: its id, and how to find its checkout on this host
#: from the spec's pin. A test puts a fake in here.
DRIVERS: dict[str, Callable[[Mapping[str, Any]], Any]] = {
    robotwin.BENCHMARK_ID: lambda pin: robotwin.RoboTwinFork.from_environment(),
}

__all__ = ["DRIVERS", "BenchmarkRefused", "Found", "discover", "load", "suites_of"]


@dataclass
class Found:
    """One benchmark the spec declares, as found on this host."""

    name: str
    pin: dict[str, Any]
    workdir: str | None = None
    python: str | None = None
    problems: list[str] = field(default_factory=list)
    #: Not problems: things an operator should know.
    notes: list[str] = field(default_factory=list)
    benchmark: Any = None

    @property
    def ok(self) -> bool:
        return not self.problems

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "pin": dict(self.pin),
            "workdir": self.workdir,
            "python": self.python,
            "found": self.benchmark is not None,
            "problems": list(self.problems),
            "notes": list(self.notes),
        }


def suites_of(spec: Any, name: str) -> list[str]:
    """The suites the spec's skills draw from `name`, in first-use order."""
    return list(
        dict.fromkeys(spec.suite(s) for s in spec.all_skills if spec.benchmark_of(s) == name)
    )


def discover(spec: Any, *, prepare: bool | Collection[str] = False) -> dict[str, Found]:
    """Every benchmark the spec declares, keyed by name.

    `prepare` (True, or the names to prepare) also asks each found checkout what a duel over the
    spec's suites needs (`Benchmark.prepare`): its task table and each suite's task config, which
    runs the checkout's own commands. Without it only the checkout's place is checked.
    """
    found: dict[str, Found] = {}
    for name, pin in spec.benchmarks.items():
        row = Found(name=name, pin=dict(pin))
        found[name] = row
        make = DRIVERS.get(name)
        if make is None:
            row.problems.append(
                f"no driver: this orchestrator drives {', '.join(sorted(DRIVERS)) or 'nothing'}"
            )
            continue
        try:
            benchmark = make(pin)
        except BenchmarkRefused as exc:
            row.problems.append(str(exc))
            continue
        except Exception as exc:  # noqa: BLE001 - reported, so the operator sees every problem
            row.problems.append(f"could not be found: {type(exc).__name__}: {exc}")
            continue
        row.problems.extend(validate_benchmark(benchmark))
        row.workdir = getattr(benchmark, "workdir", None)
        row.python = getattr(benchmark, "python", None)
        module = getattr(benchmark, "module", None) or getattr(type(benchmark), "module", None)
        if module is not None and pin.get("module") != module:
            row.problems.append(
                f"the spec pins module {pin.get('module')!r}; the driver runs {module!r}"
            )
        wanted = prepare is True or (not isinstance(prepare, bool) and name in prepare)
        if wanted and row.ok and callable(getattr(benchmark, "prepare", None)):
            try:
                benchmark.prepare(suites_of(spec, name))
            except BenchmarkRefused as exc:
                row.problems.append(str(exc))
            except Exception as exc:  # noqa: BLE001
                row.problems.append(f"prepare: {type(exc).__name__}: {exc}")
        if row.ok:
            row.benchmark = benchmark
    return found


def load(spec: Any, name: str) -> Any:
    """The benchmark `name` on this host, or `BenchmarkRefused` naming every reason it cannot be
    used. Its checkout is found, not yet asked anything: a duel prepares it (`Benchmark.prepare`)
    before it deals units."""
    row = discover(spec).get(name)
    if row is None:
        raise BenchmarkRefused(f"{name}: not declared in spec.json `benchmarks`")
    if not row.ok:
        raise BenchmarkRefused(f"{name}: " + "; ".join(row.problems))
    return row.benchmark

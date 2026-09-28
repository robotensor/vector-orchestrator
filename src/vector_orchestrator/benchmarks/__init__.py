"""Benchmarks: forks run from their checkouts, which the orchestrator drives but never imports.

`api.py` says what a benchmark is to the orchestrator; `drivers.py` finds each one the spec declares;
`robotwin.py` drives RoboTwin-Vector. Nothing in this package imports a benchmark's simulator:
everything that needs one runs as the argv a driver's command builders return.
"""

from __future__ import annotations

from .api import (
    BENCHMARK_API_VERSION,
    COMMAND_METHODS,
    PURE_METHODS,
    Benchmark,
    BenchmarkRefused,
    validate_benchmark,
)

__all__ = [
    "BENCHMARK_API_VERSION",
    "COMMAND_METHODS",
    "PURE_METHODS",
    "Benchmark",
    "BenchmarkRefused",
    "validate_benchmark",
]

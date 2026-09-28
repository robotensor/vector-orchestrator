"""`python -m vector_fake_cli ...`: the orchestrator's command line with the fake benchmark
registered as the `fake` driver, for a test that runs the orchestrator in a process of its own,
where a monkeypatched registry does not reach."""

import vector_fake_benchmark

from vector_orchestrator.benchmarks import drivers
from vector_orchestrator.cli import main

drivers.DRIVERS["fake"] = lambda pin: vector_fake_benchmark.BENCHMARK

raise SystemExit(main())

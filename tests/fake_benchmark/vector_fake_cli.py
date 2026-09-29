"""`vector-orchestrator` with the fake benchmark standing where the RoboTwin checkout stands."""

import sys

import vector_fake_benchmark

from vector_orchestrator import robotwin
from vector_orchestrator.cli import main

robotwin.RoboTwin.from_environment = classmethod(
    lambda cls, level, environ=None: vector_fake_benchmark.FakeBenchmark()
)

if __name__ == "__main__":
    sys.exit(main())

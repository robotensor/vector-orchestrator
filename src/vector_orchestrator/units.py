"""A duel's units: for each task, `runs_per_task` units, each a pair of candidate scene lists.

A unit is scored in a scene drawn from `scene_seeds` and shown a demonstration recorded in a scene
drawn from `demo_seeds` (the first candidate RoboTwin's expert solves, in order: a seed the expert
fails is skipped, and the unit is void only if it fails all of a list). The seeds come
from a sha256 counter over the duel id, the chain entropy and the task, above the level
benchmark's public seed ranges; anyone holding a record can deal them again. A task's units are
drawn in order, so a smaller duel's units are the first units of a larger one.
"""

from __future__ import annotations

import hashlib
from typing import Any

from .ids import unit_id

#: The derivation label: a change to how units are drawn changes this, and every unit.
LABEL = "vector/units/3"
#: Scenes are drawn from [SEED_LOW, SEED_HIGH): above the public ranges [0, 2M).
SEED_LOW = 2_000_000
SEED_HIGH = 2**31 - 1
#: Candidate scenes per list, tried in order until the expert solves one. Past duels' expert solved
#: about 80% of first tries, so all twenty failing is out of reach for any task it can do at all.
CANDIDATES = 20


class Counter:
    """Integers from sha256(label | material | n), n counting up from 0."""

    def __init__(self, material: str, label: str = LABEL) -> None:
        self._prefix = f"{label}|{material}|".encode()
        self._n = 0

    def below(self, bound: int) -> int:
        """Uniform in [0, bound), rejecting the draws that would bias a modulo."""
        limit = (1 << 64) - (1 << 64) % bound
        while True:
            digest = hashlib.sha256(self._prefix + str(self._n).encode("ascii")).digest()
            self._n += 1
            value = int.from_bytes(digest[:8], "big")
            if value < limit:
                return value % bound


def seed_key(duel_id: str, entropy: str | None = None) -> str:
    """What a duel's draws hang off: its id, and the chain entropy (`<block>:<hash>`) when it has one."""
    return duel_id if entropy is None else f"{duel_id}|{entropy}"


def _seeds(draws: Counter, taken: list[int]) -> list[int]:
    """`CANDIDATES` distinct seeds, none in `taken`."""
    out: list[int] = []
    while len(out) < CANDIDATES:
        seed = SEED_LOW + draws.below(SEED_HIGH - SEED_LOW)
        if seed not in out and seed not in taken:
            out.append(seed)
    return out


def deal(
    spec: Any, duel_id: str, entropy: str | None = None, runs_per_task: int | None = None
) -> list[dict[str, Any]]:
    """Every unit of a duel, tasks in spec order, runs in order."""
    runs = spec.runs_per_task if runs_per_task is None else runs_per_task
    units = []
    for task in spec.tasks:
        draws = Counter(f"{seed_key(duel_id, entropy)}|{task}")
        for run in range(runs):
            scene = _seeds(draws, [])
            units.append(
                {
                    "unit_id": unit_id(task, run),
                    "task": task,
                    "scene_seeds": scene,
                    "demo_seeds": _seeds(draws, scene),
                }
            )
    return units

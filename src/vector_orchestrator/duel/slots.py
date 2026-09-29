"""Where a duel's work runs: `workers` slots on each GPU of this host, one job in a slot at a time.

The cards are the host's: `CUDA_VISIBLE_DEVICES` when it is set (a validator's GPU lease sets it),
else every card `nvidia-smi` lists, else none - a host without a GPU gets card-less slots. Nothing
assumes a card's model or memory: `workers` is how many units one card runs at once, chosen for
its memory. Slot i is on card i mod n, so the first n slots are on n different cards and a duel
with fewer jobs than slots spreads over the cards.

A job asks for a slot with a priority - a unit's play (`PLAY`) before a new demonstration
(`MAKE`), so the units already begun finish first and both sides' results come in pairs - and
gets the lowest free slot, preferring one whose kept policy server serves its submission. Its
subprocesses see the slot's card as their only device (`Slot.env`), so a policy's `cuda:0` is that
card.
"""

from __future__ import annotations

import itertools
import os
import subprocess
import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from .runtime import ServerPool

#: Priorities: a lower one gets a free slot first.
PLAY = 0
MAKE = 1


def gpus(environ: Mapping[str, str] | None = None) -> list[str | None]:
    """The cards this host's work may use: `CUDA_VISIBLE_DEVICES`'s entries when it is set, else
    one per card `nvidia-smi -L` lists, else `[None]` (no card)."""
    environ = os.environ if environ is None else environ
    named = environ.get("CUDA_VISIBLE_DEVICES")
    if named is not None:
        cards = [card.strip() for card in named.split(",") if card.strip()]
        return list(cards) or [None]
    try:
        listing = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True, timeout=30, check=False
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return [None]
    count = sum(1 for line in listing.splitlines() if line.startswith("GPU "))
    return [str(i) for i in range(count)] or [None]


@dataclass
class Slot:
    index: int
    gpu: str | None
    #: The policy server this slot keeps between units.
    servers: ServerPool

    @property
    def env(self) -> dict[str, str]:
        """What a subprocess in this slot is started with: its card as the only visible one."""
        return {} if self.gpu is None else {"CUDA_VISIBLE_DEVICES": self.gpu}


class _Waiter:
    def __init__(self, priority: int, ticket: int, handle: str | None) -> None:
        self.priority, self.ticket, self.handle = priority, ticket, handle
        self.slot: Slot | None = None


class Slots:
    """`per_gpu` slots on each of `cards`; `take` one for a job, `close` stops the kept servers."""

    def __init__(self, per_gpu: int, cards: list[str | None], ledger_root: Path) -> None:
        cards = list(cards) or [None]
        count = max(1, int(per_gpu)) * len(cards)
        self.slots = [
            Slot(i, cards[i % len(cards)], ServerPool(Path(ledger_root) / f"slot-{i}"))
            for i in range(count)
        ]
        self._free = list(self.slots)
        self._cond = threading.Condition()
        self._waiting: list[_Waiter] = []
        self._tickets = itertools.count()

    def __len__(self) -> int:
        return len(self.slots)

    @contextmanager
    def take(self, priority: int = PLAY, handle: object = None) -> Iterator[Slot]:
        """A free slot for the length of the block, waiting for one if none is. Slots go to the
        jobs of the lowest priority first; among those, a slot goes to one that will serve what
        its kept server serves (`handle`), so a model is not built again for nothing, and
        otherwise to the one that asked first."""
        waiter = _Waiter(priority, next(self._tickets), None if handle is None else str(handle))
        with self._cond:
            self._waiting.append(waiter)
            self._dispatch()
            while waiter.slot is None:
                self._cond.wait()
        slot = waiter.slot
        try:
            yield slot
        finally:
            with self._cond:
                self._free.append(slot)
                self._free.sort(key=lambda s: s.index)
                self._dispatch()

    def _dispatch(self) -> None:
        """Hand free slots to waiting jobs. Called holding the condition."""
        while self._free and self._waiting:
            first = min(w.priority for w in self._waiting)
            ready = sorted(
                (w for w in self._waiting if w.priority == first), key=lambda w: w.ticket
            )
            slot, waiter = self._match(ready)
            self._free.remove(slot)
            self._waiting.remove(waiter)
            waiter.slot = slot
        self._cond.notify_all()

    def _match(self, ready: list[_Waiter]) -> tuple[Slot, _Waiter]:
        # A slot already serving what a job will serve, for the earliest such job.
        for waiter in ready:
            if waiter.handle is None:
                continue
            for slot in self._free:
                if slot.servers.handle == waiter.handle:
                    return slot, waiter
        # Otherwise the earliest job gets the lowest slot, preferring one that keeps no server,
        # so another job's kept server is not replaced while it could still be used.
        waiter = ready[0]
        for slot in self._free:
            if slot.servers.handle is None:
                return slot, waiter
        return self._free[0], waiter

    def close(self) -> None:
        for slot in self.slots:
            slot.servers.close()

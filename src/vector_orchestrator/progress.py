"""What the duel running now is doing, written into the store as `running.json`.

    {schema, event_id, kind, king, challenger, phase, side, current, message, started_at,
     updated_at, progress: {units, demonstrations, challenger, king}, outcome}

The dashboard reads it from the store like every other file, so it needs no endpoint of its own
and no token: whatever host serves the dashboard, the mirror is the one place both sides agree on.
It is never part of the record - the next duel overwrites it - and a duel never stops because it
could not be written.

`progress` counts what is finished: `demonstrations` the units whose prompt is made (or void),
`challenger` and `king` the units that side has played (a void unit counts as played by both);
`king` is null for a genesis, which has none. `outcome` is null until the duel ends, then says how:
`published` (with `dethroned`), `refused`, `void`, `stale` or `failed`.

Writing is cheap and happens on every change; pushing to the mirror is one commit, so the board
asks for one on a phase change and otherwise at most every `PUSH_INTERVAL_S`.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable, Mapping
from typing import Any

from .store.records import now_iso

log = logging.getLogger(__name__)

SCHEMA = 1
#: The phases of a duel, in order; `done` and `failed` end it.
PHASES = ("fetching", "checking", "materializing", "evaluating", "publishing", "done", "failed")
SIDES = ("challenger", "king")
OUTCOMES = ("published", "refused", "void", "stale", "failed")
EVENT_ID_RE = re.compile(r"[0-9a-f]{8,64}")
MAX_MESSAGE = 300
#: The longest a change waits for the mirror when no phase changes meanwhile.
PUSH_INTERVAL_S = 60.0


def counts(units: list[Mapping[str, Any]], *, genesis: bool) -> dict[str, int | None]:
    """How far the duel's units are: made, and played by each side."""

    def played(side: str) -> int:
        return sum(1 for u in units if u.get("void") or u.get(f"{side}_success") is not None)

    return {
        "units": len(units),
        "demonstrations": sum(1 for u in units if u.get("prompt_sha256") or u.get("void")),
        "challenger": played("challenger"),
        "king": None if genesis else played("king"),
    }


def build_progress(
    *,
    event_id: str,
    kind: str,
    king: dict[str, str] | None,
    challenger: dict[str, str] | None,
    phase: str,
    side: str | None,
    units: list[Mapping[str, Any]],
    current: Mapping[str, Any] | None,
    message: str,
    started_at: str,
    outcome: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if phase not in PHASES:
        raise ValueError(f"phase must be one of {PHASES}, not {phase!r}")
    if not EVENT_ID_RE.fullmatch(event_id):
        raise ValueError(f"event_id must be 8-64 lowercase hex characters, not {event_id!r}")
    if side is not None and side not in SIDES:
        raise ValueError(f"side must be one of {SIDES} or None, not {side!r}")
    if outcome is not None and outcome.get("status") not in OUTCOMES:
        raise ValueError(f"outcome status must be one of {OUTCOMES}, not {outcome.get('status')!r}")
    task = None
    if current is not None:
        task = next(
            (u.get("task") for u in units if u.get("unit_id") == current.get("unit_id")), None
        )
    return {
        "schema": SCHEMA,
        "event_id": event_id,
        "kind": kind,
        "king": king,
        "challenger": challenger,
        "phase": phase,
        "side": side,
        "current": {**current, "task": task} if current is not None else None,
        "message": message[:MAX_MESSAGE],
        "started_at": started_at,
        "updated_at": now_iso(),
        "progress": counts(list(units), genesis=king is None),
        "outcome": dict(outcome) if outcome is not None else None,
    }


class ProgressBoard:
    """Writes each state of the running duel into the store's `running.json`, and calls `push`
    (the orchestrator's mirror push) on a forced change or once `interval_s` has passed."""

    def __init__(
        self,
        store: Any,
        push: Callable[[], None] | None = None,
        *,
        interval_s: float = PUSH_INTERVAL_S,
    ) -> None:
        self.store = store
        self.push = push
        self.interval_s = interval_s
        self._pushed = 0.0

    def post(self, doc: dict[str, Any], *, force: bool = False) -> None:
        try:
            self.store.write_running(doc)
        except OSError as exc:  # a board that cannot be written never stops a duel
            log.warning("running.json not written: %s", exc)
            return
        now = time.monotonic()
        if self.push is not None and (force or now - self._pushed >= self.interval_s):
            self._pushed = now
            self.push()

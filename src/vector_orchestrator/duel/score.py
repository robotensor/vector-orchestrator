"""Scoring and the crown rule: pure functions over a duel's unit rows.

- Per task, a side's rate is its successes over the task's non-void units; a task with no scored
  unit has none. Its score (`average`) is the mean of the rates it has.
- A unit void for either side is void for both. A duel with more than `max_void` of its units void
  is void and moves nothing.
- The challenger takes the crown iff its average is at least the king's plus `margin` points.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

SIDES = ("challenger", "king")
#: Averages are means of fractions; a margin met exactly must not be missed to rounding.
SCORE_EPSILON = 1e-9


def paired_outcome(king_success: bool | None, challenger_success: bool | None) -> str:
    """Which side did better on one unit: `challenger`, `king` or `tie` (also when unscored)."""
    if not isinstance(king_success, bool) or not isinstance(challenger_success, bool):
        return "tie"
    if king_success == challenger_success:
        return "tie"
    return "challenger" if challenger_success else "king"


def side_success(unit: dict[str, Any], side: str) -> bool | None:
    value = unit.get(f"{side}_success")
    return value if isinstance(value, bool) else None


def task_rate(units: Iterable[dict[str, Any]], side: str, task: str) -> float | None:
    scored = successes = 0
    for u in units:
        if u.get("task") != task or u.get("void"):
            continue
        s = side_success(u, side)
        if s is None:
            continue
        scored += 1
        successes += int(s)
    return successes / scored if scored else None


def average(per_task: dict[str, float | None], tasks: Sequence[str]) -> float | None:
    present = [per_task[t] for t in tasks if per_task.get(t) is not None]
    return sum(present) / len(present) if present else None  # type: ignore[arg-type]


def task_scores(
    units: Iterable[dict[str, Any]], side: str, tasks: Sequence[str]
) -> dict[str, float | None]:
    """`{"average": ..., task: rate, ...}`, tasks in spec order."""
    units = list(units)
    per: dict[str, float | None] = {t: task_rate(units, side, t) for t in tasks}
    return {"average": average(per, tasks), **per}


def crown_moves(
    king_average: float | None, challenger_average: float | None, margin_points: float
) -> bool:
    if king_average is None or challenger_average is None:
        return False
    return challenger_average >= king_average + margin_points / 100.0 - SCORE_EPSILON


def void_fraction(units: Iterable[dict[str, Any]]) -> float:
    units = list(units)
    return sum(1 for u in units if u.get("void")) / len(units) if units else 0.0


def too_void(units: Iterable[dict[str, Any]], max_void_fraction: float) -> bool:
    return void_fraction(units) > max_void_fraction


@dataclass
class Tally:
    units: int = 0
    wins: int = 0
    losses: int = 0
    ties: int = 0
    decided: int = 0
    void: int = 0

    def as_dict(self) -> dict[str, int]:
        return {"wins": self.wins, "losses": self.losses, "ties": self.ties, "void": self.void}


def tally(units: Iterable[dict[str, Any]]) -> Tally:
    """The paired diagnostic, from the challenger's side. A void unit counts only as void."""
    t = Tally()
    for u in units:
        t.units += 1
        if u.get("void"):
            t.void += 1
            continue
        outcome = paired_outcome(side_success(u, "king"), side_success(u, "challenger"))
        if outcome == "challenger":
            t.wins += 1
        elif outcome == "king":
            t.losses += 1
        else:
            t.ties += 1
    t.decided = t.wins + t.losses
    return t


@dataclass
class Verdict:
    king_scores: dict[str, float | None]
    challenger_scores: dict[str, float | None]
    score_margin: float
    dethroned: bool
    #: margin-met, short-of-margin, unscored or no-units: why the crown moved or stayed.
    reason: str
    void_fraction: float = 0.0
    tally: Tally = field(default_factory=Tally)

    @property
    def delta_points(self) -> float | None:
        k, c = self.king_scores["average"], self.challenger_scores["average"]
        return None if k is None or c is None else (c - k) * 100.0

    def as_dict(self) -> dict[str, Any]:
        """What an event publishes of the verdict (with `void_fraction`, which the duel adds)."""
        return {
            "reason": self.reason,
            "delta_points": self.delta_points,
        }


def verdict(units: Iterable[dict[str, Any]], score_margin: float, tasks: Sequence[str]) -> Verdict:
    """The duel's scores and whether the crown moves. Void units are excluded from both sides."""
    units = list(units)
    king = task_scores(units, "king", tasks)
    challenger = task_scores(units, "challenger", tasks)
    moves = crown_moves(king["average"], challenger["average"], score_margin)
    if not units:
        reason = "no-units"
    elif king["average"] is None or challenger["average"] is None:
        reason = "unscored"
    elif not moves:
        reason = "short-of-margin"
    else:
        reason = "margin-met"
    return Verdict(
        king,
        challenger,
        score_margin,
        moves,
        reason,
        void_fraction(units),
        tally(units),
    )

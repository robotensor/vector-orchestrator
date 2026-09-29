"""What is published: an index record per duel and its event. The dashboard reads both field by
field, so a field here is a contract.

A unit row, as a duel keeps it and an event publishes it:

    unit_id, task, scene_seed        the unit, and the scene both sides played
    prompt_sha256, demo_video        the demonstration both sides ran from, and its clip
    void, outcome                    void for both sides, or who did better: king, challenger, tie
    <side>_success, _steps, _video, _error   for side in king, challenger
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

SIDES = ("king", "challenger")


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def unit_row(unit: dict[str, Any]) -> dict[str, Any]:
    """A unit before either side has run."""
    row: dict[str, Any] = {
        "unit_id": unit["unit_id"],
        "task": unit["task"],
        "scene_seed": None,
        "prompt_sha256": None,
        "demo_video": None,
        "void": False,
        "outcome": "tie",
    }
    for side in SIDES:
        row.update(
            {
                f"{side}_success": None,
                f"{side}_steps": None,
                f"{side}_video": None,
                f"{side}_error": None,
            }
        )
    return row


def tally(units: list[dict[str, Any]]) -> dict[str, int]:
    """From the challenger's side: a void unit counts only as void, the others by `outcome`."""
    t = {"wins": 0, "losses": 0, "ties": 0, "void": 0}
    for u in units:
        key = (
            "void"
            if u.get("void")
            else {"challenger": "wins", "king": "losses"}.get(u.get("outcome"), "ties")
        )
        t[key] += 1
    return t


def index_record(
    *,
    event_id: str,
    kind: str,
    block: int,
    finished_at: str,
    king: dict[str, str],
    challenger: dict[str, str] | None,
    king_score: float | None,
    challenger_score: float | None,
    dethroned: bool,
    units: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "seq": 0,  # assigned by the store on append
        "event_id": event_id,
        "kind": kind,
        "block": block,
        "finished_at": finished_at,
        "king": king,
        "challenger": challenger,
        "king_score": king_score,
        "challenger_score": challenger_score,
        "dethroned": dethroned,
        **tally(units),
    }


def media_shas(units: list[dict[str, Any]]) -> list[str]:
    shas = {u.get(k) for u in units for k in ("demo_video", "king_video", "challenger_video")}
    return sorted(s for s in shas if isinstance(s, str) and s)

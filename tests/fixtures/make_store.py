"""Write the fixture store: a small, byte-reproducible history for the dashboard.

    python tests/fixtures/make_store.py [OUT]      # default: tests/fixtures/store

A genesis (one run per task) and one duel (two runs per task) in which a replay challenger
dethrones a zero-action king, one unit void for the harness, and a queue with one entry waiting,
in the shapes the orchestrator publishes. Units are dealt as a duel deals them.

It is NOT a competition result: it renders the dashboard against the layout this orchestrator
writes (`VECTOR_STORE=<abs path> npm run dev`) and catches an unintended change to that layout.
After an intended change, regenerate it and commit the result.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vector_orchestrator.canon import canonical_sha256  # noqa: E402
from vector_orchestrator.duel import score  # noqa: E402
from vector_orchestrator.ids import SubmissionRef, duel_id, event_id  # noqa: E402
from vector_orchestrator.spec import load_spec_file  # noqa: E402
from vector_orchestrator.store.records import index_record, unit_row  # noqa: E402
from vector_orchestrator.store.writer import Store  # noqa: E402
from vector_orchestrator.units import deal  # noqa: E402

KING = SubmissionRef.make("robotensor/vector-zero-policy", "0" * 39 + "1")
CHALLENGER = SubmissionRef.make("robotensor/vector-replay-policy", "0" * 39 + "2")
WAITING = SubmissionRef.make("robotensor/vector-example-policy", "0" * 39 + "3")
BENCHMARK = {"commit": "0" * 40, "source_sha256": "f" * 64}
NOTE = "Fixture store written by tests/fixtures/make_store.py; not a result."


def side(ref: SubmissionRef) -> dict:
    return {**ref.as_dict(), "weights_sha256": canonical_sha256(ref.key), "refused": None}


def rows(spec, did: str, runs: int) -> list[dict]:
    """A duel's unit rows, each with a stand-in prompt hash and scene."""
    out = []
    for unit in deal(spec, did, None, runs):
        row = unit_row(unit)
        row["prompt_sha256"] = canonical_sha256({"unit_id": unit["unit_id"]})
        row["scene_seed"] = unit["scene_seeds"][0]
        out.append(row)
    return out


def publish(store, spec, record, event_fields) -> None:
    event = {
        "event_id": record["event_id"],
        "kind": record["kind"],
        "block": record["block"],
        "seed": None,
        "spec_version": spec.version,
        "finished_at": record["finished_at"],
        "dethroned": record["dethroned"],
        "benchmark": BENCHMARK,
        "notes": [NOTE],
        **event_fields,
    }
    store.write_event(event)
    store.append(record)


def build(out: Path) -> Path:
    spec = load_spec_file(ROOT / "spec.json")
    if out.exists():
        shutil.rmtree(out)
    store = Store(out, spec)
    store.init()

    # -- genesis: the zero-action policy takes the empty throne, scored on its own units
    gid = duel_id(spec.version, KING, None)
    units = rows(spec, gid, 1)
    for n, unit in enumerate(units):
        unit.update(king_success=n % 5 == 0, king_steps=100 + n)
    scores = score.task_scores(units, "king", spec.tasks)
    record = index_record(
        event_id=event_id("genesis", 1, gid),
        kind="genesis",
        block=1,
        finished_at="2026-09-13T09:00:00Z",
        king=KING.as_dict(),
        challenger=None,
        king_score=scores["average"],
        challenger_score=None,
        dethroned=False,
        units=units,
    )
    publish(store, spec, record, {
        "duel_id": gid, "runs_per_task": 1, "started_at": "2026-09-13T08:30:00Z",
        "king": side(KING), "challenger": None, "scores": {"king": scores, "challenger": None},
        "verdict": {"reason": "genesis", "void_fraction": 0.0}, "units": units,
    })  # fmt: skip

    # -- one duel: the replay challenger against the zero-action king
    did = duel_id(spec.version, CHALLENGER, KING)
    units = rows(spec, did, 2)
    for n, unit in enumerate(units):
        if n == 4:
            unit.update(void=True, challenger_error="fake: the simulator lost the scene")
            continue
        won = n % 3 != 2
        unit.update(
            king_success=False,
            challenger_success=won,
            outcome="challenger" if won else "tie",
            king_steps=unit["scene_seed"] % 300 + 100,
            challenger_steps=120 + n,
        )
    verdict = score.verdict(units, spec.margin, spec.tasks, spec.alpha)
    record = index_record(
        event_id=event_id("duel", 2, did),
        kind="duel",
        block=2,
        finished_at="2026-09-13T11:30:00Z",
        king=KING.as_dict(),
        challenger=CHALLENGER.as_dict(),
        king_score=verdict.king_scores["average"],
        challenger_score=verdict.challenger_scores["average"],
        dethroned=verdict.dethroned,
        units=units,
    )
    publish(store, spec, record, {
        "duel_id": did, "runs_per_task": 2, "started_at": "2026-09-13T10:00:00Z",
        "king": side(KING), "challenger": side(CHALLENGER),
        "scores": {"king": verdict.king_scores, "challenger": verdict.challenger_scores},
        "verdict": {**verdict.as_dict(), "void_fraction": score.void_fraction(units)},
        "units": units,
    })  # fmt: skip

    # -- the queue as the dashboard sees it
    store.write_queue({"entries": [{**WAITING.as_dict(), "block": 3}]})
    # The store's own bookkeeping (locks) is never published and never part of the fixture.
    for dotfile in out.rglob(".*"):
        if dotfile.is_file():
            dotfile.unlink()
    return out


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "store"
    print(build(target.resolve()))

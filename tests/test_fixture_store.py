"""The committed fixture store is what the orchestrator writes today, holds together, and reads
the way the dashboard reads a store."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

from store_helpers import check_store

FIXTURES = Path(__file__).resolve().parent / "fixtures"
STORE = FIXTURES / "store"
EVENT_ID = re.compile(r"^[0-9a-f]{8,64}$")


def tree(root: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()
    }


def test_the_fixture_store_is_what_the_generator_writes_today(tmp_path):
    out = tmp_path / "store"
    subprocess.run([sys.executable, str(FIXTURES / "make_store.py"), str(out)], check=True)
    assert tree(out) == tree(STORE), (
        "the fixture store is stale: if the change was intended, run "
        "`python tests/fixtures/make_store.py` and commit tests/fixtures/store"
    )


def test_the_fixture_store_holds_together_under_the_shipped_spec(spec):
    report = check_store(STORE, spec)
    assert report.ok, report.errors
    assert (report.records, report.events) == (2, 2)


def test_the_dashboard_can_read_the_fixture_store(spec):
    head = json.loads((STORE / "head.json").read_text())
    records = [json.loads(line) for line in (STORE / "index-0000.jsonl").read_text().splitlines()]
    assert [r["kind"] for r in records] == ["genesis", "duel"]
    duel = records[-1]
    assert duel["dethroned"] and head["king"] == duel["challenger"]
    for record in records:
        assert EVENT_ID.match(record["event_id"])
        event = json.loads((STORE / "events" / f"{record['event_id']}.json").read_text())
        # A unit is shown under its task, which must be one the spec names.
        assert {u["task"] for u in event["units"]} == set(spec.tasks)
        assert set(event["scores"]["king"]) == {"average", *spec.tasks}
    duel_event = json.loads((STORE / "events" / f"{duel['event_id']}.json").read_text())
    assert len(duel_event["units"]) == 2 * len(spec.tasks)
    assert sum(u["void"] for u in duel_event["units"]) == duel["void"] == 1
    queue = json.loads((STORE / "queue.json").read_text())
    assert [e["repo"] for e in queue["entries"]] == ["robotensor/vector-example-policy"]

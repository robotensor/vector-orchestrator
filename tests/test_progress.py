"""The running duel's progress: what `running.json` holds, and when the board pushes it."""

import json

import pytest

from vector_orchestrator.progress import PHASES, ProgressBoard, build_progress, counts
from vector_orchestrator.store.mirror import STORE_LAYOUT
from vector_orchestrator.store.writer import Store

KING = {"key": "k" * 16, "repo": "org/king", "revision": "1" * 40}
CHALLENGER = {"key": "c" * 16, "repo": "org/challenger", "revision": "2" * 40}


def unit(unit_id, *, prompt=None, void=False, king=None, challenger=None, task="beat_block_hammer"):
    return {
        "unit_id": unit_id,
        "task": task,
        "prompt_sha256": prompt,
        "void": void,
        "king_success": king,
        "challenger_success": challenger,
    }


def doc(**changes):
    base = dict(
        event_id="ab" * 8,
        kind="duel",
        king=KING,
        challenger=CHALLENGER,
        phase="evaluating",
        side="challenger",
        units=[],
        current=None,
        message="",
        started_at="2026-10-01T00:00:00Z",
    )
    return build_progress(**{**base, **changes})


def test_counts_what_is_made_and_what_each_side_played():
    units = [
        unit("u1", prompt="p", king=True, challenger=False),
        unit("u2", prompt="p", challenger=True),
        unit("u3", void=True),  # a void unit is played by neither, and finished for both
        unit("u4"),
    ]
    assert counts(units, genesis=False) == {
        "units": 4,
        "demonstrations": 3,
        "challenger": 3,
        "king": 2,
    }
    assert counts(units, genesis=True)["king"] is None


def test_a_progress_document_names_the_unit_running_now_and_its_task():
    built = doc(
        units=[unit("u1", task="place_shoe")],
        current={"unit_id": "u1", "side": "king"},
        message="x" * 1000,
    )
    assert built["schema"] == 1
    assert built["current"] == {"unit_id": "u1", "side": "king", "task": "place_shoe"}
    assert len(built["message"]) == 300
    assert built["progress"]["units"] == 1 and built["outcome"] is None
    assert built["updated_at"]


def test_a_genesis_has_no_king_to_count():
    assert doc(king=None, kind="genesis")["progress"]["king"] is None


@pytest.mark.parametrize(
    "changes",
    [
        {"phase": "running"},
        {"event_id": "XYZ"},
        {"side": "both"},
        {"outcome": {"status": "won"}},
    ],
)
def test_what_a_progress_document_refuses(changes):
    with pytest.raises(ValueError):
        doc(**changes)


def test_phases_run_in_order_and_end_in_done_or_failed():
    assert PHASES[:5] == ("fetching", "checking", "materializing", "evaluating", "publishing")
    assert PHASES[-2:] == ("done", "failed")


def test_the_board_writes_every_state_and_pushes_on_a_phase_change_or_the_interval(
    tmp_path, monkeypatch
):
    store = Store(tmp_path / "store", spec=None)
    pushes = []
    board = ProgressBoard(store, push=lambda: pushes.append(store.drain_touched()), interval_s=60)
    clock = iter([1000.0, 1010.0, 1020.0, 1070.0])
    monkeypatch.setattr("vector_orchestrator.progress.time.monotonic", lambda: next(clock))

    board.post(doc(phase="fetching"), force=True)  # a phase change: pushed now
    board.post(doc(message="one"))  # 10 s later: written, not pushed
    assert json.loads(store.running_path().read_text())["message"] == "one"
    board.post(doc(message="two"))  # 20 s
    board.post(doc(message="three"))  # 70 s after the last push: pushed
    assert pushes == [["running.json"], ["running.json"]]


def test_the_board_files_are_mirrored():
    for name in ("running.json", "champions.json", "queue.json"):
        assert STORE_LAYOUT.fullmatch(name), name

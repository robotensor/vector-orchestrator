"""`vector-orchestrator duel`, with the fake benchmark and the example policies served locally."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from duel_helpers import REPLAY, REPLAY_REF, ZERO, ZERO_REF
from store_helpers import check_store
from vector_orchestrator.cli import main
from vector_orchestrator.duel.side import read_results
from vector_orchestrator.store.writer import Store, store_lock

ZERO_AT = ZERO_REF.entry
REPLAY_AT = REPLAY_REF.entry


@pytest.fixture
def cli_spec(spec_doc, write_spec):
    """Three tasks, one run each, and an alpha three straight wins can clear."""
    doc = {
        **spec_doc,
        "tasks": ["place_cube_plate", "stack_two_blocks", "press_button"],
        "runs_per_task": 1,
    }
    doc["crown"] = {**doc["crown"], "alpha": 0.2}
    doc["budgets"] = {**doc["budgets"], "act": 2.0}
    return write_spec(doc, name="cli.json")


@pytest.fixture
def store(cli_spec, tmp_path, fake_module, monkeypatch, capsys):
    monkeypatch.setattr(
        "vector_orchestrator.robotwin.RoboTwin.from_environment",
        classmethod(lambda cls, level, environ=None: fake_module.FakeBenchmark()),
    )
    root = tmp_path / "store"
    assert main(["--spec", str(cli_spec.path), "store", "init", "--root", str(root)]) == 0
    capsys.readouterr()
    return root


def run(spec, store, tmp_path, *args: str, zero: Path = ZERO) -> int:
    return main(
        [
            "--spec", str(spec.path), "duel", "--store", str(store), "--run-dir", str(tmp_path / "runs"),
            "--local", f"{REPLAY_REF.repo}={REPLAY}", "--local", f"{ZERO_REF.repo}={zero}", *args,
        ]
    )  # fmt: skip


def test_a_genesis_then_a_duel_publish_every_unit_with_one_prompt_hash_for_both_sides(
    cli_spec, store, tmp_path, capsys
):
    assert run(cli_spec, store, tmp_path, "--challenger", ZERO_AT) == 0
    genesis = json.loads(capsys.readouterr().out)
    assert genesis["status"] == "published" and genesis["kind"] == "genesis"

    assert run(cli_spec, store, tmp_path, "--challenger", REPLAY_AT) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "published" and out["kind"] == "duel" and out["dethroned"] is True
    assert out["units"] == 3 and out["void"] == 0 and out["challenger_score"] == 1.0

    event = Store(store, cli_spec).event(out["event_id"])
    run_dir = Path(out["run_dir"])
    challenger, king = read_results(run_dir / "challenger"), read_results(run_dir / "king")
    for unit in event["units"]:
        sha = unit["prompt_sha256"]
        assert (
            sha
            and challenger[unit["unit_id"]]["prompt_sha256"]
            == sha
            == king[unit["unit_id"]]["prompt_sha256"]
        )
    assert [r["block"] for r in Store(store, cli_spec).iter_index()] == [1, 2]
    report = check_store(store, cli_spec)
    assert report.ok, report.errors
    # Clips counted per record: the genesis's demonstration and rollout, the duel's three each.
    assert (report.records, report.events, report.media) == (2, 2, 15)


def test_a_failed_duel_run_again_resumes_at_its_block(cli_spec, store, tmp_path, capsys):
    assert run(cli_spec, store, tmp_path, "--challenger", ZERO_AT) == 0
    capsys.readouterr()
    # The king's directory is missing: the duel fails for the harness, undecided.
    code = run(cli_spec, store, tmp_path, "--challenger", REPLAY_AT, zero=tmp_path / "no")
    assert code == 2 and "fetching the king" in capsys.readouterr().err
    assert run(cli_spec, store, tmp_path, "--challenger", REPLAY_AT) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "published" and not (Path(out["run_dir"]) / "failed.txt").exists()
    assert len(list((tmp_path / "runs").iterdir())) == 2, "the failed duel was not resumed"


def test_the_duel_command_refuses_while_another_holds_the_store(cli_spec, store, tmp_path, capsys):
    with store_lock(store):
        assert run(cli_spec, store, tmp_path, "--challenger", ZERO_AT) == 2
    assert "another orchestrator is publishing" in capsys.readouterr().err
    assert not (tmp_path / "runs").exists()


def test_what_the_duel_command_refuses(cli_spec, store, tmp_path, capsys):
    assert run(cli_spec, store, tmp_path, "--challenger", "org/nobody@main") == 2
    assert "no local directory for org/nobody" in capsys.readouterr().err
    assert run(cli_spec, store, tmp_path, "--challenger", "no-revision") == 2
    assert "is not owner/name@revision" in capsys.readouterr().err
    assert run(cli_spec, store, tmp_path, "--challenger", REPLAY_AT, "--live-url", "http://x") == 2
    assert "--live-token-env" in capsys.readouterr().err
    assert run(cli_spec, tmp_path / "nothing", tmp_path, "--challenger", REPLAY_AT) == 2
    assert "is not a store" in capsys.readouterr().err

"""Prompts produced once per duel, through the fake benchmark's real materialize command."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from vector_orchestrator.canon import sha256_file
from vector_orchestrator.duel.materialize import (
    MANIFEST_FILE,
    materialize_units,
    read_manifest,
)
from vector_orchestrator.units import deal


@pytest.fixture
def fake(fake_benchmark):
    return fake_benchmark


@pytest.fixture
def fake_spec(duel_spec):
    return duel_spec


@pytest.fixture
def units(fake_spec):
    return deal(fake_spec, "d" * 64, None, 2)


def counting(fake):
    """The fake, counting its materialize commands."""
    calls = []

    class Counted(type(fake)):
        def materialize_command(self, *, unit, out_dir):
            calls.append(unit["unit_id"])
            return super().materialize_command(unit=unit, out_dir=out_dir)

    return Counted(), calls


def test_no_prompt_is_made_past_the_duels_deadline(fake_spec, fake, units, tmp_path):
    counted, calls = counting(fake)
    out = materialize_units(
        fake_spec, units, tmp_path, benchmark=counted, deadline=time.monotonic() - 1
    )
    assert calls == [], "an expert ran after the duel's time was up"
    assert all(
        p.void and "ran out of its wall-clock budget" in p.error for p in out.prompts.values()
    )


def test_a_materialization_is_cut_at_the_duels_deadline_before_its_own_budget(
    fake_spec, fake, units, tmp_path
):
    units[0]["fake_materialize"] = "hang"
    started = time.monotonic()
    out = materialize_units(fake_spec, units[:1], tmp_path, benchmark=fake, deadline=started + 1.0)
    assert time.monotonic() - started < 30, "the expert ran on past the duel's deadline"
    (prompt,) = out.prompts.values()
    assert prompt.void and "budget" in prompt.error


def test_every_unit_gets_a_verified_prompt_and_its_hash(fake_spec, fake, units, tmp_path):
    seen = []
    out = materialize_units(
        fake_spec,
        units,
        tmp_path,
        benchmark=fake,
        on_prompt=lambda unit, prompt: seen.append(unit["unit_id"]),
    )
    assert seen == [u["unit_id"] for u in units] == list(out.prompts)
    for unit in units:
        prompt = out.prompts[unit["unit_id"]]
        assert not prompt.void, prompt.error
        assert prompt.path == tmp_path / unit["unit_id"] / "prompt.npz"
        assert prompt.sha256 == sha256_file(prompt.path)
        assert prompt.demo_sha256 == sha256_file(prompt.demo_clip)
        assert prompt.path.stat().st_mode & 0o222 == 0, "a side's benchmark could rewrite it"
    assert out.void == 0


@pytest.mark.parametrize(
    "behaviour, reason",
    [
        ("crash", "exited 3: the expert lost the GPU"),
        ("wrong", "the prompt is not the one the unit asked for: scene_seed"),
        ("expert_fails", "the demonstration did not succeed: the expert never succeeded"),
    ],
)
def test_a_unit_whose_prompt_fails_is_void_with_the_reason_and_the_rest_go_on(
    fake_spec, fake, units, tmp_path, behaviour, reason
):
    units[1]["fake_materialize"] = behaviour
    out = materialize_units(fake_spec, units, tmp_path, benchmark=fake)
    bad = out.prompts[units[1]["unit_id"]]
    assert bad.void and bad.sha256 is None and bad.path is None
    assert bad.error.startswith("fake: materialize: ") and reason in bad.error
    assert [p.void for p in out.prompts.values()] == [False, True, False, False]
    assert out.void == 1


def test_a_materialization_past_its_budget_is_void(fake_spec, fake, units, tmp_path):
    units[0]["fake_materialize"] = "hang"
    out = materialize_units(fake_spec, units[:1], tmp_path, benchmark=fake, timeout_s=1.0)
    assert out.prompts[units[0]["unit_id"]].error == "fake: materialize: exceeded its 1s budget"


def test_a_verify_prompt_that_hashes_something_else_is_refused(fake_spec, fake, units, tmp_path):
    class Hashes(type(fake)):
        def verify_prompt(self, *, path, unit):
            return {**super().verify_prompt(path=path, unit=unit), "sha256": "0" * 64}

    out = materialize_units(fake_spec, units[:1], tmp_path, benchmark=Hashes())
    assert "not the sha256 of its bytes" in out.prompts[units[0]["unit_id"]].error


def test_a_restarted_duel_reuses_its_prompts_rather_than_producing_new_ones(
    fake_spec, fake, units, tmp_path
):
    counted, calls = counting(fake)
    first = materialize_units(fake_spec, units[:2], tmp_path, benchmark=counted)
    assert calls == [units[0]["unit_id"], units[1]["unit_id"]]
    # A crash while the third was being written: its line never made it.
    again = materialize_units(fake_spec, units, tmp_path, benchmark=counted)
    assert calls == [u["unit_id"] for u in units], "a finished prompt was produced twice"
    for unit_id, prompt in first.prompts.items():
        assert again.prompts[unit_id].sha256 == prompt.sha256
    assert set(read_manifest(tmp_path)) == {u["unit_id"] for u in units}


def test_a_prompt_changed_on_disk_is_void_on_resume_not_trusted(fake_spec, fake, units, tmp_path):
    first = materialize_units(fake_spec, units[:1], tmp_path, benchmark=fake)
    path = first.prompts[units[0]["unit_id"]].path
    os.chmod(path, 0o644)
    path.write_bytes(path.read_bytes() + b"\0")
    again = materialize_units(fake_spec, units[:1], tmp_path, benchmark=fake)
    prompt = again.prompts[units[0]["unit_id"]]
    assert prompt.void and "changed after it was materialized" in prompt.error
    lines = (tmp_path / MANIFEST_FILE).read_text().splitlines()
    assert json.loads(lines[-1])["void"] is True and len(lines) == 2


def test_a_prompt_keeps_the_scene_seed_its_benchmark_chose_and_a_resumed_duel_does_too(
    fake_spec, fake, units, tmp_path
):
    """The expert rejects the unit's first candidate and builds the prompt on the second."""
    units[0]["scene_seeds"] = [7, 8]
    units[0]["fake_materialize"] = "reject_first"
    first = materialize_units(fake_spec, units[:2], tmp_path, benchmark=fake)
    chosen, plain = (first.prompts[u["unit_id"]] for u in units[:2])
    assert not chosen.void, chosen.error
    assert chosen.scene_seed == 8
    assert plain.scene_seed == units[1]["scene_seeds"][0]
    again = materialize_units(fake_spec, units[:2], tmp_path, benchmark=fake)
    assert again.prompts[units[0]["unit_id"]].scene_seed == 8


def test_a_result_naming_another_scene_seed_than_its_prompt_is_void(
    fake_spec, fake, units, tmp_path
):
    class Misremembers(type(fake)):
        def read_result(self, *, out_dir):
            return {**super().read_result(out_dir=out_dir), "scene_seed": 12345}

    out = materialize_units(fake_spec, units[:1], tmp_path, benchmark=Misremembers())
    prompt = out.prompts[units[0]["unit_id"]]
    assert prompt.void and prompt.scene_seed is None
    assert "built on scene seed 12345, but the prompt's own is" in prompt.error


@pytest.mark.parametrize("written", ["0" * 64, 7, "the file's"])
def test_a_prompt_sha256_the_result_names_must_be_the_files(
    fake_spec, fake, units, tmp_path, written
):
    """RoboTwin's materialize result names the sha256 of the prompt it wrote: when a result names
    one, it is the file's, or the prompt is not the one both sides would run from."""

    class Reporting(type(fake)):
        def read_result(self, *, out_dir):
            own = sha256_file(Path(out_dir) / "prompt.npz")
            value = own if written == "the file's" else written
            return {**super().read_result(out_dir=out_dir), "prompt_sha256": value}

    out = materialize_units(fake_spec, units[:1], tmp_path, benchmark=Reporting())
    prompt = out.prompts[units[0]["unit_id"]]
    if written == "the file's":
        assert not prompt.void and prompt.sha256 == sha256_file(prompt.path), prompt.error
        return
    assert prompt.void and prompt.sha256 is None
    assert f"its result says it wrote a prompt hashing to {str(written)[:12]}..." in prompt.error

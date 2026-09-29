"""Where a duel's work runs: slots on the host's cards, handed out by priority and kept server."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

from vector_orchestrator.duel import slots as S


def test_the_cards_are_the_visible_devices_when_they_are_named():
    assert S.gpus({"CUDA_VISIBLE_DEVICES": "2,5"}) == ["2", "5"]
    assert S.gpus({"CUDA_VISIBLE_DEVICES": " GPU-ab12 , GPU-cd34 "}) == ["GPU-ab12", "GPU-cd34"]
    assert S.gpus({"CUDA_VISIBLE_DEVICES": ""}) == [None]


def test_otherwise_every_card_nvidia_smi_lists_and_none_without_it(monkeypatch):
    listing = "GPU 0: NVIDIA GeForce RTX 5090 (UUID: a)\nGPU 1: NVIDIA GeForce RTX 5090 (UUID: b)\n"
    monkeypatch.setattr(
        S.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=listing, returncode=0)
    )
    assert S.gpus({}) == ["0", "1"]

    def missing(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(S.subprocess, "run", missing)
    assert S.gpus({}) == [None]


def test_slots_spread_over_the_cards_and_see_only_their_own(tmp_path):
    slots = S.Slots(2, ["a", "b"], tmp_path)
    assert len(slots) == 4
    assert [slot.gpu for slot in slots.slots] == ["a", "b", "a", "b"]
    assert slots.slots[1].env == {"CUDA_VISIBLE_DEVICES": "b"}
    assert S.Slots(3, [None], tmp_path).slots[0].env == {}
    with slots.take() as first, slots.take() as second:
        assert (first.gpu, second.gpu) == ("a", "b"), "two jobs landed on one card"


def _waiting(slots, priority, handle, taken):
    """A thread waiting for a slot; it records which one it got, and holds it until released."""
    release = threading.Event()

    def job():
        with slots.take(priority, handle) as slot:
            taken.append((handle, slot.index))
            release.wait(10)

    thread = threading.Thread(target=job, daemon=True)
    thread.start()
    time.sleep(0.1)  # asked in this order
    return thread, release


def test_a_play_gets_a_free_slot_before_a_demonstration_that_asked_first(tmp_path):
    slots = S.Slots(1, [None], tmp_path)
    taken: list = []
    with slots.take(S.PLAY):
        make, make_release = _waiting(slots, S.MAKE, "make", taken)
        play, play_release = _waiting(slots, S.PLAY, "play", taken)
    play_release.set()
    make_release.set()
    play.join(5)
    make.join(5)
    assert [who for who, _ in taken] == ["play", "make"]


def test_a_freed_slot_goes_to_the_job_that_serves_what_its_server_already_serves(tmp_path):
    slots = S.Slots(2, [None], tmp_path)
    kept = SimpleNamespace(handle="king", stop=lambda: None)
    taken: list = []
    with slots.take() as zero, slots.take() as one:
        zero.servers.current = kept  # slot 0 keeps the king's server
        challenger, challenger_release = _waiting(slots, S.PLAY, "challenger", taken)
        king, king_release = _waiting(slots, S.PLAY, "king", taken)
        assert one.index == 1
    # Both slots are free now: the king gets slot 0, whose server is its own, though it asked
    # after the challenger, which gets the other.
    king_release.set()
    challenger_release.set()
    king.join(5)
    challenger.join(5)
    assert dict(taken) == {"king": 0, "challenger": 1}
    zero.servers.current = None

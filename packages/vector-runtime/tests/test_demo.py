"""The benchmark's arrays as the network's prompt and frames, and its action as the benchmark's row."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")
pytest.importorskip("torch")
pytest.importorskip("scipy")

from vector_runtime import demo  # noqa: E402
from vector_runtime.net import CHUNK  # noqa: E402

#: RoboTwin's copy of XPolicyLab's image codec, the one its episode writer used.
ROBOTWIN = Path(os.environ.get("ROBOTWIN_ROOT", "/root/robotensor/vector/RoboTwin-Vector"))
IDENTITY_6D = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]


def demonstration(count: int = 33, height: int = 24, width: int = 32) -> dict:
    rng = np.random.default_rng(0)
    arrays = {
        f"frames_{camera}": rng.integers(0, 256, (count, height, width, 3), dtype=np.uint8)
        for camera in demo.BENCH_CAMERAS.values()
    }
    endpose = np.zeros((count, 16))
    endpose[:, 0] = np.linspace(-0.4, -0.3, count)
    endpose[:, 3] = 1.0  # identity quaternions
    endpose[:, 7] = np.linspace(1.0, 0.0, count)
    endpose[:, 8] = 0.4
    endpose[:, 11] = 1.0
    endpose[:, 15] = 1.0
    arrays["endpose"] = endpose
    arrays["frequency"] = np.asarray(15.0)
    for value in arrays.values():
        value.setflags(write=False)  # arrays arrive read-only
    return arrays


def test_the_prompt_keeps_one_frame_per_chunk_and_every_action():
    arrays = demonstration()
    endpose = arrays["endpose"]
    obs, action = demo.prompt(arrays)
    # 32 steps (the last frame only closes the last action) -> 3 groups of 15, 13 rows of padding
    assert action.shape == (3, CHUNK, 20) and action.dtype == np.float32
    np.testing.assert_array_equal(obs["left_ee_pos"], endpose[[0, 15, 30], :3].astype(np.float32))
    np.testing.assert_array_equal(
        obs["left_gripper_states"], endpose[[0, 15, 30], 7:8].astype(np.float32)
    )
    np.testing.assert_allclose(obs["right_ee_ori"], [IDENTITY_6D] * 3)
    rows = action.reshape(-1, 20)
    np.testing.assert_array_equal(rows[:32, :3], endpose[1:, :3].astype(np.float32))
    np.testing.assert_array_equal(rows[:32, 19], endpose[1:, 15].astype(np.float32))
    assert not rows[32:].any()
    assert obs["head_rgb"].shape == (3, 3, 224, 224) and obs["head_rgb"].dtype == np.float32


def test_prompt_frames_go_through_jpeg_then_resize():
    arrays = demonstration()
    obs, _ = demo.prompt(arrays)
    original = arrays["frames_left_camera"][15]
    expected = np.moveaxis(demo.resize(demo.jpeg(original)), -1, 0).astype(np.float32) / 255.0
    np.testing.assert_array_equal(obs["left_wrist_rgb"][1], expected)
    assert not np.array_equal(demo.jpeg(original), original)  # noise does not survive JPEG


def test_a_short_demonstration_is_one_padded_group():
    obs, action = demo.prompt(demonstration(count=5))
    assert action.shape == (1, CHUNK, 20) and not action[0, 4:].any()
    assert obs["head_rgb"].shape[0] == 1


@pytest.mark.skipif(
    not (ROBOTWIN / "data" / "decode_image_bit.py").exists(), reason="no RoboTwin checkout"
)
def test_the_jpeg_round_trip_is_xpolicylabs_encode_then_decode():
    spec = importlib.util.spec_from_file_location("codec", ROBOTWIN / "data/decode_image_bit.py")
    codec = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(codec)
    for frame in demonstration(count=3, height=240, width=320)["frames_head_camera"]:
        np.testing.assert_array_equal(
            demo.jpeg(frame), codec.decode_image_bit(codec.encode_image_bit(frame))
        )


def test_a_live_frame_is_resized_without_jpeg():
    arrays = demonstration()
    row = {k: v[2] for k, v in arrays.items() if k.startswith("frames_") or k == "endpose"}
    frame = demo.frame(row)
    np.testing.assert_array_equal(frame["right_wrist_rgb"], demo.resize(row["frames_right_camera"]))
    np.testing.assert_array_equal(
        frame["right_ee_pos"], arrays["endpose"][2, 8:11].astype(np.float32)
    )
    np.testing.assert_array_equal(
        frame["left_gripper_states"], arrays["endpose"][2, 7:8].astype(np.float32)
    )


def test_an_action_is_the_benchmarks_row():
    action = np.asarray([0.1, 0.2, 0.3, *IDENTITY_6D, 1.7, 0.4, 0.5, 0.6, *IDENTITY_6D, -0.2])
    row = demo.action_row(action)
    assert row.dtype == np.float64 and row.shape == (16,)
    expected = [0.1, 0.2, 0.3, 1, 0, 0, 0, 1.0, 0.4, 0.5, 0.6, 1, 0, 0, 0, 0.0]
    np.testing.assert_array_equal(row, np.asarray(expected, dtype=np.float32).astype(np.float64))


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda a: a.pop("frames_head_camera"), "frames_head_camera"),
        (
            lambda a: a.update(frames_left_camera=a["frames_left_camera"].astype(np.float32)),
            "uint8",
        ),
        (lambda a: a.update(endpose=a["endpose"][:, :14]), "endpose"),
        (lambda a: a.update(endpose=np.full((33, 16), np.nan)), "non-finite"),
        (lambda a: a.update(frames_right_camera=a["frames_right_camera"][:4]), "4 frames"),
    ],
)
def test_a_bad_demonstration_is_refused(change, message):
    arrays = dict(demonstration())
    change(arrays)
    with pytest.raises(demo.DemonstrationError, match=message):
        demo.prompt(arrays)


def test_one_frame_is_not_a_demonstration():
    arrays = {k: (v[:1] if k != "frequency" else v) for k, v in demonstration().items()}
    with pytest.raises(demo.DemonstrationError, match="at least 2"):
        demo.prompt(arrays)

"""The benchmark's arrays as the network's inputs, and its output as the benchmark's action.

A demonstration is laid out as RoboTwin episodes are recorded (the XPolicyLab HDF5 layout): state
is the recording's rows `[:-1]`, action its rows `[1:]`, and each stored camera frame went through
a JPEG round trip (OpenCV, default quality). Frames are resized to 224x224
(INTER_AREA). Per arm, a pose `[x y z qw qx qy qz]` becomes position and rot6d (the first two rows
of its rotation matrix, in float64) and the gripper stays as is.

The prompt keeps one frame per CHUNK steps and every action, grouped CHUNK to a frame; the last
group is padded with zeros. A live observation is converted the same way, without JPEG.

Adapted from `behavior_prompting` (MIT, Copyright (c) 2026 Austin Patel); see
THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .net import CAMERAS, CHUNK, IMAGE

#: The benchmark's camera for each of the network's.
BENCH_CAMERAS = dict(zip(CAMERAS, ("head_camera", "left_camera", "right_camera"), strict=True))
ARMS = ("left", "right")
#: The benchmark's `endpose` row (and the action answered): per arm, pose 7 then gripper 1.
ENDPOSE_DIM = 16


class DemonstrationError(ValueError):
    """The arrays are not a demonstration or an observation the network can use."""


def jpeg(rgb: np.ndarray) -> np.ndarray:
    import cv2

    ok, data = cv2.imencode(".jpg", cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR))
    if not ok:
        raise DemonstrationError(f"cannot JPEG-encode a frame of shape {rgb.shape}")
    return cv2.cvtColor(cv2.imdecode(data, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def resize(rgb: np.ndarray) -> np.ndarray:
    import cv2

    if rgb.shape[:2] == (IMAGE, IMAGE):
        return rgb.astype(np.uint8, copy=False)
    return cv2.resize(rgb, (IMAGE, IMAGE), interpolation=cv2.INTER_AREA)


def rot6d(quat_wxyz: np.ndarray) -> np.ndarray:
    """(..., 4) [qw qx qy qz] -> (..., 6), through the axis-angle the recorded episodes went through."""
    import torch
    from scipy.spatial.transform import Rotation

    q = np.asarray(quat_wxyz, dtype=np.float64)
    rotvec = (
        Rotation.from_quat(q[..., [1, 2, 3, 0]].reshape(-1, 4))
        .as_rotvec()
        .reshape(q.shape[:-1] + (3,))
    )
    aa = torch.from_numpy(rotvec)
    angle = torch.norm(aa, p=2, dim=-1, keepdim=True)
    r, i, j, k = torch.unbind(
        torch.cat([torch.cos(angle * 0.5), aa * (0.5 * torch.sinc(angle * 0.5 / torch.pi))], -1), -1
    )
    s = 2.0 / (r * r + i * i + j * j + k * k)
    rows = (1 - s * (j * j + k * k), s * (i * j - k * r), s * (i * k + j * r),
            s * (i * j + k * r), 1 - s * (i * i + k * k), s * (j * k - i * r))  # fmt: skip
    return torch.stack(rows, -1).numpy()


def quat_from_rot6d(d6: np.ndarray) -> np.ndarray:
    """(6,) -> [qw qx qy qz] with qw >= 0, through Gram-Schmidt."""
    import torch
    import torch.nn.functional as F
    from scipy.spatial.transform import Rotation

    d6 = torch.from_numpy(np.asarray(d6, dtype=np.float64))
    b1 = F.normalize(d6[:3], dim=-1)
    b2 = F.normalize(d6[3:] - (b1 * d6[3:]).sum(-1, keepdim=True) * b1, dim=-1)
    mat = torch.stack((b1, b2, torch.cross(b1, b2, dim=-1)), dim=-2).numpy()
    q = Rotation.from_matrix(mat).as_quat()[..., [3, 0, 1, 2]]
    return np.where(q[..., :1] < 0, -q, q)


def _endpose(arrays: Mapping[str, Any], ndim: int) -> np.ndarray:
    endpose = np.asarray(arrays.get("endpose", np.zeros(0)))
    if endpose.ndim != ndim or endpose.shape[-1] != ENDPOSE_DIM or endpose.dtype.kind not in "fiu":
        raise DemonstrationError(f"endpose is {endpose.dtype} {endpose.shape}, expected (..., 16)")
    endpose = endpose.astype(np.float64)
    if not np.all(np.isfinite(endpose)):
        raise DemonstrationError("endpose holds non-finite values")
    return endpose


def _frames(arrays: Mapping[str, Any], cam: str, ndim: int) -> np.ndarray:
    name = "frames_" + BENCH_CAMERAS[cam]
    frames = np.asarray(arrays.get(name, np.zeros(0)))
    if frames.dtype != np.uint8 or frames.ndim != ndim or frames.shape[-1] != 3:
        raise DemonstrationError(f"{name} is {frames.dtype} {frames.shape}, expected uint8 RGB")
    return frames


def proprio(endpose: np.ndarray) -> dict[str, np.ndarray]:
    """(..., 16) endpose rows -> per-arm position, rot6d and gripper, float32."""
    out = {}
    for n, arm in enumerate(ARMS):
        pose = endpose[..., 8 * n : 8 * n + 7]
        out[f"{arm}_ee_ori"] = rot6d(pose[..., 3:]).astype(np.float32)
        out[f"{arm}_ee_pos"] = pose[..., :3].astype(np.float32)
        out[f"{arm}_gripper_states"] = endpose[..., 8 * n + 7 : 8 * n + 8].astype(np.float32)
    return out


def prompt(arrays: Mapping[str, Any]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """A demonstration -> the prompt: `(obs, action)`, obs per key (P, ...) float32 (images
    (P, 3, 224, 224) in [0, 1]), action (P, CHUNK, 20)."""
    endpose = _endpose(arrays, 2)
    steps = len(endpose) - 1
    if steps < 1:
        raise DemonstrationError(f"a demonstration needs at least 2 frames, not {len(endpose)}")
    groups = max(1, -(-steps // CHUNK))
    keep = np.arange(0, groups * CHUNK, CHUNK)
    obs = {k: v[keep] for k, v in proprio(endpose[:-1]).items()}
    for cam in CAMERAS:
        frames = _frames(arrays, cam, 4)
        if len(frames) != len(endpose):
            raise DemonstrationError(f"{cam} has {len(frames)} frames; endpose has {len(endpose)}")
        stored = np.stack([resize(jpeg(frames[i])) for i in keep])
        obs[cam] = np.moveaxis(stored, -1, 1).astype(np.float32) / 255.0
    target = endpose[1:]
    parts = []
    for n in range(len(ARMS)):
        pose = target[:, 8 * n : 8 * n + 7]
        parts += [pose[:, :3], rot6d(pose[:, 3:]), target[:, 8 * n + 7 : 8 * n + 8]]
    action = np.concatenate(parts, axis=-1).astype(np.float32)
    action = np.concatenate(
        [action, np.zeros((groups * CHUNK - steps, action.shape[1]), np.float32)]
    )
    return obs, action.reshape(groups, CHUNK, -1)


def frame(observation: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """One benchmark observation -> one frame: images (224, 224, 3) uint8, proprioception float32."""
    out = proprio(_endpose(observation, 1))
    for cam in CAMERAS:
        out[cam] = resize(_frames(observation, cam, 3))
    return out


def action_row(action: np.ndarray) -> np.ndarray:
    """A predicted (20,) [pos 3, rot6d 6, gripper 1] x 2 -> the benchmark's (16,) float64 row."""
    a = np.asarray(action, dtype=np.float64)
    parts = []
    for n in range(len(ARMS)):
        arm = a[10 * n : 10 * n + 10]
        parts += [np.concatenate([arm[:3], quat_from_rot6d(arm[3:9])]).astype(np.float32),
                  np.asarray([np.clip(arm[9], 0.0, 1.0)], dtype=np.float32)]  # fmt: skip
    return np.concatenate(parts).astype(np.float64)

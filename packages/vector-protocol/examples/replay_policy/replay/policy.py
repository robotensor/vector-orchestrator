"""The replay policy: open-loop playback of the one demonstration.

It ignores every observation. The k-th call to `act` after `reset` returns the demonstration's
k-th action, and the last action once the demonstration has run out. The expert is
position-controlled, so its actions are the joint positions it reached: `qpos[1:]`. It is the floor a learned
policy has to beat, and the smallest thing that exercises the whole protocol.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


class ReplayPolicy:
    #: The demonstration's actions, `qpos[1:]`, are joint-position targets.
    action_type = "qpos"

    def __init__(self) -> None:
        self._actions: np.ndarray | None = None
        self._step = 0

    def set_demonstration(self, arrays: Mapping[str, np.ndarray], info: Mapping[str, Any]) -> None:
        if "qpos" not in arrays:
            raise ValueError(f"the demonstration has no 'qpos', only {sorted(arrays)}")
        qpos = np.array(arrays["qpos"], dtype=np.float64)  # a copy: arrays are read-only
        if qpos.ndim != 2 or qpos.shape[0] < 2:
            raise ValueError(
                f"'qpos' must be a (T, D) array of at least two frames, not {qpos.shape}"
            )
        self._actions = qpos[1:]
        self._step = 0

    def reset(self, seed: int) -> None:
        self._step = 0

    def act(self, observation: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        if self._actions is None:
            raise RuntimeError("act() before set_demonstration()")
        k = min(self._step, len(self._actions) - 1)
        self._step += 1
        return {"action": self._actions[k]}

    def close(self) -> None:
        self._actions = None

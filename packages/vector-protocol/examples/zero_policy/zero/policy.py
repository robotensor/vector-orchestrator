"""The zero policy: every action is zeros, as wide as the demonstration's joint positions.

It learns nothing from the demonstration but the width of an action. It is a baseline, and a check
that a benchmark copes with a policy that does nothing useful.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


class ZeroPolicy:
    #: Zeros in joint-position space, as wide as the demonstration's `qpos`.
    action_type = "qpos"

    def __init__(self) -> None:
        self._zero: np.ndarray | None = None

    def set_demonstration(self, arrays: Mapping[str, np.ndarray], info: Mapping[str, Any]) -> None:
        qpos = arrays.get("qpos")
        if qpos is None or np.ndim(qpos) != 2:
            raise ValueError("the demonstration needs a (T, D) 'qpos' array")
        self._zero = np.zeros(qpos.shape[1:], dtype=qpos.dtype)

    def reset(self, seed: int) -> None:
        pass

    def act(self, observation: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        if self._zero is None:
            raise RuntimeError("act() before set_demonstration()")
        return {"action": self._zero}

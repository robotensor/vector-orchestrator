"""`VectorPolicy`: a `vector_v1.1` weights file, served over the `vector-protocol` protocol.

    # policy.yaml
    api: 1
    policy: vector_runtime.policy:VectorPolicy
    kwargs: {weights: /abs/path/model.safetensors, device: "cuda:0"}

`reset(seed)` clears the episode and seeds the diffusion noise; `set_demonstration` encodes the
prompt once; `act` answers one observation with one (16,) `ee` row, predicting HORIZON actions
whenever the EXECUTED ones cached from the last prediction are used up.
"""

from __future__ import annotations

import logging
import os
import time
from collections import deque
from collections.abc import Mapping
from typing import Any

import numpy as np

from . import demo
from .net import CAMERAS, EXECUTED, HISTORY, MAX_PROMPT

log = logging.getLogger("vector_runtime.policy")


class VectorPolicy:
    action_type = "ee"

    def __init__(
        self, weights: str, device: str = "cuda:0", weights_sha256: str | None = None
    ) -> None:
        import torch

        from .net import load

        if not os.path.isabs(str(weights)):
            raise ValueError(f"weights must be an absolute path, not {weights!r}")
        started = time.monotonic()
        self.device = torch.device(device)
        self.net = load(weights, device=str(self.device), weights_sha256=weights_sha256)
        self._generator: Any = None
        self._memory: Any = None
        self._history: deque[dict[str, np.ndarray]] = deque(maxlen=HISTORY)
        self._chunk: deque[np.ndarray] = deque()
        self.predictions = 0
        self.load_seconds = time.monotonic() - started
        log.info("loaded %s on %s in %.1fs", weights, self.device, self.load_seconds)

    def reset(self, seed: int) -> None:
        import torch

        self._clear()
        self._generator = torch.Generator(device=self.device)
        self._generator.manual_seed(int(seed) % (1 << 64))

    def set_demonstration(self, arrays: Mapping[str, np.ndarray], info: Mapping[str, Any]) -> None:
        import torch

        dims = info.get("action_dims") if isinstance(info, Mapping) else None
        if isinstance(dims, Mapping) and dims.get("ee") not in (None, demo.ENDPOSE_DIM):
            raise ValueError(
                f"the robot takes {dims.get('ee')}-wide ee actions; this policy answers 16"
            )
        obs, action = demo.prompt(arrays)
        if len(action) > MAX_PROMPT:
            raise ValueError(
                f"the demonstration makes {len(action)} prompt frames; at most {MAX_PROMPT}"
            )
        with torch.inference_mode():
            self._memory = self.net.encode_prompt(self._tensors(obs), self._tensor(action))
        self._history.clear()
        self._chunk.clear()

    def act(self, observation: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        if self._generator is None:
            raise RuntimeError("act() before reset()")
        if self._memory is None:
            raise RuntimeError("act() before set_demonstration()")
        self._history.append(demo.frame(observation))
        if not self._chunk:
            self._chunk.extend(self.predict()[:EXECUTED])
        return {"action": demo.action_row(self._chunk.popleft())}

    def close(self) -> None:
        self._clear()
        self._generator = None

    def predict(self) -> np.ndarray:
        """One prediction on the history (the episode start padded with its first frame)."""
        import torch

        frames = list(self._history)
        frames = [frames[0]] * (HISTORY - len(frames)) + frames
        obs = {}
        for key in frames[0]:
            stack = np.stack([f[key] for f in frames])
            obs[key] = (
                np.moveaxis(stack, -1, 1).astype(np.float32) / 255.0 if key in CAMERAS else stack
            )
        with torch.inference_mode():
            chunk = self.net.predict(self._tensors(obs), self._memory, self._generator)
        self.predictions += 1
        return chunk.float().cpu().numpy()[0]

    def _tensor(self, array: np.ndarray) -> Any:
        import torch

        return torch.from_numpy(np.ascontiguousarray(array)).unsqueeze(0).to(self.device)

    def _tensors(self, arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
        return {k: self._tensor(v) for k, v in arrays.items()}

    def _clear(self) -> None:
        self._history.clear()
        self._chunk.clear()
        self._memory = None

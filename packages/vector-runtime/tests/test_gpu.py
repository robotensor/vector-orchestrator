"""Load and serve real weights. Needs torch, CUDA and `VECTOR_WEIGHTS` (a `vector_v1.1`
model.safetensors); `VECTOR_PROMPT` names a benchmark prompt.npz, else a synthetic demonstration is
used."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.gpu

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
if not torch.cuda.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)
WEIGHTS = Path(os.environ.get("VECTOR_WEIGHTS", "/nonexistent"))
if not WEIGHTS.is_file():
    pytest.skip("VECTOR_WEIGHTS names no model.safetensors", allow_module_level=True)

from vector_runtime import WEIGHTS_FILENAME  # noqa: E402
from vector_runtime.check import check  # noqa: E402
from vector_runtime.net import WeightsError, load  # noqa: E402
from vector_runtime.policy import VectorPolicy  # noqa: E402


@pytest.fixture(scope="module")
def weights():
    return WEIGHTS.parent


@pytest.fixture(scope="module")
def policy(weights):
    return VectorPolicy(weights=str(weights / WEIGHTS_FILENAME))


@pytest.fixture(scope="module")
def arrays():
    if os.environ.get("VECTOR_PROMPT"):
        with np.load(os.environ["VECTOR_PROMPT"], allow_pickle=False) as loaded:
            return {k: loaded[k] for k in loaded.files if k != "meta"}
    count, rng = 40, np.random.default_rng(0)
    t = np.linspace(0.0, 1.0, count)
    endpose = np.zeros((count, 16))
    for start, x in ((0, -0.4), (8, 0.4)):
        endpose[:, start] = x + 0.05 * np.sin(3 * t)
        endpose[:, start + 1] = -0.2 + 0.05 * t
        endpose[:, start + 2] = 0.98 - 0.1 * t
        endpose[:, start + 3] = np.cos(0.3 * t)
        endpose[:, start + 6] = np.sin(0.3 * t)
        endpose[:, start + 7] = 1.0 - t
    frames = {
        f"frames_{camera}": rng.integers(0, 256, (count, 240, 320, 3), dtype=np.uint8)
        for camera in ("head_camera", "left_camera", "right_camera")
    }
    return {**frames, "endpose": endpose}


def observation(arrays, row):
    return {k: v[row] for k, v in arrays.items() if k.startswith("frames_") or k == "endpose"}


def test_the_weights_pass_the_check():
    report = check(WEIGHTS)
    assert report.ok, report.errors
    assert report.tensor_count == 753


def test_the_policy_follows_the_protocol(policy, arrays):
    info = {"action_dims": {"ee": 16}}
    with pytest.raises(RuntimeError, match="before reset"):
        policy.act(observation(arrays, 0))
    policy.reset(0)
    with pytest.raises(RuntimeError, match="before set_demonstration"):
        policy.act(observation(arrays, 0))

    def episode(seed, steps=26):
        """Drive the policy as the harness does: one observation per call, every row executed."""
        policy.reset(seed)
        policy.set_demonstration(arrays, info)
        chunks, step = [], 0
        while step < steps:
            chunks.append(policy.act(observation(arrays, step))["action"])
            step += len(chunks[-1])
        return np.concatenate(chunks)[:steps], [len(c) for c in chunks]

    before = policy.predictions
    first, calls = episode(0)
    # A prediction's 12 actions come back as 11 at once and then 1, so the harness observes only
    # the two consecutive frames the next prediction reads.
    assert calls == [11, 1, 11, 1, 11]
    assert policy.predictions - before == 3
    assert first.shape == (26, 16) and first.dtype == np.float64
    assert np.all(np.isfinite(first))
    assert np.all((first[:, [7, 15]] >= 0) & (first[:, [7, 15]] <= 1))
    np.testing.assert_array_equal(episode(0)[0], first)  # the seed fixes the noise
    assert not np.array_equal(episode(1)[0], first)

    policy.set_demonstration(arrays, info)
    policy.reset(0)  # reset forgets the prompt
    with pytest.raises(RuntimeError, match="before set_demonstration"):
        policy.act(observation(arrays, 0))
    policy.close()  # the network stays loaded
    assert next(policy.net.parameters()).device.type == "cuda"


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("norm.action.scale", 0.0, "zero scale"),
        ("norm.left_ee_pos.offset", float("nan"), "non-finite"),
        ("unet.final_conv.1.bias", float("inf"), "non-finite"),
    ],
)
def test_the_loader_refuses_bad_values(weights, tmp_path, key, value, message):
    from safetensors.torch import load_file, save_file

    state = load_file(str(weights / WEIGHTS_FILENAME))
    state[key] = state[key].clone()
    state[key][0] = value
    bad = tmp_path / WEIGHTS_FILENAME
    save_file(state, str(bad), metadata={"format": "pt"})
    assert check(bad).ok  # a header cannot show a value
    with pytest.raises(WeightsError, match=message):
        load(bad, device="cuda:0")


def test_the_loader_refuses_a_file_that_fails_the_check(weights, tmp_path):
    from safetensors.torch import load_file, save_file

    state = load_file(str(weights / WEIGHTS_FILENAME))
    del state["norm.action.scale"]
    bad = tmp_path / WEIGHTS_FILENAME
    save_file(state, str(bad), metadata={"format": "pt"})
    with pytest.raises(WeightsError, match="is missing"):
        load(bad, device="cuda:0")
    with pytest.raises(WeightsError, match="sha256"):
        load(weights, device="cuda:0", weights_sha256="0" * 64)

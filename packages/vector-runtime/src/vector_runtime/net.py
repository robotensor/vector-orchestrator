"""The `vector_v1.1` network: a demonstration-prompted diffusion policy.

Observations are 9 tokens a frame: three CLIP ViT-B/16 camera embeddings (the CLS token) and six
proprioception projections. The demonstration is a prompt of one frame per 15 steps, each frame's
9 tokens and its 15 actions pooled into one token. The current 2 frames cross-attend the prompt in
a 6-layer decoder; both, flattened, condition a 1-D U-Net that denoises 16 actions in 16 DDIM steps.

Adapted from `behavior_prompting` (MIT, Copyright (c) 2026 Austin Patel); see
THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn

CAMERAS = ("head_rgb", "left_wrist_rgb", "right_wrist_rgb")
#: Proprioception, in token order, and its width.
PROPRIO = {
    "left_ee_ori": 6,
    "left_ee_pos": 3,
    "left_gripper_states": 1,
    "right_ee_ori": 6,
    "right_ee_pos": 3,
    "right_gripper_states": 1,
}
ACTION_DIM = 20
HORIZON = 16
#: DDIM denoising steps.
STEPS = 16
EXECUTED = 12
HISTORY = 2
CHUNK = 15
MAX_PROMPT = math.ceil(1000 / CHUNK)
WIDTH = 768
SINKS = 4
IMAGE = 224
CROP = int(IMAGE * 0.95)
VIT = "vit_base_patch16_clip_224.openai"
TOKENS = len(CAMERAS) + len(PROPRIO)


class WeightsError(ValueError):
    """The weights cannot be loaded into the network."""


class Affine(nn.Module):
    """x * scale + offset over the last dimension."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.register_buffer("scale", torch.ones(dim))
        self.register_buffer("offset", torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x.reshape(-1, self.scale.shape[0]) * self.scale + self.offset).reshape(x.shape)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return ((x.reshape(-1, self.scale.shape[0]) - self.offset) / self.scale).reshape(x.shape)


def normalizer() -> nn.ModuleDict:
    """Per-key affine normalization of the action and proprioception; images are used as is."""
    return nn.ModuleDict(
        {"action": Affine(ACTION_DIM), **{k: Affine(d) for k, d in PROPRIO.items()}}
    )


class DecoderLayer(nn.TransformerDecoderLayer):
    """Pre-norm decoder layer; cross-attention is called with `need_weights=True`."""

    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = x + self._sa_block(self.norm1(x), None, None, False)
        y, _ = self.multihead_attn(
            self.norm2(x), memory, memory, need_weights=True, average_attn_weights=False
        )
        x = x + self.dropout2(y)
        return x + self._ff_block(self.norm3(x))


class Conv1dBlock(nn.Module):
    def __init__(self, cin: int, cout: int, kernel: int, groups: int = 8) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(cin, cout, kernel, padding=kernel // 2), nn.GroupNorm(groups, cout), nn.Mish()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResBlock(nn.Module):
    """Two conv blocks with FiLM (scale and bias) from the condition between them."""

    def __init__(self, cin: int, cout: int, cond: int, kernel: int = 5) -> None:
        super().__init__()
        self.out = cout
        self.blocks = nn.ModuleList(
            [Conv1dBlock(cin, cout, kernel), Conv1dBlock(cout, cout, kernel)]
        )
        self.cond_encoder = nn.Sequential(
            nn.Mish(), nn.Linear(cond, 2 * cout), nn.Unflatten(-1, (2 * cout, 1))
        )
        self.residual_conv = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        out = self.blocks[0](x)
        film = self.cond_encoder(cond).reshape(cond.shape[0], 2, self.out, 1)
        out = film[:, 0, ...] * out + film[:, 1, ...]
        return self.blocks[1](out) + self.residual_conv(x)


class StepEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freq = torch.exp(torch.arange(half, device=t.device) * -(math.log(10000) / (half - 1)))
        emb = t[:, None] * freq[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class Unet(nn.Module):
    """1-D conditional U-Net over the action horizon (256-512-1024 channels)."""

    def __init__(
        self, cond: int, dims: tuple[int, ...] = (256, 512, 1024), step: int = 128
    ) -> None:
        super().__init__()
        self.diffusion_step_encoder = nn.Sequential(
            StepEmbedding(step), nn.Linear(step, step * 4), nn.Mish(), nn.Linear(step * 4, step)
        )
        cond += step
        pairs = list(zip((ACTION_DIM, *dims[:-1]), dims, strict=True))
        self.down_modules = nn.ModuleList(
            nn.ModuleList(
                [
                    ResBlock(cin, cout, cond),
                    ResBlock(cout, cout, cond),
                    nn.Conv1d(cout, cout, 3, 2, 1) if i < len(pairs) - 1 else nn.Identity(),
                ]
            )
            for i, (cin, cout) in enumerate(pairs)
        )
        self.mid_modules = nn.ModuleList([ResBlock(dims[-1], dims[-1], cond) for _ in range(2)])
        self.up_modules = nn.ModuleList(
            nn.ModuleList(
                [
                    ResBlock(cout * 2, cin, cond),
                    ResBlock(cin, cin, cond),
                    nn.ConvTranspose1d(cin, cin, 4, 2, 1),
                ]
            )
            for cin, cout in reversed(pairs[1:])
        )
        self.final_conv = nn.Sequential(
            Conv1dBlock(dims[0], dims[0], 5), nn.Conv1d(dims[0], ACTION_DIM, 1)
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        t = t[None].to(x.device).expand(x.shape[0])
        cond = torch.cat([self.diffusion_step_encoder(t), cond], dim=-1)
        skips = []
        for res, res2, down in self.down_modules:
            x = res2(res(x, cond), cond)
            skips.append(x)
            x = down(x)
        for mid in self.mid_modules:
            x = mid(x, cond)
        for res, res2, up in self.up_modules:
            x = up(res2(res(torch.cat((x, skips.pop()), dim=1), cond), cond))
        return self.final_conv(x).transpose(1, 2)


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        import timm
        import torchvision
        from diffusers import DDIMScheduler
        from timm.layers import AttentionPoolLatent, Mlp

        # The ViT is built empty; the loaded weights fill it. Its pixel normalization is CLIP's.
        self.vision = nn.ModuleDict(
            {
                cam: timm.create_model(VIT, pretrained=False, global_pool="", num_classes=0)
                for cam in CAMERAS
            }
        )
        pixel = timm.data.resolve_data_config(self.vision[CAMERAS[0]].pretrained_cfg)
        self.crop = nn.Sequential(
            torchvision.transforms.CenterCrop(size=CROP),
            torchvision.transforms.Resize(size=IMAGE, antialias=True),
        )
        self.pixel_norm = torchvision.transforms.Normalize(mean=pixel["mean"], std=pixel["std"])
        self.proprio = nn.ModuleDict({k: nn.Linear(d, WIDTH) for k, d in PROPRIO.items()})
        self.action_proj = Mlp(in_features=ACTION_DIM * CHUNK, out_features=WIDTH)
        self.prompt_pool = AttentionPoolLatent(
            in_features=WIDTH, num_heads=8, mlp_ratio=4.0, pool_type="token"
        )
        self.prompt_pos = nn.Parameter(torch.zeros(1, MAX_PROMPT, WIDTH))
        self.obs_pos = nn.Parameter(torch.zeros(1, HISTORY * TOKENS, WIDTH))
        self.sinks = nn.Parameter(torch.zeros(1, SINKS, WIDTH))
        layer = dict(
            d_model=WIDTH,
            nhead=8,
            dim_feedforward=4 * WIDTH,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.ModuleDict(
            {"layers": nn.ModuleList(DecoderLayer(**layer) for _ in range(6))}
        )
        self.unet = Unet(cond=2 * HISTORY * TOKENS * WIDTH)
        self.norm = normalizer()
        self.scheduler = DDIMScheduler(
            num_train_timesteps=50,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            set_alpha_to_one=True,
            steps_offset=0,
            prediction_type="epsilon",
        )

    def tokens(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """(B, T, ...) per key -> (B, T, 9, 768): the cameras, then proprioception."""
        out = []
        for cam in CAMERAS:
            img = obs[cam]
            b, t = img.shape[:2]
            img = self.pixel_norm(self.crop(img.reshape(b * t, *img.shape[2:])))
            out.append(self.vision[cam](img)[:, [0], :].reshape(b, -1, WIDTH))
        for key in PROPRIO:
            x = obs[key]
            out.append(self.proprio[key](x.reshape(*x.shape[:2], -1)))
        return torch.stack(out, dim=2)

    def encode_prompt(self, obs: dict[str, torch.Tensor], action: torch.Tensor) -> torch.Tensor:
        """A prompt of P frames -> the decoder's memory, (1, SINKS + P, 768)."""
        obs = {k: self.norm[k](v) if k in PROPRIO else v for k, v in obs.items()}
        tok = self.tokens(obs)
        b, p = tok.shape[:2]
        act = self.action_proj(self.norm["action"](action).reshape(b, p, -1)).unsqueeze(2)
        pooled = self.prompt_pool(torch.cat([tok, act], dim=2).reshape(b * p, TOKENS + 1, WIDTH))
        pooled = pooled.reshape(b, p, WIDTH) + self.prompt_pos[:, :p, :]
        return torch.cat([self.sinks.expand(b, -1, -1), pooled], dim=1)

    def predict(
        self,
        obs: dict[str, torch.Tensor],
        memory: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """The last HISTORY frames and the prompt memory -> (B, HORIZON, ACTION_DIM) actions."""
        obs = {k: self.norm[k](v) if k in PROPRIO else v for k, v in obs.items()}
        now = self.tokens(obs)
        b = now.shape[0]
        now = now.reshape(b, HISTORY * TOKENS, WIDTH)
        x = now + self.obs_pos
        memory = memory.expand(b, -1, -1)
        for layer in self.decoder["layers"]:
            x = layer(x, memory)
        cond = torch.cat([now, x], dim=1).reshape(b, -1)
        traj = torch.randn(
            (b, HORIZON, ACTION_DIM), dtype=cond.dtype, device=cond.device, generator=generator
        )
        self.scheduler.set_timesteps(STEPS)
        for t in self.scheduler.timesteps:
            traj = self.scheduler.step(
                self.unet(traj, t, cond), t, traj, generator=generator
            ).prev_sample
        return self.norm["action"].inverse(traj)


def load(weights: str | Path, device: str = "cuda:0", weights_sha256: str | None = None) -> Net:
    """The network with the weights at `weights` (a file or its directory), in eval mode on
    `device`. The file is checked against the architecture's tensors first."""
    from safetensors.torch import load_file

    from .check import check

    report = check(weights, compute_sha256=bool(weights_sha256))
    if not report.ok:
        raise WeightsError(f"{report.path}: " + "; ".join(report.errors))
    if weights_sha256 and report.weights_sha256 != weights_sha256.lower():
        raise WeightsError(
            f"{report.path} has sha256 {report.weights_sha256}, expected {weights_sha256}"
        )
    state = load_file(report.path, device=str(device))
    for problem, bad in (
        ("non-finite values", [k for k, v in state.items() if not bool(torch.isfinite(v).all())]),
        (
            "a zero scale",
            [k for k, v in state.items() if k.endswith(".scale") and bool((v == 0).any())],
        ),
    ):
        if bad:
            raise WeightsError(f"{problem} in {bad[:10]}")
    with torch.device("cpu"):
        net = Net()
    net.load_state_dict(state, strict=True)
    net.to(device).eval()
    del state
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
    return net

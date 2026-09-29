"""Synthetic safetensors files, written by hand: the host tests need neither torch nor safetensors."""

from __future__ import annotations

import json
import struct
from typing import Any

from vector_runtime.header import DTYPE_SIZES

#: A small architecture: a weight, a bias, a zero-size tensor and a normalizer scale.
TENSORS = {
    "empty": ("F32", [0]),
    "unet.weight": ("F32", [4, 3]),
    "unet.bias": ("F32", [4]),
    "norm.action.scale": ("F32", [20]),
}


def numel(shape: list[int]) -> int:
    n = 1
    for dim in shape:
        n *= dim
    return n


def safetensors_bytes(
    tensors: dict[str, tuple[str, list[int]]] | None = None,
    *,
    metadata: dict[str, str] | None = None,
    header: bytes | None = None,
    extra: bytes = b"",
    length: int | None = None,
) -> bytes:
    """A safetensors file holding zeros for `tensors`; `header`, `extra` bytes after the data
    and a false `length` field make it malformed on purpose."""
    tensors = TENSORS if tensors is None else tensors
    document: dict[str, Any] = {}
    if metadata is not None:
        document["__metadata__"] = metadata
    offset = 0
    for name, (dtype, shape) in tensors.items():
        size = numel(shape) * DTYPE_SIZES[dtype]
        document[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + size]}
        offset += size
    raw = json.dumps(document).encode() if header is None else header
    raw += b" " * (-len(raw) % 8)
    prefix = struct.pack("<Q", len(raw) if length is None else length)
    return prefix + raw + bytes(offset) + extra


def manifest(tensors: dict[str, tuple[str, list[int]]] | None = None) -> dict[str, dict[str, Any]]:
    """`tensors` as the architecture's manifest holds them."""
    tensors = TENSORS if tensors is None else tensors
    return {name: {"shape": shape, "dtype": dtype} for name, (dtype, shape) in tensors.items()}

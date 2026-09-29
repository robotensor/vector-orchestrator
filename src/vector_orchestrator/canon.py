"""Canonical JSON and hashing.

Every index line is a record's canonical JSON, and ids and fingerprints are hashes of it, so the
encoding is part of the contract: sorted keys, no whitespace, ASCII only, no NaN. Numbers are
formatted as Python formats them (`3.0` stays `3.0`, where `JSON.stringify` and RFC 8785 both give
`3`), so a reader that hashes a record compares against the bytes as published, never against a
re-encoding in another language.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def canonical_json(obj: Any) -> str:
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )


def sha256_hex(data: bytes | str) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def canonical_sha256(obj: Any) -> str:
    return sha256_hex(canonical_json(obj))

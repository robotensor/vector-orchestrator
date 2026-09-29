"""`vector-orchestrator store init` and `check`, through the console script."""

from __future__ import annotations

import json
import subprocess
import sysconfig
from pathlib import Path

from store_helpers import check_store


def cli(
    *args: str, cwd: Path | None = None, env: dict | None = None
) -> subprocess.CompletedProcess:
    script = Path(sysconfig.get_path("scripts")) / "vector-orchestrator"
    return subprocess.run([str(script), *args], capture_output=True, text=True, cwd=cwd, env=env)


def test_store_init_writes_the_manifest_and_an_empty_head(spec, tmp_path):
    root = tmp_path / "store"
    init = cli("store", "init", "--root", str(root), cwd=tmp_path)
    assert init.returncode == 0, init.stderr
    manifest = json.loads(init.stdout)
    assert manifest == json.loads((root / "manifest.json").read_text())
    assert json.loads((root / "head.json").read_text()) == {"seq": 0, "block": 0, "king": None}
    assert check_store(root, spec).ok
    again = cli("store", "init", "--root", str(root), cwd=tmp_path)
    assert again.returncode == 0 and json.loads(again.stdout) == manifest
    assert sorted(p.name for p in tmp_path.iterdir()) == ["store"], "init wrote beside the store"


def test_check_says_what_is_missing_without_a_checkout(tmp_path):
    done = cli("check", env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path)})
    assert done.returncode == 1
    report = json.loads(done.stdout)
    assert report["ok"] is False and "ROBOTWIN_BENCH_ROOT" in report["error"]

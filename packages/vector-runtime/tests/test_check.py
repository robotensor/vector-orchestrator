"""`vector-runtime check`: the architecture's tensors exactly, a sane file, and its sha256. No torch."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys

import pytest

from st_testing import TENSORS, manifest, safetensors_bytes
from vector_runtime import WEIGHTS_FILENAME
from vector_runtime import check as check_module
from vector_runtime.check import check, tensors
from vector_runtime.cli import main


@pytest.fixture(autouse=True)
def small(request, monkeypatch):
    """The small architecture of `st_testing`, except for the test of the packaged one."""
    if "packaged" not in request.node.name:
        monkeypatch.setattr(check_module, "tensors", lambda architecture=None: manifest())


def weights(tmp_path, data: bytes, name: str = WEIGHTS_FILENAME):
    directory = tmp_path / "submission"
    directory.mkdir(exist_ok=True)
    (directory / name).write_bytes(data)
    return directory


def test_a_matching_file_passes_with_its_sha256(tmp_path):
    data = safetensors_bytes(metadata={"format": "pt"})
    report = check(weights(tmp_path, data))
    assert report.ok, report.errors
    assert report.weights_sha256 == hashlib.sha256(data).hexdigest()
    assert report.tensor_count == len(TENSORS)
    assert report.param_count == 36
    assert report.file_bytes == len(data)
    assert report.metadata == {"format": "pt"}
    assert report.to_dict()["ok"] is True


def test_a_file_path_works_as_well_as_its_directory(tmp_path):
    directory = weights(tmp_path, safetensors_bytes())
    assert check(directory / WEIGHTS_FILENAME).ok


def test_a_missing_tensor_fails(tmp_path):
    tensors = {k: v for k, v in TENSORS.items() if k != "norm.action.scale"}
    report = check(weights(tmp_path, safetensors_bytes(tensors)))
    assert report.errors == ["tensor norm.action.scale is missing"]


def test_an_extra_tensor_fails(tmp_path):
    # an extra tensor fails before a byte of data is read
    tensors = {**TENSORS, "norm.evil.scale": ("F32", [3])}
    report = check(weights(tmp_path, safetensors_bytes(tensors)))
    assert report.errors == ["tensor norm.evil.scale is not part of the architecture"]


def test_a_wrong_shape_fails(tmp_path):
    tensors = {**TENSORS, "unet.weight": ("F32", [3, 4])}
    report = check(weights(tmp_path, safetensors_bytes(tensors)))
    assert report.errors == ["tensor unet.weight: shape [3, 4] != [4, 3]"]


def test_a_wrong_dtype_fails(tmp_path):
    tensors = {**TENSORS, "unet.bias": ("BF16", [4])}
    report = check(weights(tmp_path, safetensors_bytes(tensors)))
    assert report.errors == ["tensor unet.bias: dtype BF16 != F32"]


def test_a_huge_header_fails_without_being_read(tmp_path):
    report = check(weights(tmp_path, safetensors_bytes(length=1 << 40)))
    assert not report.ok
    assert "at most" in report.errors[0]


def test_a_header_over_the_cap_fails(tmp_path):
    report = check(weights(tmp_path, safetensors_bytes()), max_header_bytes=64)
    assert not report.ok
    assert "at most 64" in report.errors[0]


def test_an_oversized_file_fails_before_its_header_is_read(tmp_path):
    data = safetensors_bytes()
    report = check(weights(tmp_path, data), max_file_bytes=len(data) - 1)
    assert report.errors == [f"the file is {len(data)} bytes; at most {len(data) - 1}"]
    assert report.weights_sha256 is None


def test_garbage_fails(tmp_path):
    report = check(weights(tmp_path, b"\x80\x04\x95 not safetensors at all"))
    assert not report.ok
    assert report.errors[0].startswith("not a valid safetensors file")


def test_appended_bytes_fail(tmp_path):
    report = check(weights(tmp_path, safetensors_bytes(extra=b"PK\x03\x04")))
    assert not report.ok


def test_a_directory_without_the_file_fails(tmp_path):
    directory = weights(tmp_path, safetensors_bytes(), name="pytorch_model.bin")
    report = check(directory)
    assert not report.ok
    assert report.other_files == ["pytorch_model.bin"]


def test_other_files_are_a_warning(tmp_path):
    directory = weights(tmp_path, safetensors_bytes())
    (directory / "README.md").write_text("hello")
    report = check(directory)
    assert report.ok
    assert report.other_files == ["README.md"]
    assert report.warnings


def test_a_symbolic_link_is_followed(tmp_path):
    # a Hugging Face snapshot directory holds links into its blob store
    blob = tmp_path / "blob"
    blob.write_bytes(safetensors_bytes())
    directory = tmp_path / "snapshot"
    directory.mkdir()
    (directory / WEIGHTS_FILENAME).symlink_to(blob)
    assert check(directory).ok


def test_the_cli_prints_a_report_and_exits_by_it(tmp_path, capsys):
    good = weights(tmp_path, safetensors_bytes())
    assert main(["check", "--weights", str(good)]) == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True
    (good / WEIGHTS_FILENAME).write_bytes(safetensors_bytes({"unet.bias": ("F32", [4])}))
    assert main(["check", "--weights", str(good)]) == 1
    assert json.loads(capsys.readouterr().out)["ok"] is False


def test_the_packaged_manifest_is_the_architecture():
    packaged = tensors()
    assert len(packaged) == 753
    assert sum(_numel(t["shape"]) for t in packaged.values()) == 780_979_360
    assert {t["dtype"] for t in packaged.values()} == {"F32"}
    assert {k.split(".")[0] for k in packaged} == {
        "vision",
        "proprio",
        "action_proj",
        "prompt_pool",
        "prompt_pos",
        "obs_pos",
        "sinks",
        "decoder",
        "unet",
        "norm",
    }


def test_the_check_imports_no_torch_or_numpy():
    code = (
        "import sys, vector_runtime.check, vector_runtime.cli; "
        "print(sorted(m for m in sys.modules if m.split('.')[0] in "
        "('torch', 'numpy', 'safetensors')))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"


def _numel(shape):
    n = 1
    for dim in shape:
        n *= dim
    return n

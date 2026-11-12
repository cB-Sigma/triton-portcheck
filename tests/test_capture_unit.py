from pathlib import Path

import pytest

from triton_portcheck import capture
from triton_portcheck.manifest import dumps_toml, load_manifest

ALIGNED = [["tt.divisibility", 16]]


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        (["pytest", "-x", "tests"], ["-m", "pytest", "-x", "tests"]),
        (["python", "-m", "pytest", "tests"], ["-m", "pytest", "tests"]),
        (["/usr/bin/python3.12", "train.py", "--fast"], ["train.py", "--fast"]),
        (["python3", "-c", "print(1)"], ["-c", "print(1)"]),
        (["bench.py", "1"], ["bench.py", "1"]),
    ],
)
def test_python_argv(command, expected):
    assert capture.python_argv(command) == expected


@pytest.mark.parametrize("command", [[], ["python"], ["python", "-m"], ["python", "-O", "x.py"], ["make", "test"]])
def test_python_argv_rejects(command):
    with pytest.raises(capture.CaptureError):
        capture.python_argv(command)


def record(**over):
    base = {
        "module": "kernels",
        "qualname": "add",
        "file": "/proj/kernels.py",
        "line": 5,
        "arg_names": ["x", "y", "n", "BLOCK", "cfg"],
        "signature": {"x": "*fp32", "y": "*fp32", "n": "i32", "BLOCK": "constexpr", "cfg": ["i32", "constexpr"]},
        "constant_keys": [[3], [4, 1]],
        "constant_vals": [128, "fp16"],
        "attrs_keys": [[0], [1], [2], [3]],
        "attrs_vals": [ALIGNED, ALIGNED, [], []],
        "options": {},
    }
    return {**base, **over}


def test_records_become_specs_with_named_constant_paths():
    manifest, skipped = capture.records_to_manifest([record()], Path("/proj"))
    assert skipped == []
    (spec,) = manifest.kernels
    assert spec.name == "add"
    assert spec.target == "kernels.py:add"
    assert spec.signature == {"x": "*fp32", "y": "*fp32", "n": "i32", "cfg": ["i32", "constexpr"]}
    assert spec.constants == {"BLOCK": 128, "cfg.1": "fp16"}
    assert spec.attrs is None  # all pointers aligned is the default, so it is not spelled out


def test_duplicates_collapse_and_distinct_specs_are_numbered():
    records = [
        record(),
        record(),
        record(constant_vals=[64, "fp16"]),
        record(attrs_vals=[ALIGNED, [], [], []]),
        record(options={"num_warps": 8}),
    ]
    manifest, _ = capture.records_to_manifest(records, Path("/proj"))
    assert [k.name for k in manifest.kernels] == ["add#1", "add#2", "add#3", "add#4"]
    assert manifest.kernels[1].constants["BLOCK"] == 64
    assert manifest.kernels[2].attrs == {"x": ALIGNED}
    assert manifest.kernels[3].options == {"num_warps": 8}


def test_locator_prefers_module_for_installed_packages_and_path_otherwise():
    installed = record(module="pkg.ops", file="/venv/lib/python3.12/site-packages/pkg/ops.py")
    outside = record(file="/other/place/kernels.py", qualname="Ops.add")
    script = record(module="__main__", file="/proj/tools/run.py")
    manifest, _ = capture.records_to_manifest([installed, outside, script], Path("/proj"))
    assert [k.target for k in manifest.kernels] == [
        "pkg.ops:add",
        "/other/place/kernels.py:Ops.add",
        "tools/run.py:add",
    ]


def test_nested_kernels_are_skipped_with_a_note():
    manifest, skipped = capture.records_to_manifest([record(qualname="make.<locals>.inner"), record()], Path("/proj"))
    assert skipped == ["kernels:make.<locals>.inner"]
    assert len(manifest.kernels) == 1


def test_captured_manifest_survives_toml(tmp_path):
    manifest, _ = capture.records_to_manifest([record(), record(attrs_vals=[ALIGNED, [], [], []])], tmp_path)
    path = tmp_path / "portcheck.toml"
    path.write_text(dumps_toml(manifest))
    again = load_manifest(path)
    assert [k.to_dict() for k in again.kernels] == [k.to_dict() for k in manifest.kernels]


def test_read_records_ignores_missing_file_and_blank_lines(tmp_path):
    assert capture.read_records(tmp_path / "none.jsonl") == []
    path = tmp_path / "r.jsonl"
    path.write_text('{"a": 1}\n\n{"a": 2}\n')
    assert capture.read_records(path) == [{"a": 1}, {"a": 2}]

import json
import tomllib
from pathlib import Path

import pytest

from triton_portcheck.manifest import (
    ManifestError,
    dump_manifest,
    dumps_toml,
    find_manifest,
    load_manifest,
    parse_manifest,
)

BASIC = """
[defaults]
targets = ["cuda:80", "hip:gfx942"]
triton = ["current", "3.8.0"]
timeout = 60
pythonpath = ["src"]

[[kernel]]
target = "pkg.mod:add"
signature = { x = "*fp32", n = "i32" }
constants = { BLOCK = 64 }
options = { num_warps = 8 }
xfail = ["hip:*"]

[[kernel]]
name = "dot"
target = "k.py:dot"
signature = { a = "*fp16" }
constants = { M = 16 }
  [[kernel.variant]]
  name = "fp16"
  [[kernel.variant]]
  name = "int8"
  signature = { a = "*i8" }
  constants = { M = 32 }
  targets = ["cuda:80"]
"""


def write(tmp_path: Path, text: str, name: str = "portcheck.toml") -> Path:
    path = tmp_path / name
    path.write_text(text)
    return path


def test_load_basic(tmp_path):
    manifest = load_manifest(write(tmp_path, BASIC))
    assert manifest.targets == ["cuda:80", "hip:gfx942"]
    assert manifest.triton == ["current", "3.8.0"]
    assert manifest.timeout == 60
    assert manifest.base_dir == tmp_path.resolve()
    assert [k.name for k in manifest.kernels] == ["add", "dot[fp16]", "dot[int8]"]
    add = manifest.kernels[0]
    assert add.signature == {"x": "*fp32", "n": "i32"}
    assert add.constants == {"BLOCK": 64}
    assert add.options == {"num_warps": 8}
    assert add.xfail == ["hip:*"]
    assert add.align is True


def test_variants_merge_over_base(tmp_path):
    manifest = load_manifest(write(tmp_path, BASIC))
    fp16, int8 = manifest.kernels[1], manifest.kernels[2]
    assert fp16.signature == {"a": "*fp16"}
    assert fp16.constants == {"M": 16}
    assert int8.signature == {"a": "*i8"}
    assert int8.constants == {"M": 32}
    assert int8.targets == ["cuda:80"]
    assert fp16.targets is None


def test_child_request_shape(tmp_path):
    spec = load_manifest(write(tmp_path, BASIC)).kernels[0]
    request = spec.child_request()
    assert request["id"] == "add"
    assert request["target"] == "pkg.mod:add"
    assert "attrs" not in request


@pytest.mark.parametrize(
    ("text", "needle"),
    [
        ("", "no [[kernel]]"),
        ("[[kernel]]\nname='x'\n", "'target' must be"),
        ("[[kernel]]\ntarget='nocolon'\n", "'target' must be"),
        ("[[kernel]]\ntarget='m:f'\nsignatur={}\n", "unknown key(s) signatur"),
        ("[[kernel]]\ntarget='m:f'\n[[kernel]]\ntarget='m:f'\n", "duplicate kernel name 'f'"),
        ("[[kernel]]\ntarget='m:f'\ntargets=['cuda']\n", "invalid target"),
        ("[[kernel]]\ntarget='m:f'\nsignature='x'\n", "must be a table"),
        ("[[kernel]]\ntarget='m:f'\ntimeout=-1\n", "positive number"),
        ("[defaults]\ntarget=['cuda:80']\n[[kernel]]\ntarget='m:f'\n", "unknown key(s) target"),
        ("[[kernel]]\ntarget='m:f'\nvariant=[]\n", "non-empty array"),
    ],
)
def test_invalid_manifests_say_what_is_wrong(tmp_path, text, needle):
    with pytest.raises(ManifestError) as info:
        load_manifest(write(tmp_path, text))
    assert needle in str(info.value)


def test_toml_syntax_error_names_file(tmp_path):
    with pytest.raises(ManifestError) as info:
        load_manifest(write(tmp_path, "[[kernel"))
    assert "portcheck.toml" in str(info.value)


def test_missing_and_unsupported_files(tmp_path):
    with pytest.raises(ManifestError, match="not found"):
        load_manifest(tmp_path / "nope.toml")
    with pytest.raises(ManifestError, match="unsupported manifest type"):
        load_manifest(write(tmp_path, "x", "portcheck.ini"))


def test_toml_round_trip(tmp_path):
    manifest = load_manifest(write(tmp_path, BASIC))
    manifest.kernels[0].attrs = {"x": [["tt.divisibility", 16]]}
    text = dumps_toml(manifest, header="two\nlines")
    assert text.startswith("# two\n# lines\n")
    tomllib.loads(text)
    again = load_manifest(write(tmp_path, text, "again.toml"))
    assert [k.to_dict() for k in again.kernels] == [k.to_dict() for k in manifest.kernels]
    assert again.targets == manifest.targets


def test_toml_quotes_awkward_keys_and_rejects_null(tmp_path):
    manifest = parse_manifest({"kernel": [{"target": "m:f", "constants": {"cfg.1": 3, "name": 'a"b'}}]}, tmp_path)
    data = tomllib.loads(dumps_toml(manifest))
    assert data["kernel"][0]["constants"] == {"cfg.1": 3, "name": 'a"b'}
    manifest.kernels[0].constants["none"] = None
    with pytest.raises(ManifestError, match="null"):
        dumps_toml(manifest)


def test_json_round_trip(tmp_path):
    manifest = load_manifest(write(tmp_path, BASIC))
    manifest.kernels[0].constants["maybe"] = None
    dump_manifest(manifest, tmp_path / "m.json")
    again = load_manifest(tmp_path / "m.json")
    assert again.kernels[0].constants == {"BLOCK": 64, "maybe": None}
    assert json.loads((tmp_path / "m.json").read_text())["defaults"]["timeout"] == 60


def test_yaml_round_trip(tmp_path):
    pytest.importorskip("yaml")
    manifest = load_manifest(write(tmp_path, BASIC))
    dump_manifest(manifest, tmp_path / "m.yaml", header="hello")
    assert (tmp_path / "m.yaml").read_text().startswith("# hello\n")
    again = load_manifest(tmp_path / "m.yaml")
    assert [k.name for k in again.kernels] == ["add", "dot[fp16]", "dot[int8]"]


def test_find_manifest(tmp_path):
    assert find_manifest(tmp_path) is None
    write(tmp_path, BASIC)
    assert find_manifest(tmp_path) == tmp_path / "portcheck.toml"


def test_bundled_demo_manifest_is_valid():
    from triton_portcheck.cli import DEMO_MANIFEST

    manifest = load_manifest(DEMO_MANIFEST)
    assert len(manifest.kernels) == 9
    assert "hip:gfx942" in manifest.targets

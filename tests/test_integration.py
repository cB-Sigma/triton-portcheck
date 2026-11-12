"""Integration tests: really compile kernels with the installed Triton, on CPU, for CUDA and HIP targets.

Skipped when no interpreter with Triton is available.  The divergences asserted
here were observed with Triton 3.9.0 (and 3.8.0 for the cross-version test).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import ROOT
from triton_portcheck import baseline, capture, cli, diffing, envs, runner
from triton_portcheck import results as R
from triton_portcheck.manifest import load_manifest, parse_manifest
from triton_portcheck.report import render_terminal

pytestmark = pytest.mark.compile

CUDA = ["cuda:75", "cuda:80", "cuda:90", "cuda:100", "cuda:120"]
HIP = ["hip:gfx942", "hip:gfx950", "hip:gfx1100", "hip:gfx1250"]
EXAMPLES = ROOT / "examples"


def run_demo(versions: list[str], targets: list[str] | None = None, **settings: object) -> R.Report:
    manifest = load_manifest(cli.DEMO_MANIFEST)
    targets = targets or manifest.targets
    environments = runner.resolve_envs(versions, targets, create=False)
    return runner.run_matrix(manifest, targets, environments, runner.RunSettings(jobs=4, timeout=180, **settings))


@pytest.fixture(scope="module")
def demo(triton_python: str) -> R.Report:
    os.environ["TRITON_PORTCHECK_PYTHON"] = triton_python
    return run_demo(["current"])


def statuses(report: R.Report, kernel: str, version: str | None = None) -> dict[str, str]:
    version = version or report.versions[0]
    return {c.target: c.status for c in report.cells if c.kernel == kernel and c.triton == version}


def test_matrix_is_complete(demo):
    assert demo.targets == CUDA + HIP
    assert len(demo.cells) == 9 * 9
    assert all(c.status != R.SKIPPED for c in demo.cells)
    assert set(statuses(demo, "axpy").values()) == {R.OK}


def test_libdevice_tgamma_compiles_on_cuda_and_not_on_hip(demo):
    result = statuses(demo, "gamma_ratio")
    assert {result[t] for t in CUDA} == {R.OK}
    assert {result[t] for t in HIP} == {R.FRONTEND}
    cell = demo.cell("gamma_ratio", "hip:gfx942", demo.versions[0])
    assert cell.signature == "libdevice-missing"
    assert cell.stage == "frontend"
    assert cell.issue_url.endswith("/10375")
    assert cell.file.endswith("demo/kernels.py")
    source = Path(cell.file).read_text().splitlines()
    assert "libdevice.tgamma" in source[cell.line - 1]


def test_fp8e4nv_cast_needs_newer_nvidia_or_hip(demo):
    result = statuses(demo, "quantize_fp8")
    assert result["cuda:75"] == result["cuda:80"] == R.FRONTEND
    assert {result[t] for t in ["cuda:90", "cuda:100", "cuda:120", *HIP]} == {R.OK}
    assert demo.cell("quantize_fp8", "cuda:80", demo.versions[0]).signature == "fp8-dtype-unsupported"


def test_descriptor_scatter_is_rejected_by_the_assembler_on_some_targets(demo):
    result = statuses(demo, "scatter_rows")
    assert result["cuda:90"] == result["cuda:120"] == R.ASSEMBLER
    assert {result[t] for t in ["cuda:75", "cuda:80", "cuda:100", *HIP]} == {R.OK}
    cell = demo.cell("scatter_rows", "cuda:120", demo.versions[0])
    assert cell.stage == "cubin"
    assert cell.signature == "ptxas-unsupported-feature"
    assert "sm_120a" in cell.message
    assert cell.issue_url.endswith("/11344")
    # the stages before the assembler ran and were timed
    assert {"frontend", "ttir", "ttgir", "llir", "ptx"} <= set(cell.stage_seconds)
    assert cell.fingerprint == demo.cell("scatter_rows", "cuda:90", demo.versions[0]).fingerprint


def test_int8_dot_fails_in_an_mlir_pass_on_cuda_75_only(demo):
    result = statuses(demo, "dot_small[int8]")
    assert result.pop("cuda:75") == R.MLIR
    assert set(result.values()) == {R.OK}
    cell = demo.cell("dot_small[int8]", "cuda:75", demo.versions[0])
    assert cell.stage == "ttgir"
    assert "pass " in cell.message
    assert set(statuses(demo, "dot_small[fp16]").values()) == {R.OK}


def test_fake_driver_makes_target_guards_resolve(demo, triton_python, tmp_path):
    """With the fake driver ``is_hip()`` is true for HIP cells; without it the guard silently takes the else branch."""
    assert set(statuses(demo, "gamma_ratio_guarded").values()) == {R.OK}
    guarded = demo.cell("gamma_ratio_guarded", "hip:gfx942", demo.versions[0])
    unguarded_cuda = demo.cell("gamma_ratio_guarded", "cuda:80", demo.versions[0])
    # the HIP cell took the lgamma branch, so its code differs from the tgamma branch compiled for CUDA
    assert guarded.artifact_sizes["ttir"] != unguarded_cuda.artifact_sizes["ttir"]

    script = tmp_path / "no_driver.py"
    script.write_text(
        "import sys, triton\n"
        "from triton.backends.compiler import GPUTarget\n"
        "from triton.compiler import ASTSource\n"
        f"sys.path.insert(0, {str(cli.DEMO_MANIFEST.parent)!r})\n"
        "import kernels\n"
        "sig = {'x_ptr': '*fp32', 'out_ptr': '*fp32', 'n': 'i32', 'BLOCK': 'constexpr'}\n"
        "try:\n"
        "    triton.compile(ASTSource(kernels.gamma_ratio_guarded, sig, {'BLOCK': 64}), target=GPUTarget('hip', 'gfx942', 64))\n"
        "except Exception as exc:\n"
        "    print('FAILED', exc)\n"
        "else:\n"
        "    print('COMPILED')\n"
    )
    env = {**os.environ, "TRITON_CACHE_DIR": str(tmp_path / "cache"), "PYTHONDONTWRITEBYTECODE": "1"}
    out = subprocess.run([triton_python, str(script)], capture_output=True, text=True, env=env, check=False)
    assert "FAILED" in out.stdout
    assert "has no attribute 'tgamma'" in out.stdout


def test_metadata_is_what_the_compiler_reported(demo):
    cuda = demo.cell("dot_small[fp16]", "cuda:90", demo.versions[0])
    assert cuda.metadata["shared"] > 0
    assert cuda.metadata["ptxas_registers"] > 0
    assert cuda.metadata["name"] == "dot_small"
    assert "num_warps" not in cuda.metadata  # echoed options are not reported as findings
    assert {"ttir", "ttgir", "llir", "ptx", "cubin"} <= set(cuda.artifact_sizes)
    hip = demo.cell("dot_small[fp16]", "hip:gfx950", demo.versions[0])
    assert {"amdgcn", "hsaco"} <= set(hip.artifact_sizes)
    assert "ptxas_registers" not in hip.metadata
    assert hip.seconds > 0
    assert hip.peak_rss_bytes > 10 * 1024**2


def test_static_predictions_match_real_compiles(demo):
    predicted = {c.key for c in demo.cells if c.findings}
    failed_for_table_reasons = {
        c.key for c in demo.cells if c.signature in ("libdevice-missing", "fp8-dtype-unsupported")
    }
    assert predicted == failed_for_table_reasons
    assert len(predicted) == 6
    # the guarded kernel is not predicted to fail on HIP: the guard is evaluated per target
    assert not demo.cell("gamma_ratio_guarded", "hip:gfx942", demo.versions[0]).findings


def test_static_skip_avoids_compiling_predicted_cells(with_triton):
    report = run_demo(["current"], ["cuda:80", "hip:gfx942"], static="skip")
    cell = report.cell("gamma_ratio", "hip:gfx942", report.versions[0])
    assert cell.status == R.PREDICTED
    assert cell.seconds is None
    assert cell.signature == "libdevice-missing"
    assert report.cell("quantize_fp8", "cuda:80", report.versions[0]).status == R.PREDICTED
    assert report.cell("quantize_fp8", "hip:gfx942", report.versions[0]).status == R.OK


def test_removed_and_added_apis_on_the_installed_version(demo):
    if not demo.versions[0].startswith("3.9"):
        pytest.skip("asserts Triton 3.9 behaviour")
    assert set(statuses(demo, "rowsum_blockptr").values()) == {R.FRONTEND}
    assert demo.cell("rowsum_blockptr", "cuda:80", demo.versions[0]).signature == "api-removed"
    assert set(statuses(demo, "poll_flags").values()) == {R.OK}


def test_probe_reads_targets_and_tables_from_triton(with_triton):
    info = runner.probe(Path(with_triton), ["cuda:80", "cuda:90", "hip:gfx942"])
    backends = {b["target_backend"]: b for b in info["backends"].values()}
    assert {"80", "90", "100"} <= set(backends["cuda"]["archs"])
    assert "95" not in backends["cuda"]["archs"]
    assert {"gfx942", "gfx950"} <= set(backends["hip"]["archs"])
    assert "tgamma" in info["libdevice"]["generic"]
    hip_module = info["targets"]["hip:gfx942"]["libdevice_module"]
    assert "tgamma" not in info["libdevice"]["implementations"][hip_module]
    assert "lgamma" in info["libdevice"]["implementations"][hip_module]
    assert "fp8e4nv" not in info["targets"]["cuda:80"]["supported_fp8_dtypes"]
    assert "fp8e4nv" in info["targets"]["cuda:90"]["supported_fp8_dtypes"]
    assert info["targets"]["hip:gfx942"]["warp_size"] == 64
    assert info["targets"]["cuda:80"]["warp_size"] == 32


def test_manifest_mistakes_are_load_errors_not_compiler_failures(with_triton, tmp_path):
    shutil.copy(EXAMPLES / "large_tiles" / "kernels.py", tmp_path / "kernels.py")
    sig = {"a_ptr": "*fp32", "b_ptr": "*fp32", "c_ptr": "*fp32"}
    consts = {"M": 16, "N": 16, "K": 16}
    manifest = parse_manifest(
        {
            "kernel": [
                {"name": "missing_fn", "target": "kernels.py:nope", "signature": sig, "constants": consts},
                {"name": "missing_file", "target": "gone.py:f"},
                {
                    "name": "typo",
                    "target": "kernels.py:tile_matmul",
                    "signature": {**sig, "d_ptr": "*fp32"},
                    "constants": consts,
                },
                {"name": "incomplete", "target": "kernels.py:tile_matmul", "signature": sig},
                {
                    "name": "bad_option",
                    "target": "kernels.py:tile_matmul",
                    "signature": sig,
                    "constants": consts,
                    "options": {"num_warps": 3},
                },
                {
                    "name": "fine",
                    "target": "kernels.py:tile_matmul",
                    "signature": sig,
                    "constants": consts,
                    "options": {"num_warps": 2},
                },
            ]
        },
        tmp_path,
    )
    environments = runner.resolve_envs(["current"], ["cuda:80"])
    report = runner.run_matrix(manifest, ["cuda:80"], environments, runner.RunSettings(jobs=4))
    by_name = {c.kernel: c for c in report.cells}
    assert by_name["missing_fn"].status == R.LOAD
    assert "nope" in by_name["missing_fn"].message
    assert by_name["missing_file"].status == R.LOAD
    assert "not found" in by_name["missing_file"].message
    assert by_name["typo"].status == R.LOAD
    assert "'d_ptr' is not a parameter" in by_name["typo"].message
    assert by_name["incomplete"].status == R.LOAD
    assert "parameter 'M'" in by_name["incomplete"].message
    assert by_name["bad_option"].status == R.FRONTEND
    assert "power of 2" in by_name["bad_option"].message
    assert by_name["fine"].status == R.OK


@pytest.mark.slow
def test_real_timeout_and_memory_guard(with_triton):
    manifest = load_manifest(EXAMPLES / "large_tiles" / "portcheck.toml")
    manifest.kernels = [k for k in manifest.kernels if k.name == "tile_matmul[256]"]
    environments = runner.resolve_envs(["current"], ["cuda:80"])
    slow = runner.run_matrix(manifest, ["cuda:80"], environments, runner.RunSettings(timeout=4, static="off"))
    (cell,) = slow.cells
    assert cell.status == R.TIMEOUT
    assert cell.stage in ("llir", "ptx", "cubin")
    assert 4 <= cell.wall_seconds < 15

    tight = runner.RunSettings(timeout=120, max_rss=60 * 1024**2, static="off")
    (cell,) = runner.run_matrix(manifest, ["cuda:80"], environments, tight).cells
    assert cell.status == R.OOM


def test_saved_ir_and_textual_diff_between_targets(with_triton, tmp_path):
    report = run_demo(["current"], ["cuda:80", "cuda:100"], save_ir=tmp_path / "ir", static="off")
    cell = report.cell("dot_small[fp16]", "cuda:80", report.versions[0])
    saved = {p.name for p in Path(cell.ir_dir).iterdir()}
    assert {"kernel.ttir", "kernel.ttgir", "kernel.llir", "kernel.ptx", "kernel.cubin"} <= saved
    failed = report.cell("quantize_fp8", "cuda:80", report.versions[0])
    assert failed.status == R.FRONTEND
    diff = diffing.diff_reports(report, targets=("cuda:80", "cuda:100"), ir_stage="ttgir")
    entry = next(e for e in diff.entries if e.row == "dot_small[fp16]")
    assert entry.ir_diff.startswith("--- cuda:80.ttgir")
    assert "loc(" not in entry.ir_diff
    fixed = next(e for e in diff.entries if e.row == "quantize_fp8")
    assert fixed.kind == "fixed"


@pytest.mark.parametrize("mode", ["interpret", "record"])
def test_capture_from_a_real_test_suite_then_compile(with_triton, torch_available, tmp_path, mode, capfd):
    if not torch_available:
        pytest.skip("the example test-suite needs torch")
    project = tmp_path / "proj"
    shutil.copytree(EXAMPLES / "quickstart", project, ignore=shutil.ignore_patterns("portcheck.toml", "__pycache__"))
    old = Path.cwd()
    os.chdir(project)
    try:
        code = cli.main(
            ["capture", "--mode", mode, "-o", "portcheck.toml", "--", "pytest", "-q", "-p", "no:cacheprovider", "."]
        )
    finally:
        os.chdir(old)
    output = capfd.readouterr()
    assert code == cli.EXIT_OK
    # the interpreter really runs the kernels, so the suite passes; record mode does not execute them
    assert ("3 passed" in output.out) == (mode == "interpret")
    manifest = load_manifest(project / "portcheck.toml")
    by_name = {k.name: k for k in manifest.kernels}
    assert set(by_name) == {"scaled_add#1", "scaled_add#2", "row_softmax"}
    assert by_name["scaled_add#1"].signature["x_ptr"] == "*fp32"
    assert by_name["scaled_add#1"].constants == {"BLOCK": 128}
    assert by_name["scaled_add#2"].signature["x_ptr"] == "*fp16"
    assert by_name["scaled_add#2"].constants == {"BLOCK": 64}
    assert by_name["row_softmax"].options == {"num_warps": 2}
    assert by_name["row_softmax"].target == "kernels.py:row_softmax"

    targets = ["cuda:80", "hip:gfx942"]
    environments = runner.resolve_envs(["current"], targets)
    report = runner.run_matrix(manifest, targets, environments, runner.RunSettings(jobs=4))
    assert {c.status for c in report.cells} == {R.OK}
    assert len(report.cells) == 6


def test_committed_example_manifest_matches_capture(with_triton):
    manifest = load_manifest(EXAMPLES / "quickstart" / "portcheck.toml")
    assert [k.name for k in manifest.kernels] == ["scaled_add#1", "scaled_add#2", "row_softmax"]
    assert capture.MODES[0] == "interpret"


def test_cli_end_to_end(with_triton, tmp_path):
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONDONTWRITEBYTECODE": "1"}
    command = [sys.executable, "-m", "triton_portcheck", "check", "--demo", "-q", "-t", "cuda:80,hip:gfx942"]
    out = subprocess.run(
        [*command, "-k", "axpy", "--json", str(tmp_path / "r.json")],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert out.returncode == 0, out.stderr
    assert "2 cells: 2 ok, 0 failed" in out.stdout
    assert json.loads((tmp_path / "r.json").read_text())["environments"]
    manifest = tmp_path / "portcheck.toml"
    manifest.write_text(
        f'[[kernel]]\ntarget = "{cli.DEMO_MANIFEST.parent / "kernels.py"}:gamma_ratio"\n'
        'signature = { x_ptr = "*fp32", out_ptr = "*fp32", n = "i32" }\nconstants = { BLOCK = 64 }\n'
    )
    failing = [sys.executable, "-m", "triton_portcheck", "check", str(manifest), "-q", "-t", "cuda:80,hip:gfx942"]
    out = subprocess.run(failing, capture_output=True, text=True, env=env, check=False)
    assert out.returncode == 1
    assert "front:A" in out.stdout
    out = subprocess.run(
        [sys.executable, "-m", "triton_portcheck", "targets"], capture_output=True, text=True, env=env, check=False
    )
    assert out.returncode == 0
    assert "cuda:90" in out.stdout
    assert "hip:gfx942" in out.stdout


def _normalise(text: str) -> str:
    text = re.sub(r" · slowest compile .*", "", text)
    text = re.sub(r"\(python [\d.]+\)", "(python X)", text)
    return re.sub(r"triton-portcheck \d+\.\d+\.\d+", "triton-portcheck X", text).strip()


def _readme_block(name: str) -> tuple[str, str, str]:
    readme = (ROOT / "README.md").read_text()
    match = re.search(rf"(<!-- BEGIN {name} -->\n```text\n)(.*?)(```\n<!-- END {name} -->)", readme, re.DOTALL)
    assert match, f"README.md has no {name} block"
    return readme, match.group(2), match.group(0)


def _check_readme(name: str, command: str, rendered: str) -> None:
    readme, body, whole = _readme_block(name)
    fresh = f"$ {command}\n{rendered}"
    if os.environ.get("UPDATE_README"):
        replacement = f"<!-- BEGIN {name} -->\n```text\n{fresh}```\n<!-- END {name} -->"
        (ROOT / "README.md").write_text(readme.replace(whole, replacement))
        return
    assert _normalise(body) == _normalise(fresh), f"README {name} block is stale; rerun with UPDATE_README=1"


def test_readme_demo_output_is_real(demo):
    if demo.versions != ["3.9.0"]:
        pytest.skip("the README shows Triton 3.9.0 output")
    _check_readme("demo", "triton-portcheck check --demo", render_terminal(demo, baseline.judge(demo, None)))


@pytest.mark.versions
def test_cross_version_matrix_and_diff(with_triton):
    """Needs a managed 3.8.0 environment (`triton-portcheck env add 3.8.0`) next to a current 3.9.x."""
    if not envs.env_python(envs.env_path("3.8.0")).exists():
        pytest.skip("no managed Triton 3.8.0 environment")
    targets = ["cuda:80", "cuda:90", "hip:gfx942"]
    report = run_demo(["3.8.0", "current"], targets)
    if report.versions != ["3.8.0", "3.9.0"]:
        pytest.skip(f"asserts 3.8.0 vs 3.9.0, have {report.versions}")
    assert report.environments["3.8.0"]["managed"] is True
    assert set(statuses(report, "rowsum_blockptr", "3.8.0").values()) == {R.OK}
    assert set(statuses(report, "rowsum_blockptr", "3.9.0").values()) == {R.FRONTEND}
    assert set(statuses(report, "poll_flags", "3.8.0").values()) == {R.FRONTEND}
    assert set(statuses(report, "poll_flags", "3.9.0").values()) == {R.OK}
    assert report.cell("poll_flags", "cuda:80", "3.8.0").signature == "language-attribute-missing"
    # the portability findings hold on both versions
    for version in report.versions:
        assert statuses(report, "gamma_ratio", version)["hip:gfx942"] == R.FRONTEND
        assert statuses(report, "scatter_rows", version)["cuda:90"] == R.ASSEMBLER
        assert statuses(report, "gamma_ratio_guarded", version)["hip:gfx942"] == R.OK

    diff = diffing.diff_reports(report, versions=("3.8.0", "3.9.0"))
    assert {e.row.split(" @ ")[0] for e in diff.regressions} == {"rowsum_blockptr"}
    assert {e.row.split(" @ ")[0] for e in diff.of_kind("fixed")} == {"poll_flags"}
    grown = next(e for e in diff.entries if e.row == "gamma_ratio @ cuda:80")
    assert grown.size_changes["llir"][1] > grown.size_changes["llir"][0]
    _check_readme(
        "versions",
        "triton-portcheck check --demo --triton 3.8.0,3.9.0 -t cuda:80,cuda:90,hip:gfx942 --json report.json\n"
        "$ triton-portcheck diff report.json --versions 3.8.0 3.9.0",
        diffing.render_diff(diff),
    )

"""Runner behaviour with a stub child: isolation, limits, mapping to cells. No Triton needed."""

import subprocess
import sys
import time
from pathlib import Path

import pytest

from conftest import FIXTURES
from triton_portcheck import envs, runner
from triton_portcheck import results as R
from triton_portcheck.manifest import parse_manifest


@pytest.fixture
def stub(monkeypatch):
    monkeypatch.setattr(runner, "CHILD", FIXTURES / "stub_child.py")
    monkeypatch.setenv("TRITON_PORTCHECK_PYTHON", sys.executable)
    return Path(sys.executable)


def manifest_for(tmp_path, *behaviours, **kernel_extra):
    kernels = [{"name": f"{b}{i}", "target": f"mod:{b}", **kernel_extra} for i, b in enumerate(behaviours)]
    return parse_manifest({"kernel": kernels}, tmp_path)


def run(tmp_path, *behaviours, targets=("cuda:80",), **settings):
    manifest = manifest_for(tmp_path, *behaviours)
    environments = runner.resolve_envs(["current"], list(targets))
    return runner.run_matrix(manifest, list(targets), environments, runner.RunSettings(**settings))


@pytest.mark.parametrize(
    ("text", "expected"),
    [("8G", 8 * 1024**3), ("512M", 512 * 1024**2), ("1.5g", int(1.5 * 1024**3)), ("2GiB", 2 * 1024**3), ("100", 100)],
)
def test_parse_size(text, expected):
    assert runner.parse_size(text) == expected


def test_parse_size_off_and_invalid():
    assert runner.parse_size(None) is None
    assert runner.parse_size("0") is None
    assert runner.parse_size("off") is None
    with pytest.raises(ValueError, match="invalid size"):
        runner.parse_size("lots")


def test_ok_cell_carries_times_sizes_and_reported_metadata(tmp_path, stub):
    report = run(tmp_path, "ok")
    (cell,) = report.cells
    assert cell.status == R.OK
    assert cell.triton == "9.9.9"
    assert cell.seconds == 0.3
    assert cell.stage_seconds == {"frontend": 0.01, "ttgir": 0.02}
    assert cell.artifact_sizes == {"ttir": 10, "cubin": 20}
    assert cell.metadata == {"shared": 128, "ptxas_registers": 17}
    assert (cell.file, cell.line) == ("/repo/k.py", 4)
    assert report.versions == ["9.9.9"]
    assert "does not prove" in report.disclaimer


def test_failure_maps_stage_to_status_and_reads_stderr(tmp_path, stub):
    (cell,) = run(tmp_path, "fail").cells
    assert cell.status == R.MLIR
    assert cell.stage == "ttgir"
    assert cell.message == "op is not supported here"
    assert (cell.line, cell.column) == (9, 3)
    assert len(cell.fingerprint) == 12
    assert "stderr" in cell.detail


def test_timeout_kills_the_whole_process_tree(tmp_path, stub):
    started = time.monotonic()
    (cell,) = run(tmp_path, "sleep", timeout=1.0).cells
    assert time.monotonic() - started < 20
    assert cell.status == R.TIMEOUT
    assert cell.stage == "ttgir"
    assert cell.message == "compile exceeded the 1 s limit"
    time.sleep(0.2)
    leftovers = subprocess.run(["pgrep", "-f", "time.sleep\\(600\\)"], capture_output=True, text=True, check=False)
    assert leftovers.stdout.strip() == ""


@pytest.mark.skipif(not Path("/proc/self/statm").exists(), reason="memory guard reads /proc")
def test_memory_guard(tmp_path, stub):
    (cell,) = run(tmp_path, "alloc", timeout=60, max_rss=100 * 1024**2).cells
    assert cell.status == R.OOM
    assert "memory limit" in cell.message
    assert (cell.peak_rss_bytes or 0) > 100 * 1024**2


def test_crash_reports_signal_stage_and_stderr(tmp_path, stub):
    (cell,) = run(tmp_path, "abort").cells
    assert cell.status == R.CRASH
    assert cell.stage == "ttgir"
    assert "LLVM ERROR: something broke inside the compiler" in cell.message
    assert "SIGABRT" in cell.message


def test_one_bad_cell_does_not_affect_the_others(tmp_path, stub):
    report = run(tmp_path, "abort", "ok", "fail", "ok", jobs=3)
    assert [c.status for c in report.cells] == [R.CRASH, R.OK, R.MLIR, R.OK]


def test_matrix_shape_per_kernel_targets_skip_and_xfail(tmp_path, stub):
    manifest = parse_manifest(
        {
            "kernel": [
                {"name": "a", "target": "m:ok", "targets": ["cuda:80"]},
                {"name": "b", "target": "m:fail", "xfail": ["hip:*"], "skip": ["cuda:80"]},
            ]
        },
        tmp_path,
    )
    targets = ["cuda:80", "hip:gfx942"]
    environments = runner.resolve_envs(["current"], targets)
    report = runner.run_matrix(manifest, targets, environments, runner.RunSettings())
    by_key = {(c.kernel, c.target): c for c in report.cells}
    assert set(by_key) == {("a", "cuda:80"), ("b", "cuda:80"), ("b", "hip:gfx942")}
    assert by_key[("b", "cuda:80")].status == R.SKIPPED
    assert by_key[("b", "hip:gfx942")].status == R.MLIR
    assert by_key[("b", "hip:gfx942")].expected_failure


def test_static_modes(tmp_path, stub):
    manifest = manifest_for(tmp_path, "predict_ok", "ok")
    targets = ["cuda:80", "hip:gfx942"]
    environments = runner.resolve_envs(["current"], targets)

    annotated = runner.run_matrix(manifest, targets, environments, runner.RunSettings(static="annotate"))
    cell = annotated.cell("predict_ok0", "hip:gfx942", "9.9.9")
    assert cell.status == R.OK
    assert cell.findings[0].symbol == "tgamma"

    skipped = runner.run_matrix(manifest, targets, environments, runner.RunSettings(static="skip"))
    cell = skipped.cell("predict_ok0", "hip:gfx942", "9.9.9")
    assert cell.status == R.PREDICTED
    assert cell.signature == "libdevice-missing"
    assert cell.line == 7
    assert skipped.cell("predict_ok0", "cuda:80", "9.9.9").status == R.OK

    only = runner.run_matrix(manifest, targets, environments, runner.RunSettings(static="only"))
    assert {c.status for c in only.cells} == {R.PREDICTED, R.SKIPPED}

    off = runner.run_matrix(manifest, targets, environments, runner.RunSettings(static="off"))
    assert all(not c.findings for c in off.cells)


def test_progress_callback_sees_every_compiled_cell(tmp_path, stub):
    seen = []
    manifest = manifest_for(tmp_path, "ok", "fail")
    environments = runner.resolve_envs(["current"], ["cuda:80"])
    runner.run_matrix(manifest, ["cuda:80"], environments, runner.RunSettings(), lambda c, d, t: seen.append((d, t)))
    assert sorted(seen) == [(1, 2), (2, 2)]


def test_resolve_envs_versions(tmp_path, stub, monkeypatch):
    monkeypatch.setenv("TRITON_PORTCHECK_HOME", str(tmp_path / "home"))
    # asking for the version the current interpreter already has does not create anything
    (same,) = runner.resolve_envs(["9.9.9"], [])
    assert not same.managed
    assert len(runner.resolve_envs(["current", "9.9.9"], [])) == 1
    with pytest.raises(runner.RunnerError, match=r"env add 1\.2\.3"):
        runner.resolve_envs(["1.2.3"], [], create=False)

    created = []

    def fake_add(version, **kwargs):
        python = envs.env_python(envs.env_path(version))
        python.parent.mkdir(parents=True)
        python.symlink_to(sys.executable)
        created.append(version)
        return envs.env_path(version)

    monkeypatch.setattr(envs, "add_env", fake_add)
    (managed,) = runner.resolve_envs(["1.2.3"], [])
    assert created == ["1.2.3"]
    assert managed.managed
    assert managed.fallback_sys_path == ["/base/site"]
    (isolated,) = runner.resolve_envs(["1.2.3"], [], inherit_site=False)
    assert isolated.fallback_sys_path == []


def test_no_triton_gives_actionable_error(tmp_path, monkeypatch):
    # the real child, run by an interpreter that cannot import triton
    monkeypatch.setenv("TRITON_PORTCHECK_PYTHON", sys.executable)
    monkeypatch.setattr(
        runner, "probe", lambda python, targets=None: {"python": "3.12.0", "error": "No module named 'triton'"}
    )
    with pytest.raises(runner.RunnerError, match="Triton is not importable"):
        runner.resolve_envs(["current"], [])


def test_missing_interpreter(tmp_path):
    with pytest.raises(runner.RunnerError, match="cannot start"):
        runner.run_child(tmp_path / "no-python", "probe", {})

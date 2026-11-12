"""CLI wiring with the stub child: exit codes, outputs, baseline flow. No Triton needed."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import FIXTURES, ROOT
from triton_portcheck import cli, runner

ENV = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONDONTWRITEBYTECODE": "1"}

MANIFEST = """
[defaults]
targets = ["cuda:80", "hip:gfx942"]

[[kernel]]
name = "good"
target = "m:ok"

[[kernel]]
name = "bad"
target = "m:fail"
"""


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "CHILD", FIXTURES / "stub_child.py")
    monkeypatch.setenv("TRITON_PORTCHECK_PYTHON", sys.executable)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "portcheck.toml").write_text(MANIFEST)
    return tmp_path


def test_check_fails_on_failures_and_writes_every_output(project, capsys):
    code = cli.main(
        [
            "check",
            "-q",
            "--json",
            "out/r.json",
            "--markdown",
            "out/r.md",
            "--junit",
            "out/r.xml",
            "--sarif",
            "out/r.sarif",
            "--github",
        ]
    )
    out = capsys.readouterr().out
    assert code == cli.EXIT_FAILURES
    assert next(line for line in out.splitlines() if line.startswith("bad")).split() == ["bad", "mlir:A", "mlir:A"]
    assert "4 cells: 2 ok, 2 failed (2 new)" in out
    assert "::error " in out
    assert json.loads((project / "out/r.json").read_text())["kernels"] == ["good", "bad"]
    assert "## Triton portability matrix" in (project / "out/r.md").read_text()
    assert "<testsuites" in (project / "out/r.xml").read_text()
    assert json.loads((project / "out/r.sarif").read_text())["version"] == "2.1.0"


def test_baseline_makes_known_failures_pass_and_new_ones_fail(project, capsys):
    assert cli.main(["check", "-q", "--baseline", "baseline.json", "--update-baseline"]) == cli.EXIT_OK
    assert len(json.loads((project / "baseline.json").read_text())["failures"]) == 2
    assert cli.main(["check", "-q", "--baseline", "baseline.json"]) == cli.EXIT_OK
    assert "(0 new, 2 in baseline)" in capsys.readouterr().out
    (project / "portcheck.toml").write_text(MANIFEST + '\n[[kernel]]\nname = "worse"\ntarget = "m:abort"\n')
    assert cli.main(["check", "-q", "--baseline", "baseline.json"]) == cli.EXIT_FAILURES
    assert "(2 new, 2 in baseline)" in capsys.readouterr().out


def test_selection_flags(project, capsys):
    assert cli.main(["check", "-q", "-k", "go*", "-t", "cuda:90,cuda:100", "--format", "json"]) == cli.EXIT_OK
    data = json.loads(capsys.readouterr().out)
    assert data["targets"] == ["cuda:90", "cuda:100"]
    assert data["kernels"] == ["good"]
    assert cli.main(["check", "-q", "-k", "nothing*"]) == cli.EXIT_USAGE


def test_usage_errors(project, capsys):
    assert cli.main(["check", "missing.toml"]) == cli.EXIT_USAGE
    assert "manifest not found" in capsys.readouterr().err
    assert cli.main(["check", "-t", "cuda"]) == cli.EXIT_USAGE
    assert cli.main(["check", "--update-baseline"]) == cli.EXIT_USAGE
    assert cli.main(["check", "--max-rss", "plenty"]) == cli.EXIT_USAGE
    assert cli.main(["check", "--baseline", "nope.json"]) == cli.EXIT_USAGE
    (project / "portcheck.toml").unlink()
    assert cli.main(["check"]) == cli.EXIT_USAGE
    assert "check --demo" in capsys.readouterr().err
    assert cli.main([]) == cli.EXIT_USAGE


def test_environment_error_exit_code(project, monkeypatch, capsys):
    monkeypatch.setattr(
        runner, "probe", lambda python, targets=None: {"python": "3.12.0", "error": "No module named 'triton'"}
    )
    assert cli.main(["check", "-q"]) == cli.EXIT_ENVIRONMENT
    assert "pip install triton" in capsys.readouterr().err


def test_report_and_diff_commands(project, capsys):
    cli.main(["check", "-q", "--format", "quiet", "--json", "a.json"])
    (project / "portcheck.toml").write_text(MANIFEST.replace("m:ok", "m:fail"))
    cli.main(["check", "-q", "--format", "quiet", "--json", "b.json"])
    capsys.readouterr()

    assert cli.main(["report", "a.json", "--format", "markdown"]) == cli.EXIT_FAILURES
    assert "| `good` | ✅ | ✅ |" in capsys.readouterr().out

    assert cli.main(["diff", "a.json", "b.json"]) == cli.EXIT_OK
    assert "2 regressed" in capsys.readouterr().out
    assert cli.main(["diff", "a.json", "b.json", "--fail-on-regression", "--format", "json"]) == cli.EXIT_FAILURES
    assert json.loads(capsys.readouterr().out)["axis"] == "run"
    assert cli.main(["diff", "a.json", "--targets", "cuda:80", "hip:gfx942", "--all"]) == cli.EXIT_OK
    assert "Unchanged" in capsys.readouterr().out
    assert cli.main(["diff", "a.json", "--versions", "1.0", "2.0"]) == cli.EXIT_USAGE
    assert cli.main(["diff", "a.json"]) == cli.EXIT_USAGE


def test_env_commands(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("TRITON_PORTCHECK_HOME", str(tmp_path))
    assert cli.main(["env", "path"]) == cli.EXIT_OK
    assert capsys.readouterr().out.strip() == str(tmp_path / "envs")
    assert cli.main(["env", "list"]) == cli.EXIT_OK
    assert "no environments" in capsys.readouterr().out
    assert cli.main(["env", "remove", "3.8.0"]) == cli.EXIT_ENVIRONMENT


def test_capture_rejects_unsupported_commands(capsys):
    assert cli.main(["capture", "--", "make", "test"]) == cli.EXIT_USAGE
    assert "must start with python, pytest or a .py file" in capsys.readouterr().err
    assert cli.main(["capture"]) == cli.EXIT_USAGE


def test_version_and_help():
    out = subprocess.run(
        [sys.executable, "-m", "triton_portcheck", "--version"], capture_output=True, text=True, check=True, env=ENV
    )
    assert out.stdout.startswith("triton-portcheck ")
    helped = subprocess.run(
        [sys.executable, "-m", "triton_portcheck", "check", "--help"],
        capture_output=True,
        text=True,
        check=True,
        env=ENV,
    )
    assert "does not prove numerical correctness" in " ".join(helped.stdout.split())
    assert "exit codes" in helped.stdout


def test_core_never_imports_triton():
    """The parent process must work (and stay fast) without Triton."""
    code = (
        "import sys, triton_portcheck.cli, triton_portcheck.runner, triton_portcheck.report, "
        "triton_portcheck.diffing, triton_portcheck.capture, triton_portcheck.envs\n"
        "bad = [m for m in sys.modules if m == 'triton' or m.startswith(('triton.', 'torch'))]\n"
        "assert not bad, bad"
    )
    subprocess.run([sys.executable, "-c", code], check=True, env=ENV)
    offenders = []
    for path in (ROOT / "src" / "triton_portcheck").rglob("*.py"):
        if path.name == "_child.py" or path.parent.name == "demo":
            continue
        for line in path.read_text().splitlines():
            stripped = line.strip()
            if stripped.startswith(("import triton", "from triton ", "from triton.", "import torch")):
                offenders.append(f"{path.name}: {stripped}")
    assert offenders == []


def test_child_is_stdlib_only():
    source = (ROOT / "src" / "triton_portcheck" / "_child.py").read_text()
    assert "import triton_portcheck" not in source
    assert "from triton_portcheck" not in source
    assert "from ." not in source
    assert Path(runner.CHILD).name in ("_child.py", "stub_child.py")

"""Shared fixtures.

Unit tests need nothing but the standard library.  Tests marked ``compile``
really compile kernels and are skipped when no interpreter with Triton is
available (``TRITON_PORTCHECK_PYTHON`` or the one running pytest).
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

from triton_portcheck import results as R

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _has_module(python: str, name: str) -> bool:
    if python == sys.executable:
        return importlib.util.find_spec(name) is not None
    code = f"import importlib.util,sys;sys.exit(0 if importlib.util.find_spec('{name}') else 1)"
    try:
        return subprocess.run([python, "-c", code], check=False, timeout=60).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


@pytest.fixture(scope="session")
def triton_python() -> str:
    """Interpreter that has Triton, or skip."""
    python = os.environ.get("TRITON_PORTCHECK_PYTHON") or sys.executable
    if not _has_module(python, "triton"):
        pytest.skip("Triton is not installed (set TRITON_PORTCHECK_PYTHON to an interpreter that has it)")
    return python


@pytest.fixture
def with_triton(triton_python: str, monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setenv("TRITON_PORTCHECK_PYTHON", triton_python)
    return triton_python


@pytest.fixture(scope="session")
def torch_available(triton_python: str) -> bool:
    return _has_module(triton_python, "torch")


def make_cell(kernel: str, target: str, triton: str = "3.9.0", status: str = R.OK, **extra: object) -> R.Cell:
    cell = R.Cell(kernel=kernel, target=target, triton=triton, status=status)
    if status == R.OK:
        cell.seconds = 0.5
        cell.artifact_sizes = {"ttir": 1000, "ptx": 2000, "cubin": 3000}
        cell.metadata = {"shared": 0, "name": kernel}
    for key, value in extra.items():
        setattr(cell, key, value)
    return cell


@pytest.fixture
def sample_report() -> R.Report:
    """2 kernels x 2 targets x 2 versions with a mix of outcomes."""
    cells = []
    for version in ("3.8.0", "3.9.0"):
        cells.append(make_cell("axpy", "cuda:80", version, file="/repo/k.py", line=3))
        cells.append(make_cell("axpy", "hip:gfx942", version, file="/repo/k.py", line=3))
        cells.append(make_cell("gamma", "cuda:80", version, file="/repo/k.py", line=10))
        cells.append(
            make_cell(
                "gamma",
                "hip:gfx942",
                version,
                R.FRONTEND,
                stage="frontend",
                message="libdevice.tgamma is not implemented for the hip backend",
                fingerprint="aaaaaaaaaaaa",
                signature="libdevice-missing",
                explanation="HIP libdevice lacks it.",
                issue_url="https://github.com/triton-lang/triton/issues/10375",
                file="/repo/k.py",
                line=14,
                column=5,
            )
        )
    return R.Report(
        kernels=["axpy", "gamma"],
        targets=["cuda:80", "hip:gfx942"],
        versions=["3.8.0", "3.9.0"],
        cells=cells,
        environments={"3.8.0": {"python": "3.13.1"}, "3.9.0": {"python": "3.13.1"}},
        created="2026-01-02T03:04:05Z",
    )

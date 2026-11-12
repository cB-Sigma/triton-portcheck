"""The interpret-mode numpy pre-check."""

from __future__ import annotations

import sys

from triton_portcheck import cli


def test_has_module_finds_a_stdlib_module() -> None:
    assert cli._has_module(sys.executable, "json")


def test_has_module_reports_a_missing_module() -> None:
    assert not cli._has_module(sys.executable, "surely_not_an_installed_module_xyz")


def test_has_module_survives_a_missing_interpreter() -> None:
    assert not cli._has_module("/nonexistent/python", "json")


def test_capture_interpret_stops_when_numpy_is_missing(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli.envs, "base_python", lambda: sys.executable)
    monkeypatch.setattr(cli, "_has_module", lambda python, name: False)
    code = cli.main(["capture", "--mode", "interpret", "-o", "unused.toml", "--", "pytest", "-q"])
    assert code == cli.EXIT_ENVIRONMENT
    assert "needs numpy" in capsys.readouterr().err

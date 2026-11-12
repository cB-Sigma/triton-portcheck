"""Baseline: remember known failures so CI only fails on new breakage."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .results import Cell, Report

BASELINE_SCHEMA = 1


@dataclass
class Verdict:
    """How a report compares with what was expected."""

    new_failures: list[Cell] = field(default_factory=list)
    known_failures: list[Cell] = field(default_factory=list)
    expected_failures: list[Cell] = field(default_factory=list)
    fixed: list[tuple[str, str, str]] = field(default_factory=list)
    unexpected_passes: list[Cell] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.new_failures


def load_baseline(path: str | Path) -> dict[tuple[str, str, str], dict[str, str]]:
    """Read a baseline file (or a full JSON report) into ``{cell key: entry}``."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "cells" in data:  # a full report works as a baseline too
        entries = [c for c in Report.from_dict(data).cells if c.failed]
        return {c.key: {"status": c.status, "fingerprint": c.fingerprint} for c in entries}
    if data.get("baseline_schema") != BASELINE_SCHEMA:
        raise ValueError(f"{path}: not a triton-portcheck baseline (baseline_schema {BASELINE_SCHEMA} expected)")
    return {
        (e["kernel"], e["target"], e["triton"]): {"status": e["status"], "fingerprint": e.get("fingerprint", "")}
        for e in data["failures"]
    }


def write_baseline(report: Report, path: str | Path) -> int:
    """Write every currently failing cell; returns how many were recorded."""
    failures = [
        {
            "kernel": c.kernel,
            "target": c.target,
            "triton": c.triton,
            "status": c.status,
            "fingerprint": c.fingerprint,
            "message": c.message,
        }
        for c in sorted(report.failures(), key=lambda c: c.key)
    ]
    payload = {"baseline_schema": BASELINE_SCHEMA, "tool_version": report.tool_version, "failures": failures}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return len(failures)


def judge(report: Report, baseline: dict[tuple[str, str, str], dict[str, str]] | None) -> Verdict:
    """Split failing cells into new, known (baseline) and expected (manifest ``xfail``)."""
    verdict = Verdict()
    baseline = baseline or {}
    current = report.index()
    for cell in report.cells:
        if cell.failed:
            if cell.expected_failure:
                verdict.expected_failures.append(cell)
            elif cell.key in baseline:
                verdict.known_failures.append(cell)
            else:
                verdict.new_failures.append(cell)
        elif cell.status == "ok" and cell.expected_failure:
            verdict.unexpected_passes.append(cell)
    for key in sorted(baseline):
        cell_now = current.get(key)
        if cell_now is not None and cell_now.status == "ok":
            verdict.fixed.append(key)
    return verdict

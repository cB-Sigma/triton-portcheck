"""Compare two columns of a matrix: two Triton versions, two targets or two runs."""

from __future__ import annotations

import difflib
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from . import DISCLAIMER
from .results import OK, Cell, Report


class DiffError(ValueError):
    """Raised when the requested comparison cannot be made."""


@dataclass
class Thresholds:
    time_ratio: float = 1.5
    time_min_seconds: float = 1.0
    size_ratio: float = 0.10


@dataclass
class Entry:
    row: str
    a_status: str
    b_status: str
    kind: str  # regression | fixed | changed | same | only-a | only-b
    a_message: str = ""
    b_message: str = ""
    a_seconds: float | None = None
    b_seconds: float | None = None
    time_regression: bool = False
    size_changes: dict[str, tuple[int, int]] = field(default_factory=dict)
    metadata_changes: dict[str, tuple[Any, Any]] = field(default_factory=dict)
    ir_diff: str = ""

    @property
    def interesting(self) -> bool:
        return (
            self.kind != "same"
            or self.time_regression
            or bool(self.size_changes or self.metadata_changes or self.ir_diff)
        )


@dataclass
class Diff:
    label_a: str
    label_b: str
    axis: str
    entries: list[Entry]
    disclaimer: str = DISCLAIMER

    def of_kind(self, kind: str) -> list[Entry]:
        return [e for e in self.entries if e.kind == kind]

    @property
    def regressions(self) -> list[Entry]:
        return self.of_kind("regression")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalise_ir(text: str, stage: str) -> str:
    """Remove noise that changes between runs without changing the code."""
    lines = []
    for line in text.splitlines():
        if stage in ("ttir", "ttgir", "source"):
            line = re.sub(r"\s*loc\((?:[^()]|\([^()]*\))*\)", "", line)
            if line.startswith("#loc"):
                continue
        elif stage == "llir":
            if line.startswith(("!", "source_filename", "; ModuleID")):
                continue
            line = re.sub(r",? !dbg !\d+", "", line)
            line = re.sub(r"\s*#dbg_\w+\(.*\)$", "", line)
        elif stage == "ptx":
            if re.match(r"\s*\.(loc|file)\b", line) or line.lstrip().startswith("//"):
                continue
            line = re.sub(r"\s*//.*$", "", line)
        elif stage == "amdgcn":
            if re.match(r"\s*\.(loc|file|ident)\b", line) or line.lstrip().startswith(";"):
                continue
            line = re.sub(r"\s*;.*$", "", line)
        if line.strip():
            lines.append(line.rstrip())
    return "\n".join(lines) + "\n"


def _ir_diff(a: Cell, b: Cell, stage: str, la: str, lb: str, root_a: Path, root_b: Path, limit: int) -> str:
    if not a.ir_dir or not b.ir_dir:
        return ""
    file_a, file_b = root_a / a.ir_dir / f"kernel.{stage}", root_b / b.ir_dir / f"kernel.{stage}"
    if not file_a.is_file() or not file_b.is_file():
        return ""
    text_a = normalise_ir(file_a.read_text(encoding="utf-8", errors="replace"), stage).splitlines(keepends=True)
    text_b = normalise_ir(file_b.read_text(encoding="utf-8", errors="replace"), stage).splitlines(keepends=True)
    lines = list(difflib.unified_diff(text_a, text_b, f"{la}.{stage}", f"{lb}.{stage}", n=2))
    if len(lines) > limit:
        lines = [*lines[:limit], f"... {len(lines) - limit} more diff lines not shown\n"]
    return "".join(lines)


def compare_cells(a: Cell | None, b: Cell | None, row: str, thresholds: Thresholds) -> Entry:
    if a is None or b is None:
        present = a or b
        assert present is not None
        return Entry(
            row,
            a.status if a else "-",
            b.status if b else "-",
            "only-a" if a else "only-b",
            a.message if a else "",
            b.message if b else "",
        )
    if a.status == b.status:
        kind = "same" if a.status == OK or a.fingerprint == b.fingerprint else "changed"
    elif a.status == OK:
        kind = "regression"
    elif b.status == OK:
        kind = "fixed"
    else:
        kind = "changed"
    entry = Entry(row, a.status, b.status, kind, a.message, b.message, a.seconds, b.seconds)
    if a.status == OK and b.status == OK:
        if a.seconds is not None and b.seconds is not None:
            slower = b.seconds - a.seconds
            entry.time_regression = (
                slower >= thresholds.time_min_seconds and b.seconds >= a.seconds * thresholds.time_ratio
            )
        for ext in a.artifact_sizes.keys() & b.artifact_sizes.keys():
            size_a, size_b = a.artifact_sizes[ext], b.artifact_sizes[ext]
            if size_a != size_b and abs(size_b - size_a) >= thresholds.size_ratio * max(size_a, 1):
                entry.size_changes[ext] = (size_a, size_b)
        for key in sorted(a.metadata.keys() & b.metadata.keys()):
            if a.metadata[key] != b.metadata[key] and isinstance(a.metadata[key], (int, float)):
                entry.metadata_changes[key] = (a.metadata[key], b.metadata[key])
    return entry


def diff_reports(
    report_a: Report,
    report_b: Report | None = None,
    *,
    versions: tuple[str, str] | None = None,
    targets: tuple[str, str] | None = None,
    thresholds: Thresholds | None = None,
    ir_stage: str | None = None,
    root_a: Path = Path("."),
    root_b: Path | None = None,
    ir_limit: int = 200,
) -> Diff:
    """Build a diff along one axis.

    ``versions=(a, b)``  compare two Triton versions inside one report
    ``targets=(a, b)``   compare two targets inside one report
    two reports          compare the same cells between two runs
    """
    thresholds = thresholds or Thresholds()
    root_b = root_b or root_a
    pairs: list[tuple[str, Cell | None, Cell | None]] = []
    if versions is not None:
        other = report_b or report_a
        for name, rep, wanted in (("A", report_a, versions[0]), ("B", other, versions[1])):
            if wanted not in rep.versions:
                raise DiffError(f"version {wanted} ({name}) is not in the report; it has: {', '.join(rep.versions)}")
        index_a, index_b = report_a.index(), other.index()
        for kernel in report_a.kernels:
            for target in report_a.targets:
                a, b = index_a.get((kernel, target, versions[0])), index_b.get((kernel, target, versions[1]))
                if a or b:
                    pairs.append((f"{kernel} @ {target}", a, b))
        label_a, label_b, axis = f"triton {versions[0]}", f"triton {versions[1]}", "version"
    elif targets is not None:
        for wanted in targets:
            if wanted not in report_a.targets:
                raise DiffError(f"target {wanted} is not in the report; it has: {', '.join(report_a.targets)}")
        index_a = report_a.index()
        for kernel in report_a.kernels:
            for version in report_a.versions:
                a, b = index_a.get((kernel, targets[0], version)), index_a.get((kernel, targets[1], version))
                if a or b:
                    suffix = f" (triton {version})" if len(report_a.versions) > 1 else ""
                    pairs.append((f"{kernel}{suffix}", a, b))
        label_a, label_b, axis = targets[0], targets[1], "target"
    elif report_b is not None:
        index_a, index_b = report_a.index(), report_b.index()
        for key in list(index_a) + [k for k in index_b if k not in index_a]:
            pairs.append((f"{key[0]} @ {key[1]} (triton {key[2]})", index_a.get(key), index_b.get(key)))
        label_a, label_b, axis = "A", "B", "run"
    else:
        raise DiffError("nothing to compare: give two reports, --versions A B or --targets A B")

    entries = []
    for row, a, b in pairs:
        entry = compare_cells(a, b, row, thresholds)
        if ir_stage and a is not None and b is not None:
            entry.ir_diff = _ir_diff(a, b, ir_stage, label_a, label_b, root_a, root_b, ir_limit)
        entries.append(entry)
    return Diff(label_a, label_b, axis, entries)


def render_diff(diff: Diff, fmt: str = "text", show_all: bool = False) -> str:
    """Render as plain text or Markdown."""
    md = fmt == "markdown"
    out: list[str] = []
    title = f"{diff.label_a} -> {diff.label_b}"
    out.append(f"## triton-portcheck diff: {title}" if md else f"triton-portcheck diff: {title}")
    out.append("")
    counts = {k: len(diff.of_kind(k)) for k in ("regression", "fixed", "changed", "same")}
    slow = [e for e in diff.entries if e.time_regression]
    out.append(
        f"{counts['regression']} regressed, {counts['fixed']} fixed, {counts['changed']} fail differently, "
        f"{counts['same']} unchanged, {len(slow)} slower to compile"
    )
    sections = (
        ("regression", f"Compiles on {diff.label_a}, fails on {diff.label_b}"),
        ("fixed", f"Fails on {diff.label_a}, compiles on {diff.label_b}"),
        ("changed", "Fails on both, differently"),
        ("only-a", f"Only present for {diff.label_a}"),
        ("only-b", f"Only present for {diff.label_b}"),
    )
    for kind, heading in sections:
        entries = diff.of_kind(kind)
        if not entries:
            continue
        out += ["", f"### {heading}" if md else f"{heading}:"]
        for e in entries:
            line = f"{e.row}: {e.a_status} -> {e.b_status}"
            reason = e.b_message if kind in ("regression", "changed", "only-b") else e.a_message
            out.append(
                f"- `{e.row}`: {e.a_status} → {e.b_status}" + (f" — {reason}" if reason else "") if md else f"  {line}"
            )
            if reason and not md:
                out.append(f"      {reason}")
    if slow:
        out += ["", "### Compile-time regressions" if md else "Compile-time regressions:"]
        for e in slow:
            out.append(f"{'- ' if md else '  '}{e.row}: {e.a_seconds or 0:.2f} s -> {e.b_seconds or 0:.2f} s")
    sized = [e for e in diff.entries if e.size_changes or e.metadata_changes]
    if sized:
        out += ["", "### Size and metadata changes" if md else "Size and metadata changes:"]
        for e in sized:
            parts = [f"{ext} {a} -> {b} ({(b - a) / max(a, 1):+.0%})" for ext, (a, b) in sorted(e.size_changes.items())]
            parts += [f"{key} {a} -> {b}" for key, (a, b) in e.metadata_changes.items()]
            out.append(f"{'- ' if md else '  '}{e.row}: " + ", ".join(parts))
    with_ir = [e for e in diff.entries if e.ir_diff]
    if with_ir:
        out += ["", "### IR differences" if md else "IR differences:"]
        for e in with_ir:
            out += [
                f"{'**' if md else ''}{e.row}{'**' if md else ''}",
                "```diff" if md else "",
                e.ir_diff.rstrip(),
                "```" if md else "",
            ]
    if show_all:
        out += ["", "### Unchanged" if md else "Unchanged:"]
        out += [f"{'- ' if md else '  '}{e.row}: {e.a_status}" for e in diff.of_kind("same")]
    out += ["", f"_{diff.disclaimer}_" if md else diff.disclaimer, ""]
    return "\n".join(out)

"""Human-readable renderings: terminal matrix and Markdown."""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field

from .. import results as R
from ..baseline import Verdict

_COLOURS = {
    R.OK: "32",
    R.SKIPPED: "2",
    R.PREDICTED: "33",
    R.TIMEOUT: "35",
    R.OOM: "35",
    R.CRASH: "1;31",
}


@dataclass
class Group:
    """Cells that failed the same way."""

    label: str
    fingerprint: str
    status: str
    stage: str | None
    message: str
    explanation: str
    issue_url: str
    cells: list[R.Cell] = field(default_factory=list)

    def where(self, with_versions: bool = False) -> str:
        """``kernel on t1, t2`` compressed per kernel; ``target@version`` in multi-version reports."""
        by_kernel: dict[str, list[str]] = {}
        for cell in self.cells:
            name = f"{cell.target}@{cell.triton}" if with_versions else cell.target
            by_kernel.setdefault(cell.kernel, []).append(name)
        return "; ".join(f"{kernel} on {', '.join(names)}" for kernel, names in by_kernel.items())


def _label(index: int) -> str:
    letters = ""
    index += 1
    while index:
        index, rem = divmod(index - 1, 26)
        letters = chr(ord("A") + rem) + letters
    return letters


def failure_groups(report: R.Report) -> list[Group]:
    groups: dict[str, Group] = {}
    for cell in report.cells:
        if not cell.failed:
            continue
        key = cell.fingerprint or f"{cell.status}|{cell.message}"
        if key not in groups:
            groups[key] = Group(
                _label(len(groups)), key, cell.status, cell.stage, cell.message, cell.explanation, cell.issue_url
            )
        groups[key].cells.append(cell)
    return list(groups.values())


def _cell_text(cell: R.Cell | None, labels: dict[tuple[str, str, str], str], times: bool) -> str:
    if cell is None:
        return ""
    if cell.status == R.OK:
        if times and cell.seconds is not None:
            return f"{cell.seconds:.1f}s"
        return "ok" + ("!" if cell.expected_failure else "")
    if cell.status == R.SKIPPED:
        return "-"
    return f"{R.SHORT[cell.status]}:{labels.get(cell.key, '?')}"


def _matrix(report: R.Report, times: bool) -> tuple[list[list[str]], list[list[R.Cell | None]], list[Group]]:
    groups = failure_groups(report)
    labels = {cell.key: group.label for group in groups for cell in group.cells}
    index = report.index()
    multi = len(report.versions) > 1
    header_top = ["kernel"]
    header_sub = [""]
    for target in report.targets:
        for position, version in enumerate(report.versions):
            header_top.append(target if position == 0 else "")
            header_sub.append(version)
    rows = [header_top, header_sub] if multi else [header_top]
    cells: list[list[R.Cell | None]] = [[None] * len(header_top) for _ in rows]
    for kernel in report.kernels:
        row = [kernel]
        row_cells: list[R.Cell | None] = [None]
        for target in report.targets:
            for version in report.versions:
                cell = index.get((kernel, target, version))
                row.append(_cell_text(cell, labels, times))
                row_cells.append(cell)
        rows.append(row)
        cells.append(row_cells)
    return rows, cells, groups


def _paint(text: str, cell: R.Cell | None, colour: bool) -> str:
    if not colour or cell is None or not text.strip():
        return text
    code = _COLOURS.get(cell.status, "31")
    return f"\x1b[{code}m{text}\x1b[0m"


def summary_line(report: R.Report, verdict: Verdict | None = None) -> str:
    total = len(report.cells)
    ok = sum(1 for c in report.cells if c.status == R.OK)
    failed = sum(1 for c in report.cells if c.failed)
    skipped = total - ok - failed
    text = f"{total} cells: {ok} ok, {failed} failed"
    if skipped:
        text += f", {skipped} not run"
    if verdict is not None and failed:
        parts = [f"{len(verdict.new_failures)} new"]
        if verdict.known_failures:
            parts.append(f"{len(verdict.known_failures)} in baseline")
        if verdict.expected_failures:
            parts.append(f"{len(verdict.expected_failures)} expected (xfail)")
        text += f" ({', '.join(parts)})"
    return text


def _environment_line(report: R.Report) -> str:
    parts = []
    for version in report.versions:
        env = report.environments.get(version, {})
        parts.append(f"triton {version} (python {env.get('python', '?')})")
    return ", ".join(parts)


def render_terminal(
    report: R.Report, verdict: Verdict | None = None, *, colour: bool = False, times: bool = False
) -> str:
    """The matrix, the failure groups and the compile-only notice."""
    rows, cells, groups = _matrix(report, times)
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    multi = len(report.versions) > 1
    lines = [f"triton-portcheck {report.tool_version} · {_environment_line(report)} · no GPU involved", ""]
    for row_index, (row, row_cells) in enumerate(zip(rows, cells, strict=True)):
        parts = [row[0].ljust(widths[0])]
        for i in range(1, len(row)):
            new_target = multi and (i - 1) % len(report.versions) == 0
            parts.append(("  " if new_target else "") + _paint(row[i].ljust(widths[i]), row_cells[i], colour))
        lines.append("  ".join(parts).rstrip())
        if row_index == (1 if multi else 0):
            lines.append("-" * len("  ".join(p for p in parts)))
    if groups:
        lines.append("")
    for group in groups:
        stage = f" in stage {group.stage}" if group.stage and group.stage != "static" else ""
        lines.append(f"[{group.label}] {group.status}{stage}: {group.message}")
        lines += textwrap.wrap(group.where(multi), 100, initial_indent="    ", subsequent_indent="      ")
        if group.explanation:
            lines += textwrap.wrap(
                f"why: {group.explanation}", 100, initial_indent="    ", subsequent_indent="         "
            )
        if group.issue_url:
            lines.append(f"    see: {group.issue_url}")
    predicted = [c for c in report.cells if c.findings and c.status != R.PREDICTED]
    wrong = [c for c in predicted if c.status == R.OK]
    if predicted:
        agree = len(predicted) - len(wrong)
        lines.append("")
        lines += textwrap.wrap(
            f"static pre-checks (tables read from the installed Triton) predicted {len(predicted)} failing "
            f"cells before compiling; {agree} of them did fail",
            100,
        )
        for cell in wrong:
            lines.append(f"    predicted but compiled: {cell.kernel} on {cell.target}: {cell.findings[0].detail}")
    if verdict is not None:
        for cell in verdict.unexpected_passes:
            lines.append(f"note: {cell.kernel} on {cell.target} is marked xfail but compiled")
        for kernel, target, version in verdict.fixed:
            lines.append(f"note: {kernel} on {target} (triton {version}) is in the baseline but now compiles")
    slow = sorted((c for c in report.cells if c.seconds), key=lambda c: -(c.seconds or 0))[:1]
    lines.append("")
    summary = summary_line(report, verdict)
    if slow and slow[0].seconds:
        summary += f" · slowest compile {slow[0].seconds:.1f} s ({slow[0].kernel} on {slow[0].target})"
    lines.append(summary)
    lines += textwrap.wrap(report.disclaimer, 100)
    return "\n".join(lines) + "\n"


def _md_escape(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def render_markdown(report: R.Report, verdict: Verdict | None = None, *, times: bool = False) -> str:
    """Markdown for job summaries and pull-request comments."""
    rows, _cells, groups = _matrix(report, times)
    symbol = {"ok": "✅", "-": "➖", "": ""}
    lines = ["## Triton portability matrix", "", f"{_environment_line(report)} · compiled without a GPU", ""]
    multi = len(report.versions) > 1
    header = rows[0]
    if multi:
        header = [rows[0][0], *[f"{t}<br>{v}" for t in report.targets for v in report.versions]]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] + [":---:"] * (len(header) - 1)) + "|")
    for row in rows[2 if multi else 1 :]:
        rendered = [f"`{row[0]}`"]
        for text in row[1:]:
            rendered.append(symbol.get(text, "✅" if text.endswith("s") or text == "ok!" else f"❌ {text}"))
        lines.append("| " + " | ".join(rendered) + " |")
    lines.append("")
    if groups:
        lines += ["### Failures", ""]
    for group in groups:
        stage = f", stage `{group.stage}`" if group.stage and group.stage != "static" else ""
        lines.append(f"- **{group.label}** — {group.status}{stage}: {_md_escape(group.message)}")
        lines.append(f"  - where: {_md_escape(group.where(multi))}")
        if group.explanation:
            lines.append(f"  - why: {_md_escape(group.explanation)}")
        if group.issue_url:
            lines.append(f"  - upstream: {group.issue_url}")
    if groups:
        lines.append("")
    lines += [f"**{summary_line(report, verdict)}**", "", f"_{report.disclaimer}_", ""]
    return "\n".join(lines)

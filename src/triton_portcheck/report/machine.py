"""Machine-readable renderings: JUnit XML, SARIF 2.1.0 and GitHub annotations."""

from __future__ import annotations

import json
import os
from pathlib import Path
from xml.etree import ElementTree as ET

from .. import results as R
from ..baseline import Verdict

_INFO_URI = "https://github.com/cb-Sigma/triton-portcheck"


def _suppressed(cell: R.Cell, verdict: Verdict | None) -> bool:
    """A failing cell that should not fail CI (xfail or already in the baseline)."""
    if cell.expected_failure:
        return True
    return verdict is not None and any(cell is known for known in verdict.known_failures)


def render_junit(report: R.Report, verdict: Verdict | None = None) -> str:
    """One test suite per Triton version, one test case per kernel x target."""
    root = ET.Element("testsuites", name="triton-portcheck")
    for version in report.versions:
        cells = [c for c in report.cells if c.triton == version]
        failures = [c for c in cells if c.failed and not _suppressed(c, verdict)]
        skipped = [c for c in cells if c.status == R.SKIPPED or (c.failed and _suppressed(c, verdict))]
        suite = ET.SubElement(
            root,
            "testsuite",
            name=f"triton-{version}",
            tests=str(len(cells)),
            failures=str(len(failures)),
            errors="0",
            skipped=str(len(skipped)),
            time=f"{sum(c.wall_seconds or 0 for c in cells):.3f}",
            timestamp=report.created.rstrip("Z"),
        )
        properties = ET.SubElement(suite, "properties")
        ET.SubElement(properties, "property", name="disclaimer", value=report.disclaimer)
        for cell in cells:
            case = ET.SubElement(
                suite,
                "testcase",
                classname=f"triton-{version}.{cell.target}",
                name=cell.kernel,
                time=f"{cell.wall_seconds or 0:.3f}",
            )
            if cell.file:
                case.set("file", cell.file)
            if cell.status == R.SKIPPED:
                ET.SubElement(case, "skipped", message=cell.message or "not run")
            elif cell.failed and _suppressed(cell, verdict):
                reason = "expected failure (xfail)" if cell.expected_failure else "known failure (baseline)"
                ET.SubElement(case, "skipped", message=f"{reason}: {cell.message}")
            elif cell.failed:
                failure = ET.SubElement(case, "failure", message=cell.message, type=cell.status)
                body = [f"stage: {cell.stage}", f"fingerprint: {cell.fingerprint}"]
                if cell.explanation:
                    body.append(f"why: {cell.explanation}")
                if cell.issue_url:
                    body.append(f"see: {cell.issue_url}")
                if cell.detail:
                    body += ["", cell.detail]
                failure.text = "\n".join(body)
    ET.indent(root)
    return '<?xml version="1.0" encoding="utf-8"?>\n' + ET.tostring(root, encoding="unicode") + "\n"


def _relative(path: str | None, base: Path) -> str | None:
    if not path:
        return None
    try:
        return Path(path).resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return None


def render_sarif(report: R.Report, verdict: Verdict | None = None, base: Path | None = None) -> str:
    """SARIF 2.1.0 for GitHub code scanning; one result per failing cell."""
    base = base or Path.cwd()
    rules: dict[str, dict[str, object]] = {}
    sarif_results = []
    for cell in report.failures():
        rule_id = cell.signature or cell.status
        if rule_id not in rules:
            rule: dict[str, object] = {
                "id": rule_id,
                "name": rule_id,
                "shortDescription": {"text": R.STATUS_HELP.get(cell.status, cell.status)},
                "fullDescription": {"text": cell.explanation or R.STATUS_HELP.get(cell.status, cell.status)},
                "helpUri": cell.issue_url or _INFO_URI,
            }
            rules[rule_id] = rule
        result: dict[str, object] = {
            "ruleId": rule_id,
            "level": "note" if _suppressed(cell, verdict) else "error",
            "message": {
                "text": f"{cell.kernel} does not compile for {cell.target} with Triton {cell.triton}: {cell.message}"
            },
            "partialFingerprints": {"portcheck/v1": f"{cell.kernel}|{cell.target}|{cell.triton}|{cell.fingerprint}"},
            "properties": {"target": cell.target, "triton": cell.triton, "stage": cell.stage, "status": cell.status},
        }
        uri = _relative(cell.file, base)
        if uri:
            region = {"startLine": cell.line or 1}
            if cell.column:
                region["startColumn"] = cell.column
            result["locations"] = [{"physicalLocation": {"artifactLocation": {"uri": uri}, "region": region}}]
        sarif_results.append(result)
    document = {
        "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
        "version": "2.1.0",
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": "triton-portcheck",
                        "version": report.tool_version,
                        "informationUri": _INFO_URI,
                        "rules": list(rules.values()),
                    }
                },
                "results": sarif_results,
                "properties": {"disclaimer": report.disclaimer},
            }
        ],
    }
    return json.dumps(document, indent=2) + "\n"


def _escape(text: str, is_property: bool = False) -> str:
    text = text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
    return text.replace(":", "%3A").replace(",", "%2C") if is_property else text


def render_github(report: R.Report, verdict: Verdict | None = None, base: Path | None = None) -> str:
    """GitHub Actions workflow commands: one annotation per failing cell."""
    base = base or Path(os.environ.get("GITHUB_WORKSPACE") or Path.cwd())
    lines = []
    for cell in report.failures():
        level = "warning" if _suppressed(cell, verdict) else "error"
        properties = [f"title={_escape(f'{cell.kernel}: {cell.target} / Triton {cell.triton}', True)}"]
        uri = _relative(cell.file, base)
        if uri:
            properties.insert(0, f"file={_escape(uri, True)}")
            properties.insert(1, f"line={cell.line or 1}")
        message = f"{cell.status}: {cell.message}"
        if cell.issue_url:
            message += f" ({cell.issue_url})"
        lines.append(f"::{level} {','.join(properties)}::{_escape(message)}")
    lines.append(f"::notice title=triton-portcheck::{_escape(report.disclaimer)}")
    return "\n".join(lines) + "\n"

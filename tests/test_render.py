import json
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

from conftest import make_cell
from triton_portcheck import DISCLAIMER, baseline
from triton_portcheck import results as R
from triton_portcheck.report import (
    failure_groups,
    render_github,
    render_junit,
    render_markdown,
    render_sarif,
    render_terminal,
)


def test_terminal_matrix_has_version_subcolumns(sample_report):
    text = render_terminal(sample_report)
    lines = text.splitlines()
    assert lines[0].startswith("triton-portcheck")
    header, sub = lines[2], lines[3]
    assert header.split() == ["kernel", "cuda:80", "hip:gfx942"]
    assert sub.split() == ["3.8.0", "3.9.0", "3.8.0", "3.9.0"]
    gamma = next(line for line in lines if line.startswith("gamma"))
    assert gamma.split() == ["gamma", "ok", "ok", "front:A", "front:A"]
    assert "[A] frontend-error in stage frontend: libdevice.tgamma is not implemented" in text
    assert "gamma on hip:gfx942@3.8.0, hip:gfx942@3.9.0" in text
    assert "see: https://github.com/triton-lang/triton/issues/10375" in text
    assert "8 cells: 6 ok, 2 failed" in text
    assert " ".join(text.split()).endswith(DISCLAIMER)
    assert "\x1b[" not in text


def test_terminal_single_version_has_one_header_row(sample_report):
    sample_report.versions = ["3.9.0"]
    sample_report.cells = [c for c in sample_report.cells if c.triton == "3.9.0"]
    lines = render_terminal(sample_report).splitlines()
    assert lines[2].split() == ["kernel", "cuda:80", "hip:gfx942"]
    assert set(lines[3]) == {"-"}
    assert "gamma on hip:gfx942" in "\n".join(lines)


def test_terminal_colour_and_times(sample_report):
    text = render_terminal(sample_report, colour=True, times=True)
    assert "\x1b[32m0.5s" in text
    assert "\x1b[31mfront:A" in text


def test_every_renderer_carries_the_disclaimer(sample_report):
    for text in (
        render_terminal(sample_report),
        render_markdown(sample_report),
        render_junit(sample_report),
        render_sarif(sample_report),
        render_github(sample_report),
        json.dumps(sample_report.to_dict()),
    ):
        assert "does not prove numerical correctness" in " ".join(text.split())


def test_failure_groups_merge_by_fingerprint(sample_report):
    sample_report.cells.append(
        make_cell("axpy", "cuda:80", "3.7.0", R.TIMEOUT, fingerprint="bbbbbbbbbbbb", message="slow")
    )
    groups = failure_groups(sample_report)
    assert [(g.label, len(g.cells)) for g in groups] == [("A", 2), ("B", 1)]


def test_markdown_table(sample_report):
    text = render_markdown(sample_report)
    assert "| kernel | cuda:80<br>3.8.0 | cuda:80<br>3.9.0 | hip:gfx942<br>3.8.0 | hip:gfx942<br>3.9.0 |" in text
    assert "| `gamma` | ✅ | ✅ | ❌ front:A | ❌ front:A |" in text
    assert "- **A** — frontend-error, stage `frontend`: libdevice.tgamma" in text
    assert "upstream: https://github.com/triton-lang/triton/issues/10375" in text


def test_junit_counts_and_baseline_suppression(sample_report):
    root = ET.fromstring(render_junit(sample_report))
    suites = root.findall("testsuite")
    assert [s.get("name") for s in suites] == ["triton-3.8.0", "triton-3.9.0"]
    assert [s.get("tests") for s in suites] == ["4", "4"]
    assert [s.get("failures") for s in suites] == ["1", "1"]
    failure = suites[0].find("testcase/failure")
    assert failure is not None
    assert failure.get("type") == "frontend-error"
    assert "fingerprint: aaaaaaaaaaaa" in (failure.text or "")

    known = {("gamma", "hip:gfx942", "3.8.0"): {"status": R.FRONTEND, "fingerprint": "aaaaaaaaaaaa"}}
    verdict = baseline.judge(sample_report, known)
    root = ET.fromstring(render_junit(sample_report, verdict))
    assert [s.get("failures") for s in root.findall("testsuite")] == ["0", "1"]
    assert [s.get("skipped") for s in root.findall("testsuite")] == ["1", "0"]


def test_sarif_structure_and_locations(sample_report):
    document = json.loads(render_sarif(sample_report, base=Path("/repo")))
    assert document["version"] == "2.1.0"
    run = document["runs"][0]
    assert [r["id"] for r in run["tool"]["driver"]["rules"]] == ["libdevice-missing"]
    assert len(run["results"]) == 2
    result = run["results"][0]
    assert result["level"] == "error"
    location = result["locations"][0]["physicalLocation"]
    assert location["artifactLocation"]["uri"] == "k.py"
    assert location["region"] == {"startLine": 14, "startColumn": 5}
    assert "Triton 3.8.0" in result["message"]["text"]
    outside = json.loads(render_sarif(sample_report, base=Path("/elsewhere")))
    assert "locations" not in outside["runs"][0]["results"][0]


def test_github_annotations_escape_and_levels(sample_report):
    sample_report.cells[3].message = "50% broken\nsecond line"
    known = {("gamma", "hip:gfx942", "3.9.0"): {"status": R.FRONTEND, "fingerprint": ""}}
    text = render_github(sample_report, baseline.judge(sample_report, known), base=Path("/repo"))
    lines = text.splitlines()
    assert lines[0].startswith("::error file=k.py,line=14,title=gamma%3A hip%3Agfx942 / Triton 3.8.0::")
    assert "50%25 broken%0Asecond line" in lines[0]
    assert lines[1].startswith("::warning ")
    assert lines[-1].startswith("::notice title=triton-portcheck::")


def test_report_json_round_trip(tmp_path, sample_report):
    sample_report.cells[3].findings = [R.Finding("libdevice-missing", "tgamma", "missing", "hip.libdevice", 14)]
    path = tmp_path / "out" / "report.json"
    sample_report.dump(path)
    again = R.Report.load(path)
    assert again.to_dict() == sample_report.to_dict()
    assert again.cell("gamma", "hip:gfx942", "3.9.0").findings == []
    assert again.cell("gamma", "hip:gfx942", "3.8.0").findings[0].symbol == "tgamma"
    data = json.loads(path.read_text())
    data["schema_version"] = 99
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="unsupported report schema"):
        R.Report.load(path)


def test_static_agreement_note(sample_report):
    finding = R.Finding("libdevice-missing", "tgamma", "libdevice.tgamma is not defined in hip", "t", 3)
    sample_report.cells[3].findings = [finding]
    sample_report.cells[1].findings = [finding]  # predicted, but it compiled
    text = render_terminal(sample_report)
    assert "predicted 2 failing cells before compiling; 1 of them did fail" in " ".join(text.split())
    assert "predicted but compiled: axpy on hip:gfx942" in text


def test_format_bytes():
    assert R.format_bytes(None) == "?"
    assert R.format_bytes(512) == "512 B"
    assert R.format_bytes(1536) == "1.5 KiB"
    assert R.format_bytes(3 * 1024**3) == "3.0 GiB"

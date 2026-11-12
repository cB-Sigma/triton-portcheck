import json

import pytest

from conftest import make_cell
from triton_portcheck import baseline
from triton_portcheck import results as R


def test_without_baseline_every_failure_is_new(sample_report):
    verdict = baseline.judge(sample_report, None)
    assert len(verdict.new_failures) == 2
    assert not verdict.ok


def test_baseline_round_trip_then_only_new_breakage_fails(tmp_path, sample_report):
    path = tmp_path / "ci" / "baseline.json"
    assert baseline.write_baseline(sample_report, path) == 2
    data = json.loads(path.read_text())
    assert data["baseline_schema"] == 1
    assert {f["target"] for f in data["failures"]} == {"hip:gfx942"}

    known = baseline.load_baseline(path)
    verdict = baseline.judge(sample_report, known)
    assert verdict.ok
    assert len(verdict.known_failures) == 2

    sample_report.cells.append(make_cell("axpy", "cuda:90", "3.9.0", R.ASSEMBLER, fingerprint="cccccccccccc"))
    verdict = baseline.judge(sample_report, known)
    assert [c.target for c in verdict.new_failures] == ["cuda:90"]
    assert not verdict.ok


def test_fixed_cells_are_reported(sample_report):
    known = {("axpy", "cuda:80", "3.9.0"): {"status": R.MLIR, "fingerprint": "x"}}
    verdict = baseline.judge(sample_report, known)
    assert verdict.fixed == [("axpy", "cuda:80", "3.9.0")]


def test_xfail_is_expected_and_unexpected_pass_is_noted(sample_report):
    for cell in sample_report.cells:
        if cell.target == "hip:gfx942":
            cell.expected_failure = True
    verdict = baseline.judge(sample_report, None)
    assert verdict.ok
    assert len(verdict.expected_failures) == 2
    assert {c.kernel for c in verdict.unexpected_passes} == {"axpy"}


def test_full_report_can_be_used_as_baseline(tmp_path, sample_report):
    path = tmp_path / "report.json"
    sample_report.dump(path)
    assert set(baseline.load_baseline(path)) == {("gamma", "hip:gfx942", "3.8.0"), ("gamma", "hip:gfx942", "3.9.0")}


def test_foreign_json_is_rejected(tmp_path):
    path = tmp_path / "x.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="not a triton-portcheck baseline"):
        baseline.load_baseline(path)

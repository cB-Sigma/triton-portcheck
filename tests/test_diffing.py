import json

import pytest

from conftest import make_cell
from triton_portcheck import diffing
from triton_portcheck import results as R


def fail(kernel, target, version, fp="f1", status=R.FRONTEND, message="boom"):
    return make_cell(kernel, target, version, status, stage="frontend", fingerprint=fp, message=message)


def report(cells, versions=("3.8.0", "3.9.0"), targets=("cuda:80", "hip:gfx942")):
    kernels = list(dict.fromkeys(c.kernel for c in cells))
    return R.Report(kernels=kernels, targets=list(targets), versions=list(versions), cells=cells)


def test_version_diff_classifies_status_changes():
    rep = report(
        [
            make_cell("same", "cuda:80", "3.8.0"),
            make_cell("same", "cuda:80", "3.9.0"),
            make_cell("regress", "cuda:80", "3.8.0"),
            fail("regress", "cuda:80", "3.9.0", message="removed"),
            fail("fixed", "cuda:80", "3.8.0", message="missing"),
            make_cell("fixed", "cuda:80", "3.9.0"),
            fail("changed", "cuda:80", "3.8.0", "f1"),
            fail("changed", "cuda:80", "3.9.0", "f2", R.ASSEMBLER),
            fail("stillbad", "cuda:80", "3.8.0", "f1"),
            fail("stillbad", "cuda:80", "3.9.0", "f1"),
        ]
    )
    diff = diffing.diff_reports(rep, versions=("3.8.0", "3.9.0"))
    kinds = {e.row: e.kind for e in diff.entries}
    assert kinds == {
        "same @ cuda:80": "same",
        "regress @ cuda:80": "regression",
        "fixed @ cuda:80": "fixed",
        "changed @ cuda:80": "changed",
        "stillbad @ cuda:80": "same",
    }
    assert [e.row for e in diff.regressions] == ["regress @ cuda:80"]
    text = diffing.render_diff(diff)
    assert "1 regressed, 1 fixed, 1 fail differently" in text
    assert "Compiles on triton 3.8.0, fails on triton 3.9.0" in text
    assert "removed" in text
    assert "Compile-only check" in text


def test_compile_time_regression_needs_ratio_and_absolute_margin():
    a = make_cell("k", "cuda:80", "3.8.0", seconds=2.0)
    assert diffing.compare_cells(
        a, make_cell("k", "cuda:80", "3.9.0", seconds=4.0), "k", diffing.Thresholds()
    ).time_regression
    # 3x slower but only 0.2 s: noise
    quick = make_cell("k", "cuda:80", "3.8.0", seconds=0.1)
    assert not diffing.compare_cells(
        quick, make_cell("k", "cuda:80", "3.9.0", seconds=0.3), "k", diffing.Thresholds()
    ).time_regression
    # +1.2 s but only 1.2x
    slow = make_cell("k", "cuda:80", "3.8.0", seconds=6.0)
    assert not diffing.compare_cells(
        slow, make_cell("k", "cuda:80", "3.9.0", seconds=7.2), "k", diffing.Thresholds()
    ).time_regression
    assert diffing.compare_cells(
        slow, make_cell("k", "cuda:80", "3.9.0", seconds=7.2), "k", diffing.Thresholds(1.1, 1.0)
    ).time_regression


def test_size_and_metadata_changes_respect_threshold():
    a = make_cell(
        "k", "cuda:80", "3.8.0", artifact_sizes={"ptx": 1000, "cubin": 1000}, metadata={"shared": 0, "name": "k"}
    )
    b = make_cell(
        "k",
        "cuda:80",
        "3.9.0",
        artifact_sizes={"ptx": 1050, "cubin": 1500, "extra": 1},
        metadata={"shared": 64, "name": "k"},
    )
    entry = diffing.compare_cells(a, b, "k", diffing.Thresholds())
    assert entry.size_changes == {"cubin": (1000, 1500)}
    assert entry.metadata_changes == {"shared": (0, 64)}
    assert entry.interesting
    assert "cubin 1000 -> 1500 (+50%)" in diffing.render_diff(diffing.Diff("a", "b", "version", [entry]))


def test_target_diff_and_markdown():
    rep = report(
        [make_cell("k", "cuda:80", "3.9.0"), fail("k", "hip:gfx942", "3.9.0", message="no tgamma")],
        versions=("3.9.0",),
    )
    diff = diffing.diff_reports(rep, targets=("cuda:80", "hip:gfx942"))
    assert diff.axis == "target"
    assert diff.entries[0].kind == "regression"
    markdown = diffing.render_diff(diff, "markdown")
    assert markdown.startswith("## triton-portcheck diff: cuda:80 -> hip:gfx942")
    assert "- `k`: ok → frontend-error — no tgamma" in markdown


def test_two_run_diff_reports_added_and_removed_cells():
    old = report([make_cell("k", "cuda:80", "3.9.0"), make_cell("gone", "cuda:80", "3.9.0")], versions=("3.9.0",))
    new = report([fail("k", "cuda:80", "3.9.0"), make_cell("new", "cuda:80", "3.9.0")], versions=("3.9.0",))
    diff = diffing.diff_reports(old, new)
    kinds = {e.row.split(" ")[0]: e.kind for e in diff.entries}
    assert kinds == {"k": "regression", "gone": "only-a", "new": "only-b"}
    json.dumps(diff.to_dict())


def test_bad_requests_raise():
    rep = report([make_cell("k", "cuda:80", "3.9.0")], versions=("3.9.0",))
    with pytest.raises(diffing.DiffError, match="not in the report"):
        diffing.diff_reports(rep, versions=("3.8.0", "3.9.0"))
    with pytest.raises(diffing.DiffError, match="not in the report"):
        diffing.diff_reports(rep, targets=("cuda:80", "cuda:90"))
    with pytest.raises(diffing.DiffError, match="nothing to compare"):
        diffing.diff_reports(rep)


def test_normalise_ir_strips_locations_and_debug_noise():
    ttgir = '%1 = tt.load %0 : tensor<4xf32> loc(#loc3)\n#loc3 = loc("k.py":3:4)\n'
    assert diffing.normalise_ir(ttgir, "ttgir") == "%1 = tt.load %0 : tensor<4xf32>\n"
    llir = '; ModuleID = "x"\n  %2 = add i32 %0, %1, !dbg !17\n!17 = !DILocation(line: 3)\n'
    assert diffing.normalise_ir(llir, "llir") == "  %2 = add i32 %0, %1\n"
    ptx = "// header comment\n\t.loc\t1 3 4\n\tadd.s32 %r1, %r2, %r3; // comment\n"
    assert diffing.normalise_ir(ptx, "ptx") == "\tadd.s32 %r1, %r2, %r3;\n"
    amd = '\t.file\t1 "k.py"\n\ts_add_u32 s0, s0, s1 ; note\n'
    assert diffing.normalise_ir(amd, "amdgcn") == "\ts_add_u32 s0, s0, s1\n"


def test_ir_diff_reads_saved_files(tmp_path):
    for version, body in (("3.8.0", "a\nb\n"), ("3.9.0", "a\nc\n")):
        folder = tmp_path / version
        folder.mkdir()
        (folder / "kernel.ttgir").write_text(body)
    a = make_cell("k", "cuda:80", "3.8.0", ir_dir="3.8.0")
    b = make_cell("k", "cuda:80", "3.9.0", ir_dir="3.9.0")
    rep = report([a, b], targets=("cuda:80",))
    diff = diffing.diff_reports(rep, versions=("3.8.0", "3.9.0"), ir_stage="ttgir", root_a=tmp_path)
    assert "-b\n+c\n" in diff.entries[0].ir_diff
    assert "```diff" in diffing.render_diff(diff, "markdown")
    short = diffing.diff_reports(rep, versions=("3.8.0", "3.9.0"), ir_stage="ttgir", root_a=tmp_path, ir_limit=2)
    assert "more diff lines not shown" in short.entries[0].ir_diff
    none = diffing.diff_reports(rep, versions=("3.8.0", "3.9.0"), ir_stage="llir", root_a=tmp_path)
    assert none.entries[0].ir_diff == ""

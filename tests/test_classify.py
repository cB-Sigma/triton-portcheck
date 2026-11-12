import pytest

from triton_portcheck import classify
from triton_portcheck import results as R

PTXAS = """PTXAS error: Internal Triton PTX codegen error
`ptxas` stderr:
ptxas-blackwell /tmp/tmpp6ns1hqp.ptx, line 206; error   : Feature '.tile::scatter4' not supported on .target 'sm_90a'
ptxas-blackwell /tmp/tmpp6ns1hqp.ptx, line 211; error   : Feature '.tile::scatter4' not supported on .target 'sm_90a'
ptxas-blackwell fatal   : Ptx assembly aborted due to errors

Repro command: /x/ptxas-blackwell -lineinfo -v --gpu-name=sm_90a /tmp/tmpp6ns1hqp.ptx -o /tmp/tmpp6ns1hqp.ptx.o
"""

MLIR_STDERR = """/src/k.py:48:9: error: 'arith.extf' op operand #0 must be floating-point-like, but got 'tensor<32x32xi8, #ttg.dot_op<{opIdx = 0, parent = #ttg.blocked<{sizePerThread = [2, 2]}>}>>'
    c = tl.dot(a, b)
        ^
/src/k.py:44:1: error: Failures have been detected while processing an MLIR pass pipeline
/src/k.py:44:1: note: Pipeline failed while executing [`TritonGPUAccelerateMatmul` on 'builtin.module' operation]: reproducer
"""


@pytest.mark.parametrize(
    ("stage", "status"),
    [
        ("load", R.LOAD),
        ("options", R.FRONTEND),
        ("frontend", R.FRONTEND),
        ("ttir", R.MLIR),
        ("ttgir", R.MLIR),
        ("llir", R.MLIR),
        ("ptx", R.CODEGEN),
        ("amdgcn", R.CODEGEN),
        ("cubin", R.ASSEMBLER),
        ("hsaco", R.ASSEMBLER),
        (None, R.CRASH),
    ],
)
def test_stage_to_status(stage, status):
    assert classify.status_for_stage(stage) == status


def test_unknown_backend_stages_are_positional():
    order = ["frontend", "ttir", "spirv", "zebin"]
    assert classify.status_for_stage("zebin", order) == R.ASSEMBLER
    assert classify.status_for_stage("spirv", order) == R.CODEGEN


def classified(status, stage, error, stderr=""):
    cell = R.Cell("k", "cuda:90", "3.9.0", status=status, stage=stage)
    classify.apply_classification(cell, error, stderr)
    return cell


def test_libdevice_missing_on_hip():
    error = {
        "type": "triton.compiler.errors.CompilationError",
        "message": "at 6:29:\ndef k():\n    ^\nAttributeError(\"module 'triton.language.extra.hip.libdevice' has no attribute 'tgamma'\")",
        "short": "AttributeError(\"module 'triton.language.extra.hip.libdevice' has no attribute 'tgamma'\")",
    }
    cell = classified(R.FRONTEND, "frontend", error)
    assert cell.signature == "libdevice-missing"
    assert cell.message == "libdevice.tgamma is not implemented for the hip backend"
    assert cell.issue_url.endswith("/issues/10375")
    assert "tgamma" in cell.explanation


def test_fp8_dtype_unsupported():
    short = "ValueError(\"type fp8e4nv not supported in this architecture. The supported fp8 dtypes are ('fp8e4b15', 'fp8e5')\")"
    cell = classified(R.FRONTEND, "frontend", {"message": "at 1:0:\n" + short, "short": short})
    assert cell.signature == "fp8-dtype-unsupported"
    assert "fp8e4nv" in cell.message
    assert cell.issue_url == ""


def test_ptxas_scatter_links_issue_and_drops_ptx_dump():
    cell = classified(R.ASSEMBLER, "cubin", {"type": "triton.runtime.errors.PTXASError", "message": PTXAS})
    assert cell.signature == "ptxas-unsupported-feature"
    assert cell.message == "ptxas: .tile::scatter4 is not supported on sm_90a"
    assert cell.issue_url.endswith("/issues/11344")


def test_other_ptxas_feature_has_no_issue_link():
    message = PTXAS.replace(".tile::scatter4", ".some::thing")
    cell = classified(R.ASSEMBLER, "cubin", {"message": message})
    assert cell.signature == "ptxas-unsupported-feature"
    assert cell.issue_url == ""


def test_mlir_failure_uses_stderr_diagnostic_and_pass_name():
    cell = classified(
        R.MLIR, "ttgir", {"type": "builtins.RuntimeError", "message": "PassManager::run failed"}, MLIR_STDERR
    )
    assert cell.signature == "mlir-pass-failed"
    assert cell.message.startswith("'arith.extf' op operand #0 must be floating-point-like")
    assert "(pass TritonGPUAccelerateMatmul)" in cell.message
    assert "#ttg" not in cell.message


def test_llvm_fatal_error_is_surfaced():
    stderr = "LLVM ERROR: Unsupported rounding mode for conversion.\n/x.py:1:1: error: Failures have been detected while processing an MLIR pass pipeline\n"
    cell = classified(R.MLIR, "llir", {"message": "PassManager::run failed"}, stderr)
    assert cell.signature == "fp8-rounding-mode"
    assert cell.issue_url.endswith("/issues/12726")


def test_version_specific_api_signatures():
    removed = classified(
        R.FRONTEND, "frontend", {"short": "Block pointers have been removed in favor of the tensor descriptor API"}
    )
    assert removed.signature == "api-removed"
    assert "tensor descriptor API" in removed.explanation
    missing = classified(
        R.FRONTEND, "frontend", {"short": "AttributeError(\"module 'triton.language' has no attribute 'atomic_load'\")"}
    )
    assert missing.signature == "language-attribute-missing"
    assert missing.message == "triton.language.atomic_load does not exist in this Triton version"


def test_unknown_error_keeps_cleaned_message():
    cell = classified(
        R.FRONTEND, "options", {"type": "builtins.ValueError", "message": "num_ctas > 1 not supported on gfx942"}
    )
    assert cell.signature == ""
    assert cell.message == "num_ctas > 1 not supported on gfx942"
    assert len(cell.fingerprint) == 12


def test_crash_message_from_stderr_and_generic_explanation():
    cell = classified(R.CRASH, "llir", None, "warning: x\nLLVM ERROR: out of memory\nStack dump:\n")
    assert cell.message == "LLVM ERROR: out of memory"
    assert "exited abnormally" in cell.explanation


def test_fingerprint_is_stable_across_targets_paths_and_numbers():
    a = classify.fingerprint(R.ASSEMBLER, "cubin", "ptxas: .tile::scatter4 is not supported on sm_90a")
    b = classify.fingerprint(R.ASSEMBLER, "cubin", "ptxas: .tile::scatter4 is not supported on sm_120a")
    assert a == b
    c = classify.fingerprint(R.MLIR, "ttgir", "/home/a/k.py:12:3 bad tensor<32x32xi8> at 0xdeadbeef %12")
    d = classify.fingerprint(R.MLIR, "ttgir", "/tmp/b/other.py:99:1 bad tensor<64x64xi8> at 0x1234 %7")
    assert c == d
    assert a != classify.fingerprint(R.ASSEMBLER, "cubin", "ptxas: something else")
    assert a != classify.fingerprint(R.CODEGEN, "ptx", "ptxas: .tile::scatter4 is not supported on sm_90a")


def test_long_messages_are_truncated():
    cell = classified(R.FRONTEND, "frontend", {"short": "x" * 5000})
    assert len(cell.message) <= 300

"""Turn raw compiler output into a status, a short message and a stable fingerprint.

The known-signature table only links an upstream issue where the issue text
was read and the failure was reproduced with a released Triton wheel.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any

from . import results as R

_ISSUES = "https://github.com/triton-lang/triton/issues/"

#: Stage name -> status when that stage raised.
STAGE_STATUS = {
    "import": R.LOAD,
    "target": R.LOAD,
    "load": R.LOAD,
    "specialise": R.LOAD,
    "harness": R.LOAD,
    "options": R.FRONTEND,
    "frontend": R.FRONTEND,
    "ttir": R.MLIR,
    "ttgir": R.MLIR,
    "llir": R.MLIR,
    "ptx": R.CODEGEN,
    "amdgcn": R.CODEGEN,
    "cubin": R.ASSEMBLER,
    "hsaco": R.ASSEMBLER,
}


def status_for_stage(stage: str | None, stage_order: list[str] | None = None) -> str:
    """Map the failing stage to a status; unknown stages of other backends are positional."""
    if stage is None:
        return R.CRASH
    if stage in STAGE_STATUS:
        return STAGE_STATUS[stage]
    if stage_order and stage == stage_order[-1]:
        return R.ASSEMBLER
    return R.CODEGEN


@dataclass(frozen=True)
class Signature:
    id: str
    pattern: re.Pattern[str]
    title: str
    explanation: str
    issue: str = ""

    def render(self, match: re.Match[str]) -> tuple[str, str]:
        groups = {k: v or "" for k, v in match.groupdict().items()}
        return self.title.format(**groups), self.explanation.format(**groups)


SIGNATURES: tuple[Signature, ...] = (
    Signature(
        "libdevice-missing",
        re.compile(r"module 'triton\.language\.extra\.(?P<backend>\w+)\.libdevice' has no attribute '(?P<symbol>\w+)'"),
        "libdevice.{symbol} is not implemented for the {backend} backend",
        "The kernel calls libdevice.{symbol}; Triton maps triton.language.extra.libdevice to a per-backend "
        "module and the {backend} one does not define this function. Guard the call with a target check or "
        "express it with functions both backends have.",
        _ISSUES + "10375",
    ),
    Signature(
        "fp8-dtype-unsupported",
        re.compile(r"type (?P<symbol>fp8\w+) not supported in this architecture"),
        "dtype {symbol} is not supported on this architecture",
        "The backend's supported_fp8_dtypes for this target does not include {symbol}, so the frontend rejects "
        "any kernel that names it (on NVIDIA, fp8e4nv needs compute capability 8.9 or newer).",
    ),
    Signature(
        "ptxas-unsupported-feature",
        re.compile(r"Feature '(?P<symbol>\.tile::scatter\w*)' not supported on \.target '(?P<backend>[^']+)'"),
        "ptxas: {symbol} is not supported on {backend}",
        "Triton lowered a tensor-descriptor scatter to an instruction this target does not implement; the error "
        "only appears when the assembler runs.",
        _ISSUES + "11344",
    ),
    Signature(
        "ptxas-unsupported-feature",
        re.compile(r"Feature '(?P<symbol>[^']+)' not supported on \.target '(?P<backend>[^']+)'"),
        "ptxas: {symbol} is not supported on {backend}",
        "Triton emitted PTX that uses a feature the assembler does not accept for this target.",
    ),
    Signature(
        "fp8-arithmetic-lowering",
        re.compile(r"'llvm\.f\w+' op operand #\d+ must be floating point LLVM type.*but got 'i8'"),
        "arithmetic on fp8 values fails in LLVM lowering",
        "Operations on two operands of the same fp8 type pass the frontend and fail when lowered, because fp8 "
        "is stored as i8. Cast to fp16 or wider before computing.",
        _ISSUES + "11898",
    ),
    Signature(
        "fp8-rounding-mode",
        re.compile(r"LLVM ERROR: Unsupported rounding mode for conversion"),
        "fp8 downcast with this rounding mode is not lowered",
        "The frontend accepts fp_downcast_rounding for this fp8 conversion but the NVIDIA lowering has no "
        "implementation for it.",
        _ISSUES + "12726",
    ),
    Signature(
        "api-removed",
        re.compile(r"(?P<symbol>Block pointers) have been removed in favor of (?P<backend>.+)"),
        "block pointers were removed from this Triton version",
        "tl.make_block_ptr / tl.advance no longer exist in this release; Triton points to {backend} instead.",
    ),
    Signature(
        "language-attribute-missing",
        re.compile(r"module 'triton\.language(?P<backend>[\w.]*)' has no attribute '(?P<symbol>\w+)'"),
        "triton.language{backend}.{symbol} does not exist in this Triton version",
        "The kernel uses an API this Triton release does not provide (added later or removed earlier).",
    ),
    Signature(
        "mlir-pass-failed",
        re.compile(r"Pipeline failed while executing \[`(?P<symbol>\w+)`"),
        "MLIR pass {symbol} failed",
        "A compiler pass rejected the IR for this target. This is usually an unsupported combination of "
        "operation, dtype and architecture rather than a mistake in the kernel.",
    ),
)

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _first(pattern: str, text: str, flags: int = 0) -> str | None:
    match = re.search(pattern, text, flags)
    return match.group(1).strip() if match else None


def _unwrap_repr(text: str) -> str:
    """``AttributeError("x")`` -> ``x`` (Triton stores the repr of the cause)."""
    match = re.fullmatch(r"\w+\((['\"])(.*)\1\)", text.strip(), re.DOTALL)
    return match.group(2).replace("\\'", "'") if match else text.strip()


def clean_message(error: dict[str, Any] | None, stderr: str) -> str:
    """One readable line describing the failure."""
    stderr = _ANSI.sub("", stderr or "")
    error = error or {}
    message = str(error.get("message") or "")
    short = error.get("short")
    if "ptxas" in message and "error" in message:
        lines = re.findall(r"^ptxas\S*\s+(?:\S+, line \d+; )?(?:error|fatal)\s*: (.+)$", message, re.MULTILINE)
        unique = list(dict.fromkeys(line.strip() for line in lines if "aborted due to errors" not in line))
        if unique:
            return "ptxas: " + "; ".join(unique[:3])
    if "PassManager::run failed" in message or (not message and stderr):
        fatal = _first(r"(LLVM ERROR: .+)", stderr)
        diagnostics = [d for d in re.findall(r"(?:^|: )error: (.+)", stderr) if "Failures have been detected" not in d]
        failed_pass = _first(r"Pipeline failed while executing \[`(\w+)`", stderr)
        head = fatal or (diagnostics[0] if diagnostics else None)
        if head:
            return _truncate(head + (f" (pass {failed_pass})" if failed_pass else ""))
        if failed_pass:
            return f"MLIR pass {failed_pass} failed"
    if isinstance(short, str) and short.strip():
        return _truncate(_unwrap_repr(short))
    if message.strip():
        last = [line for line in message.strip().splitlines() if line.strip()]
        text = last[-1] if error.get("type", "").endswith("CompilationError") else last[0]
        return _truncate(_unwrap_repr(text))
    for pattern in (r"(LLVM ERROR: .+)", r"(.*Assertion .+ failed.*)", r"(.*[Ss]egmentation fault.*)"):
        found = _first(pattern, stderr)
        if found:
            return _truncate(found)
    tail = [line for line in stderr.strip().splitlines() if line.strip()]
    return _truncate(tail[-1]) if tail else ""


def _truncate(text: str, limit: int = 300) -> str:
    text = " ".join(text.split())
    text = re.sub(r"(tensor<[^,>']+), #[^']*>", r"\1, ...>", text)  # layout attributes are noise here
    return text if len(text) <= limit else text[: limit - 1] + "…"


def normalise_for_fingerprint(text: str) -> str:
    text = re.sub(r"(/[\w.\-+]+)+(\.\w+)?", "<path>", text)
    text = re.sub(r"0x[0-9a-fA-F]+", "<hex>", text)
    text = re.sub(r"\bsm_?\d+[a-z]?\b", "<sm>", text)
    text = re.sub(r"\bgfx\w+\b", "<gfx>", text)
    text = re.sub(r"%[\w.]+", "%v", text)
    text = re.sub(r"\d+", "N", text)
    return " ".join(text.split()).lower()


def fingerprint(status: str, stage: str | None, message: str) -> str:
    """Stable id for grouping the same failure across targets and versions."""
    basis = f"{status}|{stage or ''}|{normalise_for_fingerprint(message)}"
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()[:12]


def match_signature(*texts: str) -> tuple[str, str, str, str] | None:
    """Return ``(id, title, explanation, issue_url)`` of the first known signature."""
    haystack = "\n".join(t for t in texts if t)
    for signature in SIGNATURES:
        match = signature.pattern.search(haystack)
        if match:
            title, explanation = signature.render(match)
            return signature.id, title, explanation, signature.issue
    return None


GENERIC_EXPLANATION = {
    R.TIMEOUT: "The compile did not finish within the time limit. Large tiles can take minutes in ptxas; raise "
    "--timeout if this is expected.",
    R.OOM: "The compiler process tree exceeded the memory limit (--max-rss) and was killed.",
    R.CRASH: "The compiler process exited abnormally. The stage shown is the last one that started.",
}


def apply_classification(cell: R.Cell, error: dict[str, Any] | None, stderr: str) -> None:
    """Fill ``message``, ``fingerprint`` and the known-signature fields of a failed cell."""
    if not cell.message:
        cell.message = clean_message(error, stderr)
    raw = "\n".join(str((error or {}).get(k) or "") for k in ("message", "short", "cause_message"))
    known = match_signature(raw, _ANSI.sub("", stderr or ""), cell.message)
    if known:
        cell.signature, title, cell.explanation, cell.issue_url = known
        if cell.signature != "mlir-pass-failed":
            cell.message = title
    elif cell.status in GENERIC_EXPLANATION:
        cell.explanation = GENERIC_EXPLANATION[cell.status]
    cell.fingerprint = fingerprint(cell.status, cell.stage, cell.message)

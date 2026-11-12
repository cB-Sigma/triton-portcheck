"""Result model shared by the runner, the renderers and the diff engine."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from . import DISCLAIMER, __version__

SCHEMA_VERSION = 1

OK = "ok"
FRONTEND = "frontend-error"
MLIR = "mlir-error"
CODEGEN = "codegen-error"
ASSEMBLER = "assembler-error"
TIMEOUT = "timeout"
OOM = "oom"
CRASH = "crash"
LOAD = "load-error"
PREDICTED = "predicted-fail"
SKIPPED = "skipped"

STATUSES = (OK, FRONTEND, MLIR, CODEGEN, ASSEMBLER, TIMEOUT, OOM, CRASH, LOAD, PREDICTED, SKIPPED)
FAILING = frozenset({FRONTEND, MLIR, CODEGEN, ASSEMBLER, TIMEOUT, OOM, CRASH, LOAD, PREDICTED})

#: Short labels used in the matrix views.
SHORT = {
    OK: "ok",
    FRONTEND: "front",
    MLIR: "mlir",
    CODEGEN: "llvm",
    ASSEMBLER: "asm",
    TIMEOUT: "time",
    OOM: "oom",
    CRASH: "crash",
    LOAD: "load",
    PREDICTED: "pred",
    SKIPPED: "-",
}

STATUS_HELP = {
    OK: "compiled to a device binary",
    FRONTEND: "rejected while turning Python into Triton IR",
    MLIR: "a Triton/MLIR pass or the lowering to LLVM IR failed",
    CODEGEN: "the LLVM backend failed to emit assembly",
    ASSEMBLER: "the assembler or linker (ptxas / AMD lld) rejected the code",
    TIMEOUT: "the compile exceeded the time limit and was killed",
    OOM: "the compile exceeded the memory limit and was killed",
    CRASH: "the compiler process died (abort or signal)",
    LOAD: "the kernel could not be imported or specialised",
    PREDICTED: "not compiled: a table in the installed Triton says it cannot work",
    SKIPPED: "not run",
}


@dataclass
class Finding:
    """A static, table-derived prediction for one cell."""

    rule: str
    symbol: str
    detail: str
    table: str = ""
    line: int | None = None


@dataclass
class Cell:
    """One kernel specialisation x target x Triton version."""

    kernel: str
    target: str
    triton: str
    status: str = SKIPPED
    stage: str | None = None
    message: str = ""
    fingerprint: str = ""
    signature: str = ""
    explanation: str = ""
    issue_url: str = ""
    seconds: float | None = None
    wall_seconds: float | None = None
    stage_seconds: dict[str, float] = field(default_factory=dict)
    artifact_sizes: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    peak_rss_bytes: int | None = None
    file: str | None = None
    line: int | None = None
    column: int | None = None
    notes: list[str] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    expected_failure: bool = False
    detail: str = ""
    ir_dir: str | None = None

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.kernel, self.target, self.triton)

    @property
    def failed(self) -> bool:
        return self.status in FAILING

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Cell:
        data = dict(data)
        findings = [Finding(**f) for f in data.pop("findings", [])]
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(findings=findings, **known)


@dataclass
class Report:
    """A full matrix run."""

    kernels: list[str]
    targets: list[str]
    versions: list[str]
    cells: list[Cell]
    environments: dict[str, dict[str, Any]] = field(default_factory=dict)
    created: str = ""
    tool_version: str = __version__
    schema_version: int = SCHEMA_VERSION
    disclaimer: str = DISCLAIMER
    host: dict[str, Any] = field(default_factory=dict)
    settings: dict[str, Any] = field(default_factory=dict)

    def cell(self, kernel: str, target: str, triton: str) -> Cell | None:
        for cell in self.cells:
            if cell.key == (kernel, target, triton):
                return cell
        return None

    def index(self) -> dict[tuple[str, str, str], Cell]:
        return {cell.key: cell for cell in self.cells}

    def failures(self) -> list[Cell]:
        return [cell for cell in self.cells if cell.failed]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def dump(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=False) + "\n", encoding="utf-8")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Report:
        if data.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported report schema {data.get('schema_version')!r}; this build reads schema {SCHEMA_VERSION}"
            )
        data = dict(data)
        cells = [Cell.from_dict(c) for c in data.pop("cells")]
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(cells=cells, **known)

    @classmethod
    def load(cls, path: str | Path) -> Report:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def format_bytes(count: float | None) -> str:
    if count is None:
        return "?"
    value = float(count)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024 or unit == "GiB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GiB"

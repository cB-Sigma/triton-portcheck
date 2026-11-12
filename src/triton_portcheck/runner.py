"""Matrix runner: every cell compiles in its own child process.

The parent never imports Triton.  A cell is one ``_child.py compile`` process,
started in its own session so the whole process tree (including ``ptxas``) can
be killed on timeout or when it exceeds the memory limit.
"""

from __future__ import annotations

import contextlib
import datetime as _dt
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import classify, envs
from . import results as R
from .manifest import KernelSpec, Manifest
from .targets import matches

CHILD = Path(__file__).with_name("_child.py")
_PAGE = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096
_SIZE_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([kmgt]?)i?b?\s*$", re.IGNORECASE)


class RunnerError(RuntimeError):
    """Raised when the run cannot start (no Triton, bad environment...)."""


def parse_size(text: str | int | None) -> int | None:
    """``"8G"`` -> bytes; ``None``/``"0"``/``"off"`` disable the limit."""
    if text is None:
        return None
    if isinstance(text, int):
        return text or None
    if text.strip().lower() in ("", "0", "off", "none"):
        return None
    match = _SIZE_RE.match(text)
    if not match:
        raise ValueError(f"invalid size {text!r}: use something like 8G or 512M")
    factor = {"": 1, "k": 1024, "m": 1024**2, "g": 1024**3, "t": 1024**4}[match[2].lower()]
    return int(float(match[1]) * factor)


@dataclass
class ChildOutcome:
    events: list[dict[str, Any]]
    returncode: int | None
    stderr: str
    stdout: str
    wall_seconds: float
    timed_out: bool = False
    over_memory: bool = False
    peak_rss_bytes: int | None = None

    @property
    def payload(self) -> dict[str, Any] | None:
        for event in reversed(self.events):
            if event.get("ev") == "result":
                payload = event.get("payload")
                return payload if isinstance(payload, dict) else None
        return None

    def last_stage(self) -> tuple[str | None, list[str]]:
        """Last stage that began without ending, and the order of stages seen."""
        open_stage: str | None = None
        order: list[str] = []
        for event in self.events:
            if event.get("ev") == "stage_begin":
                open_stage = event["stage"]
                order.append(event["stage"])
            elif event.get("ev") == "stage_end":
                open_stage = None
        return open_stage, order


def _session_rss(session_id: int) -> int | None:
    """Sum of resident memory of every process in a session (Linux /proc only)."""
    total = 0
    found = False
    try:
        entries = os.listdir("/proc")
    except OSError:
        return None
    for entry in entries:
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/stat", encoding="ascii", errors="replace") as handle:
                fields = handle.read().rpartition(")")[2].split()
            if int(fields[3]) != session_id:
                continue
            with open(f"/proc/{entry}/statm", encoding="ascii") as handle:
                total += int(handle.read().split()[1]) * _PAGE
            found = True
        except (OSError, ValueError, IndexError):
            continue
    return total if found else None


def _kill_tree(proc: subprocess.Popen[bytes]) -> None:
    with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
        if hasattr(os, "killpg"):
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()


def run_child(
    python: Path,
    mode: str,
    request: dict[str, Any],
    *,
    timeout: float | None = None,
    max_rss: int | None = None,
    extra_env: dict[str, str] | None = None,
    cwd: Path | None = None,
    inherit_output: bool = False,
) -> ChildOutcome:
    """Run ``_child.py <mode>`` and collect its event stream."""
    workdir = Path(tempfile.mkdtemp(prefix="tpc-"))
    try:
        request_path = workdir / "request.json"
        events_path = workdir / "events.jsonl"
        request_path.write_text(json.dumps(request), encoding="utf-8")
        events_path.touch()
        env = dict(os.environ)
        env.pop("TRITON_INTERPRET", None)
        env.update(
            {
                "TRITON_CACHE_DIR": str(workdir / "cache"),
                "TRITON_ALWAYS_COMPILE": "1",
                "TRITON_DUMP_PTXAS_LOG": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
                "TMPDIR": str(workdir),
            }
        )
        env.update(extra_env or {})
        command = [str(python), str(CHILD), mode, str(request_path), str(events_path)]
        start = time.monotonic()
        timed_out = over_memory = False
        peak: int | None = None
        with open(workdir / "stdout", "wb") as out, open(workdir / "stderr", "wb") as err:
            try:
                proc = subprocess.Popen(
                    command,
                    stdout=None if inherit_output else out,
                    stderr=None if inherit_output else err,
                    stdin=subprocess.DEVNULL,
                    env=env,
                    cwd=cwd,
                    start_new_session=True,
                )
            except OSError as exc:
                raise RunnerError(f"cannot start {python}: {exc}") from exc
            last_poll = 0.0
            while True:
                try:
                    proc.wait(timeout=0.05)
                    break
                except subprocess.TimeoutExpired:
                    pass
                now = time.monotonic()
                if timeout is not None and now - start > timeout:
                    timed_out = True
                elif max_rss is not None and now - last_poll >= 0.2:
                    last_poll = now
                    rss = _session_rss(proc.pid)
                    if rss is not None:
                        peak = max(peak or 0, rss)
                        over_memory = rss > max_rss
                if timed_out or over_memory:
                    _kill_tree(proc)
                    proc.wait()
                    break
            _kill_tree(proc)  # reap stragglers such as an orphaned assembler
        wall = time.monotonic() - start
        events = []
        for line in events_path.read_text(encoding="utf-8", errors="replace").splitlines():
            with contextlib.suppress(json.JSONDecodeError):
                events.append(json.loads(line))
        return ChildOutcome(
            events=events,
            returncode=proc.returncode,
            stderr=(workdir / "stderr").read_text(encoding="utf-8", errors="replace"),
            stdout=(workdir / "stdout").read_text(encoding="utf-8", errors="replace"),
            wall_seconds=round(wall, 3),
            timed_out=timed_out,
            over_memory=over_memory,
            peak_rss_bytes=peak,
        )
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def probe(python: Path, targets: list[str] | None = None) -> dict[str, Any]:
    """Describe the Triton installed for ``python`` (version, backends, tables)."""
    outcome = run_child(python, "probe", {"targets": targets or []}, timeout=180)
    payload = outcome.payload
    if payload is None:
        tail = outcome.stderr.strip().splitlines()[-3:]
        raise RunnerError(f"could not probe {python}: " + (" | ".join(tail) or f"exit code {outcome.returncode}"))
    return payload


# --------------------------------------------------------------------------
# Environments
# --------------------------------------------------------------------------


@dataclass
class ResolvedEnv:
    label: str
    python: Path
    managed: bool
    info: dict[str, Any]
    fallback_sys_path: list[str] = field(default_factory=list)


def resolve_envs(
    versions: list[str],
    targets: list[str],
    *,
    create: bool = True,
    inherit_site: bool = True,
    log: Callable[[str], None] = lambda _msg: None,
) -> list[ResolvedEnv]:
    """Map version requests (``current`` or ``X.Y.Z``) to interpreters."""
    base = envs.base_python()
    base_info: dict[str, Any] | None = None

    def base_probe() -> dict[str, Any]:
        nonlocal base_info
        if base_info is None:
            base_info = probe(base, targets)
        return base_info

    resolved: list[ResolvedEnv] = []
    for spec in versions or ["current"]:
        if spec == "current":
            info = base_probe()
            if "triton_version" not in info:
                raise RunnerError(
                    f"Triton is not importable from {base} ({info.get('error', 'unknown error')}). Install it "
                    "there (pip install triton), point TRITON_PORTCHECK_PYTHON at an interpreter that has it, "
                    "or name a release to manage, e.g. --triton 3.9.0"
                )
            entry = ResolvedEnv(info["triton_version"], base, False, info)
        elif base_probe().get("triton_version") == spec:
            entry = ResolvedEnv(spec, base, False, base_probe())
        else:
            path = envs.env_path(spec)
            if not envs.env_python(path).exists():
                if not create:
                    raise RunnerError(
                        f"no environment for triton {spec}; create it with `triton-portcheck env add {spec}`"
                    )
                envs.add_env(spec, log=log)
            python = envs.env_python(path)
            info = probe(python, targets)
            if "triton_version" not in info:
                raise RunnerError(f"environment {path} is broken: {info.get('error')}")
            fallback: list[str] = []
            same_minor = info["python"].split(".")[:2] == base_probe()["python"].split(".")[:2]
            if inherit_site and same_minor:
                fallback = list(base_probe()["site_paths"])
            entry = ResolvedEnv(info["triton_version"], python, True, info, fallback)
        if any(e.label == entry.label for e in resolved):
            continue
        resolved.append(entry)
    return resolved


# --------------------------------------------------------------------------
# Cells
# --------------------------------------------------------------------------


@dataclass
class RunSettings:
    timeout: float = 300.0
    max_rss: int | None = 8 * 1024**3
    jobs: int = 2
    static: str = "annotate"  # off | annotate | skip | only
    save_ir: Path | None = None
    sys_path: list[str] = field(default_factory=list)


def cell_slug(cell: R.Cell) -> str:
    return re.sub(r"[^A-Za-z0-9_.+-]+", "_", f"{cell.triton}/{cell.target}/{cell.kernel}".replace("/", "__"))


def outcome_to_cell(cell: R.Cell, outcome: ChildOutcome, timeout: float, max_rss: int | None) -> None:
    """Fill a cell from what the child reported (or failed to report)."""
    payload = outcome.payload
    cell.wall_seconds = outcome.wall_seconds
    open_stage, order = outcome.last_stage()
    error: dict[str, Any] | None = None
    if outcome.timed_out or outcome.over_memory:
        cell.status = R.TIMEOUT if outcome.timed_out else R.OOM
        cell.stage = open_stage or (order[-1] if order else None)
        cell.peak_rss_bytes = outcome.peak_rss_bytes
        if outcome.timed_out:
            cell.message = f"compile exceeded the {timeout:g} s limit"
        else:
            cell.message = f"compile exceeded the {R.format_bytes(max_rss)} memory limit"
    elif payload is None:
        cell.status = R.CRASH
        cell.stage = open_stage or (order[-1] if order else None)
        code = outcome.returncode
        if code is not None and code < 0:
            with contextlib.suppress(ValueError):
                cell.detail = f"killed by {signal.Signals(-code).name}"
        cell.detail = cell.detail or f"exit code {code}"
    else:
        for key in ("seconds", "stage_seconds", "artifact_sizes", "metadata", "notes"):
            if payload.get(key) is not None:
                setattr(cell, key, payload[key])
        if payload.get("ptxas"):
            cell.metadata = {**cell.metadata, **{f"ptxas_{k}": v for k, v in payload["ptxas"].items()}}
        cell.peak_rss_bytes = payload.get("peak_rss_bytes") or outcome.peak_rss_bytes
        location = payload.get("location") or {}
        cell.file, cell.line = location.get("file"), location.get("line")
        if payload.get("ok"):
            cell.status = R.OK
            return
        error = payload.get("error") or {}
        cell.stage = payload.get("stage")
        cell.status = classify.status_for_stage(cell.stage, order)
        if error.get("line"):
            cell.line, cell.column = error["line"], error.get("column")
    classify.apply_classification(cell, error, outcome.stderr)
    if cell.status == R.CRASH and cell.detail:
        cell.message = f"{cell.message} ({cell.detail})" if cell.message else f"compiler process died: {cell.detail}"
    if not cell.message:
        cell.message = R.STATUS_HELP[cell.status]
    cell.detail = _detail(error, outcome)


def _detail(error: dict[str, Any] | None, outcome: ChildOutcome) -> str:
    """Longer text kept in the JSON report for debugging a failure."""
    parts = []
    if error and error.get("message"):
        parts.append(str(error["message"])[:3000])
    stderr = "\n".join(
        line for line in outcome.stderr.splitlines() if "UserWarning" not in line and "warn_once" not in line
    ).strip()
    if stderr:
        parts.append("--- stderr (tail) ---\n" + stderr[-3000:])
    return "\n".join(parts)


def _static_pass(
    env: ResolvedEnv, specs: list[KernelSpec], targets: list[str], base_dir: Path, settings: RunSettings
) -> dict[str, dict[str, list[R.Finding]]]:
    request = {
        "kernels": [s.child_request() for s in specs],
        "targets": targets,
        "base_dir": str(base_dir),
        "sys_path": settings.sys_path,
        "fallback_sys_path": env.fallback_sys_path,
    }
    outcome = run_child(env.python, "static", request, timeout=300, cwd=base_dir)
    payload = outcome.payload or {}
    out: dict[str, dict[str, list[R.Finding]]] = {}
    for kernel_id, entry in (payload.get("kernels") or {}).items():
        out[kernel_id] = {
            target: [R.Finding(**{k: f.get(k) for k in ("rule", "symbol", "detail", "table", "line")}) for f in found]
            for target, found in (entry.get("targets") or {}).items()
        }
    return out


def _predicted(cell: R.Cell) -> None:
    first = cell.findings[0]
    cell.status = R.PREDICTED
    cell.stage = "static"
    cell.message = first.detail
    known = classify.match_signature(
        f"module 'triton.language.extra.{cell.target.split(':')[0]}.libdevice' has no attribute '{first.symbol}'"
        if first.rule == "libdevice-missing"
        else f"type {first.symbol} not supported in this architecture"
    )
    if known:
        cell.signature, cell.message, cell.explanation, cell.issue_url = known
    cell.fingerprint = classify.fingerprint(cell.status, cell.stage, cell.message)
    if first.line:
        cell.line = first.line


def run_matrix(
    manifest: Manifest,
    targets: list[str],
    environments: list[ResolvedEnv],
    settings: RunSettings,
    progress: Callable[[R.Cell, int, int], None] | None = None,
) -> R.Report:
    """Compile every kernel x target x version cell and return the report."""
    cells: list[R.Cell] = []
    work: list[tuple[R.Cell, KernelSpec, ResolvedEnv]] = []
    for env in environments:
        findings: dict[str, dict[str, list[R.Finding]]] = {}
        if settings.static != "off":
            findings = _static_pass(env, manifest.kernels, targets, manifest.base_dir, settings)
        for spec in manifest.kernels:
            for target in targets:
                if spec.targets is not None and target not in spec.targets:
                    continue
                cell = R.Cell(kernel=spec.name, target=target, triton=env.label)
                cell.expected_failure = any(matches(p, target, env.label) for p in spec.xfail)
                cell.findings = findings.get(spec.name, {}).get(target, [])
                cells.append(cell)
                if any(matches(p, target, env.label) for p in spec.skip):
                    cell.message = "skipped by the manifest"
                elif settings.static == "only" or (settings.static == "skip" and cell.findings):
                    if cell.findings:
                        _predicted(cell)
                    else:
                        cell.message = "not compiled (static checks only)"
                else:
                    work.append((cell, spec, env))

    def compile_cell(item: tuple[R.Cell, KernelSpec, ResolvedEnv]) -> R.Cell:
        cell, spec, env = item
        timeout = spec.timeout or settings.timeout
        request: dict[str, Any] = {
            "kernel": spec.child_request(),
            "target": cell.target,
            "base_dir": str(manifest.base_dir),
            "sys_path": settings.sys_path,
            "fallback_sys_path": env.fallback_sys_path,
        }
        if settings.save_ir is not None:
            cell.ir_dir = str(settings.save_ir / cell_slug(cell))
            request["save_ir_dir"] = str((settings.save_ir / cell_slug(cell)).resolve())
        try:
            outcome = run_child(
                env.python, "compile", request, timeout=timeout, max_rss=settings.max_rss, cwd=manifest.base_dir
            )
            outcome_to_cell(cell, outcome, timeout, settings.max_rss)
        except RunnerError as exc:
            cell.status, cell.stage, cell.message = R.LOAD, "harness", str(exc)
            cell.fingerprint = classify.fingerprint(cell.status, cell.stage, cell.message)
        return cell

    done = 0
    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=max(1, settings.jobs)) as pool:
        futures = [pool.submit(compile_cell, item) for item in work]
        for future in as_completed(futures):
            cell = future.result()
            with lock:
                done += 1
                if progress is not None:
                    progress(cell, done, len(work))

    return R.Report(
        kernels=[s.name for s in manifest.kernels],
        targets=list(targets),
        versions=[e.label for e in environments],
        cells=cells,
        environments={
            e.label: {
                "python": e.info.get("python"),
                "executable": str(e.python),
                "managed": e.managed,
                "backends": e.info.get("backends", {}),
            }
            for e in environments
        },
        created=_dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        host={"platform": platform.platform(), "machine": platform.machine(), "python": sys.version.split()[0]},
        settings={
            "timeout": settings.timeout,
            "max_rss": settings.max_rss,
            "jobs": settings.jobs,
            "static": settings.static,
        },
    )

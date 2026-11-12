"""Child-side entry point of triton-portcheck.

This file is executed *by path* inside the interpreter that has Triton
installed (possibly a per-version virtual environment in which
``triton_portcheck`` itself is not installed).  It therefore only uses the
standard library plus Triton, and never imports the rest of the package.

Usage::

    python _child.py <mode> <request.json> <events.jsonl>

Modes:

``probe``    describe the installed Triton (version, backends, targets, tables)
``compile``  compile one kernel specialisation for one target
``static``   table-driven pre-checks for many kernels and targets, no compile
``capture``  run a user command and record the specialisations it launches

Progress is written to ``events.jsonl`` one JSON object per line, flushed
immediately, so the parent still knows which stage was running when the
process is killed or crashes inside LLVM.
"""

from __future__ import annotations

import ast
import contextlib
import functools
import importlib
import inspect
import io
import json
import os
import re
import runpy
import sys
import textwrap
import time
import traceback
from typing import Any

GENERIC_LIBDEVICE = "triton.language.extra.libdevice"

_events_fh: Any = None


def emit(event: str, **fields: Any) -> None:
    """Append one event to the events file and flush it."""
    record = {"ev": event, "t": time.time(), **fields}
    if _events_fh is None:
        return
    _events_fh.write(json.dumps(record, default=_json_default) + "\n")
    _events_fh.flush()


def _json_default(value: Any) -> Any:
    if isinstance(value, (set, frozenset, tuple)):
        return list(value)
    if isinstance(value, bytes):
        return f"<{len(value)} bytes>"
    if hasattr(value, "__dict__"):
        return {k: v for k, v in vars(value).items() if not k.startswith("_")}
    return repr(value)


def _jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, default=_json_default))


# --------------------------------------------------------------------------
# Fake driver
# --------------------------------------------------------------------------


class FakeDriver:
    """Stand-in for a Triton device driver.

    ``triton.compile(..., target=...)`` happily cross-compiles without a
    device, but ``triton.language.target_info`` (``is_cuda()``, ``is_hip()``,
    ``cuda_capability_geq()`` ...) asks the *active driver* for the current
    target.  With no GPU there is no active driver, so those helpers all
    return ``False`` and target-guarded kernels silently compile the wrong
    branch.  Installing this object as the active driver makes the guards see
    the target that is actually being compiled.
    """

    def __init__(self, target: Any) -> None:
        self._target = target

    def get_current_target(self) -> Any:
        return self._target

    def get_current_device(self) -> int:
        return 0

    def get_current_stream(self, device: Any = None) -> int:
        return 0

    def get_active_torch_device(self) -> Any:
        import torch

        return torch.device("cpu")

    def __getattr__(self, name: str) -> Any:
        raise RuntimeError(
            f"triton-portcheck: the compile-only driver has no '{name}': "
            "there is no device here, kernels are compiled but never launched"
        )


def resolve_target(spec: str) -> Any:
    """Turn ``cuda:90`` / ``hip:gfx942`` / ``hip:gfx942:64`` into a GPUTarget.

    The warp size is asked from the backend's own option parser unless it is
    given explicitly.
    """
    from triton.backends.compiler import GPUTarget
    from triton.compiler.compiler import make_backend

    parts = spec.split(":")
    if len(parts) not in (2, 3):
        raise ValueError(f"target must look like 'cuda:90' or 'hip:gfx942', got {spec!r}")
    backend, arch_text = parts[0], parts[1]
    arch: Any = int(arch_text) if arch_text.isdigit() else arch_text
    if len(parts) == 3:
        warp_size = int(parts[2])
    else:
        probe_backend = make_backend(GPUTarget(backend, arch, 0))
        warp_size = int(getattr(probe_backend.parse_options({}), "warp_size", 32))
    return GPUTarget(backend, arch, warp_size)


def install_fake_driver(target: Any) -> None:
    from triton.runtime.driver import driver

    driver.set_active(FakeDriver(target))


# --------------------------------------------------------------------------
# Loading kernels
# --------------------------------------------------------------------------


def setup_sys_path(request: dict[str, Any]) -> None:
    for entry in reversed(request.get("sys_path") or []):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    for entry in request.get("fallback_sys_path") or []:
        if entry not in sys.path:
            sys.path.append(entry)


def _module_name_for_file(path: str) -> tuple[str, str]:
    """Return ``(sys.path root, dotted module name)`` for a source file.

    Walks up while ``__init__.py`` exists so files inside packages are
    imported under their real dotted name and relative imports keep working.
    """
    path = os.path.abspath(path)
    directory, filename = os.path.split(path)
    parts = [os.path.splitext(filename)[0]]
    while os.path.isfile(os.path.join(directory, "__init__.py")):
        directory, package = os.path.split(directory)
        parts.insert(0, package)
    if parts[-1] == "__init__":
        parts.pop()
    return directory, ".".join(parts)


def load_kernel(locator: str, base_dir: str) -> Any:
    """Import ``module:attr`` or ``path/to/file.py:attr`` and return the JITFunction."""
    module_part, sep, attr = locator.rpartition(":")
    if not sep or not module_part or not attr:
        raise ValueError(f"kernel must be 'module:function' or 'file.py:function', got {locator!r}")
    if module_part.endswith(".py") or "/" in module_part or os.sep in module_part:
        path = module_part if os.path.isabs(module_part) else os.path.join(base_dir, module_part)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"kernel source file not found: {path}")
        root, module_name = _module_name_for_file(path)
        if root not in sys.path:
            sys.path.insert(0, root)
    else:
        module_name = module_part
    module = importlib.import_module(module_name)
    obj = functools.reduce(getattr, attr.split("."), module)
    seen = 0
    while not hasattr(obj, "preload") and hasattr(obj, "fn") and seen < 8:
        obj = obj.fn  # unwrap triton.autotune / triton.heuristics wrappers
        seen += 1
    if not hasattr(obj, "preload") or not hasattr(obj, "params"):
        raise TypeError(
            f"{locator} is a {type(obj).__name__}, not a @triton.jit function "
            "(is TRITON_INTERPRET set, or is the name wrong?)"
        )
    return obj


def kernel_location(fn: Any) -> dict[str, Any]:
    info: dict[str, Any] = {"file": None, "line": None}
    try:
        info["file"] = inspect.getsourcefile(fn.fn)
        lines, start = inspect.getsourcelines(fn.fn)
        offset = 0
        for index, line in enumerate(lines):
            if line.lstrip().startswith("def "):
                offset = index
                break
        info["line"] = start + offset
    except (OSError, TypeError):
        pass
    return info


# --------------------------------------------------------------------------
# Building Triton's own specialisation record
# --------------------------------------------------------------------------


def _encode_constant(value: Any) -> Any:
    """Mirror ``triton.runtime.jit.serialize_specialization_data`` for one value."""
    cls = value.__class__.__name__
    if cls == "dtype":
        return str(value)
    if cls == "constexpr":
        return {"constexpr": value.value}
    if cls == "JITFunction":
        return {"jit_function": f"{value.module}:{value.fn.__qualname__}"}
    return value


def _path_key(path: list[int], names: list[str]) -> str:
    return ".".join([names[path[0]], *[str(p) for p in path[1:]]])


def build_specialization(fn: Any, backend: Any, target: Any, kernel: dict[str, Any]) -> tuple[str, list[str]]:
    """Build the JSON blob ``JITFunction.preload`` understands.

    Returns the blob and a list of human-readable notes.
    """
    from triton.runtime.jit import native_specialize_impl

    notes: list[str] = []
    names = [p.name for p in fn.params]
    signature_in = dict(kernel.get("signature") or {})
    constants_in = dict(kernel.get("constants") or {})
    attrs_in = kernel.get("attrs")
    align = kernel.get("align", True)

    known = set(names)
    for key in list(signature_in) + [k.split(".")[0] for k in constants_in] + [k.split(".")[0] for k in attrs_in or {}]:
        if key not in known:
            raise KeyError(f"'{key}' is not a parameter of {fn.fn.__qualname__}; parameters are: {', '.join(names)}")

    signature: dict[str, Any] = {}
    constant_keys: list[list[int]] = []
    constant_vals: list[Any] = []
    attrs_keys: list[list[int]] = []
    attrs_vals: list[Any] = []
    default_attr = backend.parse_attr("D")

    for index, param in enumerate(fn.params):
        name = param.name
        if name in constants_in:
            signature[name] = "constexpr"
            constant_keys.append([index])
            constant_vals.append(constants_in[name])
        elif name in signature_in:
            signature[name] = signature_in[name]
        elif param.has_default:
            if param.is_constexpr:
                signature[name] = "constexpr"
                constant_keys.append([index])
                constant_vals.append(_encode_constant(param.default))
            else:
                spec = native_specialize_impl(backend, param.default, param.is_const, True, True)
                if spec[0] == "constexpr":
                    signature[name] = "constexpr"
                    constant_keys.append([index])
                    constant_vals.append(_encode_constant(param.default))
                else:
                    signature[name] = param.annotation_type or spec[0]
            notes.append(f"parameter '{name}' not listed; using its default {param.default!r}")
        else:
            raise KeyError(
                f"no type (signature) or value (constants) given for parameter '{name}' of {fn.fn.__qualname__}"
            )
        ty = signature[name]
        if attrs_in is None and align and isinstance(ty, str) and ty.startswith("*") and default_attr:
            attrs_keys.append([index])
            attrs_vals.append(default_attr)

    for key, value in constants_in.items():
        if "." in key:  # nested constant inside a tuple argument, e.g. "cfg.1"
            head, *rest = key.split(".")
            constant_keys.append([names.index(head), *[int(r) for r in rest]])
            constant_vals.append(value)
    for key, value in (attrs_in or {}).items():
        head, *rest = key.split(".")
        attrs_keys.append([names.index(head), *[int(r) for r in rest]])
        attrs_vals.append(value)

    options = dict(kernel.get("options") or {})
    option_fields = getattr(type(backend.parse_options({})), "__dataclass_fields__", None)
    if option_fields is not None:
        for key in options:
            if key not in option_fields:
                notes.append(f"option '{key}' is not an option of the {target.backend} backend and is ignored there")

    blob = {
        "name": fn._fn_name,
        "signature": signature,
        "constant_keys": constant_keys,
        "constant_vals": constant_vals,
        "attrs_keys": attrs_keys,
        "attrs_vals": attrs_vals,
        "options": options,
        "key": "triton-portcheck",
        "target": dict(vars(target)),
    }
    return json.dumps(blob), notes


# --------------------------------------------------------------------------
# compile mode
# --------------------------------------------------------------------------


class StageRecorder:
    """Wraps every stage Triton's backend registers so we can time and name it."""

    def __init__(self, keep_text: bool) -> None:
        self.keep_text = keep_text
        self.current: str | None = None
        self.seconds: dict[str, float] = {}
        self.order: list[str] = []
        self.texts: dict[str, str] = {}

    @contextlib.contextmanager
    def stage(self, name: str) -> Any:
        self.current = name
        self.order.append(name)
        emit("stage_begin", stage=name)
        start = time.perf_counter()
        yield
        self.seconds[name] = round(time.perf_counter() - start, 6)
        emit("stage_end", stage=name, seconds=self.seconds[name])
        self.current = None

    def wrap(self, name: str, func: Any) -> Any:
        def run_stage(src: Any, metadata: Any) -> Any:
            with self.stage(name):
                out = func(src, metadata)
            if self.keep_text and not isinstance(out, bytes):
                with contextlib.suppress(Exception):
                    self.texts[name] = str(out)
            return out

        return run_stage

    def install(self, backend_cls: Any) -> None:
        from triton.compiler import compiler as tc

        recorder = self
        original_add_stages = backend_cls.add_stages

        def add_stages(self: Any, stages: Any, *args: Any, **kwargs: Any) -> Any:
            result = original_add_stages(self, stages, *args, **kwargs)
            for stage_name in list(stages):
                stages[stage_name] = recorder.wrap(stage_name, stages[stage_name])
            return result

        backend_cls.add_stages = add_stages

        original_make_ir = tc.ASTSource.make_ir

        def make_ir(self: Any, *args: Any, **kwargs: Any) -> Any:
            with recorder.stage("frontend"):
                return original_make_ir(self, *args, **kwargs)

        tc.ASTSource.make_ir = make_ir


def _exception_record(exc: BaseException, fn_line: int | None) -> dict[str, Any]:
    cause = exc.__cause__ or exc.__context__
    record: dict[str, Any] = {
        "type": f"{type(exc).__module__}.{type(exc).__qualname__}",
        "message": str(exc)[:20000],
        "short": None,
        "cause_type": None if cause is None else f"{type(cause).__module__}.{type(cause).__qualname__}",
        "cause_message": None if cause is None else str(cause)[:4000],
        "line": None,
        "column": None,
    }
    short = getattr(exc, "error_message", None)
    if isinstance(short, str):
        record["short"] = short[:4000]
    node = getattr(exc, "node", None)
    lineno = getattr(node, "lineno", None)
    if isinstance(lineno, int) and fn_line is not None:
        record["line"] = fn_line + lineno - 1
        record["column"] = getattr(node, "col_offset", 0) + 1
    tb = traceback.format_exception(type(exc), exc, exc.__traceback__)
    record["traceback"] = "".join(tb)[-6000:]
    return record


def _reported_metadata(compiled: Any, backend: Any) -> dict[str, Any]:
    """Keep only what the compiler stages reported; drop echoed options and env."""
    meta = compiled.metadata._asdict()
    echoed = set(vars(backend.parse_options({}))) | {"hash", "target", "triton_version"}
    out: dict[str, Any] = {}
    for key, value in meta.items():
        if key in echoed or key.startswith("TRITON_") or key.isupper():
            continue
        out[key] = _jsonable(value)
    return out


def _peak_rss_bytes() -> int | None:
    try:
        import resource
    except ImportError:
        return None
    scale = 1 if sys.platform == "darwin" else 1024
    own = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    children = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    return int(max(own, children) * scale)


def run_compile(request: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {"ok": False, "stage": "import", "notes": []}
    setup_sys_path(request)
    try:
        import triton
    except Exception as exc:
        result["error"] = _exception_record(exc, None)
        result["stage"] = "import"
        return result
    result["triton_version"] = triton.__version__
    emit("triton", version=triton.__version__)

    kernel = request["kernel"]
    save_dir = request.get("save_ir_dir")
    recorder = StageRecorder(keep_text=bool(save_dir))
    fn = None
    fn_line = None
    stdout_capture = io.StringIO()
    started = time.perf_counter()
    try:
        from triton.compiler.compiler import make_backend

        result["stage"] = "target"
        target = resolve_target(request["target"])
        backend = make_backend(target)
        result["target"] = _jsonable(vars(target))
        install_fake_driver(target)

        result["stage"] = "load"
        fn = load_kernel(kernel["target"], request.get("base_dir") or os.getcwd())
        location = kernel_location(fn)
        fn_line = location["line"]
        result["location"] = location
        emit("loaded", **location)

        result["stage"] = "specialise"
        blob, notes = build_specialization(fn, backend, target, kernel)
        result["notes"] = notes
        result["stage"] = "options"
        recorder.install(type(backend))
        emit("compile_begin")
        started = time.perf_counter()
        with contextlib.redirect_stdout(stdout_capture):
            compiled = fn.preload(blob)
        result["seconds"] = round(time.perf_counter() - started, 6)
        result["ok"] = True
        result["stage"] = None
        result["stage_seconds"] = recorder.seconds
        result["artifact_sizes"] = {ext: len(text) for ext, text in compiled.asm.items()}
        result["metadata"] = _reported_metadata(compiled, backend)
        result["binary_ext"] = getattr(make_backend(target), "binary_ext", None)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            for ext, text in compiled.asm.items():
                mode = "wb" if isinstance(text, bytes) else "w"
                with open(os.path.join(save_dir, f"kernel.{ext}"), mode) as handle:
                    handle.write(text)
    except BaseException as exc:
        result["seconds"] = round(time.perf_counter() - started, 6)
        if recorder.order:
            result["stage"] = recorder.current or recorder.order[-1]
        result["stage_seconds"] = recorder.seconds
        result["error"] = _exception_record(exc, fn_line)
        if save_dir and recorder.texts:
            os.makedirs(save_dir, exist_ok=True)
            for ext, text in recorder.texts.items():
                with open(os.path.join(save_dir, f"kernel.{ext}"), "w") as handle:
                    handle.write(text)
    captured = stdout_capture.getvalue()
    registers = re.findall(r"Used (\d+) registers", captured)
    if registers:
        result["ptxas"] = {"registers": max(int(r) for r in registers)}
        spills = re.findall(r"(\d+) bytes spill stores", captured)
        if spills:
            result["ptxas"]["spill_store_bytes"] = max(int(s) for s in spills)
    result["stdout_tail"] = captured[-4000:] if not result["ok"] else ""
    result["peak_rss_bytes"] = _peak_rss_bytes()
    return result


# --------------------------------------------------------------------------
# probe mode
# --------------------------------------------------------------------------


def _public_functions(module: Any) -> list[str]:
    """Top-level function names defined in a module's source file."""
    try:
        with open(module.__file__, encoding="utf-8") as handle:
            tree = ast.parse(handle.read())
    except (OSError, SyntaxError, TypeError, AttributeError):
        return sorted(n for n in dir(module) if not n.startswith("_") and callable(getattr(module, n)))
    return sorted(
        node.name for node in tree.body if isinstance(node, ast.FunctionDef) and not node.name.startswith("_")
    )


def _ptxas_gpu_names(path: str) -> set[str]:
    import subprocess

    try:
        out = subprocess.run([path, "--help"], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return set()
    return set(re.findall(r"'(sm_\d+[a-z]?)'", out))


def _cuda_capabilities(compiler_module: Any) -> tuple[list[int], str]:
    """Capabilities whose ``--gpu-name`` the bundled assembler accepts.

    Uses the backend's own ``get_ptxas`` and ``sm_arch_from_capability`` so the
    answer matches the command line Triton would really run.
    """
    get_ptxas = getattr(compiler_module, "get_ptxas", None)
    to_arch = getattr(compiler_module, "sm_arch_from_capability", None)
    if get_ptxas is None or to_arch is None:
        return [], "this Triton does not expose get_ptxas/sm_arch_from_capability"
    names_by_tool: dict[str, set[str]] = {}
    capabilities = []
    for capability in range(50, 200):
        try:
            tool = get_ptxas(capability).path
        except Exception:  # noqa: S112 - no assembler for this capability
            continue
        if tool not in names_by_tool:
            names_by_tool[tool] = _ptxas_gpu_names(tool)
        if to_arch(capability) in names_by_tool[tool]:
            capabilities.append(capability)
    return capabilities, "capabilities whose --gpu-name the bundled ptxas accepts"


def _hip_archs(triton_root: str, backend_cls: Any) -> tuple[list[str], str]:
    """Architecture names the installed AMD backend source mentions.

    Triton keeps no list of valid HIP architectures: any name is handed to
    LLVM.  We collect the names its own sources special-case, which is the
    best table-derived answer available.
    """
    from triton.backends.compiler import GPUTarget

    found: set[str] = set()
    roots = [
        os.path.join(triton_root, "backends", "amd"),
        os.path.join(triton_root, "language"),
        os.path.join(triton_root, "experimental", "gluon", "language", "amd"),
    ]
    for root in roots:
        for directory, _dirs, files in os.walk(root):
            for filename in files:
                if not filename.endswith(".py"):
                    continue
                try:
                    with open(os.path.join(directory, filename), encoding="utf-8") as handle:
                        found.update(re.findall(r"\bgfx(?:\d{2}[a-z]|\d{3,4})\b", handle.read()))
                except OSError:
                    continue
    names = []
    for name in sorted(found):
        if any(other != name and other.startswith(name) for other in found) or re.fullmatch(r"gfx1\d\d", name):
            continue  # a prefix used for family matching, e.g. "gfx110" in arch
        try:
            backend_cls(GPUTarget("hip", name, 0)).parse_options({})
        except Exception:  # noqa: S112 - the backend rejects this name
            continue
        names.append(name)
    return names, "architecture names special-cased in the installed AMD backend source; LLVM accepts others"


def _backend_tables(request_targets: list[str]) -> dict[str, Any]:
    from triton.compiler.compiler import make_backend

    tables: dict[str, Any] = {}
    for spec in request_targets:
        entry: dict[str, Any] = {}
        try:
            target = resolve_target(spec)
            backend = make_backend(target)
            options = backend.parse_options({})
            entry["warp_size"] = target.warp_size
            fp8 = getattr(options, "supported_fp8_dtypes", None)
            entry["supported_fp8_dtypes"] = None if fp8 is None else sorted(fp8)
            module_map = backend.get_module_map()
            impl = module_map.get(GENERIC_LIBDEVICE)
            entry["libdevice_module"] = None if impl is None else impl.__name__
        except Exception as exc:
            entry["error"] = f"{type(exc).__name__}: {exc}"
        tables[spec] = entry
    return tables


def run_probe(request: dict[str, Any]) -> dict[str, Any]:
    import sysconfig

    info: dict[str, Any] = {
        "python": ".".join(str(v) for v in sys.version_info[:3]),
        "executable": sys.executable,
        "site_paths": sorted({sysconfig.get_paths()["purelib"], sysconfig.get_paths()["platlib"]}),
        "platform": sys.platform,
    }
    import warnings

    warnings.simplefilter("ignore")
    try:
        import triton
    except Exception as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"
        return info
    info["triton_version"] = triton.__version__
    triton_root = os.path.dirname(os.path.abspath(triton.__file__))
    info["triton_path"] = triton_root

    from triton.backends import backends

    info["backends"] = {}
    for name, backend in backends.items():
        compiler_module = sys.modules[backend.compiler.__module__]
        entry: dict[str, Any] = {"compiler": backend.compiler.__name__}
        options_cls = next(
            (
                obj
                for obj in vars(compiler_module).values()
                if isinstance(obj, type) and hasattr(obj, "__dataclass_fields__") and obj.__name__.endswith("Options")
            ),
            None,
        )
        field = getattr(options_cls, "__dataclass_fields__", {}).get("backend_name")
        entry["target_backend"] = getattr(field, "default", name)
        if entry["target_backend"] == "cuda":
            archs, how = _cuda_capabilities(compiler_module)
            entry["archs"] = [str(a) for a in archs]
            entry["archs_source"] = how
            tools = {}
            for capability in (80, 100):
                with contextlib.suppress(Exception):
                    tool = compiler_module.get_ptxas(capability)
                    tools[os.path.basename(tool.path)] = tool.version
            entry["assembler"] = tools
        elif entry["target_backend"] == "hip":
            archs_hip, how = _hip_archs(triton_root, backend.compiler)
            entry["archs"] = archs_hip
            entry["archs_source"] = how
        else:
            entry["archs"] = []
            entry["archs_source"] = "out-of-tree backend: no target table is read"
        info["backends"][name] = entry

    libdevice: dict[str, Any] = {}
    try:
        generic = importlib.import_module(GENERIC_LIBDEVICE)
        libdevice["generic"] = _public_functions(generic)
    except Exception as exc:
        libdevice["error"] = f"{type(exc).__name__}: {exc}"
    libdevice["implementations"] = {}
    info["libdevice"] = libdevice
    info["targets"] = _backend_tables(request.get("targets") or [])
    for entry in info["targets"].values():
        module_name = entry.get("libdevice_module")
        if module_name and module_name not in libdevice["implementations"]:
            libdevice["implementations"][module_name] = _public_functions(importlib.import_module(module_name))
    return info


# --------------------------------------------------------------------------
# static mode
# --------------------------------------------------------------------------


def _resolve_chain(node: ast.AST, scope: dict[str, Any]) -> Any:
    """Resolve ``a.b.c`` against a scope; returns a sentinel when unknown."""
    if isinstance(node, ast.Name):
        return scope.get(node.id, _UNKNOWN)
    if isinstance(node, ast.Attribute):
        base = _resolve_chain(node.value, scope)
        if base is _UNKNOWN:
            return _UNKNOWN
        try:
            return inspect.getattr_static(base, node.attr)
        except AttributeError:
            return _MISSING if inspect.ismodule(base) else _UNKNOWN
    return _UNKNOWN


_UNKNOWN = object()
_MISSING = object()


class _RefScanner(ast.NodeVisitor):
    """Collects libdevice functions and fp8 dtypes named by reachable code.

    ``if is_hip():`` style guards built from ``triton.language.target_info``
    are evaluated against the active (fake) driver, exactly as Triton's
    frontend does, so only the branch taken for the current target is read.
    """

    def __init__(self, scope: dict[str, Any], first_line: int, refs: dict[str, Any], seen: set[int]) -> None:
        self.scope = scope
        self.first_line = first_line
        self.refs = refs
        self.seen = seen

    def _guard(self, test: ast.AST) -> bool | None:
        if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
            inner = self._guard(test.operand)
            return None if inner is None else not inner
        if isinstance(test, ast.Call) and not test.keywords:
            raw = getattr(_resolve_chain(test.func, self.scope), "fn", None)
            if inspect.isfunction(raw) and raw.__module__ == "triton.language.target_info":
                try:
                    return bool(raw(*[ast.literal_eval(arg) for arg in test.args]))
                except Exception:
                    return None
        return None

    def visit_If(self, node: ast.If) -> None:
        taken = self._guard(node.test)
        if taken is None:
            self.generic_visit(node)
            return
        for statement in node.body if taken else node.orelse:
            self.visit(statement)

    def visit_IfExp(self, node: ast.IfExp) -> None:
        taken = self._guard(node.test)
        if taken is None:
            self.generic_visit(node)
        else:
            self.visit(node.body if taken else node.orelse)

    def _record(self, node: ast.AST) -> None:
        line = self.first_line + getattr(node, "lineno", 1) - 1
        if isinstance(node, ast.Attribute):
            base = _resolve_chain(node.value, self.scope)
            if inspect.ismodule(base) and base.__name__ == GENERIC_LIBDEVICE:
                self.refs["libdevice"].setdefault(node.attr, line)
                return
        obj = _resolve_chain(node, self.scope)
        if obj is _UNKNOWN or obj is _MISSING:
            return
        if obj.__class__.__name__ == "dtype" and str(getattr(obj, "name", "")).startswith("fp8"):
            self.refs["fp8"].setdefault(obj.name, line)
        elif hasattr(obj, "cache_key") and inspect.isfunction(getattr(obj, "fn", None)):
            inner = scan_kernel(obj, self.seen)
            for kind in ("libdevice", "fp8"):
                for key, value in inner[kind].items():
                    self.refs[kind].setdefault(key, value)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        self._record(node)
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        self._record(node)


def scan_kernel(fn: Any, seen: set[int] | None = None) -> dict[str, Any]:
    """Find libdevice functions and fp8 dtypes a kernel (and its callees) names.

    Syntactic apart from target guards: a reference inside any other kind of
    dead branch is still reported, which is why findings are worded as
    predictions.
    """
    seen = set() if seen is None else seen
    refs: dict[str, Any] = {"libdevice": {}, "fp8": {}}
    func = getattr(fn, "fn", fn)
    if id(func) in seen:
        return refs
    seen.add(id(func))
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
        _lines, first_line = inspect.getsourcelines(func)
    except (OSError, TypeError, SyntaxError):
        return refs
    scope = dict(getattr(func, "__globals__", {}))
    closure = getattr(func, "__closure__", None) or ()
    for name, cell in zip(getattr(func.__code__, "co_freevars", ()), closure, strict=False):
        with contextlib.suppress(ValueError):
            scope[name] = cell.cell_contents
    _RefScanner(scope, first_line, refs, seen).visit(tree)
    return refs


def _fp8_in_spec(kernel: dict[str, Any]) -> set[str]:
    found: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, str):
            found.update(re.findall(r"fp8\w*", value))
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)

    visit(kernel.get("signature"))
    visit(kernel.get("constants"))
    return found


def run_static(request: dict[str, Any]) -> dict[str, Any]:
    setup_sys_path(request)
    import warnings

    warnings.simplefilter("ignore")
    import triton
    from triton.compiler.compiler import make_backend

    out: dict[str, Any] = {"triton_version": triton.__version__, "kernels": {}}
    backends: dict[str, Any] = {}
    for spec in request["targets"]:
        try:
            target = resolve_target(spec)
            backend = make_backend(target)
            options = backend.parse_options({})
            impl = backend.get_module_map().get(GENERIC_LIBDEVICE)
            backends[spec] = (target, impl, getattr(options, "supported_fp8_dtypes", None))
        except Exception as exc:
            backends[spec] = exc

    for kernel in request["kernels"]:
        entry: dict[str, Any] = {"targets": {}, "error": None}
        out["kernels"][kernel["id"]] = entry
        try:
            fn = load_kernel(kernel["target"], request.get("base_dir") or os.getcwd())
            location = kernel_location(fn)
        except Exception as exc:
            entry["error"] = f"{type(exc).__name__}: {exc}"
            continue
        entry["location"] = location
        spec_fp8 = _fp8_in_spec(kernel)
        for spec, tables in backends.items():
            findings: list[dict[str, Any]] = []
            entry["targets"][spec] = findings
            if isinstance(tables, Exception):
                continue
            target, impl, fp8_supported = tables
            install_fake_driver(target)  # so target guards resolve as they will when compiling
            refs = scan_kernel(fn)
            if impl is not None:
                for name, line in sorted(refs["libdevice"].items()):
                    if not hasattr(impl, name):
                        findings.append(
                            {
                                "rule": "libdevice-missing",
                                "symbol": name,
                                "line": line,
                                "table": impl.__name__,
                                "detail": f"libdevice.{name} is not defined in {impl.__name__}",
                            }
                        )
            if fp8_supported is not None:
                used = dict.fromkeys(sorted(spec_fp8), None) | refs["fp8"]
                for name, line in sorted(used.items()):
                    if name not in fp8_supported:
                        findings.append(
                            {
                                "rule": "fp8-dtype-unsupported",
                                "symbol": name,
                                "line": line,
                                "table": "supported_fp8_dtypes",
                                "detail": f"{name} is not in supported_fp8_dtypes {tuple(sorted(fp8_supported))}",
                            }
                        )
    return out


# --------------------------------------------------------------------------
# capture mode
# --------------------------------------------------------------------------


class Recorder:
    """Records every specialisation Triton is asked to compile.

    Uses ``knobs.runtime.jit_cache_hook``: Triton calls it right before a
    kernel would be compiled and hands over its own serialised specialisation
    (the same blob ``JITFunction.preload`` consumes), so nothing about how
    Triton keys specialisations is re-derived here.
    """

    def __init__(self, out_path: str, skip_compile: bool) -> None:
        self.out_path = out_path
        self.skip_compile = skip_compile
        self.count = 0
        self._defaults: dict[str, Any] = {}

    def _default_options(self, jit_fn: Any) -> dict[str, Any]:
        from triton.compiler.compiler import make_backend
        from triton.runtime.driver import driver

        target = driver.active.get_current_target()
        key = repr(target)
        if key not in self._defaults:
            base = {"debug": False, "instrumentation_mode": "", "fpsan_homomorphic_casts": False}
            self._defaults[key] = _jsonable(vars(make_backend(target).parse_options(base)))
        return dict(self._defaults[key])

    def __call__(self, *, key: Any, repr: Any, fn: Any, compile: Any, **_unused: Any) -> bool:
        try:
            self._record(fn, compile)
        except Exception as exc:  # never break the user's run because of recording
            sys.stderr.write(f"triton-portcheck capture: could not record {repr}: {exc}\n")
        return self.skip_compile

    def _record(self, info: Any, compile_args: dict[str, Any]) -> None:
        data = json.loads(compile_args["specialization_data"])
        jit_fn = info.jit_function
        defaults = self._default_options(jit_fn)
        options = {k: v for k, v in _jsonable(data["options"]).items() if defaults.get(k, _UNKNOWN) != v}
        location = kernel_location(jit_fn)
        record = {
            "module": info.module,
            "qualname": info.name,
            "file": location["file"],
            "line": location["line"],
            "arg_names": [p.name for p in jit_fn.params],
            "signature": data["signature"],
            "constant_keys": data["constant_keys"],
            "constant_vals": data["constant_vals"],
            "attrs_keys": data["attrs_keys"],
            "attrs_vals": data["attrs_vals"],
            "options": options,
        }
        with open(self.out_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=_json_default) + "\n")
        self.count += 1


def install_capture(mode: str, out_path: str, reference_target: str = "cuda:80") -> Recorder:
    """Install the recording hook in this process.

    ``interpret``  kernels run in Triton's CPU interpreter and every launch is
                   also fed to a shadow JITFunction whose compile is skipped.
    ``record``     kernels are specialised but neither compiled nor launched.
    ``live``       nothing is faked: needs a real device, records as it compiles.
    """
    if mode not in ("interpret", "record", "live"):
        raise ValueError(f"unknown capture mode {mode!r}")
    if mode == "interpret":
        os.environ["TRITON_INTERPRET"] = "1"
    else:
        os.environ.pop("TRITON_INTERPRET", None)
    import triton
    from triton import knobs

    recorder = Recorder(out_path, skip_compile=mode != "live")
    knobs.runtime.jit_cache_hook = recorder
    if mode != "live":
        install_fake_driver(resolve_target(reference_target))
    if mode == "interpret":
        from triton.runtime import interpreter
        from triton.runtime.jit import JITFunction

        shadows: dict[int, Any] = {}
        original_run = interpreter.InterpretedFunction.run

        def run(self: Any, *args: Any, grid: Any, warmup: Any, **kwargs: Any) -> Any:
            try:
                shadow = shadows.get(id(self))
                if shadow is None:
                    shadow = shadows[id(self)] = JITFunction(self.fn, **self.kwargs)
                shadow.run(*args, grid=grid, warmup=True, **kwargs)
            except Exception as exc:
                sys.stderr.write(f"triton-portcheck capture: could not specialise {self.fn.__qualname__}: {exc}\n")
            return original_run(self, *args, grid=grid, warmup=warmup, **kwargs)

        interpreter.InterpretedFunction.run = run
    del triton
    return recorder


def run_capture(request: dict[str, Any]) -> int:
    setup_sys_path(request)
    recorder = install_capture(request["mode"], request["out"], request.get("reference_target") or "cuda:80")
    argv = list(request["argv"])
    emit("capture_begin", argv=argv)
    code = 0
    try:
        if argv[0] == "-m":
            sys.argv = [argv[1], *argv[2:]]
            runpy.run_module(argv[1], run_name="__main__", alter_sys=True)
        elif argv[0] == "-c":
            sys.argv = ["-c", *argv[2:]]
            exec(compile(argv[1], "<string>", "exec"), {"__name__": "__main__"})
        else:
            sys.argv = argv
            sys.path.insert(0, os.path.dirname(os.path.abspath(argv[0])))
            runpy.run_path(argv[0], run_name="__main__")
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
    except BaseException:
        traceback.print_exc()
        code = 1
    emit("capture_end", returncode=code, recorded=recorder.count)
    return code


# --------------------------------------------------------------------------


def main(argv: list[str]) -> int:
    global _events_fh
    if len(argv) != 4:
        sys.stderr.write(__doc__ or "")
        return 2
    mode, request_path, events_path = argv[1], argv[2], argv[3]
    with open(request_path, encoding="utf-8") as handle:
        request = json.load(handle)
    _events_fh = open(events_path, "a", encoding="utf-8")  # noqa: SIM115 - lives for the process
    emit("start", mode=mode, pid=os.getpid())
    if mode == "capture":
        return run_capture(request)
    try:
        if mode == "compile":
            payload = run_compile(request)
        elif mode == "probe":
            payload = run_probe(request)
        elif mode == "static":
            payload = run_static(request)
        else:
            raise ValueError(f"unknown mode {mode!r}")
    except BaseException as exc:
        emit("result", payload={"ok": False, "stage": "harness", "error": _exception_record(exc, None)})
        return 3
    emit("result", payload=payload)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

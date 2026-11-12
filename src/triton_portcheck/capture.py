"""Record the specialisations a test-suite or script really launches.

The recording itself happens in the child (``_child.py capture``) through
Triton's ``jit_cache_hook``.  This module prepares the command, then turns the
recorded specialisation blobs into a manifest.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

from .manifest import KernelSpec, Manifest

MODES = ("interpret", "record", "live")
_PYTHON_RE = re.compile(r"^python(\d+(\.\d+)?)?(\.exe)?$")
_DEFAULT_ALIGN_ATTR = [["tt.divisibility", 16]]


class CaptureError(ValueError):
    """Raised when the command cannot be run under the recorder."""


def python_argv(command: list[str]) -> list[str]:
    """Rewrite a user command into arguments for the in-process runner.

    ``pytest -x tests``          -> ``-m pytest -x tests``
    ``python -m pytest tests``   -> ``-m pytest tests``
    ``python train.py --fast``   -> ``train.py --fast``
    ``train.py``                 -> ``train.py``
    """
    if not command:
        raise CaptureError("no command given: use `triton-portcheck capture -- pytest tests/` or `-- python script.py`")
    head = os.path.basename(command[0])
    rest = command[1:]
    if _PYTHON_RE.match(head):
        if not rest:
            raise CaptureError("`python` needs a script, -m module or -c code to run")
        if rest[0] in ("-m", "-c"):
            if len(rest) < 2:
                raise CaptureError(f"`python {rest[0]}` needs an argument")
            return rest
        if rest[0].startswith("-"):
            raise CaptureError(f"interpreter option {rest[0]} is not supported; pass a script, -m module or -c code")
        return rest
    if head in ("pytest", "py.test"):
        return ["-m", "pytest", *rest]
    if command[0].endswith(".py"):
        return command
    raise CaptureError(
        f"cannot run '{command[0]}' under the recorder: the command must start with python, pytest or a .py file. "
        "For anything else call triton_portcheck.capture hooks from your own entry point (see docs/capture.md)."
    )


def read_records(path: Path) -> list[dict[str, Any]]:
    records = []
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    return records


def _key(path: list[int], names: list[str]) -> str:
    return ".".join([names[path[0]], *[str(p) for p in path[1:]]])


def _locator(record: dict[str, Any], base_dir: Path) -> str:
    source = record.get("file")
    module = record["module"]
    in_site = source is None or any(part in ("site-packages", "dist-packages") for part in Path(source).parts)
    if in_site and module != "__main__":
        return f"{module}:{record['qualname']}"
    path = Path(source or "")
    try:
        shown = path.resolve().relative_to(base_dir.resolve()).as_posix()
    except ValueError:
        shown = str(path)
    return f"{shown}:{record['qualname']}"


def _pointer_paths(signature: dict[str, Any]) -> list[str]:
    return [name for name, ty in signature.items() if isinstance(ty, str) and ty.startswith("*")]


def records_to_manifest(records: list[dict[str, Any]], base_dir: Path) -> tuple[Manifest, list[str]]:
    """Deduplicate recorded specialisations and build a manifest.

    Returns the manifest and a list of kernels that could not be included
    (functions defined inside other functions cannot be imported again).
    """
    unique: dict[str, tuple[dict[str, Any], KernelSpec]] = {}
    skipped: list[str] = []
    for record in records:
        if "<locals>" in record["qualname"]:
            label = f"{record['module']}:{record['qualname']}"
            if label not in skipped:
                skipped.append(label)
            continue
        names = record["arg_names"]
        signature = record["signature"]
        constants = {_key(p, names): v for p, v in zip(record["constant_keys"], record["constant_vals"], strict=True)}
        attrs = {_key(p, names): v for p, v in zip(record["attrs_keys"], record["attrs_vals"], strict=True) if v}
        pointers = _pointer_paths(signature)
        default_attrs = dict.fromkeys(pointers, _DEFAULT_ALIGN_ATTR)
        spec = KernelSpec(
            name=record["qualname"],
            target=_locator(record, base_dir),
            signature={k: v for k, v in signature.items() if v != "constexpr"},
            constants=constants,
            attrs=None if attrs == default_attrs else attrs,
            options=record.get("options") or {},
        )
        identity = json.dumps(spec.to_dict(), sort_keys=True, default=str)
        unique.setdefault(identity, (record, spec))
    specs = [spec for _record, spec in unique.values()]
    totals = Counter(spec.name for spec in specs)
    seen: Counter[str] = Counter()
    for spec in specs:
        if totals[spec.name] > 1:
            seen[spec.name] += 1
            spec.name = f"{spec.name}#{seen[spec.name]}"
    return Manifest(kernels=specs, base_dir=base_dir), skipped

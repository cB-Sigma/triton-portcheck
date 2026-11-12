"""Manifest: which kernels to compile, with which specialisations.

The value formats are Triton's own.  ``signature`` holds the type strings
Triton derives for runtime arguments (``*fp32``, ``i32``, ``tensordesc<...>``)
and ``constants`` holds values in the form
``triton.runtime.jit.serialize_specialization_data`` writes them, so a captured
manifest and a hand-written one are the same thing.
"""

from __future__ import annotations

import json
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .targets import parse_targets


class ManifestError(ValueError):
    """Raised for an invalid manifest; the message names the offending entry."""


_KERNEL_KEYS = {
    "name",
    "target",
    "signature",
    "constants",
    "attrs",
    "align",
    "options",
    "targets",
    "xfail",
    "skip",
    "variant",
    "timeout",
}
_VARIANT_KEYS = _KERNEL_KEYS - {"target", "variant"}
_DEFAULT_KEYS = {"targets", "triton", "timeout", "max_rss", "jobs", "pythonpath"}
_LOCATOR_RE = re.compile(r"^[^:]+:[A-Za-z_][\w.]*$")


@dataclass
class KernelSpec:
    """One specialisation of one kernel: a row of the matrix."""

    name: str
    target: str
    signature: dict[str, Any] = field(default_factory=dict)
    constants: dict[str, Any] = field(default_factory=dict)
    attrs: dict[str, Any] | None = None
    align: bool = True
    options: dict[str, Any] = field(default_factory=dict)
    targets: list[str] | None = None
    xfail: list[str] = field(default_factory=list)
    skip: list[str] = field(default_factory=list)
    timeout: float | None = None

    def child_request(self) -> dict[str, Any]:
        """The part of the spec the child process needs."""
        request: dict[str, Any] = {
            "id": self.name,
            "target": self.target,
            "signature": self.signature,
            "constants": self.constants,
            "align": self.align,
            "options": self.options,
        }
        if self.attrs is not None:
            request["attrs"] = self.attrs
        return request

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"name": self.name, "target": self.target}
        if self.signature:
            out["signature"] = self.signature
        if self.constants:
            out["constants"] = self.constants
        if self.attrs is not None:
            out["attrs"] = self.attrs
        if not self.align:
            out["align"] = False
        if self.options:
            out["options"] = self.options
        if self.targets is not None:
            out["targets"] = self.targets
        if self.xfail:
            out["xfail"] = self.xfail
        if self.skip:
            out["skip"] = self.skip
        if self.timeout is not None:
            out["timeout"] = self.timeout
        return out


@dataclass
class Manifest:
    kernels: list[KernelSpec]
    base_dir: Path
    targets: list[str] = field(default_factory=list)
    triton: list[str] = field(default_factory=list)
    timeout: float | None = None
    max_rss: str | None = None
    jobs: int | None = None
    pythonpath: list[str] = field(default_factory=list)
    path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        defaults: dict[str, Any] = {}
        if self.targets:
            defaults["targets"] = self.targets
        if self.triton:
            defaults["triton"] = self.triton
        if self.timeout is not None:
            defaults["timeout"] = self.timeout
        if self.max_rss is not None:
            defaults["max_rss"] = self.max_rss
        if self.jobs is not None:
            defaults["jobs"] = self.jobs
        if self.pythonpath:
            defaults["pythonpath"] = self.pythonpath
        out: dict[str, Any] = {}
        if defaults:
            out["defaults"] = defaults
        out["kernel"] = [k.to_dict() for k in self.kernels]
        return out


def _table(value: Any, where: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ManifestError(f"{where} must be a table, got {type(value).__name__}")
    return dict(value)


def _str_list(value: Any, where: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ManifestError(f"{where} must be a list of strings")
    return list(value)


def _check_keys(entry: dict[str, Any], allowed: set[str], where: str) -> None:
    unknown = sorted(set(entry) - allowed)
    if unknown:
        raise ManifestError(f"{where}: unknown key(s) {', '.join(unknown)}; allowed: {', '.join(sorted(allowed))}")


def _build_spec(base: dict[str, Any], override: dict[str, Any], name: str, where: str) -> KernelSpec:
    merged = dict(base)
    for key in ("signature", "constants", "options"):
        merged[key] = {**_table(base.get(key), f"{where}.{key}"), **_table(override.get(key), f"{where}.{key}")}
    for key in ("attrs", "align", "targets", "xfail", "skip", "timeout"):
        if key in override:
            merged[key] = override[key]
    attrs = merged.get("attrs")
    targets = merged.get("targets")
    try:
        parsed_targets = None if targets is None else parse_targets(_str_list(targets, f"{where}.targets"))
    except ValueError as exc:
        raise ManifestError(f"{where}: {exc}") from exc
    if not isinstance(merged.get("align", True), bool):
        raise ManifestError(f"{where}.align must be true or false")
    timeout = merged.get("timeout")
    if timeout is not None and (not isinstance(timeout, (int, float)) or timeout <= 0):
        raise ManifestError(f"{where}.timeout must be a positive number of seconds")
    return KernelSpec(
        name=name,
        target=merged["target"],
        signature=merged["signature"],
        constants=merged["constants"],
        attrs=None if attrs is None else _table(attrs, f"{where}.attrs"),
        align=merged.get("align", True),
        options=merged["options"],
        targets=parsed_targets,
        xfail=_str_list(merged.get("xfail"), f"{where}.xfail"),
        skip=_str_list(merged.get("skip"), f"{where}.skip"),
        timeout=None if timeout is None else float(timeout),
    )


def parse_manifest(data: dict[str, Any], base_dir: Path, path: Path | None = None) -> Manifest:
    """Validate a decoded manifest and expand variants into flat specs."""
    if not isinstance(data, dict):
        raise ManifestError("manifest must be a table at top level")
    _check_keys(data, {"defaults", "kernel"}, "manifest")
    defaults = _table(data.get("defaults"), "[defaults]")
    _check_keys(defaults, _DEFAULT_KEYS, "[defaults]")
    entries = data.get("kernel")
    if not isinstance(entries, list) or not entries:
        raise ManifestError("manifest has no [[kernel]] entries")

    specs: list[KernelSpec] = []
    seen: set[str] = set()
    for position, entry in enumerate(entries, start=1):
        where = f"kernel #{position}"
        if not isinstance(entry, dict):
            raise ManifestError(f"{where} must be a table")
        _check_keys(entry, _KERNEL_KEYS, where)
        locator = entry.get("target")
        if not isinstance(locator, str) or not _LOCATOR_RE.match(locator):
            raise ManifestError(
                f"{where}: 'target' must be 'module:function' or 'path/file.py:function', got {locator!r}"
            )
        base_name = entry.get("name") or locator.rpartition(":")[2]
        where = f"kernel '{base_name}'"
        variants = entry.get("variant")
        if variants is None:
            pairs: list[tuple[str, dict[str, Any]]] = [(base_name, {})]
        else:
            if not isinstance(variants, list) or not variants:
                raise ManifestError(f"{where}: 'variant' must be a non-empty array of tables")
            pairs = []
            for index, variant in enumerate(variants, start=1):
                if not isinstance(variant, dict):
                    raise ManifestError(f"{where}: variant #{index} must be a table")
                _check_keys(variant, _VARIANT_KEYS, f"{where} variant #{index}")
                pairs.append((f"{base_name}[{variant.get('name') or index}]", variant))
        for name, override in pairs:
            if name in seen:
                raise ManifestError(f"duplicate kernel name '{name}': give each entry a unique 'name'")
            seen.add(name)
            specs.append(_build_spec(entry, override, name, where))

    try:
        targets = parse_targets(_str_list(defaults.get("targets"), "[defaults].targets"))
    except ValueError as exc:
        raise ManifestError(f"[defaults]: {exc}") from exc
    return Manifest(
        kernels=specs,
        base_dir=base_dir,
        targets=targets,
        triton=[str(v) for v in _str_list(defaults.get("triton"), "[defaults].triton")],
        timeout=defaults.get("timeout"),
        max_rss=None if defaults.get("max_rss") is None else str(defaults["max_rss"]),
        jobs=defaults.get("jobs"),
        pythonpath=_str_list(defaults.get("pythonpath"), "[defaults].pythonpath"),
        path=path,
    )


def load_manifest(path: str | Path) -> Manifest:
    """Load a ``.toml``, ``.json`` or ``.yaml`` manifest."""
    path = Path(path)
    if not path.is_file():
        raise ManifestError(f"manifest not found: {path}")
    suffix = path.suffix.lower()
    try:
        if suffix == ".toml":
            data = tomllib.loads(path.read_text(encoding="utf-8"))
        elif suffix == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
        elif suffix in (".yaml", ".yml"):
            try:
                import yaml
            except ImportError as exc:
                raise ManifestError(
                    "YAML manifests need PyYAML: pip install 'triton-portcheck[yaml]' (or use TOML)"
                ) from exc
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        else:
            raise ManifestError(f"unsupported manifest type '{suffix}': use .toml, .yaml or .json")
    except (tomllib.TOMLDecodeError, json.JSONDecodeError) as exc:
        raise ManifestError(f"{path}: {exc}") from exc
    return parse_manifest(data, path.resolve().parent, path)


def find_manifest(start: Path) -> Path | None:
    """Look for the conventional manifest names in a directory."""
    for name in ("portcheck.toml", "triton-portcheck.toml", "portcheck.yaml", "portcheck.yml", "portcheck.json"):
        candidate = start / name
        if candidate.is_file():
            return candidate
    return None


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------

_BARE_KEY = re.compile(r"^[A-Za-z0-9_-]+$")


def _toml_key(key: str) -> str:
    return key if _BARE_KEY.match(key) else json.dumps(key)


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    if isinstance(value, dict):
        return (
            "{ " + ", ".join(f"{_toml_key(str(k))} = {_toml_value(v)}" for k, v in value.items()) + " }"
            if value
            else "{}"
        )
    if value is None:
        raise ManifestError("TOML cannot store a null value; write this manifest as .json or .yaml instead")
    raise ManifestError(f"cannot write a value of type {type(value).__name__} to TOML")


def dumps_toml(manifest: Manifest, header: str = "") -> str:
    """Serialise a manifest to TOML (only the subset of TOML this schema needs)."""
    data = manifest.to_dict()
    lines: list[str] = [f"# {line}".rstrip() for line in header.splitlines()]
    if lines:
        lines.append("")
    if "defaults" in data:
        lines.append("[defaults]")
        lines += [f"{_toml_key(k)} = {_toml_value(v)}" for k, v in data["defaults"].items()]
        lines.append("")
    for kernel in data["kernel"]:
        lines.append("[[kernel]]")
        lines += [f"{_toml_key(k)} = {_toml_value(v)}" for k, v in kernel.items()]
        lines.append("")
    return "\n".join(lines)


def dump_manifest(manifest: Manifest, path: str | Path, header: str = "") -> None:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".json":
        text = json.dumps(manifest.to_dict(), indent=2) + "\n"
    elif suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as exc:
            raise ManifestError("writing YAML needs PyYAML: pip install 'triton-portcheck[yaml]'") from exc
        text = "".join(f"# {line}\n" for line in header.splitlines()) + yaml.safe_dump(
            manifest.to_dict(), sort_keys=False
        )
    else:
        text = dumps_toml(manifest, header)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")

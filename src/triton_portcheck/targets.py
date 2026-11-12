"""Target names: ``cuda:<capability>``, ``hip:<arch>``, optional ``:<warp size>``."""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass

_TARGET_RE = re.compile(r"^(?P<backend>[a-z][a-z0-9_]*):(?P<arch>[A-Za-z0-9_.\-]+)(?::(?P<warp>\d+))?$")

#: Used when neither the command line nor the manifest names targets.
DEFAULT_TARGETS = ("cuda:80", "cuda:90", "cuda:100", "hip:gfx942", "hip:gfx950")


class TargetError(ValueError):
    """Raised for a malformed target name."""


@dataclass(frozen=True)
class Target:
    backend: str
    arch: str
    warp_size: int | None = None

    def __str__(self) -> str:
        base = f"{self.backend}:{self.arch}"
        return base if self.warp_size is None else f"{base}:{self.warp_size}"


def parse_target(text: str) -> Target:
    """Parse one target name, normalising ``cuda:sm_90`` / ``cuda:9.0`` to ``cuda:90``."""
    raw = text.strip()
    match = _TARGET_RE.match(raw.lower() if raw.lower().startswith("cuda") else raw)
    if match is None:
        raise TargetError(
            f"invalid target {text!r}: expected 'cuda:<capability>' or 'hip:<arch>', e.g. cuda:90, hip:gfx942"
        )
    backend, arch = match["backend"], match["arch"]
    if backend == "cuda":
        arch = re.sub(r"^sm_?", "", arch).replace(".", "").rstrip("a")
        if not arch.isdigit():
            raise TargetError(
                f"invalid CUDA capability in {text!r}: expected digits such as cuda:90 (compute capability 9.0)"
            )
    return Target(backend, arch, int(match["warp"]) if match["warp"] else None)


def parse_targets(items: list[str] | tuple[str, ...]) -> list[str]:
    """Parse a list that may contain comma-separated names; keep order, drop duplicates."""
    out: list[str] = []
    for item in items:
        for part in item.split(","):
            if part.strip():
                name = str(parse_target(part))
                if name not in out:
                    out.append(name)
    return out


def matches(pattern: str, target: str, triton: str = "") -> bool:
    """Glob match used by ``xfail`` and ``skip``: ``hip:*``, ``cuda:7?``, ``cuda:90@3.8.*``."""
    target_pattern, _, version_pattern = pattern.partition("@")
    if not fnmatch.fnmatchcase(target, target_pattern.strip() or "*"):
        return False
    return not version_pattern or fnmatch.fnmatchcase(triton, version_pattern.strip())

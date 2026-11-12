"""Per-version virtual environments holding one Triton release each."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .results import format_bytes

VERSION_RE = re.compile(r"^\d+\.\d+(\.\d+)?([a-z0-9.+\-]*)$")

#: Installed size divided by wheel size, measured for the 3.8.0 manylinux x86_64 wheel.
INSTALL_RATIO = 3.6
MIN_FREE_BYTES = 2 * 1024**3


class EnvError(RuntimeError):
    """Raised when an environment cannot be created, found or used."""


@dataclass(frozen=True)
class Environment:
    label: str
    python: Path
    managed: bool


def home_dir() -> Path:
    override = os.environ.get("TRITON_PORTCHECK_HOME")
    if override:
        return Path(override).expanduser()
    cache = os.environ.get("XDG_CACHE_HOME")
    return (Path(cache).expanduser() if cache else Path.home() / ".cache") / "triton-portcheck"


def envs_dir() -> Path:
    return home_dir() / "envs"


def env_path(version: str) -> Path:
    return envs_dir() / f"triton-{version}"


def env_python(path: Path) -> Path:
    return path / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def base_python() -> Path:
    """Interpreter whose Triton is ``current`` and that seeds new environments."""
    return Path(os.environ.get("TRITON_PORTCHECK_PYTHON") or sys.executable)


def dir_size(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.lstat(os.path.join(root, name)).st_size
            except OSError:
                continue
    return total


def free_bytes(path: Path) -> int:
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return shutil.disk_usage(probe).free


def list_envs() -> list[tuple[str, Path, int]]:
    root = envs_dir()
    if not root.is_dir():
        return []
    out = []
    for child in sorted(root.iterdir()):
        if child.name.startswith("triton-") and env_python(child).exists():
            out.append((child.name.removeprefix("triton-"), child, dir_size(child)))
    return out


def wheel_size(version: str, python_tag: str, timeout: float = 6.0) -> int | None:
    """Size of the matching wheel on PyPI, or None when it cannot be determined."""
    url = f"https://pypi.org/pypi/triton/{version}/json"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            files = json.load(response).get("urls", [])
    except (urllib.error.URLError, OSError, ValueError):
        return None
    machine = os.uname().machine if hasattr(os, "uname") else ""
    for entry in files:
        name = entry.get("filename", "")
        if f"-{python_tag}-" in name and machine in name:
            return int(entry["size"])
    return None


def _run(command: list[str], log: Callable[[str], None]) -> None:
    log("  $ " + " ".join(command))
    proc = subprocess.run(command, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        tail = "\n".join((proc.stdout + proc.stderr).strip().splitlines()[-15:])
        raise EnvError(f"command failed ({proc.returncode}): {' '.join(command)}\n{tail}")


def add_env(
    version: str,
    *,
    python: Path | None = None,
    extra: list[str] | None = None,
    installer: str = "auto",
    force: bool = False,
    log: Callable[[str], None] = print,
) -> Path:
    """Create ``envs/triton-<version>`` and install that Triton release into it."""
    if not VERSION_RE.match(version):
        raise EnvError(f"'{version}' is not a Triton version (expected something like 3.8.0)")
    path = env_path(version)
    if env_python(path).exists():
        log(f"environment for triton {version} already exists: {path} ({format_bytes(dir_size(path))})")
        return path
    seed = python or base_python()
    tag_out = subprocess.run(
        [str(seed), "-c", "import sys;print(f'cp{sys.version_info[0]}{sys.version_info[1]}')"],
        capture_output=True,
        text=True,
        check=False,
    )
    if tag_out.returncode != 0:
        raise EnvError(f"cannot run {seed}: {tag_out.stderr.strip()}")
    free = free_bytes(envs_dir())
    wheel = wheel_size(version, tag_out.stdout.strip())
    if wheel is None:
        estimate = "roughly 1 GiB (wheel size could not be read from PyPI)"
        needed = MIN_FREE_BYTES
    else:
        needed = int(wheel * INSTALL_RATIO)
        estimate = f"about {format_bytes(needed)} (wheel is {format_bytes(wheel)})"
    log(f"creating environment for triton {version} in {path}")
    log(f"  disk: needs {estimate}; {format_bytes(free)} free")
    if free < max(needed, MIN_FREE_BYTES) and not force:
        raise EnvError(
            f"not enough free disk for triton {version}: {format_bytes(free)} free, "
            f"{format_bytes(max(needed, MIN_FREE_BYTES))} wanted. Free some space, set TRITON_PORTCHECK_HOME "
            "to another volume, or pass --force."
        )
    use_uv = installer == "uv" or (installer == "auto" and shutil.which("uv") is not None)
    requirements = [f"triton=={version}", *(extra or [])]
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if use_uv:
            _run(["uv", "venv", "--quiet", "--python", str(seed), str(path)], log)
            _run(["uv", "pip", "install", "--quiet", "--python", str(env_python(path)), *requirements], log)
        else:
            _run([str(seed), "-m", "venv", str(path)], log)
            _run([str(env_python(path)), "-m", "pip", "install", "--quiet", "--no-input", *requirements], log)
    except BaseException:
        shutil.rmtree(path, ignore_errors=True)
        raise
    log(f"  done: {format_bytes(dir_size(path))} on disk; remove with `triton-portcheck env remove {version}`")
    return path


def remove_env(version: str) -> int:
    """Delete a managed environment; returns the bytes freed."""
    path = env_path(version)
    if not path.is_dir():
        raise EnvError(f"no environment for triton {version} (looked in {path})")
    size = dir_size(path)
    shutil.rmtree(path)
    return size

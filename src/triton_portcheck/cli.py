"""Command line interface."""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path

from . import DISCLAIMER, __version__, baseline, capture, diffing, envs, runner
from . import results as R
from .manifest import Manifest, ManifestError, dump_manifest, find_manifest, load_manifest
from .report import render_github, render_junit, render_markdown, render_sarif, render_terminal
from .targets import DEFAULT_TARGETS, TargetError, parse_targets

EXIT_OK, EXIT_FAILURES, EXIT_USAGE, EXIT_ENVIRONMENT = 0, 1, 2, 3
DEMO_MANIFEST = Path(__file__).parent / "demo" / "manifest.toml"

EPILOG = f"""\
exit codes: 0 no new failures · 1 new compile failures (or regressions in `diff`) ·
            2 bad arguments or manifest · 3 Triton/environment problem

{DISCLAIMER}
"""


def _err(message: str) -> None:
    sys.stderr.write(message.rstrip() + "\n")


def _use_colour(choice: str) -> bool:
    if choice == "always":
        return True
    if choice == "never" or os.environ.get("NO_COLOR"):
        return False
    return sys.stdout.isatty()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="triton-portcheck",
        description="Compile Triton kernels across GPU targets and Triton versions on a machine with no GPU, "
        "and report where each combination breaks.",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"triton-portcheck {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="command")

    check = sub.add_parser(
        "check",
        help="compile the kernel x target x version matrix",
        description="Compile every kernel specialisation in the manifest for every target and Triton version. "
        "Each cell runs in its own subprocess with a timeout and a memory limit.",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    check.add_argument("manifest", nargs="?", help="manifest file (default: ./portcheck.toml)")
    check.add_argument("--demo", action="store_true", help="run the bundled example kernels instead of a manifest")
    check.add_argument(
        "-t",
        "--target",
        action="append",
        default=[],
        metavar="TARGET",
        help="target such as cuda:90 or hip:gfx942; repeat or comma-separate",
    )
    check.add_argument(
        "--triton",
        action="append",
        default=[],
        metavar="VERSION",
        help="Triton version(s): 'current' or a release such as 3.8.0; repeat or comma-separate",
    )
    check.add_argument("-k", "--kernel", action="append", default=[], metavar="GLOB", help="only kernels matching")
    check.add_argument("-j", "--jobs", type=int, help="cells compiled in parallel (default: 2)")
    check.add_argument("--timeout", type=float, help="seconds allowed per cell (default: 300)")
    check.add_argument("--max-rss", metavar="SIZE", help="memory limit per cell, e.g. 8G; 0 disables (default: 8G)")
    check.add_argument(
        "--static",
        choices=("off", "annotate", "skip", "only"),
        default="annotate",
        help="table-driven pre-checks: annotate cells (default), skip compiling predicted failures, "
        "run only the pre-checks, or turn them off",
    )
    check.add_argument("--static-only", action="store_const", const="only", dest="static", help="same as --static only")
    check.add_argument("--baseline", metavar="FILE", help="known failures; only new ones fail the run")
    check.add_argument("--update-baseline", action="store_true", help="write current failures to --baseline and exit 0")
    check.add_argument(
        "--format",
        choices=("terminal", "markdown", "json", "github", "quiet"),
        default="terminal",
        help="what to print on stdout (default: terminal)",
    )
    check.add_argument("--json", metavar="FILE", help="also write the full JSON report")
    check.add_argument("--markdown", metavar="FILE", help="also write a Markdown summary")
    check.add_argument("--junit", metavar="FILE", help="also write JUnit XML")
    check.add_argument("--sarif", metavar="FILE", help="also write SARIF 2.1.0")
    check.add_argument("--github", action="store_true", help="also print GitHub Actions annotations")
    check.add_argument("--times", action="store_true", help="show compile seconds instead of 'ok'")
    check.add_argument("--save-ir", metavar="DIR", help="keep the IR/assembly of every cell (needed for `diff --ir`)")
    check.add_argument("--no-create-envs", action="store_true", help="fail instead of creating a missing version env")
    check.add_argument(
        "--no-inherit-site",
        action="store_true",
        help="do not let version envs fall back to the current interpreter's site-packages",
    )
    check.add_argument("--color", choices=("auto", "always", "never"), default="auto")
    check.add_argument("-q", "--quiet", action="store_true", help="no progress on stderr")

    report = sub.add_parser("report", help="re-render a saved JSON report")
    report.add_argument("report", help="JSON report written by `check --json`")
    report.add_argument(
        "--format", choices=("terminal", "markdown", "junit", "sarif", "github", "json"), default="terminal"
    )
    report.add_argument("--baseline", metavar="FILE")
    report.add_argument("--times", action="store_true")
    report.add_argument("--color", choices=("auto", "always", "never"), default="auto")

    diff = sub.add_parser(
        "diff",
        help="compare two Triton versions, two targets or two runs",
        description="Compare along one axis. With one report use --versions or --targets; with two reports the "
        "same cells are compared between the runs.",
    )
    diff.add_argument("report", help="JSON report (A)")
    diff.add_argument("other", nargs="?", help="second JSON report (B)")
    diff.add_argument("--versions", nargs=2, metavar=("A", "B"), help="compare two Triton versions")
    diff.add_argument("--targets", nargs=2, metavar=("A", "B"), help="compare two targets")
    diff.add_argument(
        "--ir", metavar="STAGE", help="textual diff of normalised IR at a stage: ttir, ttgir, llir, ptx, amdgcn"
    )
    diff.add_argument("--time-ratio", type=float, default=1.5, help="flag compiles slower by this factor (default 1.5)")
    diff.add_argument("--time-min", type=float, default=1.0, help="...and by at least this many seconds (default 1.0)")
    diff.add_argument(
        "--size-ratio", type=float, default=0.10, help="flag artifact size changes above this (default 0.10)"
    )
    diff.add_argument("--format", choices=("text", "markdown", "json"), default="text")
    diff.add_argument("--all", action="store_true", help="also list unchanged cells")
    diff.add_argument("--fail-on-regression", action="store_true", help="exit 1 when a cell goes from ok to failing")

    cap = sub.add_parser(
        "capture",
        help="record the specialisations a test-suite or script uses and write a manifest",
        description="Run a command and record every kernel specialisation it launches. "
        "Example: triton-portcheck capture -o portcheck.toml -- pytest tests/",
    )
    cap.add_argument("-o", "--output", default="portcheck.toml", help="manifest to write (default: portcheck.toml)")
    cap.add_argument(
        "--mode",
        choices=capture.MODES,
        default="interpret",
        help="interpret: run kernels in Triton's CPU interpreter (default); record: specialise only, "
        "kernels do not run; live: real device, record while compiling normally",
    )
    cap.add_argument(
        "--reference-target",
        default="cuda:80",
        metavar="TARGET",
        help="target the fake driver reports while capturing (default: cuda:80)",
    )
    cap.add_argument("cmd", nargs=argparse.REMAINDER, metavar="-- command", help="python/pytest command to run")

    env = sub.add_parser("env", help="manage per-version Triton environments")
    env_sub = env.add_subparsers(dest="env_command", metavar="action")
    env_add = env_sub.add_parser("add", help="create an environment with one Triton release")
    env_add.add_argument("version", nargs="+")
    env_add.add_argument(
        "--with",
        dest="extra",
        action="append",
        default=[],
        metavar="REQ",
        help="extra requirement to install, e.g. --with torch",
    )
    env_add.add_argument("--installer", choices=("auto", "pip", "uv"), default="auto")
    env_add.add_argument("--force", action="store_true", help="skip the free-disk check")
    env_sub.add_parser("list", help="list environments and their disk usage")
    env_remove = env_sub.add_parser("remove", help="delete an environment")
    env_remove.add_argument("version", nargs="+")
    env_sub.add_parser("path", help="print the directory environments live in")

    targets = sub.add_parser("targets", help="list the targets and tables the installed Triton exposes")
    targets.add_argument("--triton", action="append", default=[], metavar="VERSION")
    targets.add_argument("--json", action="store_true", dest="as_json")
    return parser


def _split(values: list[str]) -> list[str]:
    return [part.strip() for value in values for part in value.split(",") if part.strip()]


def _load_for_check(args: argparse.Namespace) -> Manifest:
    if args.demo:
        if args.manifest:
            raise ManifestError("--demo runs the bundled kernels; do not pass a manifest as well")
        return load_manifest(DEMO_MANIFEST)
    path = Path(args.manifest) if args.manifest else find_manifest(Path.cwd())
    if path is None:
        raise ManifestError(
            "no manifest given and no portcheck.toml here. Write one (see docs/manifest.md), generate one with "
            "`triton-portcheck capture -- pytest`, or try `triton-portcheck check --demo`."
        )
    return load_manifest(path)


def _warn_unknown_targets(targets: list[str], environments: list[runner.ResolvedEnv]) -> None:
    for env in environments:
        known: dict[str, set[str]] = {}
        for backend in env.info.get("backends", {}).values():
            known.setdefault(backend.get("target_backend", ""), set()).update(backend.get("archs", []))
        for target in targets:
            backend_name, arch = target.split(":")[:2]
            if backend_name not in known:
                _err(f"warning: triton {env.label} has no '{backend_name}' backend (has: {', '.join(sorted(known))})")
            elif backend_name == "cuda" and known["cuda"] and arch not in known["cuda"]:
                _err(f"warning: the ptxas bundled with triton {env.label} does not list a gpu-name for {target}")


def cmd_check(args: argparse.Namespace) -> int:
    try:
        manifest = _load_for_check(args)
        targets = parse_targets(args.target) or manifest.targets or list(DEFAULT_TARGETS)
        max_rss = runner.parse_size(args.max_rss if args.max_rss is not None else manifest.max_rss or "8G")
    except (ManifestError, TargetError, ValueError) as exc:
        _err(f"error: {exc}")
        return EXIT_USAGE
    if args.kernel:
        manifest.kernels = [k for k in manifest.kernels if any(fnmatch.fnmatchcase(k.name, g) for g in args.kernel)]
        if not manifest.kernels:
            _err(f"error: no kernel matches {', '.join(args.kernel)}")
            return EXIT_USAGE
    if args.update_baseline and not args.baseline:
        _err("error: --update-baseline needs --baseline FILE")
        return EXIT_USAGE
    known = None
    if args.baseline and not args.update_baseline:
        try:
            known = baseline.load_baseline(args.baseline)
        except (OSError, ValueError, KeyError) as exc:
            _err(f"error: cannot read baseline {args.baseline}: {exc}")
            return EXIT_USAGE

    versions = _split(args.triton) or manifest.triton or ["current"]
    quiet = args.quiet or args.format in ("json", "github", "quiet")
    try:
        environments = runner.resolve_envs(
            versions,
            targets,
            create=not args.no_create_envs,
            inherit_site=not args.no_inherit_site,
            log=_err,
        )
    except (runner.RunnerError, envs.EnvError) as exc:
        _err(f"error: {exc}")
        return EXIT_ENVIRONMENT
    _warn_unknown_targets(targets, environments)

    base = manifest.base_dir
    settings = runner.RunSettings(
        timeout=args.timeout or manifest.timeout or 300.0,
        max_rss=max_rss,
        jobs=args.jobs or manifest.jobs or 2,
        static=args.static,
        save_ir=Path(args.save_ir) if args.save_ir else None,
        sys_path=[str((base / p).resolve()) for p in manifest.pythonpath],
    )

    def progress(cell: R.Cell, done: int, total: int) -> None:
        if not quiet:
            took = f"{cell.wall_seconds:.1f}s" if cell.wall_seconds is not None else ""
            _err(f"[{done}/{total}] {cell.status:<15} {cell.kernel} · {cell.target} · triton {cell.triton} {took}")

    report = runner.run_matrix(manifest, targets, environments, settings, progress)

    if args.update_baseline:
        count = baseline.write_baseline(report, args.baseline)
        _err(f"baseline written: {args.baseline} ({count} known failures)")
    verdict = baseline.judge(report, known)

    if args.json:
        report.dump(args.json)
    files: list[tuple[str | None, Callable[[], str]]] = [
        (args.markdown, lambda: render_markdown(report, verdict, times=args.times)),
        (args.junit, lambda: render_junit(report, verdict)),
        (args.sarif, lambda: render_sarif(report, verdict)),
    ]
    for path, render in files:
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(render(), encoding="utf-8")

    if args.format == "terminal":
        sys.stdout.write(render_terminal(report, verdict, colour=_use_colour(args.color), times=args.times))
    elif args.format == "markdown":
        sys.stdout.write(render_markdown(report, verdict, times=args.times))
    elif args.format == "json":
        sys.stdout.write(json.dumps(report.to_dict(), indent=2) + "\n")
    if args.github or args.format == "github":
        sys.stdout.write(render_github(report, verdict))

    if args.demo or args.update_baseline:
        return EXIT_OK
    return EXIT_OK if verdict.ok else EXIT_FAILURES


def cmd_report(args: argparse.Namespace) -> int:
    try:
        report = R.Report.load(args.report)
        known = baseline.load_baseline(args.baseline) if args.baseline else None
    except (OSError, ValueError, KeyError) as exc:
        _err(f"error: {exc}")
        return EXIT_USAGE
    verdict = baseline.judge(report, known)
    renderers = {
        "terminal": lambda: render_terminal(report, verdict, colour=_use_colour(args.color), times=args.times),
        "markdown": lambda: render_markdown(report, verdict, times=args.times),
        "junit": lambda: render_junit(report, verdict),
        "sarif": lambda: render_sarif(report, verdict),
        "github": lambda: render_github(report, verdict),
        "json": lambda: json.dumps(report.to_dict(), indent=2) + "\n",
    }
    sys.stdout.write(renderers[args.format]())
    return EXIT_OK if verdict.ok else EXIT_FAILURES


def cmd_diff(args: argparse.Namespace) -> int:
    try:
        report_a = R.Report.load(args.report)
        report_b = R.Report.load(args.other) if args.other else None
        result = diffing.diff_reports(
            report_a,
            report_b,
            versions=tuple(args.versions) if args.versions else None,
            targets=tuple(parse_targets(args.targets)) if args.targets else None,  # type: ignore[arg-type]
            thresholds=diffing.Thresholds(args.time_ratio, args.time_min, args.size_ratio),
            ir_stage=args.ir,
        )
    except (OSError, ValueError, KeyError) as exc:
        _err(f"error: {exc}")
        return EXIT_USAGE
    if args.ir and not any(e.ir_diff for e in result.entries) and not any(c.ir_dir for c in report_a.cells):
        _err("note: no saved IR in this report; run `check --save-ir DIR --json ...` to enable --ir")
    if args.format == "json":
        sys.stdout.write(json.dumps(result.to_dict(), indent=2) + "\n")
    else:
        sys.stdout.write(diffing.render_diff(result, args.format, show_all=args.all))
    return EXIT_FAILURES if args.fail_on_regression and result.regressions else EXIT_OK


def _has_module(python: str | os.PathLike[str], name: str) -> bool:
    """Whether ``name`` can be found by the interpreter at ``python`` (without importing it)."""
    probe = f"import importlib.util, sys; sys.exit(0 if importlib.util.find_spec({name!r}) else 1)"
    try:
        return subprocess.run([str(python), "-c", probe], capture_output=True, timeout=60, check=False).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def cmd_capture(args: argparse.Namespace) -> int:
    command = list(args.cmd)
    if command and command[0] == "--":
        command = command[1:]
    try:
        argv = capture.python_argv(command)
    except capture.CaptureError as exc:
        _err(f"error: {exc}")
        return EXIT_USAGE
    output = Path(args.output)
    python = envs.base_python()
    if args.mode == "interpret" and not _has_module(python, "numpy"):
        _err(
            "error: interpret mode needs numpy: Triton's CPU interpreter imports it, but Triton does not depend on it "
            f"and it is not installed for {python}. Install numpy there, or use `--mode record`."
        )
        return EXIT_ENVIRONMENT
    with tempfile.TemporaryDirectory(prefix="tpc-capture-") as tmp:
        records_path = Path(tmp) / "records.jsonl"
        request = {
            "mode": args.mode,
            "out": str(records_path),
            "argv": argv,
            "reference_target": args.reference_target,
        }
        _err(f"capture ({args.mode} mode): running {' '.join(command)}")
        try:
            outcome = runner.run_child(
                python, "capture", request, inherit_output=True, extra_env={"TRITON_ALWAYS_COMPILE": "0"}
            )
        except runner.RunnerError as exc:
            _err(f"error: {exc}")
            return EXIT_ENVIRONMENT
        records = capture.read_records(records_path)
    if outcome.returncode not in (0, None):
        _err(f"note: the command exited with status {outcome.returncode}; specialisations seen before that are kept")
    if not records:
        _err(
            "error: no kernel launch was recorded. The command must reach `kernel[grid](...)` on this machine; "
            "without a GPU that means CPU tensors (see docs/capture.md)."
        )
        return EXIT_ENVIRONMENT
    manifest, skipped = capture.records_to_manifest(records, output.resolve().parent)
    header = (
        f"Captured by `triton-portcheck capture --mode {args.mode} -- {' '.join(command)}`.\n"
        f"{len(records)} launches recorded, {len(manifest.kernels)} distinct specialisations."
    )
    try:
        dump_manifest(manifest, output, header)
    except ManifestError as exc:
        _err(f"error: {exc}")
        return EXIT_USAGE
    for label in skipped:
        _err(f"note: skipped {label}: kernels defined inside a function cannot be imported for compilation")
    _err(f"wrote {output}: {len(manifest.kernels)} specialisations from {len(records)} launches")
    return EXIT_OK


def cmd_env(args: argparse.Namespace) -> int:
    try:
        if args.env_command == "add":
            for version in args.version:
                envs.add_env(version, extra=args.extra, installer=args.installer, force=args.force, log=print)
        elif args.env_command == "remove":
            for version in args.version:
                freed = envs.remove_env(version)
                print(f"removed triton {version} ({R.format_bytes(freed)} freed)")
        elif args.env_command == "path":
            print(envs.envs_dir())
        else:
            found = envs.list_envs()
            free = R.format_bytes(envs.free_bytes(envs.envs_dir()))
            if not found:
                print(f"no environments in {envs.envs_dir()} ({free} free). Add one: triton-portcheck env add 3.8.0")
            for version, path, size in found:
                print(f"triton {version:<10} {R.format_bytes(size):>10}  {path}")
            if found:
                total = R.format_bytes(sum(size for _v, _p, size in found))
                print(f"total {total} in {envs.envs_dir()} · {free} free")
    except envs.EnvError as exc:
        _err(f"error: {exc}")
        return EXIT_ENVIRONMENT
    return EXIT_OK


def cmd_targets(args: argparse.Namespace) -> int:
    try:
        environments = runner.resolve_envs(_split(args.triton) or ["current"], [], create=False)
    except (runner.RunnerError, envs.EnvError) as exc:
        _err(f"error: {exc}")
        return EXIT_ENVIRONMENT
    if args.as_json:
        print(json.dumps({e.label: e.info for e in environments}, indent=2))
        return EXIT_OK
    for env in environments:
        print(f"triton {env.label} · python {env.info.get('python')} · {env.python}")
        for name, backend in env.info.get("backends", {}).items():
            prefix = backend.get("target_backend", name)
            archs = ", ".join(f"{prefix}:{arch}" for arch in backend.get("archs", [])) or "(none listed)"
            print(f"  {name}: {archs}")
            print(f"      source: {backend.get('archs_source')}")
            for tool, version in (backend.get("assembler") or {}).items():
                print(f"      {tool} {version}")
        libdevice = env.info.get("libdevice", {})
        if libdevice.get("generic"):
            print(f"  libdevice interface: {len(libdevice['generic'])} functions")
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handlers = {
        "check": cmd_check,
        "report": cmd_report,
        "diff": cmd_diff,
        "capture": cmd_capture,
        "env": cmd_env,
        "targets": cmd_targets,
    }
    if args.command is None:
        parser.print_help()
        return EXIT_USAGE
    try:
        return handlers[args.command](args)
    except KeyboardInterrupt:
        _err("interrupted")
        return 130
    except BrokenPipeError:
        return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())

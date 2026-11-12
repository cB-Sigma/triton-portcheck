# Changelog

All notable changes are recorded here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] - 2026-11-12

First release.

### Added

- `check`: compile a kernel × target × Triton-version matrix, one isolated subprocess per cell with a timeout and
  a memory limit over the process tree; fake active driver so `target_info` guards resolve for the cell's target.
- Manifest (TOML, YAML, JSON) with specialisations in Triton's own serialised form, variants, `xfail` and `skip`.
- `capture`: record the specialisations a test-suite or script launches (`interpret`, `record`, `live` modes)
  through `jit_cache_hook` and write a manifest.
- `env add/list/remove`: per-version Triton environments with disk-usage reporting.
- Per-cell status, failing stage, cleaned message, stable fingerprint, stage times, artifact sizes and the
  metadata the compiler reports; known-signature explanations with links to reproduced upstream issues.
- Static pre-checks from the installed Triton's libdevice and fp8 tables (`--static annotate|skip|only`).
- `diff` between two versions, two targets or two runs: status changes, compile-time regressions, size and
  metadata changes, optional textual diff of normalised IR.
- Outputs: terminal matrix, Markdown, JSON, JUnit, SARIF, GitHub annotations; baseline file; exit codes for CI.
- Composite GitHub Action and pre-commit hooks.
- `check --demo` with bundled kernels that show real divergences.

### Tested with

Triton 3.9.0 and 3.8.0 wheels on x86_64 Linux, Python 3.13, for `cuda:75/80/90/100/120` and
`hip:gfx942/gfx950/gfx1100/gfx1250`. Nothing was run on a GPU.

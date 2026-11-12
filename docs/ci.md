# CI usage

## GitHub Actions

```yaml
jobs:
  portability:
    runs-on: ubuntu-latest          # no GPU needed
    steps:
      - uses: actions/checkout@v7
      - uses: actions/setup-python@v7
        with:
          python-version: "3.12"
      - run: pip install -e . triton==3.9.0
      - uses: cb-Sigma/triton-portcheck@main
        with:
          manifest: portcheck.toml
          targets: cuda:80,cuda:90,cuda:100,hip:gfx942,hip:gfx950
          triton: current,3.8.0
          baseline: ci/portcheck-baseline.json
```

Inputs: `manifest`, `targets`, `triton`, `install-triton`, `baseline`, `args`, `working-directory`.
Outputs: `report` (path of the JSON report) and `exit-code`.

To see results in the code-scanning tab, upload the SARIF file the action leaves next to the JSON report
(`github/codeql-action/upload-sarif`).

Cache `~/.cache/triton-portcheck/envs` if you check several versions; each one is close to 1 GiB.

## Baselines

```bash
triton-portcheck check --baseline ci/portcheck-baseline.json --update-baseline   # record today's failures
triton-portcheck check --baseline ci/portcheck-baseline.json                     # exit 1 only on new ones
```

A baseline lists failing cells (kernel, target, Triton version). Cells that are in the baseline and now compile are
mentioned so you can tighten it. For failures you consider permanent, `xfail` in the manifest is clearer than a
baseline entry.

## Outputs

| flag | format |
|---|---|
| `--format terminal\|markdown\|json\|github\|quiet` | what goes to stdout |
| `--json FILE` | full report: every cell with stage times, artifact sizes, metadata, cleaned and raw error |
| `--markdown FILE` | table and failure list for job summaries or PR comments |
| `--junit FILE` | one suite per Triton version, one case per kernel × target |
| `--sarif FILE` | SARIF 2.1.0, one result per failing cell at the kernel's source line |
| `--github` | workflow commands (`::error file=...`) |

`triton-portcheck report report.json --format markdown` re-renders a saved report.

## Exit codes

| code | meaning |
|---|---|
| 0 | no new failures |
| 1 | at least one failing cell that is neither `xfail` nor in the baseline (or, for `diff --fail-on-regression`, a regression) |
| 2 | bad arguments or an invalid manifest |
| 3 | Triton could not be imported, or a version environment could not be prepared |

## pre-commit

```yaml
- repo: https://github.com/cb-Sigma/triton-portcheck
  rev: main
  hooks:
    - id: triton-portcheck
      additional_dependencies: ["triton==3.9.0"]
```

A full matrix takes seconds for small kernels and minutes for large ones; `triton-portcheck-static` runs only the
table-driven checks. Kernel modules that import torch need it in `additional_dependencies` too.

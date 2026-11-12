# Examples

| directory | shows |
|---|---|
| [`quickstart/`](quickstart) | two kernels, CPU tests, and the manifest `capture` wrote from those tests |
| [`large_tiles/`](large_tiles) | a kernel that is slow to assemble: the `timeout` status on a real compile |
| [`ci/`](ci) | a workflow file to copy into your repository |

The kernels the `--demo` matrix uses live in the package: `src/triton_portcheck/demo/`.

All examples need Triton; `quickstart` also needs torch (CPU is fine).

# triton-portcheck

**Compile your [Triton](https://github.com/triton-lang/triton) kernels for every GPU you ship to — NVIDIA and AMD, several Triton versions — on a machine with no GPU, and see exactly where each combination breaks.**

[![CI](https://github.com/cb-Sigma/triton-portcheck/actions/workflows/ci.yml/badge.svg)](https://github.com/cb-Sigma/triton-portcheck/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Fcb-Sigma%2Ftriton-portcheck%2Fmain%2Fpyproject.toml)](pyproject.toml)

A Triton kernel that works on your A100 can fail to *compile* on a Hopper card, a Blackwell laptop, or any AMD GPU,
and you usually find out from a user. `triton-portcheck` runs Triton's real compiler pipeline, down to `ptxas` and
the AMD assembler, for a grid of targets and Triton versions, each cell in an isolated subprocess, on an ordinary CPU
runner.

This is the unedited output of the bundled demo on a laptop-class CPU box (Triton 3.9.0, 4 cores, no GPU; 81 compiles in about 25 s):

<!-- BEGIN demo -->
```text
$ triton-portcheck check --demo
triton-portcheck 0.1.0 · triton 3.9.0 (python 3.13.18) · no GPU involved

kernel               cuda:75  cuda:80  cuda:90  cuda:100  cuda:120  hip:gfx942  hip:gfx950  hip:gfx1100  hip:gfx1250
--------------------------------------------------------------------------------------------------------------------
axpy                 ok       ok       ok       ok        ok        ok          ok          ok           ok
gamma_ratio          ok       ok       ok       ok        ok        front:A     front:A     front:A      front:A
gamma_ratio_guarded  ok       ok       ok       ok        ok        ok          ok          ok           ok
quantize_fp8         front:B  front:B  ok       ok        ok        ok          ok          ok           ok
scatter_rows         ok       ok       asm:C    ok        asm:C     ok          ok          ok           ok
dot_small[fp16]      ok       ok       ok       ok        ok        ok          ok          ok           ok
dot_small[int8]      mlir:D   ok       ok       ok        ok        ok          ok          ok           ok
rowsum_blockptr      front:E  front:E  front:E  front:E   front:E   front:E     front:E     front:E      front:E
poll_flags           ok       ok       ok       ok        ok        ok          ok          ok           ok

[A] frontend-error in stage frontend: libdevice.tgamma is not implemented for the hip backend
    gamma_ratio on hip:gfx942, hip:gfx950, hip:gfx1100, hip:gfx1250
    why: The kernel calls libdevice.tgamma; Triton maps triton.language.extra.libdevice to a per-
         backend module and the hip one does not define this function. Guard the call with a target
         check or express it with functions both backends have.
    see: https://github.com/triton-lang/triton/issues/10375
[B] frontend-error in stage frontend: dtype fp8e4nv is not supported on this architecture
    quantize_fp8 on cuda:75, cuda:80
    why: The backend's supported_fp8_dtypes for this target does not include fp8e4nv, so the
         frontend rejects any kernel that names it (on NVIDIA, fp8e4nv needs compute capability 8.9
         or newer).
[C] assembler-error in stage cubin: ptxas: .tile::scatter4 is not supported on sm_90a
    scatter_rows on cuda:90, cuda:120
    why: Triton lowered a tensor-descriptor scatter to an instruction this target does not
         implement; the error only appears when the assembler runs.
    see: https://github.com/triton-lang/triton/issues/11344
[D] mlir-error in stage ttgir: 'arith.extf' op operand #0 must be floating-point-like, but got 'tensor<32x32xi8, ...>' (pass TritonGPUAccelerateMatmul)
    dot_small[int8] on cuda:75
    why: A compiler pass rejected the IR for this target. This is usually an unsupported combination
         of operation, dtype and architecture rather than a mistake in the kernel.
[E] frontend-error in stage frontend: block pointers were removed from this Triton version
    rowsum_blockptr on cuda:75, cuda:80, cuda:90, cuda:100, cuda:120, hip:gfx942, hip:gfx950,
      hip:gfx1100, hip:gfx1250
    why: tl.make_block_ptr / tl.advance no longer exist in this release; Triton points to the tensor
         descriptor API instead.

static pre-checks (tables read from the installed Triton) predicted 6 failing cells before
compiling; 6 of them did fail

81 cells: 63 ok, 18 failed (18 new) · slowest compile 0.6 s (gamma_ratio on cuda:90)
Compile-only check: a passing cell means Triton produced a binary for that target. It does not prove
numerical correctness, that the kernel fits launch-time resource limits, or anything about
performance on the device. Nothing was executed on a GPU.
```
<!-- END demo -->

Every failure above is real compiler output, grouped by a stable fingerprint, with the stage that failed and — where
we have reproduced it and read the upstream report — a link.

> **What this is not.** A green cell means *Triton produced a binary for that target*. It says nothing about
> numerical correctness, about whether the kernel fits the device's launch-time limits, or about performance.
> **Nothing in this project has been run on a real GPU.** See [what a green cell proves](#what-a-green-cell-does-and-does-not-prove).

## Why

Portability failures in Triton are compile-time failures more often than people expect, and they are specific to a
target or a release:

- **AMD has a much smaller libdevice.** The HIP backend implements 59 of the 198 `libdevice` functions the CUDA
  backend has in Triton 3.9.0, so `libdevice.tgamma(x)` compiles for every NVIDIA target and for no AMD target
  ([triton#10375](https://github.com/triton-lang/triton/issues/10375)).
- **Dtypes depend on the architecture.** `fp8e4nv` is rejected on Ampere and older, which blocks whole models on
  A100-class cards ([vllm#50576](https://github.com/vllm-project/vllm/issues/50576) lists "Triton on SM80 being unable
  to emit `fp8e4nv`" among its blockers). Other fp8 paths pass the frontend and die in lowering
  ([triton#11898](https://github.com/triton-lang/triton/issues/11898),
  [#12259](https://github.com/triton-lang/triton/issues/12259),
  [#12726](https://github.com/triton-lang/triton/issues/12726)).
- **Some errors only appear in the assembler.** `desc.scatter()` lowers to an instruction that `ptxas` refuses on
  some targets ([triton#11344](https://github.com/triton-lang/triton/issues/11344), reported for sm_12x; here it
  also reproduces for sm_90).
- **Releases move.** Block pointers compile on 3.8.0 and are gone in 3.9.0; fixes on `main` are missing from
  release wheels ([triton#11735](https://github.com/triton-lang/triton/issues/11735)); compile times and tuned
  optima shift between versions ([triton#11373](https://github.com/triton-lang/triton/issues/11373)).

The two things this tool cannot see are also real and worth knowing about: kernels that compile and then exceed a
device's shared memory at launch ([triton#12494](https://github.com/triton-lang/triton/issues/12494)), and
performance regressions. Those need hardware.

## Install

```bash
pip install git+https://github.com/cb-Sigma/triton-portcheck
```

Python 3.11+ on Linux. The tool itself has no dependencies and never imports Triton in its own process; it drives
the Triton you already have (`pip install triton`), or releases it installs into separate environments on request.
It is not on PyPI.

## Quick start

```bash
triton-portcheck check --demo                 # the matrix above
triton-portcheck targets                      # targets and tables your Triton exposes
triton-portcheck capture -- pytest tests/     # record the specialisations your tests use -> portcheck.toml
triton-portcheck check                        # compile them for every target
triton-portcheck check --triton current,3.8.0 --json report.json
triton-portcheck diff report.json --versions 3.8.0 3.9.0
```

## The manifest

A manifest lists kernel functions and the specialisations to compile. Triton compiles a kernel per *specialisation*
(argument types, `tl.constexpr` values, a few options), so that is what a row of the matrix is.

```toml
[defaults]
targets = ["cuda:80", "cuda:90", "cuda:100", "hip:gfx942", "hip:gfx950"]
triton  = ["current"]            # or ["current", "3.8.0"]
timeout = 300

[[kernel]]
name      = "attention-fp16"
target    = "mypkg.kernels:attention"         # module:function, or path/to/file.py:function
signature = { q_ptr = "*fp16", k_ptr = "*fp16", out_ptr = "*fp16", seq_len = "i32" }
constants = { BLOCK = 64, HEAD_DIM = 128 }
options   = { num_warps = 8, num_stages = 3 }
xfail     = ["hip:gfx1100"]                   # known not to work there; does not fail CI

  [[kernel.variant]]                          # same kernel, another specialisation
  name      = "bf16"
  signature = { q_ptr = "*bf16", k_ptr = "*bf16", out_ptr = "*bf16" }
```

The value formats are Triton's own: `signature` uses the type strings Triton derives for arguments, and `constants`
uses the encoding of `triton.runtime.jit.serialize_specialization_data`. The child process hands the result to
`JITFunction.preload`, so the compile goes through the same code path as a real launch. Full reference:
[docs/manifest.md](docs/manifest.md).

## Capture: let your tests write the manifest

Writing signatures by hand is tedious and easy to get subtly wrong (was `n` specialised to a constant because it
was 1? was the pointer 16-byte aligned?). `capture` runs your test-suite or script and records what Triton was
actually asked to compile, through Triton's own `jit_cache_hook`:

```bash
triton-portcheck capture -o portcheck.toml -- pytest tests/
triton-portcheck capture --mode record -- python bench.py --small
```

| mode | needs a GPU | what happens to kernels |
|---|---|---|
| `interpret` (default) | no | run in Triton's CPU interpreter (`TRITON_INTERPRET=1`), so tests that check results pass |
| `record` | no | specialised and recorded, not compiled, not executed (fastest; result-checking tests fail) |
| `live` | yes | compiled and launched as usual on your device while being recorded |

Both CPU modes need the code path to reach `kernel[grid](...)` with CPU tensors. See [docs/capture.md](docs/capture.md).

## Versions

```bash
triton-portcheck env add 3.8.0        # prints wheel size, expected and actual disk use
triton-portcheck env list
triton-portcheck env remove 3.8.0
```

Each release lives in its own virtual environment under `~/.cache/triton-portcheck/envs` (override with
`TRITON_PORTCHECK_HOME`). A Triton wheel is large: 3.8.0 measured 893 MiB installed. `check --triton 3.8.0` creates
a missing environment on demand and says so. These environments hold only Triton; your other imports (torch, your
package) are found through the invoking interpreter's site-packages as a fallback (`--no-inherit-site` turns that
off, `env add 3.8.0 --with torch` installs them properly instead).

<!-- BEGIN versions -->
```text
$ triton-portcheck check --demo --triton 3.8.0,3.9.0 -t cuda:80,cuda:90,hip:gfx942 --json report.json
$ triton-portcheck diff report.json --versions 3.8.0 3.9.0
triton-portcheck diff: triton 3.8.0 -> triton 3.9.0

3 regressed, 3 fixed, 0 fail differently, 21 unchanged, 0 slower to compile

Compiles on triton 3.8.0, fails on triton 3.9.0:
  rowsum_blockptr @ cuda:80: ok -> frontend-error
      block pointers were removed from this Triton version
  rowsum_blockptr @ cuda:90: ok -> frontend-error
      block pointers were removed from this Triton version
  rowsum_blockptr @ hip:gfx942: ok -> frontend-error
      block pointers were removed from this Triton version

Fails on triton 3.8.0, compiles on triton 3.9.0:
  poll_flags @ cuda:80: frontend-error -> ok
      triton.language.atomic_load does not exist in this Triton version
  poll_flags @ cuda:90: frontend-error -> ok
      triton.language.atomic_load does not exist in this Triton version
  poll_flags @ hip:gfx942: frontend-error -> ok
      triton.language.atomic_load does not exist in this Triton version

Size and metadata changes:
  gamma_ratio @ cuda:80: cubin 32496 -> 47480 (+46%), llir 56350 -> 164474 (+192%), ptx 23042 -> 38282 (+66%), ptxas_registers 21 -> 22
  gamma_ratio @ cuda:90: cubin 33368 -> 47840 (+43%), llir 56350 -> 164474 (+192%), ptx 23043 -> 37194 (+61%), ptxas_registers 21 -> 22
  gamma_ratio_guarded @ cuda:80: cubin 32624 -> 47736 (+46%), llir 56374 -> 164498 (+192%), ptx 23130 -> 38370 (+66%), ptxas_registers 21 -> 22
  gamma_ratio_guarded @ cuda:90: cubin 33496 -> 47968 (+43%), llir 56374 -> 164498 (+192%), ptx 23131 -> 37282 (+61%), ptxas_registers 21 -> 22

Compile-only check: a passing cell means Triton produced a binary for that target. It does not prove numerical correctness, that the kernel fits launch-time resource limits, or anything about performance on the device. Nothing was executed on a GPU.
```
<!-- END versions -->

`diff` also compares two targets (`--targets cuda:80 cuda:90`) or two saved runs (`diff old.json new.json`), flags
compile-time regressions beyond a threshold, and with `check --save-ir DIR` can show a textual diff of normalised
IR at a chosen stage (`--ir ttgir`).

## How it works

- **One subprocess per cell.** Compiles can hang or take minutes (one 256³ `tl.dot` tile took 39 s for `cuda:80`
  here). Every cell runs in its own process and session with a timeout (`--timeout`) and a resident-memory limit
  over the whole process tree (`--max-rss`, Linux), so `ptxas` is killed too. A crash inside LLVM is one red cell,
  not a dead run, and the report still names the stage that was running.
- **The fake active driver.** `triton.compile(..., target=...)` cross-compiles without a device, but
  `triton.language.target_info.is_hip()`, `is_cuda()` and `cuda_capability_geq()` ask the *active driver* for the
  current target. With no GPU there is none, they all return `False`, and a kernel with `if is_hip(): ... else: ...`
  silently compiles the wrong branch. The child installs a stand-in driver that reports the cell's target, so
  guards resolve as they will on the device. `gamma_ratio_guarded` in the demo exists to test exactly this.
- **Stages are Triton's.** The child wraps the stages the backend registers (`ttir → ttgir → llir → ptx → cubin`,
  or `… → amdgcn → hsaco`) to time them and to know which one raised; nothing about the pipeline is re-implemented.
- **Static pre-checks come from Triton's tables.** Before compiling, the installed Triton is asked which libdevice
  functions each backend defines and which fp8 dtypes each target supports. Predictions are shown next to the real
  result, `--static skip` saves the compile, and `--static-only` gives an instant first pass. No table is copied
  into this repository.

More detail in [docs/how-it-works.md](docs/how-it-works.md).

## CI

```yaml
- uses: actions/setup-python@v7
  with: { python-version: "3.12" }
- run: pip install triton==3.9.0 -e .          # your package and the Triton you support
- uses: cb-Sigma/triton-portcheck@main
  with:
    targets: cuda:80,cuda:90,cuda:100,hip:gfx942,hip:gfx950
    baseline: ci/portcheck-baseline.json       # optional: fail only on new breakage
```

The action writes the matrix to the job summary, annotates failing kernels at their source line, and leaves JSON,
JUnit and SARIF reports behind. Without the action: `triton-portcheck check --github --junit report.xml --sarif
report.sarif`. Exit codes: `0` no new failures, `1` new failures, `2` bad arguments or manifest, `3` Triton or
environment problem. A [pre-commit hook](.pre-commit-hooks.yaml) is included. See [docs/ci.md](docs/ci.md).

## What a green cell does and does not prove

| | |
|---|---|
| ✅ proves | Triton's frontend, MLIR passes, LLVM backend and the vendor assembler/linker bundled with the wheel all accepted this specialisation for this target |
| ❌ does not prove | the kernel computes the right values on that device |
| ❌ does not prove | it fits launch-time limits (shared memory, threads, tensor memory): those are checked by the driver at load time, on the device |
| ❌ does not prove | it is fast, or even reasonably scheduled |
| ❌ does not prove | the driver, runtime or a different `ptxas` on the target machine agrees |

Every report format the tool writes repeats this, because a green matrix is easy to over-read.

## What has actually been tested

| | |
|---|---|
| Triton versions compiled with | **3.9.0** and **3.8.0** (PyPI wheels, x86_64 Linux, Python 3.13) |
| NVIDIA targets compiled for | `cuda:75`, `cuda:80`, `cuda:90`, `cuda:100`, `cuda:120` |
| AMD targets compiled for | `hip:gfx942`, `hip:gfx950`, `hip:gfx1100`, `hip:gfx1250` |
| Capture modes exercised | `interpret` and `record` with CPU torch 2.15 |
| On a real GPU | **nothing** — not the kernels, not `live` capture |

Other versions and targets may well work — target lists are read from the installed Triton — but they are
untested. Older Triton releases without `JITFunction.preload` are not supported.

## FAQ

**Is this an official Triton tool?** No. It is an independent project, not affiliated with or endorsed by OpenAI,
AMD, NVIDIA or the Triton maintainers.

**Does it need CUDA or ROCm installed?** No. The Triton wheel bundles LLVM, `ptxas` and the AMD code generator.

**My kernel module imports torch. Do version environments need it?** By default they borrow it from the
interpreter you ran the tool with. Use `env add X --with torch` plus `--no-inherit-site` for full isolation.

**A cell is red but the kernel runs fine on that GPU.** Check that the specialisation in the manifest is the one
you launch (capture it), and that the Triton version matches. If it still disagrees, please open an issue.

**A cell is green but the kernel fails on that GPU.** Expected for launch-time limits and runtime faults; see the
table above.

**Why is a failure not linked to an upstream issue?** Links are only added for failures that were reproduced here
and whose upstream report was read. Use the "New failure signature" issue template to add one.

**Can the static checks be wrong?** They are predictions from tables. Target guards built from
`triton.language.target_info` are evaluated; other dead branches are not, so a prediction can be a false alarm. The
report says when a predicted cell compiled anyway.

## Related work

- [tritonparse](https://github.com/meta-pytorch/tritonparse) (Meta) traces and visualises compilations and can
  capture for a virtual target, but its tracing needs a host GPU. Use it to *understand* a compile; use this to
  gate many targets and versions on a CPU runner.
- [ttsem](https://github.com/alepot55/ttsem) executes Triton IR on CPU to find out-of-bounds accesses and races.
  It checks behaviour, for CUDA, above the LLVM conversion; this tool checks that the whole pipeline including the
  assembler succeeds, for CUDA and AMD. They complement each other.
- [TileLens / triton-viz](https://github.com/Deep-Learning-Profiling-Tools/triton-viz) visualises and sanitises
  kernels through the interpreter.
- [tritonbench](https://github.com/meta-pytorch/tritonbench) measures performance on real GPUs — the thing this
  tool cannot do.

## Roadmap

- Out-of-tree backends (Intel XPU, CPU, Metal): the runner is backend-agnostic and reads backends from Triton's
  entry points, but none has been tried.
- Gluon kernels.
- A pytest plugin for capture, and capture from arbitrary executables.
- More reproduced failure signatures.
- Memory guard on non-Linux hosts.

## Contributing

Issues and pull requests are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md). Unit tests need no Triton
(`pytest -m "not compile"`); the integration tests really compile and take about a minute.

## License

Apache-2.0. See [LICENSE](LICENSE).

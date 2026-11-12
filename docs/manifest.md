# Manifest reference

A manifest is a TOML file (YAML and JSON are accepted too; YAML needs `pip install 'triton-portcheck[yaml]'`).
`triton-portcheck check` looks for `portcheck.toml` in the current directory when no path is given.

## `[defaults]`

| key | meaning |
|---|---|
| `targets` | list of targets, e.g. `["cuda:80", "hip:gfx942"]`. Overridden by `--target`. |
| `triton` | list of versions: `"current"` (the interpreter running the tool) or a release such as `"3.8.0"`. |
| `timeout` | seconds allowed per cell (default 300). |
| `max_rss` | memory limit per cell, e.g. `"8G"` (default 8G; `"0"` disables). |
| `jobs` | cells compiled in parallel (default 2). |
| `pythonpath` | directories, relative to the manifest, put on `sys.path` before importing kernels. |

## `[[kernel]]`

| key | meaning |
|---|---|
| `target` | **required.** `module:function` or `path/to/file.py:function` (relative to the manifest). Dotted attributes work (`mod:Ops.kernel`); `triton.autotune` wrappers are unwrapped. |
| `name` | row label; defaults to the function name. Must be unique. |
| `signature` | type of every runtime argument, as Triton spells it (table below). |
| `constants` | value of every `tl.constexpr` argument, and of any argument Triton would specialise to a constant. |
| `options` | compile options such as `num_warps`, `num_stages`, `num_ctas`, `enable_fp_fusion`. Options a backend does not have are ignored by that backend and noted in the report. |
| `align` | `true` (default): pointer arguments are assumed 16-byte aligned, which is what Triton sees for ordinary torch tensors. |
| `attrs` | explicit per-argument attributes in Triton's form, e.g. `{ x_ptr = [["tt.divisibility", 16]] }`. Overrides `align`. Written by `capture`. |
| `targets` | restrict this kernel to some targets. |
| `xfail` | patterns of cells expected to fail; they do not fail the run. `"hip:*"`, `"cuda:7?"`, `"cuda:90@3.8.*"`. |
| `skip` | patterns of cells not to run at all. |
| `timeout` | per-kernel override. |
| `variant` | array of tables overriding any of the above (except `target`) to add specialisations of the same kernel. Rows are named `name[variant]`. |

Arguments with a default value may be omitted.

### Type strings

| Python argument at launch | type string |
|---|---|
| `torch.float32` tensor | `*fp32` |
| `float16` / `bfloat16` / `float64` tensor | `*fp16` / `*bf16` / `*fp64` |
| `int8` / `int32` / `int64` / `uint8` tensor | `*i8` / `*i32` / `*i64` / `*u8` |
| fp8 tensors | `*fp8e4nv`, `*fp8e5`, `*fp8e4b15`, ... |
| Python `int` | `i32` (or `i64` when it does not fit) |
| Python `float` | `fp32` |
| Python `bool` | `u1` |
| tensor descriptor | `tensordesc<fp32[1,32]>` |
| tuple argument | a list of the element types |

When in doubt, run `capture`: it writes exactly what Triton derived.

### Constant values

Plain TOML values, plus Triton's serialised forms:

| value | meaning |
|---|---|
| `128`, `1.5`, `true`, `"mean"` | passed as is |
| `"fp16"` (any dtype name) | becomes `tl.dtype("fp16")`, as in Triton's own `preload` |
| `{ constexpr = [1, 2] }` | a `tl.constexpr` wrapping the value |
| `{ jit_function = "pkg.mod:helper" }` | another `@triton.jit` function passed as an argument |
| key `"cfg.1"` | element 1 of the tuple argument `cfg` |

TOML has no null; a manifest that needs `None` constants should be written as JSON or YAML.

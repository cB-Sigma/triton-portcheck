# How it works

```
triton-portcheck (parent, never imports triton)
 ├─ probe      one child per environment: version, backends, targets, tables
 ├─ static     one child per environment: table-driven predictions for all cells
 └─ compile    one child per cell: kernel × target × Triton version
```

The child is a single file, `_child.py`, that only needs the standard library and Triton. It is executed by path
with the interpreter of the environment under test, so per-version environments do not need this package installed.

## A compile cell

1. The parent writes a request file and starts the child in a new session with `TRITON_CACHE_DIR` pointing at a
   private temporary directory and `TRITON_ALWAYS_COMPILE=1`, so no cell can be satisfied from a cache.
2. The child resolves the target. The warp size is asked from the backend's own option parser.
3. It installs the **fake active driver** (see below) and imports the kernel.
4. It builds the specialisation blob in the format of `serialize_specialization_data` and calls
   `JITFunction.preload(blob)`, which constructs `ASTSource` and calls `triton.compile` exactly as a launch would.
5. `backend.add_stages` is wrapped so each stage Triton registers is timed and announced on an event stream that
   is flushed line by line. If the process dies or is killed, the parent still knows the stage.
6. On success the child reports artifact sizes, stage times and the metadata keys the compiler stages added
   (`shared`, `tmem_size`, `global_scratch_size`, `tensordesc_meta`, ...). Options echoed back into the metadata
   are not reported. On NVIDIA the register count comes from the `ptxas -v` log that Triton prints when
   `TRITON_DUMP_PTXAS_LOG=1`. Registers are otherwise only known after loading the binary on a device.

## Statuses

| status | meaning |
|---|---|
| `ok` | a device binary was produced |
| `frontend-error` | option validation or Python → Triton IR failed |
| `mlir-error` | a pass in the `ttir`, `ttgir` or `llir` stage failed |
| `codegen-error` | the LLVM backend failed to emit assembly (`ptx`, `amdgcn`) |
| `assembler-error` | `ptxas` or the AMD assembler/linker rejected the code (`cubin`, `hsaco`) |
| `timeout` / `oom` | the cell exceeded `--timeout` / `--max-rss` and its process tree was killed |
| `crash` | the compiler process died without reporting (abort, signal) |
| `load-error` | the kernel could not be imported, or the manifest entry does not match its parameters |
| `predicted-fail` | not compiled; a table in the installed Triton says it cannot work (`--static skip/only`) |

For backends this tool does not know, the last registered stage counts as the assembler and earlier unknown stages
as code generation.

## The fake driver

`triton.language.target_info` implements `is_cuda()`, `is_hip()`, `cuda_capability_geq()` and friends by calling
`driver.active.get_current_target()`. In a compile-only process on a CPU machine there is no active driver; the
helpers catch that and return `False`. A kernel such as

```python
if is_hip():
    y = tl.exp(libdevice.lgamma(a) - libdevice.lgamma(b))
else:
    y = libdevice.tgamma(a) / libdevice.tgamma(b)
```

then compiles its `else` branch for an AMD target and fails, although it is correct. The child therefore calls
`driver.set_active(...)` with an object whose `get_current_target()` returns the cell's target. The integration
test `test_fake_driver_makes_target_guards_resolve` compiles this kernel both ways.

## Isolation and limits

- Timeout and memory are enforced by the parent. Memory is the summed resident size of every process in the
  child's session, read from `/proc` five times a second, so a runaway `ptxas` counts. On hosts without `/proc`
  there is no memory guard.
- On timeout or limit the whole session is killed with `SIGKILL`.
- stdout and stderr go to files, never pipes, so a chatty compiler cannot deadlock the run.

## Fingerprints and known signatures

A failure's message is cleaned (for `ptxas`, the PTX dump is dropped and only the `error :` lines are kept; for
MLIR, the first diagnostic and the failing pass name are taken from stderr) and normalised (paths, numbers,
`sm_XX`, `gfxNNN`, SSA names) before hashing, so the same failure on two targets or versions gets one fingerprint
and one entry in the report. `classify.py` holds the table of known signatures; an upstream link is attached only
when the failure was reproduced with a released wheel and the issue was read.

## Static pre-checks

Two tables are read from the installed Triton, per target:

- `backend.get_module_map()["triton.language.extra.libdevice"]` — the module that implements libdevice for that
  backend. A `libdevice.<name>` the kernel references that the module does not define is predicted to fail.
- `backend.parse_options({}).supported_fp8_dtypes` — fp8 dtypes named by the signature or the kernel source that
  are not in it are predicted to fail.

The kernel source (and the `@triton.jit` functions it calls) is scanned syntactically. `if` statements whose
condition is a `target_info` helper are evaluated for the target, like the frontend does.

## Target lists

`triton-portcheck targets` shows what the installed Triton can be asked for. NVIDIA capabilities are those whose
`--gpu-name` the bundled `ptxas` accepts, using the backend's own `get_ptxas` and `sm_arch_from_capability`. Triton
keeps no list of AMD architectures, so the names its AMD backend sources special-case are shown and labelled as
such; any other name is passed through to LLVM.

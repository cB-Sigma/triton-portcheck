# Capturing specialisations

```bash
triton-portcheck capture [-o portcheck.toml] [--mode interpret|record|live] -- <command>
```

`<command>` must start with `python`, `pytest` or a `.py` file:

```bash
triton-portcheck capture -- pytest -x tests/kernels
triton-portcheck capture -- python -m mypkg.bench --small
triton-portcheck capture -- scripts/warmup.py
```

The command runs in-process under a small bootstrap that installs a hook, then every distinct specialisation is
written to the manifest (duplicates are collapsed; several specialisations of one kernel are named `kernel#1`,
`kernel#2`, ...).

## How it records

Triton calls `knobs.runtime.jit_cache_hook` right before it compiles a kernel and passes its own serialised
specialisation: argument types, constants, attributes and options. That blob is what gets stored, and it is the same
blob `JITFunction.preload` accepts, so nothing about specialisation keys is re-derived by this tool. Options equal to
the backend defaults are dropped so the manifest stays portable across backends.

## Modes

**`interpret`** (default) sets `TRITON_INTERPRET=1`. Triton's interpreter runs each kernel on CPU tensors, so tests
that compare results still pass. The interpreter does not specialise, so each launch is additionally fed to a shadow
`JITFunction` whose compile is skipped; that is where the record comes from. Interpret mode needs `numpy` in the environment that has Triton: the interpreter imports it, but Triton does not declare it as a dependency. `capture` stops with a clear message when it is missing.

**`record`** installs the fake driver and returns "handled" from the hook. Kernels are specialised and then neither
compiled nor launched. It is the fastest mode and works for code the interpreter cannot run, but output tensors are
left untouched, so result-checking tests fail (the command's exit status is reported and otherwise ignored).

**`live`** fakes nothing. Use it on a machine with a GPU to record what a real run compiles. It has not been
exercised by the maintainers, because there is no GPU in their test environment.

## Requirements and limits

- The code must reach `kernel[grid](...)` on this machine. Without a GPU that means the tests must be able to run
  with CPU tensors; code that calls `.cuda()` unconditionally will stop before any kernel launches.
- Kernels defined inside another function cannot be imported again and are skipped with a note.
- In the CPU modes the fake driver reports `--reference-target` (default `cuda:80`) to host code that asks.
  Host-side branches on the target therefore take the CUDA path while capturing.
- Kernels in files outside site-packages are referenced by path relative to the manifest.

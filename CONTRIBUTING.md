# Contributing

Thanks for taking the time. This is a small project; the bar is "does what it says and says what it does not do".

## Setup

```bash
git clone https://github.com/cb-Sigma/triton-portcheck
cd triton-portcheck
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
```

## Checks

```bash
ruff check . && ruff format --check .
mypy
pytest -m "not compile"        # unit tests, no Triton required, a few seconds
```

Integration tests really compile kernels. They need an interpreter with Triton (and torch for the capture tests):

```bash
pip install triton==3.9.0 torch --extra-index-url https://download.pytorch.org/whl/cpu
pytest -m compile              # about a minute on 4 cores
```

If Triton lives in another environment, point the tests at it: `TRITON_PORTCHECK_PYTHON=/path/to/python pytest`.
The cross-version test runs only when a managed 3.8.0 environment exists (`triton-portcheck env add 3.8.0`,
about 0.9 GiB).

The output blocks in the README are produced by the tests. After changing the demo or the terminal renderer:

```bash
UPDATE_README=1 pytest -m compile
```

## Ground rules

- **The parent process must not import Triton.** Anything that needs Triton goes into `src/triton_portcheck/_child.py`,
  which must stay a single file using only the standard library and Triton. Two tests enforce this.
- **No hand-copied tables.** Target lists, libdevice functions and dtype support are read from the installed
  Triton at run time.
- **Reuse Triton's machinery** (`jit_cache_hook`, `preload`, `add_stages`, `parse_options`) rather than
  re-deriving what it does; that is what keeps the tool honest across versions.
- **Upstream links need evidence.** A failure signature gets an issue link only if you reproduced the failure
  with a released wheel and the issue describes that failure. Say in the pull request how you reproduced it.
- **Do not overclaim.** Reports and docs must keep saying that this is a compile-only check. If you could not test
  something (a target, a version, real hardware), write that down.
- Example and demo kernels must be your own small originals, not copies from Triton or other projects.

## Adding a failure signature

Add a `Signature` to `SIGNATURES` in `classify.py` (order matters: first match wins), a unit test in
`tests/test_classify.py` using the real error text, and, if a small kernel can reproduce it, an assertion in
`tests/test_integration.py`.

## Releasing

Bump `__version__` in `src/triton_portcheck/__init__.py`, update `CHANGELOG.md`, tag `vX.Y.Z`. The release
workflow builds, checks and publishes through PyPI trusted publishing.

## What this changes

<!-- One or two sentences. Link the issue if there is one. -->

## How it was checked

- [ ] `ruff check . && ruff format --check . && mypy`
- [ ] `pytest -m "not compile"` (no Triton needed)
- [ ] `pytest -m compile` with Triton installed, if the change touches `_child.py`, the runner, classification or the demo kernels
- [ ] README output blocks refreshed (`UPDATE_README=1 pytest -m compile`) if the demo output changed

## Notes for the reviewer

<!-- Anything untested, any Triton version or target you could not try. -->

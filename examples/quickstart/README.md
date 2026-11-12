# Quickstart example

`kernels.py` has two kernels and the host functions that launch them; `test_kernels.py` tests them on CPU tensors.

```bash
cd examples/quickstart

# 1. run the tests under the recorder; this is how portcheck.toml in this directory was written
triton-portcheck capture -o portcheck.toml -- pytest -q .

# 2. compile every recorded specialisation for the default targets
triton-portcheck check
```

The tests launch `scaled_add` twice with different dtypes and block sizes and `row_softmax` once with
`num_warps=2`, so the manifest has three rows. The second `scaled_add` row carries explicit `attrs` because its
length (64) is divisible by 16, which Triton records as a property of that argument; the other rows match the
default (aligned pointers only) and stay short.

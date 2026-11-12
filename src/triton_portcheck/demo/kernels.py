"""Small kernels chosen because they compile on some targets and not on others.

They are deliberately tiny: the point is the portability matrix, not the
maths.  Each docstring says which divergence the kernel is here to show.
Importing this module needs Triton; nothing else in the package does.
"""

import triton
import triton.language as tl
from triton.language.extra import libdevice
from triton.language.target_info import is_hip


@triton.jit
def axpy(x_ptr, y_ptr, out_ptr, alpha, n, BLOCK: tl.constexpr):
    """Control row: plain masked element-wise arithmetic, portable everywhere."""
    start = tl.program_id(0) * BLOCK
    idx = start + tl.arange(0, BLOCK)
    live = idx < n
    x = tl.load(x_ptr + idx, mask=live, other=0.0)
    y = tl.load(y_ptr + idx, mask=live, other=0.0)
    tl.store(out_ptr + idx, alpha * x + y, mask=live)


@triton.jit
def gamma_ratio(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """Uses ``libdevice.tgamma``: present in the CUDA libdevice, absent from HIP's."""
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = idx < n
    x = tl.load(x_ptr + idx, mask=live, other=1.0)
    ratio = libdevice.tgamma(x + 0.5) / libdevice.tgamma(x)
    tl.store(out_ptr + idx, ratio, mask=live)


@triton.jit
def gamma_ratio_guarded(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """Same maths with an ``is_hip()`` guard that avoids ``tgamma`` on AMD.

    Only compiles for HIP if the guard really sees the HIP target, which in a
    compile-only process requires the fake active driver.
    """
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = idx < n
    x = tl.load(x_ptr + idx, mask=live, other=1.0)
    if is_hip():
        ratio = tl.exp(libdevice.lgamma(x + 0.5) - libdevice.lgamma(x))
    else:
        ratio = libdevice.tgamma(x + 0.5) / libdevice.tgamma(x)
    tl.store(out_ptr + idx, ratio, mask=live)


@triton.jit
def quantize_fp8(x_ptr, q_ptr, scale, n, BLOCK: tl.constexpr):
    """Casts fp32 to ``fp8e4nv``: rejected before Ada/Hopper, accepted after and on HIP."""
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = idx < n
    x = tl.load(x_ptr + idx, mask=live, other=0.0)
    tl.store(q_ptr + idx, (x * scale).to(tl.float8e4nv), mask=live)


@triton.jit
def scatter_rows(desc, row_ptr, src_ptr, ROWS: tl.constexpr, COLS: tl.constexpr):
    """Scatters rows through a tensor descriptor: the assembler rejects it on some NVIDIA targets."""
    rows = tl.load(row_ptr + tl.arange(0, ROWS))
    offs = tl.arange(0, ROWS)[:, None] * COLS + tl.arange(0, COLS)[None, :]
    desc.scatter(tl.load(src_ptr + offs), rows, 0)


@triton.jit
def dot_small(a_ptr, b_ptr, c_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    """A single ``tl.dot`` tile; with int8 operands it fails in an MLIR pass on Turing only."""
    rows = tl.arange(0, M)
    mids = tl.arange(0, K)
    cols = tl.arange(0, N)
    a = tl.load(a_ptr + rows[:, None] * K + mids[None, :])
    b = tl.load(b_ptr + mids[:, None] * N + cols[None, :])
    tl.store(c_ptr + rows[:, None] * N + cols[None, :], tl.dot(a, b))


@triton.jit
def rowsum_blockptr(a_ptr, out_ptr, M, N, stride_m, stride_n, BM: tl.constexpr, BN: tl.constexpr):
    """Walks a matrix with block pointers, an API that newer Triton releases removed."""
    block = tl.make_block_ptr(
        base=a_ptr, shape=(M, N), strides=(stride_m, stride_n), offsets=(0, 0), block_shape=(BM, BN), order=(1, 0)
    )
    total = tl.zeros((BM,), dtype=tl.float32)
    for _ in range(0, N, BN):
        total += tl.sum(tl.load(block, boundary_check=(0, 1)), axis=1)
        block = tl.advance(block, (0, BN))
    tl.store(out_ptr + tl.arange(0, BM), total)


@triton.jit
def poll_flags(flag_ptr, out_ptr, BLOCK: tl.constexpr):
    """Uses ``tl.atomic_load``, which older Triton releases do not have."""
    idx = tl.arange(0, BLOCK)
    seen = tl.atomic_load(flag_ptr + idx)
    tl.store(out_ptr + idx, seen + 1)

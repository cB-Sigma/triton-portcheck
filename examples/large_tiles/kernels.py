"""A single very large ``tl.dot`` tile: slow or impossible to assemble on some targets."""

import triton
import triton.language as tl


@triton.jit
def tile_matmul(a_ptr, b_ptr, c_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    rows = tl.arange(0, M)
    mids = tl.arange(0, K)
    cols = tl.arange(0, N)
    a = tl.load(a_ptr + rows[:, None] * K + mids[None, :])
    b = tl.load(b_ptr + mids[:, None] * N + cols[None, :])
    tl.store(c_ptr + rows[:, None] * N + cols[None, :], tl.dot(a, b))

"""Two small kernels and the host-side wrappers that launch them."""

import torch
import triton
import triton.language as tl


@triton.jit
def scaled_add(x_ptr, y_ptr, out_ptr, alpha, n, BLOCK: tl.constexpr):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = idx < n
    x = tl.load(x_ptr + idx, mask=live, other=0.0)
    y = tl.load(y_ptr + idx, mask=live, other=0.0)
    tl.store(out_ptr + idx, alpha * x + y, mask=live)


@triton.jit
def row_softmax(x_ptr, out_ptr, n_cols, stride, WIDTH: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, WIDTH)
    live = cols < n_cols
    x = tl.load(x_ptr + row * stride + cols, mask=live, other=-float("inf"))
    x = x - tl.max(x, axis=0)
    e = tl.exp(x)
    tl.store(out_ptr + row * stride + cols, e / tl.sum(e, axis=0), mask=live)


def add(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0, block: int = 128) -> torch.Tensor:
    out = torch.empty_like(x)
    n = x.numel()
    scaled_add[(triton.cdiv(n, block),)](x, y, out, alpha, n, BLOCK=block)
    return out


def softmax(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    rows, cols = x.shape
    row_softmax[(rows,)](x, out, cols, x.stride(0), WIDTH=triton.next_power_of_2(cols), num_warps=2)
    return out

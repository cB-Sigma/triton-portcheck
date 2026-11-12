"""CPU tests for the example kernels.

They pass under Triton's interpreter (``TRITON_INTERPRET=1``), which is also
how ``triton-portcheck capture`` runs them to learn the specialisations.
"""

import kernels
import torch


def test_add_fp32():
    x, y = torch.rand(300), torch.rand(300)
    torch.testing.assert_close(kernels.add(x, y, alpha=2.0), 2.0 * x + y)


def test_add_fp16_other_block():
    x, y = torch.rand(64, dtype=torch.float16), torch.rand(64, dtype=torch.float16)
    torch.testing.assert_close(kernels.add(x, y, block=64), x + y)


def test_softmax():
    x = torch.rand(4, 24)
    torch.testing.assert_close(kernels.softmax(x), torch.softmax(x, dim=1), rtol=1e-4, atol=1e-6)

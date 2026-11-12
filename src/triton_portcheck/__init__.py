"""triton-portcheck: compile Triton kernels across GPU targets and Triton versions without a GPU.

The package core never imports Triton.  Everything that touches Triton runs in
a child process (see ``_child.py``) started with the interpreter of the
environment being checked.
"""

__version__ = "0.1.0"

DISCLAIMER = (
    "Compile-only check: a passing cell means Triton produced a binary for that target. "
    "It does not prove numerical correctness, that the kernel fits launch-time resource "
    "limits, or anything about performance on the device. Nothing was executed on a GPU."
)

__all__ = ["DISCLAIMER", "__version__"]

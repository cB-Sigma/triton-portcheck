"""``python -m triton_portcheck``."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())

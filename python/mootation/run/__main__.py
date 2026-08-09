# SPDX-License-Identifier: Apache-2.0
"""Entry point for `python -m mootation.run`."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())

# SPDX-License-Identifier: Apache-2.0
"""Driving MOOtation from a TOML file, with objectives from external programs.

    python -m mootation.run --show --check run.toml

The library itself is C++ and knows nothing about any of this. This package is
the layer above it: it reads a run description, checks it before anything
expensive starts, executes the external steps that produce objective values,
and keeps an append-only journal so a run can be resumed.

Zero dependencies — `tomllib` has been in the standard library since 3.11. The
terminal UI is a separate, optional layer on top (it needs `textual`); nothing
here imports it, so `--check` and the runner work on a bare interpreter.

Design rule, from TUI_SPEC.md: the config is a FILE. Nothing in this package
writes one. A config assembled by clicking is not reproducible, cannot be
diffed, and cannot be attached to a paper.
"""

from .config import (
    Config,
    ConfigError,
    load,
    loads,
)
from .parsers import ParseError, parse_output
from .ledger import Ledger, Record

__all__ = [
    "Config",
    "ConfigError",
    "load",
    "loads",
    "ParseError",
    "parse_output",
    "Ledger",
    "Record",
]

__version__ = "0.1.0"

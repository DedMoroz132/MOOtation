# SPDX-License-Identifier: Apache-2.0
"""The terminal interface (TUI_SPEC.md §6).

Read-only by design. The config is a FILE: you edit it in your own editor, and
this shows you what it resolved to, what the validator thinks of it, and what a
run is doing. There are no forms and no wizards, because a configuration
assembled by clicking is one that cannot be diffed, copied to a cluster, or
attached to a paper.

Screens, in the order TUI_SPEC.md gives:

    Config      the parsed TOML, the validation result, the commands and paths
                as resolved for THIS platform. Read-only.
    Problems    the benchmark registry, filtered by family and objective count;
                for an external problem, the chain of steps.
    Algorithms  which of the 60 are selected, and with what parameters.
    Monitor     live progress from the journal, with a stop.

This subpackage needs `textual`; nothing else in `mootation.run` does. The
import below turns a missing dependency into a sentence rather than a
traceback, because `--check` and the runner are supposed to work on a bare
interpreter and a user who never wanted a UI should not be told to install one.
"""

from __future__ import annotations

try:
    import textual as _textual          # noqa: F401
except ImportError as _e:               # pragma: no cover - depends on machine
    raise ImportError(
        "mootation.tui needs Textual (the rest of mootation.run does not). "
        "Install it with `pip install textual`, or use `python -m mootation.run "
        "--show --check <config>`, which prints the same information without a "
        "UI."
    ) from _e

from .app import MootationApp, run           # noqa: E402

__all__ = ["MootationApp", "run"]

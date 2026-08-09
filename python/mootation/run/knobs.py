# SPDX-License-Identifier: Apache-2.0
"""The optional algorithm knobs, read out of the C++ header that defines them.

`knob_names()` in include/mootation/settings.hpp is the authority: it is what
Settings::validate() checks against and what embed.hpp's setter table applies.
Parsing it here keeps this package from carrying a second copy that silently
falls behind when a knob is added.

If the header cannot be found — an installed layout, a wheel — the list falls
back to a snapshot. The snapshot can go stale, so it is used only as a last
resort and is marked as such in `knob_source()`.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

# Last resort only. Mirrors knob_names() in settings.hpp as of 0.1.0.
_FALLBACK = (
    "eta_c", "eta_m", "pc", "pm", "T", "delta", "nr", "kappa",
    "K", "n_clusters", "theta", "alpha", "F", "CR", "div",
)

_BLOCK = re.compile(
    r"knob_names\s*\(\s*\)\s*\{.*?\{(?P<body>.*?)\}\s*;", re.S)
_STRING = re.compile(r'"([A-Za-z_][A-Za-z0-9_]*)"')


def _settings_hpp() -> Path | None:
    here = Path(__file__).resolve()
    for c in (here.parents[3] / "include" / "mootation" / "settings.hpp",
              Path.cwd() / "include" / "mootation" / "settings.hpp"):
        if c.is_file():
            return c
    return None


@lru_cache(maxsize=1)
def _read() -> tuple[tuple[str, ...], str]:
    p = _settings_hpp()
    if p is None:
        return _FALLBACK, "fallback (settings.hpp not found)"
    m = _BLOCK.search(p.read_text(encoding="utf-8"))
    if not m:
        return _FALLBACK, f"fallback (no knob_names() body in {p})"
    names = tuple(_STRING.findall(m.group("body")))
    if not names:
        return _FALLBACK, f"fallback (knob_names() body in {p} is empty)"
    return names, str(p)


def knob_names() -> tuple[str, ...]:
    """Every key accepted in `[[algorithms]] params`."""
    return _read()[0]


def knob_source() -> str:
    """Where the list came from — a path, or a note that it is the snapshot."""
    return _read()[1]

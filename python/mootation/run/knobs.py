# SPDX-License-Identifier: Apache-2.0
"""The optional algorithm knobs, read out of the C++ header that defines them.

`knob_names()` in include/mootation/settings.hpp is the authority: it is what
Settings::validate() checks against and what embed.hpp's setter table applies.
Parsing it here keeps this package from carrying a second copy that silently
falls behind when a knob is added.

A checkout has the header in include/; an installed package carries a copy
beside itself (python/CMakeLists.txt installs it into the wheel). If neither is
found — a package copied by hand — the list falls back to a snapshot. The
snapshot can go stale, so it is used only as a last resort and is marked as
such in `knob_source()`.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

# Last resort only. Mirrors knob_names() and text_knobs() in settings.hpp.
_FALLBACK = (
    "eta_c", "eta_m", "pc", "pm", "T", "delta", "nr", "kappa",
    "K", "n_clusters", "theta", "alpha", "F", "CR", "div", "normalize",
    "mutation_scale", "mixture_q", "blx_alpha", "sbx_var_prob",
)
_FALLBACK_TEXT = {
    "bound_repair": ("clip", "reflect", "random", "midpoint", "resample", "wrap", "native"),
    "crossover": ("sbx", "uniform", "blx_alpha"),
    "mutation": ("polynomial", "gaussian", "cauchy", "uniform_reset", "mixture",
                 "mixture_cauchy"),
}

_BLOCK = re.compile(
    r"knob_names\s*\(\s*\)\s*\{.*?\{(?P<body>.*?)\}\s*;", re.S)
# text_knobs(): {"name", {"word", "word", ...}}, ... inside the static vector
_TEXT_BLOCK = re.compile(
    r"text_knobs\s*\(\s*\)\s*\{.*?=\s*\{(?P<body>.*?)\}\s*;", re.S)
_TEXT_ITEM = re.compile(r'\{\s*"([A-Za-z_][A-Za-z0-9_]*)"\s*,\s*\{([^}]*)\}\s*\}')
_STRING = re.compile(r'"([A-Za-z_][A-Za-z0-9_]*)"')


def _settings_hpp() -> Path | None:
    here = Path(__file__).resolve()
    for c in (here.parents[3] / "include" / "mootation" / "settings.hpp",
              here.parents[1] / "settings.hpp",
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


@lru_cache(maxsize=1)
def _read_text() -> dict:
    p = _settings_hpp()
    if p is None:
        return dict(_FALLBACK_TEXT)
    m = _TEXT_BLOCK.search(p.read_text(encoding="utf-8"))
    if not m:
        return dict(_FALLBACK_TEXT)
    return {name: tuple(_STRING.findall(words)) for name, words in _TEXT_ITEM.findall(m.group("body"))}


def text_knobs() -> dict:
    """{knob: the words it takes} for the knobs whose value is a word."""
    return _read_text()


def knob_names() -> tuple[str, ...]:
    """Every key accepted in `[[algorithms]] params`: the numeric knobs, then the text ones."""
    return _read()[0] + tuple(k for k in text_knobs() if k not in _read()[0])


def knob_source() -> str:
    """Where the list came from — a path, or a note that it is the snapshot."""
    return _read()[1]

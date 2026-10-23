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
    "mutation_scale", "mixture_q", "blx_alpha", "sbx_var_prob", "step_share", "F_spread",
    "ref_delta", "de_p", "de_memory", "p_de", "sa_chains", "sa_reseed", "sa_t_ratio", "sa_s0",
    "sa_s_end", "sa_archive_cap", "sa_psa_alpha", "sa_adapt_window", "coords_warmup",
    "coords_update", "spu_frac", "inner_archive_cap", "archive_reuse_p", "dn_cf",
    "vmax_frac", "vel_on_bound_low", "vel_on_bound_up", "w_min", "w_max", "c_min",
    "c_max", "turbulence_k", "turbulence_p", "archive_size", "eps_archive_eta",
    "r2_weights", "r2_offset", "bc_lambda", "paes_depth", "mads_tau", "mads_delta0",
    "glods_search_every", "glods_local_tol", "rm_extension", "n_sub",
    "sub_budget", "fd_step",
    "local_tol", "cma_sigma0", "eps_objective", "polish_share", "polish_points", "mosqp_tau",
    "mosqp_mu", "mosqp_beta", "mosqp_spread", "n_samples",
)
_FALLBACK_TEXT = {
    "bound_repair": ("clip", "reflect", "random", "midpoint", "resample", "wrap", "native", "r_reflect"),
    "crossover": ("sbx", "uniform", "blx_alpha", "spx", "rex", "undx", "pcx", "de", "de_pbest",
                  "portfolio", "none"),
    "mutation": ("polynomial", "gaussian", "cauchy", "uniform_reset", "mixture",
                 "mixture_cauchy", "none"),
    "de_adapt": ("none", "shade"),
    "de_base": ("rand1", "current1", "current_to_pbest1"),
    "de_index": ("wor", "wr", "wpr"),
    "de_adapt_weight": ("equal", "contribution"),
    "de_archive": ("yes", "no"),
    "portfolio": ("sbx", "uniform", "blx_alpha", "spx", "rex", "undx", "pcx", "de", "de_pbest"),
    "portfolio_credit": ("uniform", "borg"),
    "sa_energy": ("cmb", "eps", "dom", "tch"),
    "sa_step": ("mix", "laplace", "de", "sbx", "adaptive"),
    "sa_weights": ("psa", "fixed"),
    "sa_tch_accept": ("weighted_sum", "chebyshev"),
    "coords": ("axis", "pca", "pca_nd", "randrot"),
    "survival": ("greedy", "stochastic"),
    "inner_archive": ("off", "nd"),
    "bin_crossover": ("uniform", "none"),
    "mating": ("tournament", "each"),
    "crowding_space": ("objectives", "decision"),
    "dms_init": ("line", "single", "random"),
    "dms_poll": ("first", "random"),
    "smpso_preset": ("jmetal", "platemo", "fair", "omopso"),
    "chi_mode": ("signed", "abs", "none"),
    "turbulence": ("every_k", "prob", "thirds", "off"),
    "r_per_dim": ("no", "yes"),
    "archive_prune": ("one_by_one", "one_shot"),
    "leader_select": ("crowding_tournament", "random"),
    "leader_offer": ("all", "improved"),
    "r2_mode": ("exact", "discrete"),
    "r2_space": ("weight", "angle"),
    "bc_selection": ("bc", "bf", "ba"),
    "mads_select": ("spread", "first"),
    "mads_poll": ("complete", "opportunistic"),
    "mads_search": ("speculative", "none"),
    "mads_cache": ("off", "on", "force"),
    "dms_cache": ("off", "on", "force"),
    "paes_cache": ("off", "on", "force"),
    "cache_check": ("off", "on"),
    "boost_scalarization": ("eps", "minmax"),
    "scalarization": ("eps", "tch", "nbi", "nnc", "ps"),
    "constraint_mode": ("feasibility", "eps_level", "none"),
    "local": ("sqp", "nm", "cmaes"),
    "sub_plan": ("grid", "laumanns", "gap"),
    "tch_weights": ("inverse", "lattice"),
    "hv_algo": ("auto", "wfg", "box"),
    "hv_contrib": ("iwfg", "hvc", "wfg", "race", "mc"),
    "polish": ("off", "sqp", "nm"),
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


# The text knobs that take a LIST of their words (settings.hpp, is_list_knob):
# a TOML list, or the words separated by commas.
LIST_KNOBS = ("portfolio",)


def accepts(name: str, value) -> bool:
    """Is `value` a valid value of the text knob `name`?"""
    words = text_knobs().get(name)
    if words is None:
        return False
    if name in LIST_KNOBS:
        items = value if isinstance(value, (list, tuple)) else str(value).split(",")
        return bool(items) and all(isinstance(w, str) and w in words for w in items)
    return value in words


# The run layer's own knobs: minimize's keywords, not a core's — the integer
# and discrete variables' repair and mutation index (mootation/var_types.py;
# task 7, §10). Not in settings.hpp: a C++ caller declares no variable types.
PY_KNOBS = ("eta_int",)
PY_TEXT = {"int_repair": ("lamarck", "baldwin")}


def text_knobs() -> dict:
    """{knob: the words it takes} for the knobs whose value is a word."""
    return {**_read_text(), **PY_TEXT}


def knob_names() -> tuple[str, ...]:
    """Every key accepted in `[[algorithms]] params`: the numeric knobs, then the text ones."""
    return _read()[0] + PY_KNOBS + tuple(k for k in text_knobs() if k not in _read()[0])


def knob_source() -> str:
    """Where the list came from — a path, or a note that it is the snapshot."""
    return _read()[1]

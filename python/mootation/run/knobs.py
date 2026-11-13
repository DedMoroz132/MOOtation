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

import json
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
    "grad_cost", "lm_ftol", "lm_xtol", "lm_gtol", "lm_f_scale", "lbfgsb_m", "lbfgsb_pgtol", "lbfgsb_ftol",
    "np", "jade_p", "jade_c", "shade_h", "shade_p_max", "lshade_p", "lshade_r_arc", "lshade_n_min", "lshade_max_nfe", "ga_tournament", "g3_mu", "g3_lambda", "g3_sigma", "chc_divergence", "chc_bits", "pso_phi", "spso_k",
    "noise_frac", "eki_ensemble", "eki_sigma0", "eki_gamma", "eki_noise", "eki_tau", "n_starts", "threads",
    "min_dist", "min_df", "start_share", "bipop_large", "mlsl_n", "mlsl_sigma", "mlsl_gamma", "mlsl_tau",
    "mlsl_nu", "start_budget", "bh_temperature", "bh_step", "bh_accept_rate", "bh_interval", "ils_step",
    "ils_restart", "ils_temperature", "memetic_every", "memetic_points", "local_share",
    "ga_block_uniform", "ga_block_mutation",
    "tiktak_n", "tiktak_share",
    "dc_share",
    "pap_interval", "pap_size",
    "noise_frac", "eki_ensemble", "eki_sigma0", "eki_gamma", "eki_noise", "eki_tau", "gng_eps",
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
    "sbx_bounds": ("bounded", "clip"),
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
    "theta_ties": ("slot", "letter"),
    "k_share": ("pp", "dp"),
    "polish": ("off", "sqp", "nm"),
    "lm_loss": ("linear", "huber", "cauchy"),
    "pso_topology": ("ring", "global"),
    "pso_v0": ("random", "zero"),
    "pm_literal": ("on", "off"),
    "flag1": ("off", "on"),
    "flag2": ("off", "on"),
    "fitness_reading": ("swap", "ibea"),
    "extreme_weights": ("nsga3", "literal"),
    "slot_exclusive": ("off", "on"),
    "nm_coefficients": ("adaptive", "standard"),
    "lm_gauge": ("off", "on"),
    "eki_init": ("around", "sobol", "lhs"),
    "eki_perturb": ("yes", "no"),
    "kpcm_pm": ("off", "on"),
    "edv_tournament": ("fitness", "random"),
    "cma_weights": ("active", "positive"),
    "cma_cmu": ("tutorial2016", "hansen2009"),
    "cma_stop": ("spread", "h09"),
    "cma_stagnation": ("h09", "either", "off"),
    "start_points": ("sobol", "lhs", "uniform", "halton_rand", "sobol_owen"),
    "cma_bounds": ("penalty", "none"),
    "ils_accept": ("better", "rw", "restart", "lsmc"),
    "memetic_pick": ("best", "spread"),
    "keep_points": ("all", "best"),
    "sym": ("off", "canon", "align"),
    "cv_rule": ("angle", "lmea", "platemo"),
    "cv_norm": ("population", "nondominated"),
    "nadir_source": ("intercepts", "archive"),
    "drs_guard": ("off", "normalize", "extreme"),
    "clia_norm": ("raw", "intercepts"),
    "clia_rpil": ("on", "off"),
    "gng_ties": ("first", "random"),
    "arv_linkage": ("single", "average", "complete"),
    "min_sym": ("off", "canon", "align"),
    "aprd_pbi": ("canonical", "printed"),
    "rd_ideal": ("running", "origin"),
    "if_range": ("pool", "alg1"),
    "nrv_archive": ("letter", "platemo"),
    "nrv_pool": ("letter", "platemo"),
}
_FALLBACK_SOLVER = ("inner", "local_solver")
_FALLBACK_JSON = ("dc_stages", "pap_members")
_JSON_BLOCK = re.compile(
    r"json_knobs\s*\(\s*\)\s*\{.*?\{(?P<body>.*?)\}\s*;", re.S)

_BLOCK = re.compile(
    r"knob_names\s*\(\s*\)\s*\{.*?\{(?P<body>.*?)\}\s*;", re.S)
# text_knobs(): {"name", {"word", "word", ...}}, ... inside the static vector
_TEXT_BLOCK = re.compile(
    r"text_knobs\s*\(\s*\)\s*\{.*?=\s*\{(?P<body>.*?)\}\s*;", re.S)
_TEXT_ITEM = re.compile(r'\{\s*"([A-Za-z_][A-Za-z0-9_]*)"\s*,\s*\{([^}]*)\}\s*\}')
_SOLVER_BLOCK = re.compile(
    r"solver_knobs\s*\(\s*\)\s*\{.*?\{(?P<body>.*?)\}\s*;", re.S)
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


@lru_cache(maxsize=1)
def _read_solver() -> tuple[str, ...]:
    p = _settings_hpp()
    if p is None:
        return _FALLBACK_SOLVER
    m = _SOLVER_BLOCK.search(p.read_text(encoding="utf-8"))
    return tuple(_STRING.findall(m.group("body"))) if m else _FALLBACK_SOLVER


def solver_knobs() -> tuple[str, ...]:
    """The knobs whose value is a solver: a table {solver = name, its knobs}
    (settings.hpp, solver_knobs(); scalar/strategy.hpp)."""
    return _read_solver()


@lru_cache(maxsize=1)
def _read_json() -> tuple[str, ...]:
    p = _settings_hpp()
    if p is None:
        return _FALLBACK_JSON
    m = _JSON_BLOCK.search(p.read_text(encoding="utf-8"))
    return tuple(_STRING.findall(m.group("body"))) if m else _FALLBACK_JSON


def json_knobs() -> tuple[str, ...]:
    """The knobs whose value is a JSON array: data_continuation's stages,
    pap's members (settings.hpp, json_knobs(); task 19, §19.6)."""
    return _read_json()


def canonical(value) -> str:
    """A solver table's canonical JSON text: keys sorted, no spaces — what the
    library takes, and the same text from both engines."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def solver_issues(name: str, value, algorithms) -> list[str]:
    """What is wrong with the solver table `value` of knob `name`: it names
    no solver or one not in `algorithms`, or a knob of it is not a knob (its
    own solver tables checked the same way)."""
    if not isinstance(value, dict):
        return [f"{name} must be a table {{solver = name, its knobs ...}}, got {value!r}"]
    solver = value.get("solver")
    if not isinstance(solver, str) or solver not in algorithms:
        return [f"{name}.solver = {solver!r} is not a single-objective solver or strategy"]
    out = []
    valid, words, inner = set(knob_names()), text_knobs(), solver_knobs()
    for k, v in value.items():
        if k == "solver":
            continue
        if k in inner:
            out += solver_issues(f"{name}.{k}", v, algorithms)
        elif k not in valid:
            out.append(f"{name}.{k}: unknown parameter")
        elif k in words and not accepts(k, v):
            out.append(f"{name}.{k} = {v!r} is not one of: {', '.join(words[k])}")
        elif k not in words and not isinstance(v, (int, float, bool)):
            out.append(f"{name}.{k}: expected a number or true/false, got {type(v).__name__}")
    return out


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
    return (_read()[0] + PY_KNOBS + tuple(k for k in text_knobs() if k not in _read()[0])
            + tuple(k for k in solver_knobs() if k not in _read()[0])
            + tuple(k for k in json_knobs() if k not in _read()[0]))


def knob_source() -> str:
    """Where the list came from — a path, or a note that it is the snapshot."""
    return _read()[1]

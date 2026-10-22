# SPDX-License-Identifier: Apache-2.0
"""What the C++ side accepts, derived from the C++ side.

The algorithm list is read out of include/mootation/algorithms.def — the same
X-macro file the test suite, the Python binding and the run-time dispatch are
generated from. Nothing here is a second copy of that list, because a second
copy is a list that goes stale.

The population-size rules below ARE hand-maintained, and they are the one place
in this package that can drift from the C++ headers. They are written down
because the alternative is worse: without them a typo costs a day-long run ten
minutes in, which is exactly the loss `--check` is meant to prevent. Each
entry names the mechanism in the header it mirrors, so a reader can check it.
"""

from __future__ import annotations

import math
import re
from functools import lru_cache
from pathlib import Path

# ── The registry ────────────────────────────────────────────────────────────

_ALG_RE = re.compile(r"^\s*MOOTATION_ALG\(\s*([A-Za-z0-9_]+)\s*,")


def _algorithms_def() -> Path:
    """Locate algorithms.def relative to this file, then relative to the CWD.

    Installed layouts move things around, so both the source checkout
    (python/mootation/run/ -> ../../include/mootation/) and a run from the
    repository root are tried before giving up.
    """
    here = Path(__file__).resolve()
    # python/mootation/run/algorithms.py -> repository root, then the include
    # tree; plus an installed layout, where the .def sits beside the package.
    candidates = [
        here.parents[3] / "include" / "mootation" / "algorithms.def",
        here.parents[1] / "algorithms.def",
        Path.cwd() / "include" / "mootation" / "algorithms.def",
    ]
    for c in candidates:
        if c.is_file():
            return c
    raise FileNotFoundError(
        "cannot find include/mootation/algorithms.def; looked in: "
        + ", ".join(str(c) for c in candidates)
    )


@lru_cache(maxsize=1)
def algorithm_names() -> tuple[str, ...]:
    """Every name accepted as `[[algorithms]] name`, in registry order."""
    text = _algorithms_def().read_text(encoding="utf-8")
    names = []
    for line in text.splitlines():
        m = _ALG_RE.match(line)
        if m:
            names.append(m.group(1))
    if not names:
        raise ValueError(f"no MOOTATION_ALG entries found in {_algorithms_def()}")
    return tuple(names)


_SECTION_RE = re.compile(r"^\s*//\s*──\s*(.+?)\s*─")


@lru_cache(maxsize=1)
def algorithm_families() -> tuple[tuple[str, tuple[str, ...]], ...]:
    """The families algorithms.def groups its entries under, in file order.

    Read from the section comments of the same file the names come from, so a
    family list cannot fall behind the registry either.
    """
    groups: list = []
    current = None
    for line in _algorithms_def().read_text(encoding="utf-8").splitlines():
        m = _SECTION_RE.match(line)
        if m:
            current = (m.group(1), [])
            groups.append(current)
            continue
        m = _ALG_RE.match(line)
        if m:
            if current is None:
                current = ("algorithms", [])
                groups.append(current)
            current[1].append(m.group(1))
    return tuple((name, tuple(names)) for name, names in groups if names)


# ── Population-size rules ───────────────────────────────────────────────────

# Cores that call das_dennis::generate_exact: the population size must equal a
# Das-Dennis lattice point count for the objective count, exactly. Anything
# else throws std::invalid_argument at setup.
#
# Derived from: grep -l generate_exact include/mootation/algorithms/*.hpp
EXACT_LATTICE = frozenset({
    "a_nsga3", "adaw", "crea", "ctaea", "edv", "irea", "mbra", "moead", "moead_awa",
    "moead_dd", "moead_de", "moead_dra", "mombi2", "nsga3", "rvea", "rvea_star",
    "srv", "srv_nsga3", "theta_dea",
})

# M2M-family cores that partition the population into K subregions of equal
# size: they require pop_size % K == 0 (pop = K*S) and throw otherwise.
# K defaults to 10 in both and is settable via `params = { K = ... }`.
#
# Derived from: the `is not divisible by` throws in moead_m2m.hpp and
# sms_m2m.hpp. moead_am2m and isde_rd also carry a K, but they adapt instead of
# throwing, so they are not listed.
K_DIVISIBLE = {"moead_m2m": 10, "sms_m2m": 10}

# The cores whose number of reference points (n_ref_points) and of children
# (n_offspring) can be set apart from the population size (task 15, §2;
# NSGA3-L in nsga3.hpp and its kin).
FREE_H = frozenset({"nsga3", "a_nsga3", "theta_dea", "srv_nsga3"})


def das_dennis_count(m: int, h: int) -> int:
    """Number of points in the m-objective Das-Dennis lattice with h divisions."""
    return math.comb(h + m - 1, m - 1)


def lattice_sizes(m: int, limit: int) -> list[int]:
    """Attainable single-layer lattice sizes for m objectives, up to `limit`."""
    out = []
    h = 1
    while True:
        n = das_dennis_count(m, h)
        if n > limit:
            break
        out.append(n)
        h += 1
    return out


def nearest_lattice_sizes(m: int, pop: int) -> tuple[int | None, int | None]:
    """The attainable lattice sizes bracketing `pop`: (largest <=, smallest >).

    Returned so an error message can say "use 91 or 105" instead of merely
    "invalid", which is the difference between a fixable message and a riddle.
    """
    below = None
    h = 1
    while True:
        n = das_dennis_count(m, h)
        if n == pop:
            return (n, n)
        if n > pop:
            return (below, n)
        below = n
        h += 1


def two_layer_sizes(m: int, limit: int) -> list[int]:
    """Attainable TWO-layer lattice sizes (Deb & Jain 2014 §V: boundary H1 plus
    inner H2 < H1) for m objectives, up to `limit`. generate_exact accepts these
    as well as the single-layer sizes."""
    out = set()
    h1 = 1
    while das_dennis_count(m, h1) <= limit:
        for h2 in range(1, h1):
            s = das_dennis_count(m, h1) + das_dennis_count(m, h2)
            if s <= limit:
                out.add(s)
        h1 += 1
    return sorted(out)


def seed_independent(name: str, params: dict | None, n_vars: int, init: str | None = None) -> bool:
    """Whether a run of `name` with these knobs is the same run for every
    seed (task 17, supplement 2, §5; checked at the defaults over every core
    and Python method, TASK17 checks/seed_independent.md): DMS unless
    dms_init or dms_poll is "random" (DMS-6); MultiGLODS up to 100 variables
    (its search the unscrambled Sobol sequence; above, uniform points from the
    seed, MG-2); BoostDMS and MOSQP (a line, the box's centre, SLSQP). An
    initial plan of `init` other than the cores' own (uniform) makes any of them read
    the seed."""
    params = params or {}
    if init not in (None, "", "uniform"):
        return False
    if name == "dms":
        return params.get("dms_init", "line") != "random" and params.get("dms_poll", "first") != "random"
    if name == "multiglods":
        return n_vars <= 100
    return name in ("boostdms", "mosqp")


def check_pop(name: str, pop: int, n_objs: int, params: dict | None = None
              ) -> str | None:
    """Return a human-readable reason `pop` is unusable, or None if it is fine.

    Only the two hard constraints are checked — the ones that abort the run at
    setup. Cores that round a request up to the nearest lattice (generate_auto)
    are deliberately not flagged: they warn at run time and keep going, so
    refusing them here would be stricter than the library.
    """
    params = params or {}

    # Task 15, §2: given n_ref_points, these take any population size (their
    # reference points no longer come from it)
    if name in FREE_H and params.get("n_ref_points"):
        return None
    if name in EXACT_LATTICE:
        below, above = nearest_lattice_sizes(n_objs, pop)
        if below == pop:
            return None
        # FIX 2026-09-05: the two-layer sizes (91+..., 275 = 220+55 at M=10,
        # 135 = 120+15 at M=15) are accepted by generate_exact too; refusing
        # them here was stricter than the library.
        if pop in two_layer_sizes(n_objs, pop):
            return None
        hint = ", ".join(str(v) for v in (below, above) if v is not None)
        return (
            f"pop = {pop} is not a Das-Dennis lattice size for n_objs = {n_objs}; "
            f"{name} requires an exact lattice. Nearest attainable: {hint}"
        )

    if name in K_DIVISIBLE:
        k = int(params.get("K", params.get("n_clusters", K_DIVISIBLE[name])))
        if k < 1:
            return f"K = {k} must be >= 1"
        if pop % k != 0:
            divisors = [d for d in range(2, min(pop, 64) + 1) if pop % d == 0]
            hint = (", ".join(str(d) for d in divisors[:8]) if divisors
                    else "none in 2..64")
            return (
                f"pop = {pop} is not divisible by K = {k}; {name} partitions the "
                f"population into K equal subregions (pop = K*S). Divisors of "
                f"{pop}: {hint}"
            )

    return None


# ── How each core normalizes the objectives ─────────────────────────────────

# Task 15, §3 (after Tanabe, "Investigating normalization in preference-based
# evolutionary multi-objective optimization using a reference point", ASOC
# 2024, source normalization-pbemo_tanabe2024): the way a core puts its
# objectives on one scale, read from its header and code. Hand-maintained like
# the rules above; the core's header is the source, and docs/algorithms.md,
# "How each algorithm normalizes", is this table.
#
#   PP          ideal and nadir point: the least and greatest value of each
#               objective over the current set (P ∪ Q, P or the set the step
#               works on)
#   BP          ideal point: the least value ever evaluated; nadir point: over
#               the current set
#   hyperplane  NSGA-III's: the intercepts of the hyperplane through the
#               extreme points (by their achievement scalarizing function),
#               the set's nadir where it degenerates
#   ideal       only translated by an ideal point, not scaled
#   none        the raw objectives
#   own         a scheme of the core's own (the entry says which)
#
# Tanabe's BA (the nadir from an archive of extreme points) is the way of none
# of the cores. The second field says what the normalized values serve
# ("selection" is the environmental selection) and where they come from.
NORMALIZATION = {
    "nsga2":      ("none", "selection: crowding distance, each objective divided by its range on the front"),
    "nsga2_sdr":  ("PP", "selection: the SDR frame over P ∪ Q (sdr.hpp, make_frame)"),
    "dn_nsga2":   ("none", "selection: crowding in the decision space, raw distances"),
    "bc_spx":     ("none", "selection and mating: raw objectives"),
    "bc_rex":     ("none", "as bc_spx"),
    "spea2":      ("none", "selection: raw distances in the objective space"),
    "spea2_sde":  ("none", "selection: shifted density estimation on raw objectives"),
    "grea":       ("PP", "selection and mating: the grid over the set's least and greatest values"),
    "vaea":       ("PP", "selection: over P ∪ Q (Alg. 3)"),
    "agemoea":    ("hyperplane", "selection: ideal point of the first front of P ∪ Q (this generation), "
                                 "intercepts from its extreme points"),
    "etea":       ("none", "selection: raw objectives (the paper prescribes none)"),
    "gde3":       ("none", "selection: crowding distance divided by the front's range"),
    "nsga3":      ("hyperplane", "selection: ideal point over every generation, extreme points kept "
                                 "from earlier generations (NSGA3-N1)"),
    "a_nsga3":    ("hyperplane", "as nsga3"),
    "theta_dea":  ("hyperplane", "selection: ideal point over every generation"),
    "ar_moea":    ("PP", "mating and the reference-point adaptation: ideal and nadir of P"),
    "moead":      ("ideal", "subproblems: the least value ever evaluated (z)"),
    "moead_de":   ("ideal", "as moead"),
    "moead_dra":  ("ideal", "as moead"),
    "moead_dd":   ("ideal", "subproblems and association: the least value ever evaluated"),
    "moead_awa":  ("ideal", "subproblems: the least value ever evaluated less 1e-7"),
    "adaw":       ("own", "subproblems: the least value ever evaluated less 1e-4 (ideal); the archive and "
                          "the weight adaptation: PP over A ∪ P"),
    "moead_m2m":  ("none", "subproblems: raw objectives, shifted only where a least value is negative"),
    "moead_am2m": ("own", "directions, association and subproblems: ideal and nadir of a snapshot taken "
                          "at the start and at each adaptation (AM2M-norm)"),
    "sms_m2m":    ("none", "as moead_m2m"),
    "moead_ds":   ("hyperplane", "subproblems: ideal point of P, intercepts from its extreme points"),
    "rd_emo":     ("ideal", "subproblems: the least value ever evaluated (RDE-3)"),
    "aprd":       ("own", "subregions: each subregion's own ideal point, the archive shifted to the origin"),
    "liu_gu2011": ("ideal", "association: the least values of the internal population"),
    "isde_rd":    ("ideal", "angles: the least values of the pool (ISDE-3)"),
    "dhea":       ("PP", "selection: over the population"),
    "hlmea":      ("none", "raw objectives, shifted only where a least value is negative (HLMEA-5)"),
    "ibea_eplus": ("PP", "fitness: the indicator over P ∪ Q scaled to [0, 1] (IEP-4)"),
    "ibea_hd":    ("PP", "as ibea_eplus; reference point 2 in the scaled space"),
    "mibea":      ("PP", "fitness: over the first front of P ∪ Q"),
    "r2ibea":     ("none", "fitness: raw objectives (set_normalize(true) scales them)"),
    "hype":       ("own", "fitness: a reference point from the pool's greatest values (no scaling)"),
    "mombi2":     ("own", "selection: least value ever evaluated, nadir of the previous population with "
                          "the paper's variance rule"),
    "nimmo":      ("none", "raw objectives; the indicator values are scaled within the set"),
    "irea":       ("BP", "fitness: least value ever evaluated, greatest over S_t"),
    "edv":        ("ideal", "angles: the least value ever evaluated (EDV-3)"),
    "if_maoea":   ("PP", "mating, selection and the archive: over the set each step works on"),
    "maoea_iamd": ("PP", "mating and selection: over P, and over P ∪ Q"),
    "sms_emoa":   ("own", "contributions: no scaling; from three objectives the reference point follows "
                          "the front (SMS-7)"),
    "r2_emoa":    ("own", "R2: raw objectives, the utopian point from the least value ever evaluated and "
                          "nadir(Q) (normalize = true scales by ideal(Q) and nadir(Q))"),
    "mo_cma_es":  ("own", "contributions: no scaling; reference point nadir + (nadir − ideal) over Q (MOCMA-4)"),
    "rvea":       ("own", "selection: translated by the least values of P ∪ Q (ideal); vector adaptation: "
                          "the survivors' ranges (PP)"),
    "rvea_star":  ("own", "as rvea"),
    "maoeaarv":   ("PP", "the knee and the reference: ideal and nadir of P"),
    "mbra":       ("PP", "selection: over the pool"),
    "nrv_moea":   ("own", "selection: least and greatest over the accepted fronts, the scale kept "
                          "for f_r·MaxG generations at a time"),
    "dea_gng":    ("BP", "selection (NSGA-III's): least value ever evaluated, greatest over the first front"),
    "srv":        ("PP", "selection: over S_t"),
    "srv_nsga3":  ("hyperplane", "as nsga3"),
    "camoea":     ("PP", "clustering: within the last front"),
    "cava_moea":  ("PP", "fitness and clustering: over the pool"),
    "maoea_ac":   ("hyperplane", "selection: ideal point of S, nadir by NSGA-III's intercepts"),
    "maoea_c":    ("PP", "selection: over P ∪ Q (over its non-dominated points below four objectives)"),
    "maoea_3c":   ("hyperplane", "selection: over the set, extreme points by achievement scalarizing"),
    "emyo_c":     ("ideal", "the least value ever evaluated"),
    "crea":       ("BP", "selection: least value ever evaluated, greatest over P ∪ Q"),
    "dcea":       ("ideal", "association: the previous generation's least values"),
    "hcca":       ("own", "clustering: PP; subproblems: the least value ever evaluated"),
    "lis_lcs":    ("own", "indicator and clustering: PP over the set; the least value ever evaluated"),
    "clia":       ("own", "the classifier: points projected onto the simplex"),
    "two_arch2":  ("PP", "the diversity archive's truncation: over the pool (TA2-1)"),
    "naemo":      ("ideal", "the archive's ideal point"),
    "paes":       ("own", "the grid over the archive's least and greatest values"),
    "ctaea":      ("ideal", "the least values of the set at hand (CTAEA-2)"),
    "ccmo":       ("none", "crowding distance divided by the front's range"),
    "dms":        ("none", "raw objectives (the step is scaled to the variables' box)"),
    "dmulti_mads": ("own", "the frame centre: the gaps along each objective scaled by its range"),
    "multiglods": ("none", "raw objectives (the variables' box normalized)"),
    "smpso":      ("none", "the leaders' archive: crowding distance divided by the front's range"),
    "rm_meda":    ("none", "raw objectives"),
    "mosa":       ("own", "energies: the archive's ideal point and ranges"),
}

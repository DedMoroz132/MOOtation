# SPDX-License-Identifier: Apache-2.0
"""Quality indicators for a campaign: IGD, IGD+, and the hypervolume.

All three take the answer set as an (n x M) array of objective rows and the
problem's reference data, and return one float. They are written for the
population sizes a benchmark run produces (a few hundred points), not for
archives of tens of thousands; NumPy is the only dependency.

The hypervolume follows the registry's convention (Tanabe & Oyama 2017):
objectives are normalised by the problem's ideal and nadir, the reference
point is (1.1, ..., 1.1), and the value is divided by 1.1^M so that a perfect
front scores close to 1. Points that do not dominate the reference point
contribute nothing. Exact (the WFG algorithm) up to M = 5 objectives by
default; above that a Monte-Carlo estimate with a fixed seed, which is what the
literature does and is stated in the record as `hv_method`. Both limits are
settings of hypervolume() and of a campaign (hv_exact_max_m, hv_mc_samples).
The exact value comes from the compiled extension (include/mootation/
hypervolume.hpp, about a thousand times faster) when it is built, and from
the Python recursion below otherwise; the two agree to rounding.

THE COMPILED HALF (task 10 of 2026-10-18). The nondominated filter, the
distances behind IGD, IGD+, GD+ and the additive epsilon, and the discrete R2
are computed in C++ when the extension is built (include/mootation/
nondominated.hpp and indicators.hpp; python/run_layer_module.cpp), with the
same expressions and NumPy's order of summation, so the values are the same
bit for bit; the NumPy code below stays as the reference the tests compare
with (the functions named *_numpy) and runs where the extension is not built.
"""

from __future__ import annotations

import math

import numpy as np

HV_EXACT_MAX_M = 5
HV_MC_SAMPLES = 100_000


def _compiled(name: str):
    """The extension's function `name`, or None where the extension is not
    built (or is older than this module)."""
    try:
        from .. import _core
    except ImportError:                              # pragma: no cover - no extension
        return None
    return getattr(_core, name, None)


def _plain(*arrays) -> bool:
    """Every array 2-D with rows and columns, the shape the compiled functions take."""
    return all(a.ndim == 2 and len(a) and a.shape[1] for a in arrays)


def nondominated(F: np.ndarray) -> np.ndarray:
    """Distinct rows of F that no other row dominates (minimisation), in
    lexicographic order (np.unique's).

    Each distinct row is kept once. Equal rows do not dominate each other, so
    without this every copy survives, and the WFG recursion below — whose
    limit sets are full of rows clipped to the same values — branches on every
    copy: a 126-point set with 11 distinct rows, which a MOEA/D-AM2M population
    at generation 0 is, ran for more than ten minutes instead of milliseconds.
    In C++ (include/mootation/nondominated.hpp) when the extension is built:
    O(n log n) at two and three objectives; of two rows equal but for the sign
    of a zero the first is kept, where np.unique keeps either.
    """
    F = np.asarray(F, float)
    fn = _compiled("nondominated")
    if fn is not None and len(F) > 1 and _plain(F):
        return F[fn(F)]
    return _nondominated_numpy(F)


def _nondominated_numpy(F: np.ndarray) -> np.ndarray:
    """nondominated() in NumPy: the reference."""
    F = np.asarray(F, float)
    if len(F) > 1:
        F = np.unique(F, axis=0)
    n = len(F)
    keep = np.ones(n, bool)
    for i in range(n):
        if not keep[i]:
            continue
        dominated_by = np.all(F <= F[i], axis=1) & np.any(F < F[i], axis=1)
        dominated_by[i] = False
        if np.any(dominated_by & keep):
            keep[i] = False
    return F[keep]


def igd(F: np.ndarray, ref: np.ndarray) -> float:
    """Mean over the reference front of the distance to the nearest point of F."""
    F = np.asarray(F, float)
    ref = np.asarray(ref, float)
    if F.size == 0:
        return float("inf")
    fn = _compiled("igd_distances")
    if fn is not None and _plain(F, ref) and F.shape[1] == ref.shape[1]:
        return float(fn(F, ref).mean())
    return _igd_numpy(F, ref)


def _igd_numpy(F: np.ndarray, ref: np.ndarray) -> float:
    """igd() in NumPy: the reference."""
    F = np.asarray(F, float)
    ref = np.asarray(ref, float)
    if F.size == 0:
        return float("inf")
    near = np.empty(len(ref))
    for a in range(0, len(ref), 512):                 # bounded memory on big fronts
        R = ref[a:a + 512]
        near[a:a + 512] = np.sqrt(((R[:, None, :] - F[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
    return float(near.mean())


def igd_plus(F: np.ndarray, ref: np.ndarray) -> float:
    """IGD+ (Ishibuchi, Masuda, Tanigaki & Nojima, EMO 2015, Eqs. 12 and 18,
    p = 1): the mean over reference points z of the smallest
    d+(z, a) = ||max(a − z, 0)|| over a in F — only the objectives in which a
    is worse than z count. Minimisation, raw units. Weakly Pareto-compliant
    for non-dominated sets (§5), not strictly (Table 8)."""
    F = np.asarray(F, float)
    ref = np.asarray(ref, float)
    if F.size == 0:
        return float("inf")
    return float(dplus_distances(F, ref).mean())


# ── coverage with a tolerance ────────────────────────────────────────────────
# The same directed distances IGD+ averages: d+(z, A) = min over a in A of
# ||max(a - z, 0)||, one per reference point z. Their mean is IGD+, their
# maximum the Euclidean counterpart of the additive epsilon (which takes the
# largest component where this takes the norm); a quantile sits between the
# two on the same scale, and so does the share of the front within a
# tolerance. All of them are weakly Pareto-compliant: if A weakly dominates B,
# d+(z, A) <= d+(z, B) for every z (Eq. 22 of Ishibuchi et al. 2015, who
# prove it for the mean only; the rest follows). Read "tau90 = 0.03" as "90 %
# of the front sample lies within 0.03 (normalised) of something found" — an
# engineering tolerance, and less at the mercy of one far reference point
# than eps.

COVERAGE_TAUS = (0.01, 0.02, 0.05, 0.1, 0.2)


def dplus_distances(F: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """d+(z, F) for every reference point z: the vector IGD+ is the mean of."""
    F = np.asarray(F, float)
    ref = np.asarray(ref, float)
    fn = _compiled("front_distances")
    if fn is not None and _plain(F, ref) and F.shape[1] == ref.shape[1]:
        return fn(F, ref)[0]
    return _dplus_distances_numpy(F, ref)


def _dplus_distances_numpy(F: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """dplus_distances() in NumPy: the reference."""
    F = np.asarray(F, float)
    ref = np.asarray(ref, float)
    out = np.empty(len(ref))
    for a in range(0, len(ref), 512):                 # bounded memory on big fronts
        R = ref[a:a + 512]
        d = np.sqrt((np.maximum(F[None, :, :] - R[:, None, :], 0.0) ** 2).sum(axis=2))
        out[a:a + 512] = d.min(axis=1)
    return out


def front_distances(F: np.ndarray, ref: np.ndarray) -> tuple:
    """One pass over the differences a − z of every point a of the set and z of
    the reference: the d+ matrix minimised over the set for every z (the vector
    IGD+ is the mean of, dplus_distances), minimised over the reference for
    every a (GD+'s), and the additive epsilon's min over a of max_i (a_i − z_i)
    for every z (eps_plus's). The same arithmetic as igd_plus, gd_plus,
    dplus_distances and eps_plus, element for element, so the values are the
    same."""
    F = np.asarray(F, float)
    ref = np.asarray(ref, float)
    fn = _compiled("front_distances")
    if fn is not None and _plain(F, ref) and F.shape[1] == ref.shape[1]:
        return fn(F, ref)
    return _front_distances_numpy(F, ref)


def _front_distances_numpy(F: np.ndarray, ref: np.ndarray) -> tuple:
    """front_distances() in NumPy: the reference."""
    F = np.asarray(F, float)
    ref = np.asarray(ref, float)
    to_set = np.empty(len(ref))
    to_ref = np.full(len(F), np.inf)
    eps = np.empty(len(ref))
    for a in range(0, len(ref), 512):                 # bounded memory on big fronts
        R = ref[a:a + 512]
        diff = F[None, :, :] - R[:, None, :]
        d = np.sqrt((np.maximum(diff, 0.0) ** 2).sum(axis=2))
        to_set[a:a + 512] = d.min(axis=1)
        np.minimum(to_ref, d.min(axis=0), out=to_ref)
        eps[a:a + 512] = diff.max(axis=2).min(axis=1)
    return to_set, to_ref, eps


def tau_quantile(F, ref, q: float = 0.9) -> float:
    """The q-quantile of d+(z, F) over the reference sample (NumPy's linear
    interpolation between order statistics)."""
    return float(np.quantile(dplus_distances(F, ref), q))


def coverage_curve(F, ref, taus=COVERAGE_TAUS) -> dict:
    """C(tau): the share of the reference sample within tau of the set, per tau."""
    d = dplus_distances(F, ref)
    return {f"{t:g}": float(np.mean(d <= t)) for t in taus}


# ── hypervolume ──────────────────────────────────────────────────────────────


def _hv_wfg(pl: np.ndarray, ref: np.ndarray) -> float:
    """Exact hypervolume by the WFG recursion (While, Bradstreet & Barone 2012).

    `pl` must be nondominated and every row must dominate `ref`.
    """
    n, m = pl.shape
    if n == 0:
        return 0.0
    if m == 1:
        return float(ref[0] - pl[:, 0].min())
    if m == 2:
        order = np.argsort(pl[:, 0])
        p = pl[order]
        area = 0.0
        prev_f2 = ref[1]
        for row in p:
            area += (ref[0] - row[0]) * (prev_f2 - row[1])
            prev_f2 = row[1]
        return float(area)
    # sort so that the recursion works on ever smaller limit sets
    order = np.argsort(-pl[:, m - 1])
    p = pl[order]
    total = 0.0
    for k in range(n):
        incl = float(np.prod(ref - p[k]))
        rest = p[k + 1:]
        if len(rest) == 0:
            total += incl
            continue
        limit = np.maximum(rest, p[k])          # the part of each later box inside p[k]'s
        limit = nondominated(limit)
        total += incl - _hv_wfg(limit, ref)
    return float(total)


def _hv_exact(pl: np.ndarray, ref: np.ndarray, hv_algo: str = "auto") -> float:
    """Exact hypervolume of `pl` (nondominated, inside `ref`): C++ when built,
    by box decomposition at 3 to 8 objectives and WFG elsewhere ("auto"), by
    WFG or by boxes (hv_algo; the same value up to the order of the sums);
    without the extension WFG in Python, whatever hv_algo says."""
    if hv_algo not in ("auto", "wfg", "box"):
        raise ValueError(f"hv_algo = '{hv_algo}': auto, wfg or box")
    try:
        from .. import _core
        fn = getattr(_core, "hypervolume", None)
    except ImportError:                              # pragma: no cover - no extension
        fn = None
    if fn is not None:
        P, R = np.ascontiguousarray(pl, float), np.ascontiguousarray(ref, float)
        return float(fn(P, R, hv_algo))
    return _hv_wfg(pl, ref)


_MC_DRAWS: dict = {}
_MC_DRAWS_KEPT = 8                 # (objectives, samples, seed) kept per process

# The Monte-Carlo hypervolume's version: 2 from task 17, supplement 1 (§3, the
# owner's decision of 2026-10-23), the library's generator below instead of
# NumPy's default_rng(seed); a campaign writes it to meta.json beside
# trajectory_hv wherever the hypervolume may be estimated.
HV_MC_VERSION = 2
# its generator's stream: the seed XOR this seeds a std::mt19937 (random.hpp)
_HV_MC_STREAM = 0x6A09E667


def _mc_draws(m: int, samples: int, seed: int) -> list:
    """The uniform draws behind _hv_mc, in [0, 1)^m, in chunks of 20 000 rows:
    a std::mt19937 seeded with the seed's low 32 bits XOR _HV_MC_STREAM
    (init_genrand; NumPy's MT19937 with its legacy seeding is that generator),
    rnd::uniform01 of two 32-bit words a value (genrand_res53, random.hpp),
    row after row — the same numbers as the library's generator gives on
    every platform. Drawn once per (m, samples, seed) and process and kept (a
    trajectory asks for the same points at every record)."""
    key = (int(m), int(samples), int(seed))
    if key not in _MC_DRAWS:
        if len(_MC_DRAWS) >= _MC_DRAWS_KEPT:
            _MC_DRAWS.pop(next(iter(_MC_DRAWS)))
        bg = np.random.MT19937()
        bg._legacy_seeding((int(seed) & 0xFFFFFFFF) ^ _HV_MC_STREAM)
        chunks, done = [], 0
        while done < samples:
            k = min(20_000, samples - done)
            w = bg.random_raw(2 * k * m)
            a, b = w[0::2] >> 5, w[1::2] >> 6
            U = (a.astype(float) * 67108864.0 + b.astype(float)) * (1.0 / 9007199254740992.0)
            chunks.append(U.reshape(k, m))
            done += k
        _MC_DRAWS[key] = chunks
    return _MC_DRAWS[key]


def _hv_mc(pl: np.ndarray, ref: np.ndarray, lo: np.ndarray, samples: int, seed: int) -> float:
    """Monte-Carlo hypervolume: the box [lo, ref] times the share of `samples`
    uniform points in it that some row of `pl` weakly dominates.

    The points are the library's generator's with `seed` (_mc_draws), and
    only counted by the compiled hv_covered when the extension is built, so
    the estimate does not depend on which of the two counted it.
    """
    box, hit = _hv_mc_count(pl, ref, lo, samples, seed)
    return float(box * hit / samples) if box > 0 else 0.0


def _hv_mc_count(pl: np.ndarray, ref: np.ndarray, lo: np.ndarray, samples: int, seed: int):
    """(the box's volume, how many of the samples some row of pl covers)."""
    box = np.prod(ref - lo)
    if box <= 0:
        return box, 0
    try:
        from .. import _core
        count = getattr(_core, "hv_covered", None)
    except ImportError:                              # pragma: no cover - no extension
        count = None
    pl = np.ascontiguousarray(pl, float)
    hit = 0
    for U in _mc_draws(len(ref), samples, seed):
        k = len(U)
        s = lo + U * (ref - lo)
        if count is not None:
            hit += int(count(pl, np.ascontiguousarray(s)))
        else:
            # a sample is covered if some point dominates it (<= in every coordinate)
            covered = np.zeros(k, bool)
            for row in pl:
                covered |= np.all(row[None, :] <= s, axis=1)
            hit += int(covered.sum())
    return box, hit


def hypervolume(F: np.ndarray, ideal, nadir, *, ref_scale: float = 1.1,
                mc_samples: int = HV_MC_SAMPLES, seed: int = 0,
                exact_max_m: int = HV_EXACT_MAX_M, hv_algo: str = "auto") -> tuple[float, str]:
    """Normalised hypervolume and the method used ("exact" or "mc").

    F is normalised to (F - ideal) / (nadir - ideal); the reference point is
    ref_scale in every objective; the result is divided by ref_scale^M. Exact
    up to `exact_max_m` objectives, a Monte-Carlo estimate from `mc_samples`
    points (the same points on every call with the same seed) above. The exact
    value by box decomposition at 3 to 8 objectives and WFG elsewhere
    (hv_algo = "auto", the default; task 13 of 2026-10-19, §3a), by WFG
    ("wfg") or by box decomposition ("box"; task 11 of 2026-10-18, §6.1:
    Lacour, Klamroth & Fonseca 2017), the same up to the last bits.
    """
    if hv_algo not in ("auto", "wfg", "box"):
        raise ValueError(f"hv_algo = '{hv_algo}': auto, wfg or box")
    F = np.asarray(F, float)
    ideal = np.asarray(ideal, float)
    nadir = np.asarray(nadir, float)
    if F.size == 0:
        return 0.0, "none"
    v, method, _ = hypervolume_se(F, ideal, nadir, ref_scale=ref_scale, mc_samples=mc_samples,
                                  seed=seed, exact_max_m=exact_max_m, hv_algo=hv_algo)
    return v, method


def hypervolume_se(F, ideal, nadir, *, ref_scale: float = 1.1, mc_samples: int = HV_MC_SAMPLES,
                   seed: int = 0, exact_max_m: int = HV_EXACT_MAX_M,
                   hv_algo: str = "auto") -> tuple[float, str, float | None]:
    """hypervolume() and the Monte-Carlo estimate's standard error (task 17,
    supplement 2, §2): ref_scale^-M box sqrt(p (1 - p) / samples), p the share
    of the samples covered; None for an exact value."""
    if hv_algo not in ("auto", "wfg", "box"):
        raise ValueError(f"hv_algo = '{hv_algo}': auto, wfg or box")
    F = np.asarray(F, float)
    ideal = np.asarray(ideal, float)
    nadir = np.asarray(nadir, float)
    if F.size == 0:
        return 0.0, "none", None
    fn = _compiled("hv_normalised")
    if fn is not None and _plain(F):
        v, method, se = fn(F, ideal, nadir, float(ref_scale), int(mc_samples),
                           int(seed) & 0xFFFFFFFFFFFFFFFF, int(exact_max_m), hv_algo)
        return float(v), method, (None if math.isnan(se) else float(se))
    return _hypervolume_numpy(F, ideal, nadir, ref_scale=ref_scale, mc_samples=mc_samples, seed=seed,
                              exact_max_m=exact_max_m, hv_algo=hv_algo)


def _hypervolume_numpy(F, ideal, nadir, *, ref_scale, mc_samples, seed, exact_max_m, hv_algo):
    """hypervolume() in NumPy (its exact part the extension's where built): the
    reference; ref_scale ** M the C library's pow here, the library's own in
    C++ (INDICATOR_VERSION 2: an ulp apart at some ref_scale and M)."""
    span = nadir - ideal
    span[span <= 0] = 1.0
    G = (F - ideal) / span
    m = G.shape[1]
    ref = np.full(m, float(ref_scale))
    G = G[np.all(G < ref, axis=1)]
    if len(G) == 0:
        return 0.0, "none", None
    G = nondominated(G)
    if m <= exact_max_m:
        v = _hv_exact(G, ref, hv_algo)
        method = "exact"
    else:
        lo = np.minimum(G.min(axis=0), 0.0)
        box, hit = _hv_mc_count(G, ref, lo, mc_samples, seed)
        v = float(box * hit / mc_samples) if box > 0 else 0.0
        share = hit / mc_samples
        se = float(box * math.sqrt(share * (1.0 - share) / mc_samples)) if box > 0 else 0.0
        return float(v / ref_scale ** m), "mc", se / ref_scale ** m
    return float(v / ref_scale ** m), method, None


def eps_plus(F: np.ndarray, ref: np.ndarray) -> float:
    """Additive epsilon indicator of F against a reference front.

    The smallest eps such that every reference point is weakly dominated by
    some point of F moved by eps in every objective: the maximum over r of the
    minimum over a of max_i (a_i - r_i). Zero on the front, lower is better,
    and weakly Pareto-compliant. It sees only the worst place, so it goes with
    IGD+, not instead of it — but it is the one indicator whose value reads
    directly: no worse than the front by more than eps in any objective.
    """
    F = np.asarray(F, float)
    ref = np.asarray(ref, float)
    if F.size == 0:
        return float("inf")
    fn = _compiled("front_distances")
    if fn is not None and _plain(F, ref) and F.shape[1] == ref.shape[1]:
        return float(fn(F, ref)[2].max())
    return _eps_plus_numpy(F, ref)


def _eps_plus_numpy(F: np.ndarray, ref: np.ndarray) -> float:
    """eps_plus() in NumPy: the reference."""
    F = np.asarray(F, float)
    ref = np.asarray(ref, float)
    if F.size == 0:
        return float("inf")
    worst = np.empty(len(ref))
    for start in range(0, len(ref), 256):           # bounded memory for dense fronts
        r = ref[start:start + 256]
        worst[start:start + len(r)] = (F[None, :, :] - r[:, None, :]).max(axis=2).min(axis=1)
    return float(worst.max())


def gd_plus(F: np.ndarray, ref: np.ndarray) -> float:
    """GD+ (Ishibuchi et al. 2015): IGD+'s distance averaged over the set instead.

    The same d+(r, a) = ||max(a − r, 0)|| matrix as igd_plus, minimized over the
    reference and averaged over the set: how far the points are from the front,
    blind to how much of it they cover. Read next to IGD+, it separates "still
    converging" (both fall) from "converged, now spreading" (GD+ flat, IGD+
    falling). Not Pareto-compliant, not even weakly: a set that dominates
    another can score worse (Ishibuchi et al. 2015, Fig. 8 and Table 7).
    """
    F = np.asarray(F, float)
    ref = np.asarray(ref, float)
    if F.size == 0:
        return float("inf")
    return float(front_distances(F, ref)[1].mean())


def roi_dist(F: np.ndarray, ideal, nadir) -> float:
    """Distance of the set to the region of interest [ideal, nadir], normalised.

    min over a of ||max(G(a) − 1, 0)||, G(a) = (a − ideal)/(nadir − ideal): zero
    as soon as one point has every objective at or below the nadir. COCO scores
    bbob-biobj runs that have no hypervolume yet by this distance; it tells
    apart the many runs whose HV is exactly 0 (DTLZ3 at small budgets) and
    needs no reference front.
    """
    F = np.asarray(F, float)
    if F.size == 0:
        return float("inf")
    fn = _compiled("roi_dist")
    if fn is not None and _plain(F):
        return float(fn(F, np.asarray(ideal, float), np.asarray(nadir, float)))
    ideal = np.asarray(ideal, float)
    span = np.asarray(nadir, float) - ideal
    span = np.where(span > 0.0, span, 1.0)
    G = (F - ideal) / span
    return float(np.sqrt((np.maximum(G - 1.0, 0.0) ** 2).sum(axis=1)).min())


def range_cover_each(F: np.ndarray, ideal, nadir) -> np.ndarray:
    """Per objective: the share of [ideal_j, nadir_j] that the set spans.

    Values are clipped to the box first, so a point beyond the nadir counts as
    reaching the nadir end and a set entirely outside spans nothing.
    """
    F = np.asarray(F, float)
    fn = _compiled("range_cover_each")
    if fn is not None and _plain(F):
        return np.asarray(fn(F, np.asarray(ideal, float), np.asarray(nadir, float)), float)
    ideal = np.asarray(ideal, float)
    span = np.asarray(nadir, float) - ideal
    span = np.where(span > 0.0, span, 1.0)
    G = np.clip((F - ideal) / span, 0.0, 1.0)
    return G.max(axis=0) - G.min(axis=0)


def nn_cv(F: np.ndarray, ideal, nadir) -> float | None:
    """Evenness of the set: the coefficient of variation (standard deviation over
    mean) of the distances from each distinct non-dominated point to its nearest
    other one, objectives normalised by ideal and nadir. 0 when every point is
    as far from its nearest neighbour as every other is from its own; larger as
    the set clumps. Needs no reference front, so it reads bbob-biobj too. None
    with fewer than two such points."""
    fn = _compiled("nn_cv")
    F = np.asarray(F, float)
    if fn is not None and _plain(F):
        v = fn(F, np.asarray(ideal, float), np.asarray(nadir, float))
        return None if v is None else float(v)
    return _nn_cv_numpy(F, ideal, nadir)


def _nn_cv_numpy(F, ideal, nadir) -> float | None:
    ideal = np.asarray(ideal, float)
    span = np.asarray(nadir, float) - ideal
    span = np.where(span > 0.0, span, 1.0)
    G = nondominated((np.asarray(F, float) - ideal) / span)
    if len(G) < 2:
        return None
    d = np.sqrt(((G[:, None, :] - G[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(d, np.inf)
    nn = d.min(axis=1)
    mean = float(nn.mean())
    return float(nn.std() / mean) if mean > 0.0 else None


def nd_share(F: np.ndarray) -> float:
    """Share of rows no other row dominates; equal rows do not dominate each other."""
    F = np.asarray(F, float)
    n = len(F)
    if n == 0:
        return 0.0
    fn = _compiled("nd_share")
    if fn is not None and _plain(F):
        return float(fn(F))
    le = np.all(F[:, None, :] <= F[None, :, :], axis=2)      # le[j, i]: F_j <= F_i
    lt = np.any(F[:, None, :] < F[None, :, :], axis=2)
    dominated = np.any(le & lt, axis=0)
    return float(1.0 - dominated.mean())


def dup_share(F: np.ndarray) -> float:
    """Share of rows that repeat an objective vector already in the set (exactly)."""
    F = np.asarray(F, float)
    if len(F) == 0:
        return 0.0
    fn = _compiled("dup_share")
    if fn is not None and _plain(F):
        return float(fn(F))
    return float(1.0 - len(np.unique(F, axis=0)) / len(F))


# ── the decision space ──────────────────────────────────────────────────────
def _unit(X, bounds) -> np.ndarray:
    """Variables mapped to [0, 1] by the problem's bounds."""
    X = np.atleast_2d(np.asarray(X, float))
    lo = np.array([b[0] for b in bounds], float)
    hi = np.array([b[1] for b in bounds], float)
    return (X - lo) / np.where(hi > lo, hi - lo, 1.0)


def _decision_args(X, pareto_set, bounds, cyclic):
    """The arguments of the compiled decision-space indicators, or None where
    they do not apply (no extension, an empty or ragged set): (X, sample,
    lo, hi, cyclic) as float arrays and a list."""
    X = np.atleast_2d(np.asarray(X, float))
    P = None if pareto_set is None else np.atleast_2d(np.asarray(pareto_set, float))
    if not (X.ndim == 2 and len(X) and X.shape[1]) or (P is not None and not _plain(P)):
        return None
    lo = np.array([b[0] for b in bounds], float)
    hi = np.array([b[1] for b in bounds], float)
    return X, P, lo, hi, [int(c) for c in cyclic]


def _unit_diff(A: np.ndarray, B: np.ndarray, cyclic) -> np.ndarray:
    """B − A for every pair, (len(B), len(A), d); cyclic variables wrap at 1."""
    D = B[:, None, :] - A[None, :, :]
    if len(cyclic):
        c = list(cyclic)
        D[..., c] = np.mod(D[..., c] + 0.5, 1.0) - 0.5
    return D


def igdx(X, pareto_set, *, bounds, cyclic=()) -> float:
    """IGDX (Tanabe & Ishibuchi 2019, Eq. 5): IGD in the decision space.

    The mean over a Pareto-set sample of the distance to the nearest solution,
    here in variables normalised by the problem's bounds — so that ZCAT's,
    which span [-i/2, i/2], weigh alike; on [0, 1]^n it is Eq. 5 exactly — and
    with the wrap of the cyclic variables (shiftDTLZ's distance variables,
    read mod 1), without which a solution next to its optimum across the
    wrap reads as far from it. The extension's igdx where it is built, the
    same value (decision_indicators.hpp).
    """
    fn = _compiled("igdx")
    args = _decision_args(X, pareto_set, bounds, cyclic) if fn is not None else None
    if args is not None:
        return float(fn(*args))
    return _igdx_numpy(X, pareto_set, bounds=bounds, cyclic=cyclic)


def _igdx_numpy(X, pareto_set, *, bounds, cyclic=()) -> float:
    U = _unit(X, bounds)
    R = _unit(pareto_set, bounds)
    if U.size == 0:
        return float("inf")
    best = np.empty(len(R))
    for a in range(0, len(R), 128):
        D = _unit_diff(U, R[a:a + 128], cyclic)
        best[a:a + 128] = np.sqrt((D ** 2).sum(axis=2)).min(axis=1)
    return float(best.mean())


def cover_rate(X, pareto_set) -> float:
    """CR (Tanabe & Ishibuchi 2019, Eqs. 7-8): the share of the Pareto set's
    extent in every variable that the solutions span, (prod delta_i)^(1/2D).

    delta_i is the squared overlap of the solutions' range of x_i with the
    set's, over the set's; 1 where the set is a single value, 0 where the
    ranges do not meet. Higher is better; 1 at full cover. The extension's
    where it is built (INDICATOR_VERSION 2: the root by the library's own
    pow, the same on every platform); without it the C library's.
    """
    X = np.atleast_2d(np.asarray(X, float))
    fn = _compiled("cover_rate")
    P = np.atleast_2d(np.asarray(pareto_set, float))
    if fn is not None and _plain(X, P) and X.shape[1] == P.shape[1]:
        return float(fn(X, P))
    return _cover_rate_numpy(X, pareto_set)


def _cover_rate_numpy(X, pareto_set) -> float:
    X = np.atleast_2d(np.asarray(X, float))
    P = np.atleast_2d(np.asarray(pareto_set, float))
    if X.size == 0:
        return 0.0
    smin, smax = P.min(axis=0), P.max(axis=0)
    xmin, xmax = X.min(axis=0), X.max(axis=0)
    span = smax - smin
    live = span > 0
    delta = np.ones(X.shape[1])
    delta[live] = ((np.minimum(smax, xmax) - np.maximum(smin, xmin))[live] / span[live]) ** 2
    delta[live & ((xmin >= smax) | (xmax <= smin))] = 0.0
    return float(np.prod(np.clip(delta, 0.0, 1.0)) ** (1.0 / (2 * X.shape[1])))


def psp(X, pareto_set, *, bounds, cyclic=()) -> float:
    """PSP, the Pareto sets proximity (Yue, Qu & Liang, IEEE TEVC 22(5),
    2018, Eq. 8; source mmf_yue2018): the cover rate over IGDX, larger
    better; the CEC 2019 competition ranks by 1/PSP. IGDX as igdx measures it,
    in variables normalised by the bounds — the paper's Eq. 13 is in the raw
    variables, so on a box with one range r for every variable the paper's
    PSP is this one over r. inf where IGDX is 0."""
    d = igdx(X, pareto_set, bounds=bounds, cyclic=cyclic)
    c = cover_rate(X, pareto_set)
    return float("inf") if d == 0.0 else c / d


def pairwise_distance(X, *, bounds, cyclic=()) -> float:
    """The mean distance between two solutions in normalised variables (the
    cyclic ones wrapped): how spread the set is in the decision space, with no
    reference needed. 0 for fewer than two solutions. The extension's where it
    is built, the same value."""
    fn = _compiled("pairwise_distance")
    args = _decision_args(X, None, bounds, cyclic) if fn is not None else None
    if args is not None:
        X, _, lo, hi, cyc = args
        return float(fn(X, lo, hi, cyc))
    return _pairwise_distance_numpy(X, bounds=bounds, cyclic=cyclic)


def _pairwise_distance_numpy(X, *, bounds, cyclic=()) -> float:
    U = _unit(X, bounds)
    n = len(U)
    if n < 2:
        return 0.0
    total = 0.0
    for a in range(0, n, 128):
        D = _unit_diff(U, U[a:a + 128], cyclic)
        total += float(np.sqrt((D ** 2).sum(axis=2)).sum())
    return total / (n * (n - 1))


def ps_dmin(X, pareto_set, *, bounds, cyclic=()) -> float:
    """The smallest distance from a solution to the Pareto-set sample, in
    variables normalised by the bounds with the cyclic ones wrapped, as for
    igdx: has any solution reached the Pareto set (task 6 of 2026-10-09). The
    extension's where it is built, the same value."""
    fn = _compiled("ps_dmin")
    args = _decision_args(X, pareto_set, bounds, cyclic) if fn is not None else None
    if args is not None:
        return float(fn(*args))
    return _ps_dmin_numpy(X, pareto_set, bounds=bounds, cyclic=cyclic)


def _ps_dmin_numpy(X, pareto_set, *, bounds, cyclic=()) -> float:
    U = _unit(X, bounds)
    R = _unit(pareto_set, bounds)
    if U.size == 0:
        return float("inf")
    best = float("inf")
    for a in range(0, len(R), 128):
        D = _unit_diff(U, R[a:a + 128], cyclic)
        best = min(best, float(np.sqrt((D ** 2).sum(axis=2)).min()))
    return best


# The indicators' version: 2 from task 17, supplement 1 (stage 2a, the
# owner's decision of 2026-10-23), the library's own pow (detmath.hpp) where
# metrics.py took the C library's, which differed between platforms in the
# last bit: CR's root (and so PSP) and the hypervolume's ref_scale^M (hv,
# hv_h; at some ref_scale and M one ulp off the C library's). A campaign
# writes it to every run's meta.json.
INDICATOR_VERSION = 2

_REF_STEP: dict = {}


def reference_step(ref) -> float:
    """The resolution of a reference sample: the median over its points of the
    distance to the nearest other point, in the frame it is given in. Below it
    gdp_min says no more than "on the front" (task 6 of 2026-10-09). Once per
    sample and process. Every pair: the extension's (nearest_other) or NumPy
    in blocks, the same values (task 17, stage 2a: no SciPy KD-tree, whose
    distances were its own sums)."""
    R = np.asarray(ref, float)
    key = (R.shape, hash(R.tobytes()))
    if key not in _REF_STEP:
        fn = _compiled("nearest_other")
        if fn is not None and len(R) > 1 and _plain(R):
            nearest = np.asarray(fn(R), float)
        else:
            nearest = np.empty(len(R))
            for a in range(0, len(R), 256):
                D = np.sqrt(((R[a:a + 256, None, :] - R[None, :, :]) ** 2).sum(axis=2))
                D[np.arange(len(D)), np.arange(a, a + len(D))] = np.inf
                nearest[a:a + 256] = D.min(axis=1)
        med = _compiled("median")
        _REF_STEP[key] = float(med(nearest) if med is not None and len(nearest) else np.median(nearest))
    return _REF_STEP[key]


def _normalised(F, ref, ideal, nadir):
    ideal = np.asarray(ideal, float)
    span = np.asarray(nadir, float) - ideal
    span[span <= 0] = 1.0
    return (np.asarray(F, float) - ideal) / span, (np.asarray(ref, float) - ideal) / span


def lattice_h(m: int, pop: int) -> int:
    """The largest Das-Dennis parameter H whose lattice has at most `pop` points.

    100 points at 2 objectives give H = 99, 91 at 3 give 12, 126 at 5 give 5.
    """
    from math import comb
    if m < 2:
        return 1
    h = 1
    while comb(h + 1 + m - 1, m - 1) <= pop:
        h += 1
    return h


def das_dennis(m: int, h: int) -> np.ndarray:
    """The simplex lattice of Das & Dennis (SIAM J. Optim. 8(3), 1998): every
    weight vector whose components are multiples of 1/h and sum to 1."""
    rows: list = []

    def rec(prefix, left, k):
        if k == 1:
            rows.append(prefix + [left])
            return
        for v in range(left, -1, -1):
            rec(prefix + [v], left - v, k - 1)

    rec([], int(h), int(m))
    return np.asarray(rows, float) / float(h)


def r2_discrete(F, ideal, nadir, h: int) -> float:
    """Discrete R2 (task 2, D5): the unary R2 of Brockhoff, Wagner & Trautmann
    (Evol. Comput. 23(3), 2015, Def. 4, after Hansen & Jaszkiewicz 1998) — the
    mean over the Das-Dennis lattice of H = h of min over a of
    max_i λ_i·G(a)_i, G = (F − ideal)/(nadir − ideal), so that the ideal point
    is the utopian point of the plain weighted Tchebycheff utility; the
    normalisation (ideal to 0, nadir to 1) is Schäpermeier & Kerschke's
    (arXiv:2407.01504, 2024). Lower is better. On a finite set of weights R2 is
    only weakly Pareto-compliant, and at most as many points count as there
    are weights; the integral over all weights is Pareto-compliant
    (Schäpermeier & Kerschke, exact at two objectives; Jaszkiewicz &
    Zielniewicz, IEEE TEVC 29(4), 2025)."""
    ideal = np.asarray(ideal, float)
    span = np.asarray(nadir, float) - ideal
    span[span <= 0] = 1.0
    G = (np.asarray(F, float) - ideal) / span
    W = das_dennis(G.shape[1], h)
    fn = _compiled("r2_mean")
    if fn is not None and _plain(W) and G.ndim == 2 and G.shape[1] == W.shape[1]:
        return float(fn(np.ascontiguousarray(G), W))
    return _r2_numpy(G, W)


def _r2_numpy(G: np.ndarray, W: np.ndarray) -> float:
    """r2_discrete()'s mean over the weights in NumPy: the reference."""
    best = np.full(len(W), np.inf)
    for a in range(0, len(G), 256):                  # (weights, points) in blocks
        u = (W[:, None, :] * G[None, a:a + 256, :]).max(axis=2)
        best = np.minimum(best, u.min(axis=1))
    return float(best.mean())


def magnitude(F, ideal, nadir, *, ref_scale: float) -> float:
    """Magnitude of the dominated region (Emmerich, arXiv 2604.18147, 2026,
    Theorem 3.3): Σ over the subsets S of the objectives of 2^−|S| times the
    |S|-dimensional measure of the region's projection on S, the empty S
    counting 1. In the normalised frame, G = (F − ideal)/(nadir − ideal), with
    the anchor at ref_scale in every objective (minimisation: the region is
    ∪ [G(a), r] over the points with G(a) ≤ r); 0 when no point is inside.
    At M = 2 it is 1 + (X + Y)/2 + HV/4, X and Y the region's extents
    (Theorem 2.5). Not scale-invariant, so only in this fixed frame.
    Weakly and strictly Pareto-compliant (Corollary 3.7)."""
    import itertools
    ideal = np.asarray(ideal, float)
    span = np.asarray(nadir, float) - ideal
    span[span <= 0] = 1.0
    G = (np.asarray(F, float) - ideal) / span
    m = G.shape[1]
    r = np.full(m, float(ref_scale))
    G = G[np.all(G <= r, axis=1)]
    if not len(G):
        return 0.0
    total = 1.0
    for k in range(1, m + 1):
        for S in itertools.combinations(range(m), k):
            P = G[:, list(S)]
            ref = r[list(S)]
            P = P[np.all(P < ref, axis=1)]
            if not len(P):
                continue
            vol = float(ref[0] - P[:, 0].min()) if k == 1 else _hv_exact(nondominated(P), ref)
            total += vol / 2.0 ** k
    return float(total)


def nondominated_mask(F) -> np.ndarray:
    """Which rows of F no other row dominates (minimisation); copies all kept.
    In C++ when the extension is built (nondominated())."""
    F = np.asarray(F, float)
    fn = _compiled("nondominated_mask")
    if fn is not None and _plain(F):
        return fn(F)
    return _nondominated_mask_numpy(F)


def _nondominated_mask_numpy(F) -> np.ndarray:
    """nondominated_mask() in NumPy: the reference."""
    F = np.asarray(F, float)
    keep = np.ones(len(F), bool)
    for i in range(len(F)):
        worse = np.all(F <= F[i], axis=1) & np.any(F < F[i], axis=1)
        keep[i] = not worse.any()
    return keep


def chamfer(X, Y, *, bounds, cyclic=()) -> float:
    """Averaged Hausdorff (chamfer) distance between two solution sets in
    variables normalised by the bounds (task 2, D4):
    ½·[mean over x of min over y ‖x − y‖ + mean over y of min over x ‖x − y‖];
    cyclic variables wrap at 1, as in igdx."""
    U = _unit(X, bounds)
    V = _unit(Y, bounds)
    D = np.sqrt((_unit_diff(U, V, cyclic) ** 2).sum(axis=2))    # (|V|, |U|)
    return float(0.5 * (D.min(axis=0).mean() + D.min(axis=1).mean()))


def eps_mult(A, B):
    """Binary multiplicative ε, I_ε×(A, B) = max over b of min over a of
    max_i a_i / b_i (Zitzler, Thiele, Laumanns, Fonseca & Grunert da Fonseca,
    IEEE TEVC 7(2), 2003): the smallest factor by which A, stretched, weakly
    dominates B; A is better than B iff I(A, B) ≤ 1 < I(B, A). Needs no
    reference and no frame, and does not change when an objective is
    rescaled. For non-negative objectives only (None otherwise); a zero in b
    is matched only by a zero (a_i/0 is 0 when a_i = 0 and ∞ otherwise), so
    residuals with a natural zero are compared as they are."""
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    if not A.size or not B.size or (A < 0).any() or (B < 0).any():
        return None
    num = A[None, :, :]
    den = B[:, None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        R = np.where(den > 0, num / np.where(den > 0, den, 1.0),
                     np.where(num > 0, np.inf, 0.0))
    return float(R.max(axis=2).min(axis=1).max())


def violations(X, constraints) -> np.ndarray:
    """Every row's overall violation, the sum of its constraints' positive
    values (the library's sign: <= 0 satisfied)."""
    X = np.asarray(X, float).reshape(len(X), -1) if len(X) else np.zeros((0, 0))

    def total(values):
        # added in order: builtin sum() of floats is compensated from Python 3.12
        # on and not before, which made the last bit depend on the version
        v = 0.0
        for c in values:
            v += max(c, 0.0)
        return v
    return np.array([total(constraints(list(x))) for x in X], float)


def i_cmop(F, V, ideal, nadir, tau: float = 1.0, **hv_options) -> float | None:
    """Vodopija, Tusar and Filipic's quality indicator for a constrained
    problem (IEEE TEVC 29(1), 2025, Eq. 12; source
    cmop-performance-space_vodopija2025_arxiv), smaller better. With a
    feasible point (violation V = 0): min(I_MOP, tau), I_MOP (Eq. 8) the
    negative hypervolume of the feasible points that dominate the nadir, in
    objectives normalised to [ideal, nadir] with the reference point 1
    (Eq. 9), or, while none does, their distance to that box (Eq. 10,
    roi_dist). Without one: the least violation plus tau. tau = 1, COCO's
    largest target, so every set with a feasible point scores at most 1 and
    every set without one more (their Eq. 11). None for an empty set."""
    F = np.asarray(F, float)
    V = np.asarray(V, float)
    if not len(V):
        return None
    ok = V <= 0.0
    if not ok.any():
        return float(V.min() + tau)
    G = (F[ok] - np.asarray(ideal, float)) / (np.asarray(nadir, float) - np.asarray(ideal, float))
    inside = np.all(G <= 1.0, axis=1) & np.any(G < 1.0, axis=1)      # dominates the nadir
    if inside.any():
        I = -hypervolume(F[ok][inside], ideal, nadir, ref_scale=1.0, **hv_options)[0]
    else:
        I = roi_dist(F[ok], ideal, nadir)
    return float(min(I, tau))


def feasible_mask(X, constraints) -> np.ndarray:
    """Which rows of X satisfy every constraint: constraints(x) is a list of
    values, each <= 0 when satisfied (the library's sign, BenchProblem)."""
    X = np.asarray(X, float).reshape(len(X), -1) if len(X) else np.zeros((0, 0))
    return np.array([all(c <= 0.0 for c in constraints(list(x))) for x in X], bool)


def have_frame(ideal, nadir) -> bool:
    return ideal is not None and nadir is not None


def compute(F, *, ref_front=None, ideal=None, nadir=None, which=("igd",),
            pop: int | None = None, hv_options: dict | None = None, X=None,
            pareto_set=None, bounds=None, cyclic=(), distance_fn=None,
            constraints=None) -> dict:
    """Every requested indicator in one dict; missing inputs give None.

    `pop` is the problem's default population size. hv_h places its reference
    point at 1 + 1/H with H taken from it, so every algorithm on a problem is
    measured against the same point whatever population it rounded to.
    `hv_options` goes to hypervolume() (exact_max_m, mc_samples, seed). The
    decision-space indicators need the solutions `X` and the problem's
    `bounds`, igdx, cr and ps_dmin a `pareto_set` sample besides, and g_min
    and g_median the problem's `distance_fn`.

    `constraints`, a constrained problem's (BenchProblem.constraints), needs
    `X` too: every indicator is then measured on the feasible part of the set
    alone, and feas_share, the feasible share of the set, is added (task 6 of
    2026-10-09, item 5.2). A set without a feasible point, an empty one
    included, has feas_share 0 and no value for any indicator: None, not 0 and
    not infinity (n_final 0).

    In C++ where the extension is built (run_compute.hpp; task 17, stage 2b):
    the same values in the same order; _compute_numpy is the reference.
    """
    fn = _compiled("compute_indicators")
    args = _compute_args(F, ref_front, ideal, nadir, which, pop, hv_options, X, pareto_set,
                         bounds, cyclic, distance_fn, constraints) if fn is not None else None
    if args is None:
        return _compute_numpy(F, ref_front=ref_front, ideal=ideal, nadir=nadir, which=which, pop=pop,
                              hv_options=hv_options, X=X, pareto_set=pareto_set, bounds=bounds,
                              cyclic=cyclic, distance_fn=distance_fn, constraints=constraints)
    out = dict(fn(**args))
    if "range_cover_each" in out:
        out["range_cover_each"] = [round(float(v), 6) for v in out["range_cover_each"]]
    return out


_HV_DEFAULTS = {"exact_max_m": HV_EXACT_MAX_M, "mc_samples": HV_MC_SAMPLES, "seed": 0, "hv_algo": "auto"}


def _compute_args(F, ref_front, ideal, nadir, which, pop, hv_options, X, pareto_set, bounds,
                  cyclic, distance_fn, constraints):
    """compute_indicators's arguments, or None where the compiled compute does
    not take the inputs as they are (odd shapes, an unknown name or hv
    option): _compute_numpy measures, or raises, as always."""
    from .metric_names import METRIC_NAMES
    F = np.asarray(F, float)
    if F.ndim != 2 or not F.shape[1] or any(w not in METRIC_NAMES for w in which):
        return None
    m = F.shape[1]
    hv = dict(_HV_DEFAULTS)
    if any(k not in hv for k in (hv_options or {})):
        return None
    hv.update(hv_options or {})
    if ref_front is not None:
        ref_front = np.asarray(ref_front, float)
        if not (ref_front.ndim == 2 and len(ref_front) and ref_front.shape[1] == m):
            return None
    if (ideal is None) != (nadir is None):
        return None
    for v in (ideal, nadir):
        if v is not None and np.shape(v) != (m,):
            return None
    if X is not None:
        X = np.asarray(X, float)
        if X.ndim != 2 or len(X) != len(F):
            return None
    if constraints is not None:
        if X is None:
            return None                           # _compute_numpy raises
        ok = feasible_mask(X, constraints)
    else:
        ok = None
    lo = hi = None
    if X is not None and bounds is not None:
        lo = np.array([b[0] for b in bounds], float)
        hi = np.array([b[1] for b in bounds], float)
        if lo.shape != (X.shape[1],):
            return None
    P = None
    if pareto_set is not None and np.size(pareto_set):
        P = np.atleast_2d(np.asarray(pareto_set, float))
        if X is not None and not (P.ndim == 2 and P.shape[1] == X.shape[1]):
            return None
    V = None
    if "i_cmop" in which and have_frame(ideal, nadir) and len(F) and constraints is not None:
        V = violations(X, constraints)
    g = None
    if (distance_fn is not None and X is not None and X.size
            and any(w in ("g_min", "g_median") for w in which)):
        # the problem's distance of the rows measured (the feasible ones)
        g = np.array([distance_fn(list(x)) if ok is None or ok[i] else np.nan
                      for i, x in enumerate(X)], float)
    step = float("nan")
    if ("gdp_min" in which and ref_front is not None and have_frame(ideal, nadir)
            and len(F) and (ok is None or ok.any())):
        step = reference_step(_normalised(F, ref_front, ideal, nadir)[1])
    return dict(F=F, which=list(which), ref=ref_front,
                ideal=None if ideal is None else np.asarray(ideal, float),
                nadir=None if nadir is None else np.asarray(nadir, float),
                pop=int(pop) if pop else 0, exact_max_m=int(hv["exact_max_m"]),
                mc_samples=int(hv["mc_samples"]), seed=int(hv["seed"]) & 0xFFFFFFFFFFFFFFFF,
                hv_algo=str(hv["hv_algo"]), X=X, pareto_set=P, lo=lo, hi=hi,
                cyclic=[int(c) for c in cyclic], feasible=None if ok is None else ok.astype(float),
                V=V, g=g, ref_step=step)


def _compute_numpy(F, *, ref_front=None, ideal=None, nadir=None, which=("igd",),
                   pop: int | None = None, hv_options: dict | None = None, X=None,
                   pareto_set=None, bounds=None, cyclic=(), distance_fn=None,
                   constraints=None) -> dict:
    """compute() in Python over the kernels: the reference."""
    out: dict = {}
    F = np.asarray(F, float)
    if "i_cmop" in which and have_frame(ideal, nadir) and len(F):
        # the whole set, infeasible points included (Vodopija et al., Eq. 12)
        V = (violations(X, constraints) if constraints is not None and X is not None
             else np.zeros(len(F)))
        out["i_cmop"] = i_cmop(F, V, ideal, nadir, **(hv_options or {}))
    if constraints is not None:
        if X is None:
            raise ValueError("a constrained problem's indicators need the variables X")
        ok = feasible_mask(X, constraints)
        out["feas_share"] = float(ok.mean()) if len(ok) else 0.0
        if not ok.any():
            out.update({name: (0 if name == "n_final" else None)
                        for name in which if name != "i_cmop"})
            return out
        F, X = F[ok], np.asarray(X, float)[ok]
    hvo = dict(hv_options or {})
    have_box = ideal is not None and nadir is not None
    # igdp_norm, gdp_norm, eps_norm, tau90 and gap_max read the same normalised
    # differences: one pass for all of them (front_distances; most of a
    # trajectory record's cost at five objectives otherwise), with the values
    # computing each alone gives
    shared = {}

    def dplus_norm():
        if "d" not in shared:
            G, R = _normalised(F, ref_front, ideal, nadir)
            shared["d"] = front_distances(G, R)
        return shared["d"]

    for name in which:
        if name in ("igd", "igdp", "gdp", "eps"):
            fn = {"igd": igd, "igdp": igd_plus, "gdp": gd_plus, "eps": eps_plus}[name]
            out[name] = fn(F, ref_front) if ref_front is not None else None
        elif name == "roi_dist":
            out[name] = roi_dist(F, ideal, nadir) if have_box and F.size else None
        elif name == "range_cover":
            if not have_box or F.ndim != 2 or not F.size:
                out[name] = None
            else:
                each = range_cover_each(F, ideal, nadir)
                out[name] = float(each.min())
                out["range_cover_each"] = [round(float(v), 6) for v in each]
        elif name == "nd_share":
            out[name] = nd_share(F) if F.ndim == 2 and F.size else None
        elif name == "dup_share":
            out[name] = dup_share(F) if F.ndim == 2 and F.size else None
        elif name in ("igdp_norm", "gdp_norm", "eps_norm"):
            if ref_front is None or not have_box:
                out[name] = None
            elif not F.size:
                out[name] = float("inf")
            else:
                to_set, to_ref, eps = dplus_norm()
                out[name] = float(to_set.mean() if name == "igdp_norm" else
                                  to_ref.mean() if name == "gdp_norm" else eps.max())
        elif name == "tau90":
            if ref_front is None or not have_box or not F.size:
                out[name] = None
            else:
                d = dplus_norm()[0]
                q = _compiled("quantile")
                out[name] = float(q(d, 0.9) if q is not None else np.quantile(d, 0.9))
                out["coverage_curve"] = {f"{t:g}": float(np.mean(d <= t)) for t in COVERAGE_TAUS}
        elif name == "gap_max":
            # the largest hole: max over the reference sample of d+(z, A), in
            # igdp_norm's frame — tau90's quantile taken at 100 %
            if ref_front is None or not have_box or not F.size:
                out[name] = None
            else:
                out[name] = float(dplus_norm()[0].max())
        elif name == "nn_cv":
            out[name] = nn_cv(F, ideal, nadir) if have_box and F.ndim == 2 and F.size else None
        elif name == "n_final":
            out[name] = int(len(F)) if F.ndim == 2 else 0
        elif name == "hv":
            if not have_box:
                out["hv"] = None
            else:
                v, method, se = hypervolume_se(F, ideal, nadir, **hvo)
                out["hv"] = v
                out["hv_method"] = method
                if se is not None:
                    out["hv_se"] = se
        elif name == "hv_h":
            if not have_box or not pop or F.ndim != 2:
                out["hv_h"] = None
            else:
                scale = 1.0 + 1.0 / lattice_h(F.shape[1], int(pop))
                v, method, se = hypervolume_se(F, ideal, nadir, ref_scale=scale, **hvo)
                out["hv_h"] = v
                out["hv_h_ref"] = scale
                out["hv_method"] = method
                if se is not None:
                    out["hv_h_se"] = se
        elif name == "r2":
            if not have_box or not pop or F.ndim != 2 or not F.size:
                out["r2"] = None
            else:
                h = lattice_h(F.shape[1], int(pop))
                out["r2"] = r2_discrete(F, ideal, nadir, h)
                out["r2_h"] = h
        elif name in ("igdx", "cr", "psp"):
            # an empty sample (a failing copy whose failed region takes all of
            # the Pareto set: failing.py) as no sample
            if X is None or pareto_set is None or not np.size(pareto_set) \
                    or bounds is None or not np.size(X):
                out[name] = None
            elif name == "igdx":
                out[name] = igdx(X, pareto_set, bounds=bounds, cyclic=cyclic)
            elif name == "cr":
                out[name] = cover_rate(X, pareto_set)
            else:
                out[name] = psp(X, pareto_set, bounds=bounds, cyclic=cyclic)
        elif name == "pdist":
            ok = X is not None and bounds is not None and np.size(X)
            out[name] = pairwise_distance(X, bounds=bounds, cyclic=cyclic) if ok else None
        elif name == "gdp_min":
            # convergence without coverage: the nearest point of the set to the
            # front, beside the reference's own resolution
            if ref_front is None or not have_box:
                out[name] = None
            elif not F.size:
                out[name] = float("inf")
            else:
                out[name] = float(dplus_norm()[1].min())
                step = reference_step(_normalised(F, ref_front, ideal, nadir)[1])
                out["gdp_min_ref_step"] = step
                out["gdp_min_below_step"] = bool(out[name] < step)
        elif name in ("g_min", "g_median"):
            if X is None or distance_fn is None or not np.size(X):
                out[name] = None
            else:
                if "g" not in shared:
                    shared["g"] = np.array([distance_fn(list(x))
                                            for x in np.atleast_2d(np.asarray(X, float))])
                g = shared["g"]
                med = _compiled("median")
                out[name] = float(g.min() if name == "g_min" else
                                  med(g) if med is not None and len(g) else np.median(g))
        elif name == "ps_dmin":
            if X is None or pareto_set is None or not np.size(pareto_set) \
                    or bounds is None or not np.size(X):
                out[name] = None
            else:
                out[name] = ps_dmin(X, pareto_set, bounds=bounds, cyclic=cyclic)
        elif name == "i_cmop":
            out.setdefault(name, None)              # taken above, on the whole set
        else:
            from .metric_names import METRIC_NAMES
            raise ValueError(f"unknown metric '{name}'; known: {', '.join(METRIC_NAMES)}")
    return out

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
"""

from __future__ import annotations

import numpy as np

HV_EXACT_MAX_M = 5
HV_MC_SAMPLES = 100_000


def nondominated(F: np.ndarray) -> np.ndarray:
    """Distinct rows of F that no other row dominates (minimisation).

    Each distinct row is kept once. Equal rows do not dominate each other, so
    without this every copy survives, and the WFG recursion below — whose
    limit sets are full of rows clipped to the same values — branches on every
    copy: a 126-point set with 11 distinct rows, which a MOEA/D-AM2M population
    at generation 0 is, ran for more than ten minutes instead of milliseconds.
    """
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
    d = np.sqrt(((ref[:, None, :] - F[None, :, :]) ** 2).sum(axis=2))
    return float(d.min(axis=1).mean())


def igd_plus(F: np.ndarray, ref: np.ndarray) -> float:
    """IGD+ (Ishibuchi et al. 2015): only the dominated part of the offset counts."""
    F = np.asarray(F, float)
    ref = np.asarray(ref, float)
    if F.size == 0:
        return float("inf")
    d = np.sqrt((np.maximum(F[None, :, :] - ref[:, None, :], 0.0) ** 2).sum(axis=2))
    return float(d.min(axis=1).mean())


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


def _hv_exact(pl: np.ndarray, ref: np.ndarray) -> float:
    """Exact hypervolume of `pl` (nondominated, inside `ref`): C++ when built."""
    try:
        from .. import _core
        fn = getattr(_core, "hypervolume", None)
    except ImportError:                              # pragma: no cover - no extension
        fn = None
    if fn is not None:
        return float(fn(np.ascontiguousarray(pl, float), np.ascontiguousarray(ref, float)))
    return _hv_wfg(pl, ref)


def _hv_mc(pl: np.ndarray, ref: np.ndarray, lo: np.ndarray, samples: int, seed: int) -> float:
    """Monte-Carlo hypervolume: the box [lo, ref] times the share of `samples`
    uniform points in it that some row of `pl` weakly dominates.

    The points are drawn here, from NumPy's generator with `seed`, and only
    counted by the compiled hv_covered when the extension is built, so the
    estimate does not depend on which of the two counted it.
    """
    rng = np.random.default_rng(seed)
    box = np.prod(ref - lo)
    if box <= 0:
        return 0.0
    try:
        from .. import _core
        count = getattr(_core, "hv_covered", None)
    except ImportError:                              # pragma: no cover - no extension
        count = None
    pl = np.ascontiguousarray(pl, float)
    hit = 0
    chunk = 20_000
    done = 0
    while done < samples:
        k = min(chunk, samples - done)
        s = lo + rng.random((k, len(ref))) * (ref - lo)
        if count is not None:
            hit += int(count(pl, np.ascontiguousarray(s)))
        else:
            # a sample is covered if some point dominates it (<= in every coordinate)
            covered = np.zeros(k, bool)
            for row in pl:
                covered |= np.all(row[None, :] <= s, axis=1)
            hit += int(covered.sum())
        done += k
    return float(box * hit / samples)


def hypervolume(F: np.ndarray, ideal, nadir, *, ref_scale: float = 1.1,
                mc_samples: int = HV_MC_SAMPLES, seed: int = 0,
                exact_max_m: int = HV_EXACT_MAX_M) -> tuple[float, str]:
    """Normalised hypervolume and the method used ("exact" or "mc").

    F is normalised to (F - ideal) / (nadir - ideal); the reference point is
    ref_scale in every objective; the result is divided by ref_scale^M. Exact
    up to `exact_max_m` objectives, a Monte-Carlo estimate from `mc_samples`
    points (the same points on every call with the same seed) above.
    """
    F = np.asarray(F, float)
    ideal = np.asarray(ideal, float)
    nadir = np.asarray(nadir, float)
    if F.size == 0:
        return 0.0, "none"
    span = nadir - ideal
    span[span <= 0] = 1.0
    G = (F - ideal) / span
    m = G.shape[1]
    ref = np.full(m, float(ref_scale))
    G = G[np.all(G < ref, axis=1)]
    if len(G) == 0:
        return 0.0, "none"
    G = nondominated(G)
    if m <= exact_max_m:
        v = _hv_exact(G, ref)
        method = "exact"
    else:
        lo = np.minimum(G.min(axis=0), 0.0)
        v = _hv_mc(G, ref, lo, mc_samples, seed)
        method = "mc"
    return float(v / ref_scale ** m), method


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
    worst = np.empty(len(ref))
    for start in range(0, len(ref), 256):           # bounded memory for dense fronts
        r = ref[start:start + 256]
        worst[start:start + len(r)] = (F[None, :, :] - r[:, None, :]).max(axis=2).min(axis=1)
    return float(worst.max())


def gd_plus(F: np.ndarray, ref: np.ndarray) -> float:
    """GD+ (Ishibuchi et al. 2015): IGD+'s distance averaged over the set instead.

    The same d+(a, r) = ||max(a − r, 0)|| matrix as igd_plus, minimized over the
    reference and averaged over the set: how far the points are from the front,
    blind to how much of it they cover. Read next to IGD+, it separates "still
    converging" (both fall) from "converged, now spreading" (GD+ flat, IGD+
    falling).
    """
    F = np.asarray(F, float)
    ref = np.asarray(ref, float)
    if F.size == 0:
        return float("inf")
    d = np.sqrt((np.maximum(F[None, :, :] - ref[:, None, :], 0.0) ** 2).sum(axis=2))
    return float(d.min(axis=0).mean())


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
    ideal = np.asarray(ideal, float)
    span = np.asarray(nadir, float) - ideal
    span = np.where(span > 0.0, span, 1.0)
    G = np.clip((F - ideal) / span, 0.0, 1.0)
    return G.max(axis=0) - G.min(axis=0)


def nd_share(F: np.ndarray) -> float:
    """Share of rows no other row dominates; equal rows do not dominate each other."""
    F = np.asarray(F, float)
    n = len(F)
    if n == 0:
        return 0.0
    le = np.all(F[:, None, :] <= F[None, :, :], axis=2)      # le[j, i]: F_j <= F_i
    lt = np.any(F[:, None, :] < F[None, :, :], axis=2)
    dominated = np.any(le & lt, axis=0)
    return float(1.0 - dominated.mean())


def dup_share(F: np.ndarray) -> float:
    """Share of rows that repeat an objective vector already in the set (exactly)."""
    F = np.asarray(F, float)
    if len(F) == 0:
        return 0.0
    return float(1.0 - len(np.unique(F, axis=0)) / len(F))


# ── the decision space ──────────────────────────────────────────────────────
def _unit(X, bounds) -> np.ndarray:
    """Variables mapped to [0, 1] by the problem's bounds."""
    X = np.atleast_2d(np.asarray(X, float))
    lo = np.array([b[0] for b in bounds], float)
    hi = np.array([b[1] for b in bounds], float)
    return (X - lo) / np.where(hi > lo, hi - lo, 1.0)


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
    wrap reads as far from it.
    """
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
    ranges do not meet. Higher is better; 1 at full cover.
    """
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


def pairwise_distance(X, *, bounds, cyclic=()) -> float:
    """The mean distance between two solutions in normalised variables (the
    cyclic ones wrapped): how spread the set is in the decision space, with no
    reference needed. 0 for fewer than two solutions."""
    U = _unit(X, bounds)
    n = len(U)
    if n < 2:
        return 0.0
    total = 0.0
    for a in range(0, n, 128):
        D = _unit_diff(U, U[a:a + 128], cyclic)
        total += float(np.sqrt((D ** 2).sum(axis=2)).sum())
    return total / (n * (n - 1))


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


def compute(F, *, ref_front=None, ideal=None, nadir=None, which=("igd",),
            pop: int | None = None, hv_options: dict | None = None, X=None,
            pareto_set=None, bounds=None, cyclic=()) -> dict:
    """Every requested indicator in one dict; missing inputs give None.

    `pop` is the problem's default population size. hv_h places its reference
    point at 1 + 1/H with H taken from it, so every algorithm on a problem is
    measured against the same point whatever population it rounded to.
    `hv_options` goes to hypervolume() (exact_max_m, mc_samples, seed). The
    decision-space indicators need the solutions `X` and the problem's
    `bounds`, and igdx and cr a `pareto_set` sample besides.
    """
    out: dict = {}
    F = np.asarray(F, float)
    hvo = dict(hv_options or {})
    have_box = ideal is not None and nadir is not None
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
        elif name in ("igdp_norm", "eps_norm"):
            if ref_front is None or not have_box:
                out[name] = None
            else:
                G, R = _normalised(F, ref_front, ideal, nadir)
                out[name] = igd_plus(G, R) if name == "igdp_norm" else eps_plus(G, R)
        elif name == "hv":
            if not have_box:
                out["hv"] = None
            else:
                v, method = hypervolume(F, ideal, nadir, **hvo)
                out["hv"] = v
                out["hv_method"] = method
        elif name == "hv_h":
            if not have_box or not pop or F.ndim != 2:
                out["hv_h"] = None
            else:
                scale = 1.0 + 1.0 / lattice_h(F.shape[1], int(pop))
                v, method = hypervolume(F, ideal, nadir, ref_scale=scale, **hvo)
                out["hv_h"] = v
                out["hv_h_ref"] = scale
                out["hv_method"] = method
        elif name in ("igdx", "cr"):
            if X is None or pareto_set is None or bounds is None or not np.size(X):
                out[name] = None
            elif name == "igdx":
                out[name] = igdx(X, pareto_set, bounds=bounds, cyclic=cyclic)
            else:
                out[name] = cover_rate(X, pareto_set)
        elif name == "pdist":
            ok = X is not None and bounds is not None and np.size(X)
            out[name] = pairwise_distance(X, bounds=bounds, cyclic=cyclic) if ok else None
        else:
            from .metric_names import METRIC_NAMES
            raise ValueError(f"unknown metric '{name}'; known: {', '.join(METRIC_NAMES)}")
    return out

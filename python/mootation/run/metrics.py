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
contribute nothing. Exact (WFG recursion) up to M = 5 objectives; above that
a Monte-Carlo estimate with a fixed seed, which is what the literature does
and is stated in the record as `hv_method`.
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


def _hv_mc(pl: np.ndarray, ref: np.ndarray, lo: np.ndarray, samples: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    box = np.prod(ref - lo)
    if box <= 0:
        return 0.0
    hit = 0
    chunk = 20_000
    done = 0
    while done < samples:
        k = min(chunk, samples - done)
        s = lo + rng.random((k, len(ref))) * (ref - lo)
        # a sample is covered if some point dominates it (<= in every coordinate)
        covered = np.zeros(k, bool)
        for row in pl:
            covered |= np.all(row[None, :] <= s, axis=1)
        hit += int(covered.sum())
        done += k
    return float(box * hit / samples)


def hypervolume(F: np.ndarray, ideal, nadir, *, ref_scale: float = 1.1,
                mc_samples: int = HV_MC_SAMPLES, seed: int = 0) -> tuple[float, str]:
    """Normalised hypervolume and the method used ("exact" or "mc").

    F is normalised to (F - ideal) / (nadir - ideal); the reference point is
    ref_scale in every objective; the result is divided by ref_scale^M.
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
    if m <= HV_EXACT_MAX_M:
        v = _hv_wfg(G, ref)
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
            pop: int | None = None) -> dict:
    """Every requested indicator in one dict; missing inputs give None.

    `pop` is the problem's default population size. hv_h places its reference
    point at 1 + 1/H with H taken from it, so every algorithm on a problem is
    measured against the same point whatever population it rounded to.
    """
    out: dict = {}
    F = np.asarray(F, float)
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
                v, method = hypervolume(F, ideal, nadir)
                out["hv"] = v
                out["hv_method"] = method
        elif name == "hv_h":
            if not have_box or not pop or F.ndim != 2:
                out["hv_h"] = None
            else:
                scale = 1.0 + 1.0 / lattice_h(F.shape[1], int(pop))
                v, method = hypervolume(F, ideal, nadir, ref_scale=scale)
                out["hv_h"] = v
                out["hv_h_ref"] = scale
                out["hv_method"] = method
        else:
            from .metric_names import METRIC_NAMES
            raise ValueError(f"unknown metric '{name}'; known: {', '.join(METRIC_NAMES)}")
    return out

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
    """Rows of F that no other row dominates (minimisation)."""
    F = np.asarray(F, float)
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


def compute(F, *, ref_front=None, ideal=None, nadir=None, which=("igd",)) -> dict:
    """Every requested indicator in one dict; missing inputs give None."""
    out: dict = {}
    F = np.asarray(F, float)
    for name in which:
        if name == "igd":
            out["igd"] = igd(F, ref_front) if ref_front is not None else None
        elif name == "igdp":
            out["igdp"] = igd_plus(F, ref_front) if ref_front is not None else None
        elif name == "hv":
            if ideal is None or nadir is None:
                out["hv"] = None
            else:
                v, method = hypervolume(F, ideal, nadir)
                out["hv"] = v
                out["hv_method"] = method
        else:
            raise ValueError(f"unknown metric '{name}'; known: igd, igdp, hv")
    return out

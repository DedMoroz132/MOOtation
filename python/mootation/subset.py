# SPDX-License-Identifier: Apache-2.0
"""k of the n points of a non-dominated set (task 15 of 2026-10-21, §4).

    subset_select(F, k, method="dss", ref=None, seed=1) -> list of indices

  dss          distance-based subset selection, the library's (dss.hpp)
  hv_greedy    the hypervolume's greedy choice, lazily (subset_select.hpp)
  hv_ls_rn     local search by swaps from hv_greedy's choice with candidate
               lists of 20 random and then 20 nearest points (Korogi & Tanabe,
               IEEE TEVC 2025, LS-RN; subset_select.hpp)
  igdp_greedy  the IGD+ greedy choice against the set itself, then single
               swaps (run/cover.py, greedy_subset: the floor's), in the set's
               own frame (each objective scaled by its range over F)

The extreme points (the least value of each objective) are chosen first and
never swapped out, by every method; dss chooses them first by its own rule.
ref: the hypervolume's reference point, which the hv methods need; a point
not below it in every objective adds nothing. The indices come in the order
the method chose them.
"""
from __future__ import annotations

METHODS = ("dss", "hv_greedy", "hv_ls_rn", "igdp_greedy")


def subset_select(F, k: int, method: str = "dss", ref=None, seed: int = 1) -> list:
    import numpy as np
    F = np.asarray(F, float)
    if method not in METHODS:
        raise ValueError(f"method = '{method}': {', '.join(METHODS)}")
    n = len(F)
    if k >= n:
        return list(range(n))
    if method == "igdp_greedy":
        from .run.cover import greedy_subset
        extremes = []
        for j in range(F.shape[1]):
            i = int(np.argmin(F[:, j]))            # the first least value
            if i not in extremes:
                extremes.append(i)
        if k <= len(extremes):
            return extremes[:k]
        lo, hi = F.min(axis=0), F.max(axis=0)
        G = (F - lo) / np.where(hi > lo, hi - lo, 1.0)
        return [int(i) for i in greedy_subset(G, k, fixed=extremes)]
    from . import _core
    if method != "dss" and ref is None:
        raise ValueError(f"method = '{method}' needs ref, the hypervolume's reference point")
    return [int(i) for i in _core.subset_select(F.tolist(), int(k), method,
                                                [] if ref is None else [float(v) for v in ref],
                                                int(seed))]

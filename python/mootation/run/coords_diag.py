# SPDX-License-Identifier: Apache-2.0
"""Does the learned basis find the distance block? ([campaign] coords_diag)

With coords = pca, pca_nd or randrot (operators/learned_coords.hpp; task 7 of
2026-10-10, supplement 1, §2) a campaign with coords_diag = true records at
every trajectory record:

  * coords_ratio, coords_lead_share — the spectrum of the last update: the
    largest eigenvalue over the smallest one above 1e-12 of it, and the share
    of the first M - 1 components (the core's own, as operator_stats gives
    them);
  * coords_angle_max, coords_angle_mean — the principal angles, in degrees,
    between the problem's distance block, span(e_k .. e_n), and the learned
    minor subspace, spanned by the last n - k columns of R (the smallest
    eigenvalues). k is the first distance variable. A rotated problem of
    benchmarks/rotated.py turns its distance block within itself, so the
    subspace its Q makes of the distance axes is the block's own span: one
    rule for the rotated problems and their bases. 0 degrees: PCA has found
    the block; 90: it has not.

The angles need a known distance block: ZDT, DTLZ, WFG, MaF, ZCAT and Polygon,
their shifted, rotated and noisy copies. Elsewhere only the spectrum.
"""

from __future__ import annotations


def distance_start(problem: str):
    """The first distance variable's index, or None where the block is not known."""
    from ..benchmarks import get
    from ..benchmarks.noisy import base_of, is_noisy
    from ..families import family
    name = base_of(problem) if is_noisy(problem) else problem
    fam = family(name)
    for head in ("r0", "c0", "rc", "r", "shift", "wide", "I", "S", "minus"):
        if fam.startswith(head) and fam[len(head):] in ("ZDT", "DTLZ", "WFG", "MaF", "ZCAT",
                                                          "Polygon"):
            fam = fam[len(head):]
            break
    p = get(name)
    if fam == "ZDT":
        return 1
    if fam == "Polygon":
        return 2
    if fam == "WFG":
        return p.n_vars - 20
    if fam == "ZCAT":
        stem = name.split("_")[0].lstrip("r")
        return 1 if stem in ("ZCAT14", "ZCAT15", "ZCAT16") else p.n_obj - 1
    if fam in ("DTLZ", "MaF"):
        stem = name.split("_")[0]
        if stem.endswith(("MaF8", "MaF9", "MaF10", "MaF11", "MaF12", "MaF13", "MaF14",
                          "MaF15")):
            return None
        return p.n_obj - 1
    return None


def principal_angles(R, start: int) -> tuple:
    """(largest, mean) principal angle in degrees between span(e_start ..
    e_n) and the span of R's last n - start columns."""
    import numpy as np
    R = np.asarray(R, float)
    n = R.shape[0]
    k = n - start
    if k <= 0:
        return None, None
    block = R[start:, n - k:]                  # the distance coordinates of the minor columns
    s = np.clip(np.linalg.svd(block, compute_uv=False), 0.0, 1.0)
    ang = np.degrees(np.arccos(s))
    return float(ang.max()), float(ang.mean())


def record(problem: str, state: dict, start=None) -> dict:
    """The diagnostics of one trajectory record from _core.coords_state()."""
    out = {}
    if not state.get("active"):
        return out
    eig = [max(float(v), 0.0) for v in state["eig"]]
    if eig and eig[0] > 0.0:
        low = [v for v in eig if v > 1e-12 * eig[0]]
        out["coords_ratio"] = eig[0] / low[-1]
        out["coords_lead_share"] = sum(eig[:max(int(state["m"]) - 1, 0)]) / sum(eig)
    if start is not None:
        out["coords_angle_max"], out["coords_angle_mean"] = principal_angles(state["R"], start)
    return out

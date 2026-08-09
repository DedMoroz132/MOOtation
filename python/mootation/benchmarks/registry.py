# SPDX-License-Identifier: Apache-2.0
# ============================================================================
# The registry of benchmark problems.
#
# Suites:
#   ZDT1-6              Zitzler, Deb, Thiele 2000 (M = 2)
#   DTLZ1-7 x M={2..6}  Deb, Thiele, Laumanns, Zitzler 2002
#   WFG1-9  x M={2..6}  Huband et al. 2006
#   MaF1-13 x M={3,5,8} Cheng et al. 2017 (irregular fronts)
#   Polygon x M={3..6}  Ishibuchi, Akedo, Nojima 2011
#   MOP1-7, BT1-9, and the inverted / scaled / minus DTLZ variants
#
# Conventions follow Tanabe & Oyama, GECCO 2017, so that numbers produced here
# are comparable with the literature rather than merely self-consistent:
#   * the archive is unbounded and feasible-only (cv > 0 is skipped);
#   * the hypervolume reference point is normalized, (1.1, ..., 1.1);
#   * hv_norm_divisor = 1.1^M;
#   * K_runs = 31, enough for a Wilcoxon test.
# ============================================================================
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import json as _json
from pathlib import Path as _Path

import numpy as np

from .wfg     import WFG_FUNCS, wfg_bounds, wfg_nadir, wfg1, wfg2, wfg9
from .dtlz    import (DTLZ_FUNCS, DTLZ_K, dtlz_n_vars,
                     dtlz_nadir, dtlz_ideal)
from .polygon import (polygon_eval, polygon_bounds,
                     polygon_nadir, polygon_ideal)
from .mop     import MOP_SPECS, mop_nadir, mop_ideal
from .dtlz_variants import (SPECS as DTLZV_SPECS, variant_n_vars,
                           hv_ref_raw as dv_hv_ref)
from .maf     import MAF_FIX, maf_n_vars
from .bt      import BT_SPECS, N as BT_N
from . import polygon_ishibuchi as _ipoly


# =============================================================
@dataclass
class BenchProblem:
    name: str
    n_vars: int
    bounds: List[Tuple[float, float]]
    n_obj: int
    evaluate: Callable[[List[float]], List[float]]
    constraints: Callable[[List[float]], List[float]]
    hv_ref_raw: Tuple[float, ...]
    hv_ref_norm: Tuple[float, ...]
    hv_norm_divisor: float
    ideal: Tuple[float, ...]
    nadir: Tuple[float, ...]
    pop_size: int
    n_gen: int
    K_runs: int
    has_cons: bool
    # A sampler of the true PF (n_points -> ndarray n x M) for IGD/IGD+/GD+.
    # None where no reference is defined yet; hypervolume still works.
    pareto_front: Optional[Callable[[int], "np.ndarray"]] = None
    # A sampler of the true Pareto SET, in decision space, for IGDX and PSP.
    # Set only on decision-space-diversity problems (Polygon, MMF).
    pareto_set: Optional[Callable[[int], "np.ndarray"]] = None

    @property
    def fevals_max(self) -> int:
        return self.pop_size * self.n_gen


def _no_cons(x: List[float]) -> List[float]:
    return []


def _budget(M: int) -> Tuple[int, int]:
    if M == 2: return (100, 500)
    if M == 3: return (91,  500)
    if M == 4: return (165, 500)
    if M == 5: return (126, 500)
    if M == 6: return (147, 500)
    if M == 7: return (112, 500)
    if M == 8: return (120, 500)
    if M == 10: return (275, 500)
    if M == 15: return (135, 500)
    return (100, 500)


def _ref_norm(M: int) -> Tuple[float, ...]:
    return tuple([1.1] * M)


def _divisor(M: int) -> float:
    return 1.1 ** M


# =============================================================
#  Helpers for the reference fronts (IGD/IGD+/GD+).
#  These reuse the Das-Dennis simplex and sphere from dtlz_variants, which
#  is also the base for IDTLZ/SDTLZ/minus-DTLZ, rather than duplicating them.
# =============================================================
from .dtlz_variants import _simplex as _dd_simplex   # Σ w = 1, Das-Dennis
from .dtlz_variants import _sphere_oct as _dd_sphere  # sum f^2 = 1, one octant
from . import polygon as _poly


def _dd_xset(M: int, n: int) -> np.ndarray:
    """A uniform grid of position parameters x in [0,1]^{M-1}.

    The DTLZ and WFG fronts are not built from the weights w themselves
    (sum w = 1) but from the position variables x_i in [0,1] that the shape
    functions take. Mapping Das-Dennis points to angles via
    x_i = (2/pi)*atan2(...) would distribute them unevenly, so a direct uniform
    grid on [0,1]^{M-1} is used instead. No filtering is needed: the shape
    functions are defined on the whole cube. At M = 2 this is just a segment.
    """
    import itertools
    if M <= 1:
        return np.zeros((1, 0))
    p = max(2, int(round(n ** (1.0 / (M - 1)))))
    grid = np.linspace(0.0, 1.0, p)
    pts = list(itertools.product(grid, repeat=M - 1))
    X = np.asarray(pts, dtype=float)
    # FIX 2026-07-09, in step with maf/dtlz_variants::_simplex: at high M the
    # grid degenerated (M=10 gave 2^9 = 512 points against the 1000 requested),
    # so it is topped up to n by uniform sampling of the cube, from a fixed
    # seed so the reference stays reproducible.
    if len(X) < n:
        rng = np.random.default_rng(20260709 + 1000 * M + n)
        X = np.vstack([X, rng.random((n - len(X), M - 1))])
    return X


# ---- DTLZ base fronts (g = 0) --------------------------------------
def _pf_dtlz1(M: int, n: int) -> np.ndarray:
    """The linear simplex sum f = 0.5, i.e. f = 0.5*w."""
    return 0.5 * _dd_simplex(M, n)


def _pf_dtlz2(M: int, n: int) -> np.ndarray:
    """The sphere sum f^2 = 1, one octant: f = w/||w||. DTLZ2/3/4 share this front."""
    return _dd_sphere(M, n)


def _pf_dtlz5(M: int, n: int) -> np.ndarray:
    """The degenerate 2-D curve of DTLZ5/6.

    On the front (g = 0): theta_0 = x_0*pi/2 and theta_i = pi/4 for i >= 1, so
    f traces a one-dimensional arc. theta_0 is sampled over [0, pi/2].
    """
    t = np.linspace(0.0, math.pi / 2.0, n)
    cos = np.cos(t); sin = np.sin(t)
    cd = math.cos(math.pi / 4.0); sd = math.sin(math.pi / 4.0)
    F = np.empty((n, M))
    # f_1 = Πcos = cos(θ0)·cd^{M-2};  f_2 = cos(θ0)·cd^{M-3}·sd; ... ; f_M = sin(θ0)
    for r in range(n):
        c0, s0 = cos[r], sin[r]
        f = np.empty(M)
        f[0] = c0 * (cd ** (M - 2)) if M > 1 else 1.0
        for i in range(2, M):
            f[i - 1] = c0 * (cd ** (M - i - 1)) * sd
        if M > 1:
            f[M - 1] = s0
        F[r] = f
    nrm = np.linalg.norm(F, axis=1, keepdims=True); nrm[nrm == 0] = 1.0
    return F / nrm


def _pf_dtlz7(M: int, n: int) -> np.ndarray:
    """The disconnected front of DTLZ7. f_1..f_{M-1} lie in [0,1] on the branches
    where the h contribution is smallest, and f_M = 2*(M - sum f_i/2*(1 +
    sin(3*pi*f_i))) at g = 0.

    Sampled the standard (PlatEMO) way: a uniform grid over the first M-1
    objectives, filtered to the nondominated subset, then f_M computed.
    """
    import itertools
    p = max(2, int(round((n * 2) ** (1.0 / (M - 1))))) if M > 1 else 1
    grid = np.linspace(0.0, 1.0, p)
    if M == 1:
        return np.array([[2.0]])
    pts = np.asarray(list(itertools.product(grid, repeat=M - 1)), float)
    # FIX 2026-07-09: at high M the grid degenerated (M=10 gave 2^9 = 512
    # points), so it is topped up with random cube points to 2n BEFORE the
    # nondominated filter, from a fixed seed.
    if len(pts) < 2 * n:
        rng = np.random.default_rng(20260709 + 1000 * M + n)
        pts = np.vstack([pts, rng.random((2 * n - len(pts), M - 1))])
    # f_M = (1+g)·h, g=0 → h = M − Σ f_i·(1+sin(3π f_i))
    h = float(M) - np.sum(pts * (1.0 + np.sin(3.0 * math.pi * pts)), axis=1)
    F = np.column_stack([pts, h])
    return _nondominated(F)


def _nondominated(F: np.ndarray) -> np.ndarray:
    """Keep the nondominated points (minimization)."""
    n = len(F)
    keep = np.ones(n, bool)
    for i in range(n):
        if not keep[i]:
            continue
        dom = np.all(F <= F[i], axis=1) & np.any(F < F[i], axis=1)
        if np.any(dom):
            keep[i] = False
    return F[keep]


# ---- WFG base fronts: f_m = 2m*h_m, with x_M = 0 on the front -------
def _pf_wfg_concave(M: int, n: int) -> np.ndarray:
    """WFG4-9: h is concave, a DTLZ2-like sphere, so f_m = 2m*sphere_m."""
    d = _dd_sphere(M, n)
    scale = 2.0 * np.arange(1, M + 1, dtype=float)
    return d * scale


def _pf_wfg1(M: int, n: int) -> np.ndarray:
    """WFG1: a convex shape h_m = prod(1 - cos)..., so f_m = 2m*h_m."""
    X = _dd_xset(M, n)
    F = np.empty((len(X), M))
    for r, x in enumerate(X):
        s = np.sin(math.pi * x / 2.0); c = np.cos(math.pi * x / 2.0)
        h = np.empty(M)
        h[0] = float(np.prod(1.0 - c)) if M > 1 else 1.0
        for m in range(1, M - 1):
            h[m] = float(np.prod(1.0 - c[:M - 1 - m])) * (1.0 - s[M - 1 - m])
        if M > 1:
            h[M - 1] = 1.0 - s[0]
        F[r] = h
    scale = 2.0 * np.arange(1, M + 1, dtype=float)
    return _nondominated(F * scale)


def _pf_wfg2(M: int, n: int) -> np.ndarray:
    """WFG2: convex and disconnected; the last objective is h_M = 1 - x1*cos^2(5*pi*x1)."""
    X = _dd_xset(M, n)
    F = np.empty((len(X), M))
    for r, x in enumerate(X):
        s = np.sin(math.pi * x / 2.0); c = np.cos(math.pi * x / 2.0)
        h = np.empty(M)
        h[0] = float(np.prod(1.0 - c)) if M > 1 else 1.0
        for m in range(1, M - 1):
            h[m] = float(np.prod(1.0 - c[:M - 1 - m])) * (1.0 - s[M - 1 - m])
        x1 = x[0]
        h[M - 1] = 1.0 - x1 * (math.cos(5.0 * math.pi * x1) ** 2)
        F[r] = h
    scale = 2.0 * np.arange(1, M + 1, dtype=float)
    return _nondominated(F * scale)


def _pf_wfg3(M: int, n: int) -> np.ndarray:
    """WFG3: degenerate, linear and one-dimensional. On the front
    x_2..x_{M-1} = 0.5 and only x_1 in [0,1] varies, giving f_m = 2m*linear_h.
    """
    t = np.linspace(0.0, 1.0, n)
    x = np.empty((n, M - 1))
    x[:, 0] = t
    if M > 2:
        x[:, 1:] = 0.5
    F = np.empty((n, M))
    for r in range(n):
        xp = x[r]
        h = np.empty(M)
        h[0] = float(np.prod(xp)) if M > 1 else 1.0
        for m in range(1, M - 1):
            h[m] = float(np.prod(xp[:M - 1 - m])) * (1.0 - xp[M - 1 - m])
        if M > 1:
            h[M - 1] = 1.0 - xp[0]
        F[r] = h
    scale = 2.0 * np.arange(1, M + 1, dtype=float)
    return _nondominated(F * scale)


_WFG_PF = {
    "WFG1": _pf_wfg1, "WFG2": _pf_wfg2, "WFG3": _pf_wfg3,
    "WFG4": _pf_wfg_concave, "WFG5": _pf_wfg_concave,
    "WFG6": _pf_wfg_concave, "WFG7": _pf_wfg_concave,
    "WFG8": _pf_wfg_concave, "WFG9": _pf_wfg_concave,
}


# ---- ZDT fronts (M = 2, closed form) -------------------------------
def _pf_zdt1(n: int) -> np.ndarray:
    f1 = np.linspace(0.0, 1.0, n)
    return np.column_stack([f1, 1.0 - np.sqrt(f1)])


def _pf_zdt2(n: int) -> np.ndarray:
    f1 = np.linspace(0.0, 1.0, n)
    return np.column_stack([f1, 1.0 - f1 ** 2])


def _pf_zdt3(n: int) -> np.ndarray:
    f1 = np.linspace(0.0, 1.0, 4 * n)
    f2 = 1.0 - np.sqrt(f1) - f1 * np.sin(10.0 * math.pi * f1)
    F = np.column_stack([f1, f2])
    return _nondominated(F)[:, :]


def _pf_zdt4(n: int) -> np.ndarray:
    # the same front as ZDT1 (convex)
    return _pf_zdt1(n)


def _pf_zdt6(n: int) -> np.ndarray:
    # f1 in [~0.2807, 1]; the minimum is reached at the x0 that maximizes the
    # sin^6 modulation. f2 = 1 - f1^2, concave.
    f1 = np.linspace(0.280775, 1.0, n)
    return np.column_stack([f1, 1.0 - f1 ** 2])


# ---- Polygon front: distance minimization, the M-gon interior -------
def _pf_polygon(M: int, n: int) -> np.ndarray:
    """The front is the image of the interior of a regular M-gon, its vertices on
    the unit circle, under f_m = ||p - v_m|| at g = 0. Points inside the
    polygon are sampled by rejection and passed through polygon_eval with the
    distance variables at zero.
    """
    ang = 2.0 * np.pi * np.arange(M) / M
    V = np.column_stack([np.cos(ang), np.sin(ang)])
    lo = V.min(0); hi = V.max(0)
    rng = np.random.default_rng(2024)
    pts = []
    while len(pts) < n:
        c = rng.uniform(lo, hi, (4 * n, 2))
        inside = _point_in_poly(c, V)
        pts.extend(c[inside].tolist())
    P = np.asarray(pts[:n], float)
    # f_m = the distance to vertex m, with g = 0 and the distance vars at 0
    F = np.empty((n, M))
    for m in range(M):
        F[:, m] = np.hypot(P[:, 0] - V[m, 0], P[:, 1] - V[m, 1])
    return F


def _point_in_poly(pts: np.ndarray, poly: np.ndarray) -> np.ndarray:
    n = len(poly); inside = np.zeros(len(pts), bool)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]; xj, yj = poly[j]
        cond = ((yi > pts[:, 1]) != (yj > pts[:, 1])) & \
               (pts[:, 0] < (xj - xi) * (pts[:, 1] - yi) / (yj - yi + 1e-30) + xi)
        inside ^= cond
        j = i
    return inside


# ---- DTLZ fronts, by name ------------------------------------------
_DTLZ_PF = {
    "DTLZ1": _pf_dtlz1, "DTLZ2": _pf_dtlz2, "DTLZ3": _pf_dtlz2,
    "DTLZ4": _pf_dtlz2, "DTLZ5": _pf_dtlz5, "DTLZ6": _pf_dtlz5,
    "DTLZ7": _pf_dtlz7,
}


# =============================================================
#  Fronts for the MaF problems that have no reference yet (MaF2, MaF6-13).
#  MaF1/3/4/5 are overridden by _register_maf_fix, which brings its own.
#  The shapes match the objective formulae in _maf_register above.
# =============================================================
def _pf_maf2(M: int, n: int) -> np.ndarray:
    """MaF2 (partially concave). On the front g = 0 and the positions lie on a
    DTLZ2-type sphere, so the concave octant sum f^2 = 1 is sampled."""
    return _dd_sphere(M, n)


def _pf_maf6(M: int, n: int) -> np.ndarray:
    """MaF6 is degenerate: a 2-D manifold in M-space, of the DTLZ5 type. On the
    front g = 0, theta_0 and theta_1 are free and the rest are pi/4. At M = 2
    or 3 it coincides with the degenerate DTLZ5 curve or surface."""
    # Only the first min(2, M-1) angles vary on the front; the rest are pi/4.
    nfree = min(2, M - 1)
    p = max(2, int(round(n ** (1.0 / max(1, nfree)))))
    import itertools
    grid = np.linspace(0.0, 1.0, p)
    combos = list(itertools.product(grid, repeat=nfree))
    sd = math.sin(math.pi / 4.0); cd = math.cos(math.pi / 4.0)
    F = []
    for cm in combos:
        theta = np.empty(M - 1)
        for i in range(M - 1):
            theta[i] = cm[i] * math.pi / 2.0 if i < nfree else math.pi / 4.0
        cos = np.cos(theta); sin = np.sin(theta)
        f = np.empty(M)
        f[0] = float(np.prod(cos)) if M > 1 else 1.0
        for i in range(2, M):
            f[i - 1] = float(np.prod(cos[:M - i])) * sin[M - i]
        if M > 1:
            f[M - 1] = sin[0]
        F.append(f)
    Fa = np.asarray(F, float)
    nrm = np.linalg.norm(Fa, axis=1, keepdims=True); nrm[nrm == 0] = 1.0
    return Fa / nrm


def _pf_maf7(M: int, n: int) -> np.ndarray:
    """MaF7 is disconnected and identical to DTLZ7; at g = 0 the fronts coincide."""
    return _pf_dtlz7(M, n)


def _pf_maf8(M: int, n: int) -> np.ndarray:
    """MaF8, multi-point distance minimisation. The vertices lie on the unit
    circle about the origin, and the front is the image of the M-gon interior."""
    ang = 2.0 * np.pi * np.arange(M) / M
    V = np.column_stack([np.cos(ang), np.sin(ang)])
    P = _sample_poly_interior(V, n)
    F = np.empty((n, M))
    for m in range(M):
        F[:, m] = np.hypot(P[:, 0] - V[m, 0], P[:, 1] - V[m, 1])
    return F


def _pf_maf9(M: int, n: int) -> np.ndarray:
    """MaF9, multi-line distance minimisation. f_m is the distance to the line
    through edge m of a regular M-gon, and the front is the interior's image."""
    Vang = 2.0 * np.pi * np.arange(M) / M
    V = np.column_stack([np.cos(Vang), np.sin(Vang)])
    P = _sample_poly_interior(V, n)
    F = np.empty((n, M))
    for m in range(M):
        ax, ay = V[m]; bx, by = V[(m + 1) % M]
        dx, dy = bx - ax, by - ay
        nrm = math.hypot(dx, dy)
        F[:, m] = np.abs(dy * (P[:, 0] - ax) - dx * (P[:, 1] - ay)) / nrm
    return F


def _sample_poly_interior(V: np.ndarray, n: int) -> np.ndarray:
    lo = V.min(0); hi = V.max(0)
    rng = np.random.default_rng(7)
    pts = []
    while len(pts) < n:
        c = rng.uniform(lo, hi, (4 * n, 2))
        pts.extend(c[_point_in_poly(c, V)].tolist())
    return np.asarray(pts[:n], float)


def _pf_maf13(M: int, n: int) -> np.ndarray:
    """MaF13: a degenerate unit sphere in 3-D, replicated into the tail. On the
    front (f1,f2,f3) lie on the sphere octant and f4..fM = f1^2+f2^10+f3^10."""
    d = _dd_sphere(3, n)          # (k,3), Σd²=1
    f1, f2, f3 = d[:, 0], d[:, 1], d[:, 2]
    tail = f1 ** 2 + f2 ** 10 + f3 ** 10
    cols = [f1, f2, f3]
    for _ in range(3, M):
        cols.append(tail)
    return np.column_stack(cols)


# The MaF problems whose objectives ARE WFG1/2/9 (see _maf_register) reuse
# the WFG fronts; the scaling matches, since both use f_m = 2m*h_m.
_MAF_EXTRA_PF = {
    "MaF2":  _pf_maf2,
    "MaF6":  _pf_maf6,
    "MaF7":  _pf_maf7,
    "MaF8":  _pf_maf8,
    "MaF9":  _pf_maf9,
    "MaF10": (lambda M, n: _pf_wfg1(M, n)),
    "MaF11": (lambda M, n: _pf_wfg2(M, n)),
    "MaF12": (lambda M, n: _pf_wfg_concave(M, n)),
    "MaF13": _pf_maf13,
}


PROBLEMS: Dict[str, BenchProblem] = {}


# =============================================================
#  ZDT1-6 (M = 2; n = 30 for ZDT1-4, n = 10 for ZDT6, n = 30 for ZDT5)
#  Zitzler, Deb, Thiele. Comparison of Multiobjective Evolutionary
#  Algorithms: Empirical Results. EMO 2000.
# =============================================================
def _zdt_register():
    pop, ng = _budget(2)

    # ── ZDT1 — convex PF ─────────────────────────────────────────────────────
    def zdt1(x):
        x = np.asarray(x); f1 = x[0]
        g = 1.0 + 9.0 * np.sum(x[1:]) / (len(x) - 1)
        f2 = g * (1.0 - math.sqrt(f1 / g))
        return [float(f1), float(f2)]

    PROBLEMS["ZDT1"] = BenchProblem(
        name="ZDT1", n_vars=30, bounds=[(0.0,1.0)]*30, n_obj=2,
        evaluate=zdt1, constraints=_no_cons,
        hv_ref_raw=(1.1, 1.1), hv_ref_norm=(1.1, 1.1), hv_norm_divisor=1.21,
        ideal=(0.0, 0.0), nadir=(1.0, 1.0),
        pop_size=pop, n_gen=ng, K_runs=21, has_cons=False,
        pareto_front=lambda n: _pf_zdt1(n))

    # ── ZDT2 — non-convex PF ─────────────────────────────────────────────────
    def zdt2(x):
        x = np.asarray(x); f1 = x[0]
        g = 1.0 + 9.0 * np.sum(x[1:]) / (len(x) - 1)
        f2 = g * (1.0 - (f1/g)**2)
        return [float(f1), float(f2)]

    PROBLEMS["ZDT2"] = BenchProblem(
        name="ZDT2", n_vars=30, bounds=[(0.0,1.0)]*30, n_obj=2,
        evaluate=zdt2, constraints=_no_cons,
        hv_ref_raw=(1.1, 1.1), hv_ref_norm=(1.1, 1.1), hv_norm_divisor=1.21,
        ideal=(0.0, 0.0), nadir=(1.0, 1.0),
        pop_size=pop, n_gen=ng, K_runs=21, has_cons=False,
        pareto_front=lambda n: _pf_zdt2(n))

    # ── ZDT3 — disconnected PF ───────────────────────────────────────────────
    def zdt3(x):
        x = np.asarray(x); f1 = x[0]
        g = 1.0 + 9.0 * np.sum(x[1:]) / (len(x) - 1)
        f2 = g * (1.0 - math.sqrt(f1/g) - (f1/g)*math.sin(10*math.pi*f1))
        return [float(f1), float(f2)]

    PROBLEMS["ZDT3"] = BenchProblem(
        name="ZDT3", n_vars=30, bounds=[(0.0,1.0)]*30, n_obj=2,
        evaluate=zdt3, constraints=_no_cons,
        hv_ref_raw=(1.1, 1.1), hv_ref_norm=(1.1, 1.1), hv_norm_divisor=1.21,
        ideal=(0.0, -0.7731), nadir=(1.0, 1.0),
        pop_size=pop, n_gen=ng, K_runs=21, has_cons=False,
        pareto_front=lambda n: _pf_zdt3(n))

    # ── ZDT4 — multimodal ────────────────────────────────────────────────────
    def zdt4(x):
        x = np.asarray(x); n = len(x); f1 = x[0]
        g = 1.0 + 10*(n-1) + sum(xi**2 - 10*math.cos(4*math.pi*xi)
                                  for xi in x[1:])
        f2 = g * (1.0 - math.sqrt(f1/g))
        return [float(f1), float(f2)]

    bnd4 = [(0.0,1.0)] + [(-5.0,5.0)]*29
    PROBLEMS["ZDT4"] = BenchProblem(
        name="ZDT4", n_vars=30, bounds=bnd4, n_obj=2,
        evaluate=zdt4, constraints=_no_cons,
        hv_ref_raw=(1.1, 1.1), hv_ref_norm=(1.1, 1.1), hv_norm_divisor=1.21,
        ideal=(0.0, 0.0), nadir=(1.0, 1.0),
        pop_size=pop, n_gen=ng, K_runs=21, has_cons=False,
        pareto_front=lambda n: _pf_zdt4(n))

    # ── ZDT6 — biased, non-uniform PF ────────────────────────────────────────
    def zdt6(x):
        x = np.asarray(x)
        f1 = 1.0 - math.exp(-4.0*x[0]) * (math.sin(6*math.pi*x[0])**6)
        g  = 1.0 + 9.0*(float(np.sum(x[1:]))/9.0)**0.25
        f2 = g * (1.0 - (f1/g)**2)
        return [float(f1), float(f2)]

    PROBLEMS["ZDT6"] = BenchProblem(
        name="ZDT6", n_vars=10, bounds=[(0.0,1.0)]*10, n_obj=2,
        evaluate=zdt6, constraints=_no_cons,
        hv_ref_raw=(1.1, 11.0), hv_ref_norm=(1.1, 1.1), hv_norm_divisor=1.21,
        ideal=(0.281, 0.0), nadir=(1.0, 0.921),
        pop_size=pop, n_gen=ng, K_runs=21, has_cons=False,
        pareto_front=lambda n: _pf_zdt6(n))


# =============================================================
#  MaF1-13 — Many-objective benchmark suite with irregular PFs
#  Cheng, Jin, Olhofer, Sendhoff.
#  "A Benchmark Test Suite for Evolutionary Many-Objective Optimization."
#  Complex & Intelligent Systems 3(1): 67-81, 2017.
#  https://doi.org/10.1007/s40747-017-0039-7
# =============================================================
def _maf_register():
    # MaF at M in {3, 5, 8}, the objective counts the papers report
    for M in (3, 5, 8, 10, 15):
        pop, ng = _budget(M)
        k_pos   = M - 1       # position params
        l_dist  = 10          # distance params
        n_vars  = k_pos + l_dist

        # ── MaF1 — inverted DTLZ1 ────────────────────────────────────────────
        def maf1(x, M=M):
            x = np.asarray(x, dtype=float)
            xm = x[M-1:]
            g  = 100.0*(len(xm) + np.sum((xm-0.5)**2 - np.cos(20*math.pi*(xm-0.5))))
            # Inverted: multiply (1-f_DTLZ1) ← reflected linear
            xp = 1.0 - x[:M-1]
            base = 0.5*(1+g)
            f = np.empty(M)
            f[0] = base * float(np.prod(xp))
            for i in range(2, M):
                f[i-1] = base * float(np.prod(xp[:M-i])) * (1-xp[M-i])
            f[M-1] = base * (1-xp[0])
            return f.tolist()

        PROBLEMS[f"MaF1_{M}D"] = BenchProblem(
            name=f"MaF1_{M}D", n_vars=n_vars,
            bounds=[(0.0,1.0)]*n_vars, n_obj=M,
            evaluate=maf1, constraints=_no_cons,
            hv_ref_raw=tuple([0.6]*M),   # inverted: PF near 0.5
            hv_ref_norm=_ref_norm(M), hv_norm_divisor=_divisor(M),
            ideal=tuple([0.0]*M), nadir=tuple([0.5]*M),
            pop_size=pop, n_gen=ng, K_runs=21, has_cons=False)

        # ── MaF2 — partial concave DTLZ2-like ────────────────────────────────
        def maf2(x, M=M):
            x = np.asarray(x, dtype=float)
            xm = x[M-1:]
            g  = float(np.sum((xm-0.5)**2))
            # Concave PF on first M//2 objectives, linear on rest
            xp = x[:M-1]
            f  = np.empty(M)
            cos = np.cos(xp * math.pi/2)
            sin = np.sin(xp * math.pi/2)
            base = 1.0 + g
            f[0] = base * float(np.prod(cos))
            for i in range(2, M):
                f[i-1] = base * float(np.prod(cos[:M-i])) * sin[M-i]
            f[M-1] = base * sin[0]
            return f.tolist()

        PROBLEMS[f"MaF2_{M}D"] = BenchProblem(
            name=f"MaF2_{M}D", n_vars=n_vars,
            bounds=[(0.0,1.0)]*n_vars, n_obj=M,
            evaluate=maf2, constraints=_no_cons,
            hv_ref_raw=tuple([1.1]*M), hv_ref_norm=_ref_norm(M),
            hv_norm_divisor=_divisor(M),
            ideal=tuple([0.0]*M), nadir=tuple([1.0]*M),
            pop_size=pop, n_gen=ng, K_runs=21, has_cons=False)

        # ── MaF3 — convex, multimodal (DTLZ3 rotated) ───────────────────────
        def maf3(x, M=M):
            x = np.asarray(x, dtype=float)
            xm = x[M-1:]
            g  = 100.0*(len(xm) + np.sum((xm-0.5)**2 - np.cos(20*math.pi*(xm-0.5))))
            xp = x[:M-1]
            cos = np.cos(xp*math.pi/2); sin = np.sin(xp*math.pi/2)
            base = 1.0 + g
            f = np.empty(M)
            f[0] = base * float(np.prod(1-cos))
            for i in range(2, M):
                f[i-1] = base * float(np.prod(1-cos[:M-i])) * (1-sin[M-i])
            f[M-1] = base*(1-sin[0])
            return f.tolist()

        nadir3 = tuple([1.0+100*10]*M)  # very loose — DTLZ3 multimodal
        PROBLEMS[f"MaF3_{M}D"] = BenchProblem(
            name=f"MaF3_{M}D", n_vars=n_vars,
            bounds=[(0.0,1.0)]*n_vars, n_obj=M,
            evaluate=maf3, constraints=_no_cons,
            hv_ref_raw=tuple([1100.0]*M), hv_ref_norm=_ref_norm(M),
            hv_norm_divisor=_divisor(M),
            ideal=tuple([0.0]*M), nadir=nadir3,
            pop_size=pop, n_gen=ng, K_runs=21, has_cons=False)

        # ── MaF4 — inverted concave (hardest convergence) ────────────────────
        def maf4(x, M=M):
            x = np.asarray(x, dtype=float)
            xm = x[M-1:]
            g  = float(np.sum((xm-0.5)**2))
            xp = 1.0 - x[:M-1]
            cos = np.cos(xp*math.pi/2); sin = np.sin(xp*math.pi/2)
            base = 1.0 + g
            f = np.empty(M)
            f[0] = base * float(np.prod(sin))
            for i in range(2, M):
                f[i-1] = base * float(np.prod(sin[:M-i])) * cos[M-i]
            f[M-1] = base * cos[0]
            return f.tolist()

        PROBLEMS[f"MaF4_{M}D"] = BenchProblem(
            name=f"MaF4_{M}D", n_vars=n_vars,
            bounds=[(0.0,1.0)]*n_vars, n_obj=M,
            evaluate=maf4, constraints=_no_cons,
            hv_ref_raw=tuple([1.1]*M), hv_ref_norm=_ref_norm(M),
            hv_norm_divisor=_divisor(M),
            ideal=tuple([0.0]*M), nadir=tuple([1.0]*M),
            pop_size=pop, n_gen=ng, K_runs=21, has_cons=False)

        # ── MaF5 — concave PF (WFG-based) ────────────────────────────────────
        def maf5(x, M=M):
            # Concave, biased — similar to DTLZ4 (alpha=100)
            x = np.asarray(x, dtype=float)
            xm = x[M-1:]
            g   = float(np.sum((xm-0.5)**2))
            xp  = x[:M-1] ** 100.0  # bias
            cos = np.cos(xp*math.pi/2); sin = np.sin(xp*math.pi/2)
            base = 1.0+g; f = np.empty(M)
            f[0] = base * float(np.prod(cos))
            for i in range(2, M):
                f[i-1] = base * float(np.prod(cos[:M-i])) * sin[M-i]
            f[M-1] = base * sin[0]
            return f.tolist()

        PROBLEMS[f"MaF5_{M}D"] = BenchProblem(
            name=f"MaF5_{M}D", n_vars=n_vars,
            bounds=[(0.0,1.0)]*n_vars, n_obj=M,
            evaluate=maf5, constraints=_no_cons,
            hv_ref_raw=tuple([1.1]*M), hv_ref_norm=_ref_norm(M),
            hv_norm_divisor=_divisor(M),
            ideal=tuple([0.0]*M), nadir=tuple([1.0]*M),
            pop_size=pop, n_gen=ng, K_runs=21, has_cons=False)

        # ── MaF6 — degenerate (m−1 dimensional PF embedded in M-space) ───────
        def maf6(x, M=M):
            x = np.asarray(x, dtype=float)
            xm = x[M-1:]
            g  = float(np.sum((xm-0.5)**2))
            # Degenerate: theta_i modified like DTLZ5
            theta = np.empty(M-1)
            theta[0] = x[0]*math.pi/2
            if M > 2:
                denom = 4.0*(1+g)
                for i in range(1, M-1):
                    theta[i] = (math.pi/denom)*(1+2*g*x[i])
            cos = np.cos(theta); sin = np.sin(theta)
            base = 1.0+g; f = np.empty(M)
            f[0] = base*float(np.prod(cos))
            for i in range(2, M):
                f[i-1] = base*float(np.prod(cos[:M-i]))*sin[M-i]
            f[M-1] = base*sin[0]
            return f.tolist()

        PROBLEMS[f"MaF6_{M}D"] = BenchProblem(
            name=f"MaF6_{M}D", n_vars=n_vars,
            bounds=[(0.0,1.0)]*n_vars, n_obj=M,
            evaluate=maf6, constraints=_no_cons,
            hv_ref_raw=tuple([1.1]*M), hv_ref_norm=_ref_norm(M),
            hv_norm_divisor=_divisor(M),
            ideal=tuple([0.0]*M), nadir=tuple([1.0]*M),
            pop_size=pop, n_gen=ng, K_runs=21, has_cons=False)

        # ── MaF7 — disconnected PF (DTLZ7-like) ──────────────────────────────
        def maf7(x, M=M):
            x = np.asarray(x, dtype=float)
            xm = x[M-1:]
            g = 1.0 + 9.0/len(xm)*float(np.sum(xm))
            f = np.empty(M)
            for i in range(M-1):
                f[i] = x[i]
            h = float(M) - float(np.sum(
                (f[:M-1]/(1+g))*(1+np.sin(3*math.pi*f[:M-1]))))
            f[M-1] = (1+g)*h
            return f.tolist()

        PROBLEMS[f"MaF7_{M}D"] = BenchProblem(
            name=f"MaF7_{M}D", n_vars=n_vars,
            bounds=[(0.0,1.0)]*n_vars, n_obj=M,
            evaluate=maf7, constraints=_no_cons,
            hv_ref_raw=tuple([1.0]*(M-1) + [2.0*M*1.1]),
            hv_ref_norm=_ref_norm(M), hv_norm_divisor=_divisor(M),
            ideal=tuple([0.0]*M), nadir=tuple([1.0]*(M-1)+[2.0*M]),
            pop_size=pop, n_gen=ng, K_runs=21, has_cons=False)

        # ── MaF8 — Multi-Point Distance Minimisation ─────────────────────────
        # Cheng et al. 2017, section 2.8. A two-dimensional decision space:
        # for a point x = (x1, x2), f_i is the Euclidean distance to vertex i
        # of a regular M-gon centred at the origin with radius 1. The front is
        # the polygon's interior — a 2-D manifold whatever M is.
        # The paper uses x in [-10000, 10000]^2; [-2, 2]^2 is used here, wide
        # enough to contain a radius-1 polygon without overflowing the
        # hypervolume normalization.
        def maf8(x, M=M):
            x = np.asarray(x, dtype=float)
            f = np.empty(M)
            for m in range(M):
                th = 2.0*math.pi*m/M
                vx, vy = math.cos(th), math.sin(th)
                f[m] = math.hypot(x[0]-vx, x[1]-vy)
            return f.tolist()

        PROBLEMS[f"MaF8_{M}D"] = BenchProblem(
            name=f"MaF8_{M}D", n_vars=2,
            bounds=[(-2.0,2.0)]*2, n_obj=M,
            evaluate=maf8, constraints=_no_cons,
            hv_ref_raw=tuple([3.0]*M),       # max distance on [-2,2]^2 is 2*sqrt(2)+1 < 3
            hv_ref_norm=_ref_norm(M), hv_norm_divisor=_divisor(M),
            ideal=tuple([0.0]*M), nadir=tuple([2.0]*M),
            pop_size=pop, n_gen=ng, K_runs=21, has_cons=False)

        # ── MaF9 — Multi-Line Distance Minimisation ──────────────────────────
        # Cheng et al. 2017, section 2.9. A two-dimensional decision space
        # again; f_i is the distance from x to the line through edge
        # A_i A_{i+1} of a regular M-gon. The front is the polygon's interior,
        # and interior points and their objective images are geometrically
        # similar, which makes the decision-space distribution easy to look at.
        # The paper uses x in [-10000, 10000]^2; [-2, 2]^2 is used here. The
        # infeasible zones outside the polygon (the paper, M >= 5) are a
        # constraint: the point must lie inside the convex M-gon.
        def _maf9_vertices(M):
            return [(math.cos(2*math.pi*m/M), math.sin(2*math.pi*m/M))
                    for m in range(M)]

        def maf9(x, M=M):
            x = np.asarray(x, dtype=float)
            V = _maf9_vertices(M)
            f = np.empty(M)
            for m in range(M):
                ax, ay = V[m]
                bx, by = V[(m+1) % M]
                # distance from the point to the line through (ax,ay),(bx,by)
                dx, dy = bx-ax, by-ay
                norm = math.hypot(dx, dy)
                f[m] = abs(dy*(x[0]-ax) - dx*(x[1]-ay)) / norm
            return f.tolist()

        def maf9_cons(x, M=M):
            # The constraint: the point must lie inside the convex M-gon.
            # For a regular polygon that holds exactly when, for every edge,
            # the point is on the same side as the centre. cv > 0 is a
            # violation.
            x = np.asarray(x, dtype=float)
            V = _maf9_vertices(M)
            cv = 0.0
            for m in range(M):
                ax, ay = V[m]
                bx, by = V[(m+1) % M]
                dx, dy = bx-ax, by-ay
                # signed offset of the point and of the centre from the edge
                side_pt = dx*(x[1]-ay) - dy*(x[0]-ax)
                side_c  = dx*(0.0-ay) - dy*(0.0-ax)
                if side_pt*side_c < 0.0:          # opposite sides -> outside
                    cv += abs(side_pt)
            return [cv]

        PROBLEMS[f"MaF9_{M}D"] = BenchProblem(
            name=f"MaF9_{M}D", n_vars=2,
            bounds=[(-2.0,2.0)]*2, n_obj=M,
            evaluate=maf9, constraints=maf9_cons,
            hv_ref_raw=tuple([2.0]*M),
            hv_ref_norm=_ref_norm(M), hv_norm_divisor=_divisor(M),
            ideal=tuple([0.0]*M), nadir=tuple([1.0]*M),
            pop_size=pop, n_gen=ng, K_runs=21, has_cons=True)

        # ── MaF10 — WFG1 (mixed, biased PF) ──────────────────────────────────
        # Cheng et al. 2017, section 2.10. Identical to WFG1; the dimension
        # parameter is the only difference.
        maf_K = M - 1
        maf_L = 10
        maf_n = maf_K + maf_L
        maf_bounds = [(0.0, 2.0*(i+1)) for i in range(maf_n)]

        def maf10(x, M=M, k=maf_K):
            return wfg1(list(x), M, k)

        PROBLEMS[f"MaF10_{M}D"] = BenchProblem(
            name=f"MaF10_{M}D", n_vars=maf_n,
            bounds=maf_bounds, n_obj=M,
            evaluate=maf10, constraints=_no_cons,
            hv_ref_raw=tuple([2.0*(i+1)*1.1 for i in range(M)]),
            hv_ref_norm=_ref_norm(M), hv_norm_divisor=_divisor(M),
            ideal=tuple([0.0]*M),
            nadir=tuple([2.0*(i+1) for i in range(M)]),
            pop_size=pop, n_gen=ng, K_runs=21, has_cons=False)

        # ── MaF11 — WFG2 (convex, disconnected, non-separable PF) ────────────
        def maf11(x, M=M, k=maf_K):
            return wfg2(list(x), M, k)

        PROBLEMS[f"MaF11_{M}D"] = BenchProblem(
            name=f"MaF11_{M}D", n_vars=maf_n,
            bounds=maf_bounds, n_obj=M,
            evaluate=maf11, constraints=_no_cons,
            hv_ref_raw=tuple([2.0*(i+1)*1.1 for i in range(M)]),
            hv_ref_norm=_ref_norm(M), hv_norm_divisor=_divisor(M),
            ideal=tuple([0.0]*M),
            nadir=tuple([2.0*(i+1) for i in range(M)]),
            pop_size=pop, n_gen=ng, K_runs=21, has_cons=False)

        # ── MaF12 — WFG9 (concave, non-separable, biased deceptive PF) ───────
        def maf12(x, M=M, k=maf_K):
            return wfg9(list(x), M, k)

        PROBLEMS[f"MaF12_{M}D"] = BenchProblem(
            name=f"MaF12_{M}D", n_vars=maf_n,
            bounds=maf_bounds, n_obj=M,
            evaluate=maf12, constraints=_no_cons,
            hv_ref_raw=tuple([2.0*(i+1)*1.1 for i in range(M)]),
            hv_ref_norm=_ref_norm(M), hv_norm_divisor=_divisor(M),
            ideal=tuple([0.0]*M),
            nadir=tuple([2.0*(i+1) for i in range(M)]),
            pop_size=pop, n_gen=ng, K_runs=21, has_cons=False)

        # ── MaF13 — PF7 (concave, degenerate, complex variable linkage) ──────
        # Cheng et al. 2017, section 2.13. D = 5 is fixed and does not depend
        # on M. The front is always the unit sphere. Decision space: x1, x2 in
        # [0,1] and x3..x5 in [-2,2], coupled non-linearly through y_i.
        def maf13(x, M=M):
            x = np.asarray(x, dtype=float)
            D = len(x)
            y = np.empty(D)
            for i in range(D):
                y[i] = x[i] - 2.0*x[1]*math.sin(2*math.pi*x[0] + (i+1)*math.pi/D)
            J1 = [j for j in range(2, D) if (j+1) % 3 == 1]
            J2 = [j for j in range(2, D) if (j+1) % 3 == 2]
            J3 = [j for j in range(2, D) if (j+1) % 3 == 0]
            J4 = [j for j in range(3, D)]
            def s(J): return float(np.sum(y[J]**2)) if J else 0.0
            f = np.empty(M)
            f1 = math.sin(math.pi/2*x[0]) + 2.0/max(len(J1),1)*s(J1)
            f2 = (math.cos(math.pi/2*x[0])*math.sin(math.pi/2*x[1])
                  + 2.0/max(len(J2),1)*s(J2))
            f3 = (math.cos(math.pi/2*x[0])*math.cos(math.pi/2*x[1])
                  + 2.0/max(len(J3),1)*s(J3))
            f[0], f[1], f[2] = f1, f2, f3
            # f4..fM are all identical: the front degenerates to a 2-D manifold
            tail = f1**2 + f2**10 + f3**10 + 2.0/max(len(J4),1)*s(J4)
            for i in range(3, M):
                f[i] = tail
            return f.tolist()

        maf13_bounds = [(0.0,1.0), (0.0,1.0), (-2.0,2.0), (-2.0,2.0), (-2.0,2.0)]
        PROBLEMS[f"MaF13_{M}D"] = BenchProblem(
            name=f"MaF13_{M}D", n_vars=5,
            bounds=maf13_bounds, n_obj=M,
            evaluate=maf13, constraints=_no_cons,
            hv_ref_raw=tuple([1.1]*M),
            hv_ref_norm=_ref_norm(M), hv_norm_divisor=_divisor(M),
            ideal=tuple([0.0]*M), nadir=tuple([1.0]*M),
            pop_size=pop, n_gen=ng, K_runs=21, has_cons=False)

        # ── Reference fronts for MaF2 and MaF6-13. MaF1/3/4/5 are overridden
        #    by _register_maf_fix with its own. These are attached to the
        #    already-constructed records rather than changing how they are built.
        for _mname, _pffn in _MAF_EXTRA_PF.items():
            _key = f"{_mname}_{M}D"
            if _key in PROBLEMS:
                PROBLEMS[_key].pareto_front = (
                    lambda n, _g=_pffn, _M=M: _g(_M, n))


# =============================================================
#  WFG1-9 × M={2,3,4,5,6,10}
#  (M = 10 added 2026-07-09: the biased WFG1/9 and the WFG2/4 balance need a
#   high-M slice.)
def _register_wfg():
    for M in (2, 3, 4, 5, 6, 10):
        k = 2*(M-1); l = 20; n_vars = k+l
        bounds = wfg_bounds(M, k=k, l=l)
        nadir  = wfg_nadir(M)
        ideal  = tuple([0.0]*M)
        pop, ng = _budget(M)
        for name, func in WFG_FUNCS.items():
            def _eval(x, _f=func, _M=M): return _f(list(x), _M)
            def _pf(n, _g=_WFG_PF[name], _M=M): return _g(_M, n)
            prob_name = f"{name}_{M}D"
            PROBLEMS[prob_name] = BenchProblem(
                name=prob_name, n_vars=n_vars, bounds=bounds, n_obj=M,
                evaluate=_eval, constraints=_no_cons,
                hv_ref_raw=tuple(v*1.1 for v in nadir),
                hv_ref_norm=_ref_norm(M), hv_norm_divisor=_divisor(M),
                ideal=ideal, nadir=nadir,
                pop_size=pop, n_gen=ng, K_runs=21, has_cons=False,
                pareto_front=_pf)


# =============================================================
#  DTLZ1-7 × M={2,3,4,5,6}
# =============================================================
def _register_dtlz():
    for M in (2, 3, 4, 5, 6, 10, 15):
        pop, ng = _budget(M)
        for name, func in DTLZ_FUNCS.items():
            n_vars = dtlz_n_vars(name, M)
            bounds = [(0.0,1.0)]*n_vars
            nadir  = dtlz_nadir(name, M)
            ideal  = dtlz_ideal(name, M)
            def _eval(x, _f=func, _M=M): return _f(list(x), _M)
            def _pf(n, _g=_DTLZ_PF[name], _M=M): return _g(_M, n)
            prob_name = f"{name}_{M}D"
            PROBLEMS[prob_name] = BenchProblem(
                name=prob_name, n_vars=n_vars, bounds=bounds, n_obj=M,
                evaluate=_eval, constraints=_no_cons,
                hv_ref_raw=tuple(v*1.1 for v in nadir),
                hv_ref_norm=_ref_norm(M), hv_norm_divisor=_divisor(M),
                ideal=ideal, nadir=nadir,
                pop_size=pop, n_gen=ng, K_runs=21, has_cons=False,
                pareto_front=_pf)


# =============================================================
#  Polygon × M={3,4,5,6}
# =============================================================
def _register_polygon():
    n_vars = 10
    for M in (3, 4, 5, 6):
        bounds  = polygon_bounds(n_vars)
        nadir   = polygon_nadir(M)
        ideal   = polygon_ideal(M)
        pop, ng = _budget(M)
        def _eval(x, _M=M): return polygon_eval(list(x), _M)
        def _pf(n, _M=M): return _pf_polygon(_M, n)
        prob_name = f"Polygon_{M}D"
        PROBLEMS[prob_name] = BenchProblem(
            name=prob_name, n_vars=n_vars, bounds=bounds, n_obj=M,
            evaluate=_eval, constraints=_no_cons,
            hv_ref_raw=tuple(v*1.1 for v in nadir),
            hv_ref_norm=_ref_norm(M), hv_norm_divisor=_divisor(M),
            ideal=ideal, nadir=nadir,
            pop_size=pop, n_gen=ng, K_runs=21, has_cons=False,
            pareto_front=_pf)


# =============================================================
#  MOP1-7, the imbalanced problems (Liu, Gu, Zhang 2014, MOEA/D-M2M).
def _register_mop():
    n_vars = 10
    for name, (efn, M, pf) in MOP_SPECS.items():
        pop, ng = _budget(M)
        nadir   = mop_nadir(M)
        ideal   = mop_ideal(M)
        def _eval(x, _f=efn): return _f(list(x))
        def _pf(n, _p=pf): return _p(n)
        PROBLEMS[name] = BenchProblem(
            name=name, n_vars=n_vars, bounds=[(0.0, 1.0)] * n_vars, n_obj=M,
            evaluate=_eval, constraints=_no_cons,
            hv_ref_raw=tuple(v * 1.1 for v in nadir),
            hv_ref_norm=_ref_norm(M), hv_norm_divisor=_divisor(M),
            ideal=ideal, nadir=nadir,
            pop_size=pop, n_gen=ng, K_runs=21, has_cons=False,
            pareto_front=_pf)


# =============================================================
#  DTLZ variants: IDTLZ1/2 (inverted), minus-DTLZ2, SDTLZ1/2 (scaled), at
#  M = {3,5,8,10}. The fronts are analytic, so IGD has a real reference.
#  (M = 10 added 2026-07-09: the inverted and scaled group needs a high-M
#   slice, minusDTLZ2_10D and SDTLZ2_10D.)
def _register_dtlz_variants():
    for M in (3, 5, 8, 10):
        pop, ng = _budget(M)
        for name, spec in DTLZV_SPECS.items():
            n_vars = variant_n_vars(spec["base"], M)
            ideal  = spec["ideal"](M)
            nadir  = spec["nadir"](M)
            def _eval(x, _f=spec["eval"], _M=M): return _f(list(x), _M)
            def _pf(n, _p=spec["pf"], _M=M): return _p(_M, n)
            prob_name = f"{name}_{M}D"
            PROBLEMS[prob_name] = BenchProblem(
                name=prob_name, n_vars=n_vars, bounds=[(0.0, 1.0)] * n_vars, n_obj=M,
                evaluate=_eval, constraints=_no_cons,
                hv_ref_raw=dv_hv_ref(ideal, nadir),
                hv_ref_norm=_ref_norm(M), hv_norm_divisor=_divisor(M),
                ideal=ideal, nadir=nadir,
                pop_size=pop, n_gen=ng, K_runs=21, has_cons=False,
                pareto_front=_pf)


# =============================================================
#  The MaF1/3/4/5 correction (Cheng 2017): overwrites the entries built by
#  the older _maf_register with the correct formulae and a reference front,
#  at M = {3,5,8,10,15}. Nadir and ideal are taken from sampling the true
#  front.
#  M = 10 and 15 were added 2026-07-09; before that MaF1/3/4/5 at those sizes
#  still used the OLD, incorrect formulae, which matters most for the MaF3 and
#  MaF4 objective-count sweep. Results for those problems at M >= 10 produced
#  before this date are NOT comparable with results produced after it.
def _register_maf_fix():
    for M in (3, 5, 8, 10, 15):
        pop, ng = _budget(M)
        n_vars  = maf_n_vars(M)
        for name, (ev, pf) in MAF_FIX.items():
            Z = pf(M, 1000)
            ideal = tuple(float(v) for v in Z.min(0))
            nadir = tuple(float(v) for v in Z.max(0))
            def _eval(x, _f=ev, _M=M): return _f(list(x), _M)
            def _pf(n, _p=pf, _M=M): return _p(_M, n)
            key = f"{name}_{M}D"
            PROBLEMS[key] = BenchProblem(
                name=key, n_vars=n_vars, bounds=[(0.0, 1.0)] * n_vars, n_obj=M,
                evaluate=_eval, constraints=_no_cons,
                hv_ref_raw=tuple(nd + 0.1 * (nd - id_) for id_, nd in zip(ideal, nadir)),
                hv_ref_norm=_ref_norm(M), hv_norm_divisor=_divisor(M),
                ideal=ideal, nadir=nadir,
                pop_size=pop, n_gen=ng, K_runs=21, has_cons=False,
                pareto_front=_pf)


# =============================================================
#  BT1-9 — biased suite (Li, Zhang, Deng 2017). BT1-8: M=2, BT9: M=3.
#  n = 30. The front, ideal and nadir come from sampling the true front.
# =============================================================
def _register_bt():
    for name, (ev, M, pf, bd) in BT_SPECS.items():
        pop, ng = _budget(M)
        Z = pf(2000 if M == 2 else 1000)
        ideal = tuple(float(v) for v in Z.min(0))
        nadir = tuple(float(v) for v in Z.max(0))
        bounds = bd()
        def _eval(x, _f=ev): return _f(list(x))
        def _pf(n, _p=pf): return _p(n)
        PROBLEMS[name] = BenchProblem(
            name=name, n_vars=BT_N, bounds=bounds, n_obj=M,
            evaluate=_eval, constraints=_no_cons,
            hv_ref_raw=tuple(nd + 0.1 * (nd - id_) for id_, nd in zip(ideal, nadir)),
            hv_ref_norm=_ref_norm(M), hv_norm_divisor=_divisor(M),
            ideal=ideal, nadir=nadir,
            pop_size=pop, n_gen=ng, K_runs=21, has_cons=False,
            pareto_front=_pf)


_zdt_register()
_register_wfg()
_register_dtlz()
_register_polygon()
_maf_register()
_register_mop()
# =============================================================
#  IPolygon, Ishibuchi's multi-polygon problem (decision-space diversity).
#  M = the vertex count = the objective count; n_vars = 2 over [0,100]^2.
#  It defines both pareto_front (for IGD) and pareto_set (for IGDX and PSP).
def _register_ipolygon():
    for name, sp in _ipoly.specs((3, 4, 8)).items():
        M = sp["M"]; pop, ng = _budget(M)
        Z = sp["pf"](1000)
        ideal = tuple(float(v) for v in Z.min(0))
        nadir = tuple(float(v) for v in Z.max(0))
        def _eval(x, _f=sp["eval"]): return _f(list(x))
        def _pf(n, _p=sp["pf"]): return _p(n)
        def _ps(n, _p=sp["ps"]): return _p(n)
        PROBLEMS[name] = BenchProblem(
            name=name, n_vars=2, bounds=_ipoly.bounds(), n_obj=M,
            evaluate=_eval, constraints=_no_cons,
            hv_ref_raw=tuple(nd + 0.1 * (nd - id_) for id_, nd in zip(ideal, nadir)),
            hv_ref_norm=_ref_norm(M), hv_norm_divisor=_divisor(M),
            ideal=ideal, nadir=nadir,
            pop_size=pop, n_gen=ng, K_runs=21, has_cons=False,
            pareto_front=_pf, pareto_set=_ps)


_register_dtlz_variants()
_register_maf_fix()
_register_bt()
_register_ipolygon()


# ── HV/IGD reference frame = the sampled reference PF (PlatEMO convention) ──
# Computed LAZILY, per problem, on first lookup.
#
# Doing all 216 eagerly at import cost 136 seconds, which makes the package
# unusable as a library: `import mootation.benchmarks` to list names should
# not sample a Pareto front. A shipped cache file makes the common case free;
# a cache miss falls back to sampling that one problem, in memory. Nothing is
# ever written back — a library that writes into its own installation
# directory breaks on any read-only or shared install.
_REFRAME_CACHE_FILE = _Path(__file__).with_name("reference_frames.json")
_REFRAMED: set = set()
_REFRAME_TABLE: Optional[dict] = None


def _reframe_table() -> dict:
    global _REFRAME_TABLE
    if _REFRAME_TABLE is None:
        try:
            with open(_REFRAME_CACHE_FILE, encoding="utf-8") as fh:
                _REFRAME_TABLE = _json.load(fh)
        except (OSError, ValueError):
            _REFRAME_TABLE = {}
    return _REFRAME_TABLE


def _reframe_one(name: str, p: "BenchProblem", n: int = 500) -> None:
    """Set p.ideal / p.nadir from the true PF, once."""
    if name in _REFRAMED:
        return
    _REFRAMED.add(name)

    table = _reframe_table()
    if name in table:
        idl, nad = tuple(table[name][0]), tuple(table[name][1])
    else:
        f = getattr(p, "pareto_front", None)
        if not callable(f):
            return
        try:
            pf = np.asarray(f(n), float)
        except Exception:
            return
        if pf.ndim != 2 or len(pf) < 2:
            return
        idl, nad = tuple(pf.min(0)), tuple(pf.max(0))

    try:
        p.ideal, p.nadir = idl, nad
    except Exception:                      # frozen dataclass
        object.__setattr__(p, "ideal", idl)
        object.__setattr__(p, "nadir", nad)


def reframe_all(n: int = 500) -> None:
    """Force every problem's reference frame. Used to regenerate the cache."""
    for name, p in PROBLEMS.items():
        _reframe_one(name, p, n)


# ── Public lookup ───────────────────────────────────────────────────────────
# The registry itself is a plain dict; these wrap it so callers never mutate it
# and so a typo gets a message naming the near misses instead of a KeyError.

def names() -> list:
    """Every registered problem name, sorted."""
    return sorted(PROBLEMS)


def get(name: str) -> "BenchProblem":
    """Look up one problem by name, e.g. "DTLZ2_M3"."""
    try:
        p = PROBLEMS[name]
    except KeyError:
        stem = name.split("_")[0].upper()
        near = [n for n in sorted(PROBLEMS) if n.upper().startswith(stem[:4])]
        hint = f"; did you mean: {', '.join(near[:6])}" if near else ""
        raise KeyError(f"unknown benchmark problem '{name}'{hint}") from None
    _reframe_one(name, p)
    return p


def families() -> dict:
    """Problem names grouped by family stem: {"DTLZ": [...], "WFG": [...]}."""
    out: Dict[str, List[str]] = {}
    for n in sorted(PROBLEMS):
        stem = n.split("_")[0]
        stem = "".join(c for c in stem if not c.isdigit()) or stem
        out.setdefault(stem, []).append(n)
    return out

# SPDX-License-Identifier: Apache-2.0
# ============================================================================
# The registry of benchmark problems.
#
# Suites:
#   ZDT1-6              Zitzler, Deb, Thiele 2000 (M = 2)
#   DTLZ1-7 x M={2..6}  Deb, Thiele, Laumanns, Zitzler 2002
#   WFG1-9  x M={2..6}  Huband et al. 2006
#   MaF1-13 x M={3,5,8} Cheng et al. 2017 (irregular fronts)
#   ZCAT1-20 x 4 sizes  Zapotecas-Martinez et al. 2023 (tunable difficulty)
#   bbob-biobj x n=5,10 Brockhoff et al. 2022 (55 pairs of bbob functions)
#   Polygon x M={3..6}  Ishibuchi, Akedo, Nojima 2011
#   MOP1-7, BT1-9, and the inverted / scaled / minus DTLZ variants
#   shiftDTLZ1-4        DTLZ1-4 with the distance optimum off the centre (ours)
#
# Reference data follow Tanabe & Oyama, GECCO 2017, so that numbers produced
# here are comparable with the literature rather than merely self-consistent:
#   * the hypervolume reference point is normalized, (1.1, ..., 1.1);
#   * hv_norm_divisor = 1.1^M;
#   * K_runs = 21 for every problem.
# Their archive convention (score an unbounded, feasible-only archive of every
# solution evaluated) is not applied here: solve() and campaigns score the
# population the algorithm returns.
# ============================================================================
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import json as _json
from pathlib import Path as _Path

import warnings
import numpy as np

from .wfg     import WFG_FUNCS, wfg_bounds, wfg_nadir, wfg1, wfg2, wfg9
from .dtlz    import (DTLZ_FUNCS, DTLZ_K, dtlz_n_vars,
                     dtlz_nadir, dtlz_ideal)
from .polygon import (polygon_eval, polygon_bounds,
                     polygon_nadir, polygon_ideal)
from .mop     import MOP_SPECS, mop_nadir, mop_ideal
from .dtlz_variants import (SPECS as DTLZV_SPECS, variant_n_vars,
                           hv_ref_raw as dv_hv_ref, SHIFT_BASES, shift_dtlz,
                           shift_centres)
from .maf     import MAF_FIX, maf_n_vars
from .bt      import BT_SPECS, N as BT_N
from . import polygon_ishibuchi as _ipoly
from . import zcat as _zcat
from . import bbob_biobj as _bbobbiobj
from . import uninformative as _uninf
from . import pareto_sets as _psets


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
    # A sampler of the true Pareto SET, in decision space, for IGDX and CR
    # (benchmarks/pareto_sets.py): IPolygon, Polygon, DTLZ1-4, shiftDTLZ1-4,
    # ZCAT. None elsewhere, and the decision-space indicators say nothing there.
    pareto_set: Optional[Callable[[int], "np.ndarray"]] = None
    # Variables whose distance wraps with the period of their range: the
    # problem reads them mod 1 (shiftDTLZ's distance variables).
    cyclic_vars: Tuple[int, ...] = ()
    # A problem whose values depend on the run, not only on x, hands each run
    # its own evaluator: make_evaluator(seed) -> callable (uninformative.py).
    # None for every ordinary problem, where `evaluate` is the function.
    make_evaluator: Optional[Callable[[int], Callable]] = None

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
    """The disconnected front of DTLZ7 (Deb et al. 2002, Eq. 6.25).

    On the front x_M = 0, so g = 1 + 9/k * 0 = 1 (NOT 0: g has the additive
    constant 1), hence 1 + g = 2 and
        f_M = (1 + g) * h = 2 * (M - sum f_i/2 * (1 + sin(3 pi f_i)))
            = 2M - sum f_i * (1 + sin(3 pi f_i)).
    FIX 2026-09-05: the sampler used to compute M - sum f_i (1 + sin), i.e.
    it dropped the factor (1 + g) = 2, which put the reference front a full
    M below the true one (f_M even went negative). Every DTLZ7/MaF7 IGD
    computed before this date is wrong; the cached reference frame carried
    nadir f_M = 3 for M = 3 where the true value is 6.

    Sampled the standard (PlatEMO) way — a uniform grid over the first M-1
    objectives, f_M computed, the nondominated subset kept — but on the
    Pareto SET instead of the cube (FIX 2026-09-22). With f_i = x_i and
    f_M = 2M − sum phi(x_i), phi(x) = x (1 + sin 3 pi x), a point is
    Pareto-optimal exactly when every x_i is a left record of phi:
    phi(x_i) > phi(t) for all t < x_i. A smaller t with phi(t) >= phi(x_i)
    would dominate it, and with every coordinate a record any x' <= x has
    sum phi(x') < sum phi(x) unless x' = x, so nothing can. The set is a
    product of intervals, [0, 0.251] u [0.632, 0.859] up to the grid's
    resolution; x = 1 and x = 1/2 are not in it. The cube grid put most of
    its points outside: at M = 15 it was {0, 1}^14, all 16 384 points of which
    the filter kept and only x = 0 is Pareto-optimal; at M = 10, 1973 of the
    1990 kept points were dominated. So the grid runs over the records, spaced
    by their measure and with both ends of the set; where that grid would be
    its corners alone (p = 2, from M = 10) the positions are drawn from the
    records at random instead, with the corners of the set added — the extremes
    the frame is taken from.
    """
    import itertools
    if M == 1:
        return np.array([[2.0]])
    p = max(2, int(round((n * 2) ** (1.0 / (M - 1)))))
    rec = _dtlz7_record_cells()
    if p >= 3 and p ** (M - 1) <= 4 * n:
        grid = rec[np.linspace(0, len(rec) - 1, p).round().astype(int)]
        pts = np.asarray(list(itertools.product(grid, repeat=M - 1)), float)
    else:
        rng = np.random.default_rng(20260709 + 1000 * M + n)
        ends = np.array([rec[0], rec[-1]])
        corners = ends[(np.arange(1 << (M - 1))[:, None] >> np.arange(M - 1)) & 1] \
            if M - 1 <= 10 else np.vstack([np.full(M - 1, ends[0]), np.full(M - 1, ends[1]),
                                           ends[0] + np.eye(M - 1) * (ends[1] - ends[0])])
        pts = np.vstack([corners, _dtlz7_records(rng, max(0, 2 * n - len(corners)), M - 1)])    # f_M = (1+g)·h with g = 1 on the front → 2M − Σ f_i·(1+sin(3π f_i))
    fM = 2.0 * float(M) - np.sum(pts * (1.0 + np.sin(3.0 * math.pi * pts)), axis=1)
    F = np.column_stack([pts, fM])
    keep = _nondominated_mask(F)
    pts, F = pts[keep], F[keep]
    return F[_checked(pts, F, lambda X: _img_dtlz7(M, X), 20260922 + M)]


# ── Partially degenerate problems: their FULL fronts ────────────────────────
# DTLZ5, DTLZ6, MaF6 and WFG3 were designed to have a degenerate (curve-shaped)
# Pareto front, and the samplers below build exactly that curve. Ishibuchi,
# Masuda & Nojima, "Pareto Fronts of Many-Objective Degenerate Test Problems",
# IEEE TEC 20(5):807-813, 2016 (doi:10.1109/TEVC.2015.2505784) showed that the
# TRUE fronts also have a non-degenerate part (as the 2026-09-22 task sums the
# paper up; the paper itself is not in the local library):
#
#     DTLZ5, DTLZ6   non-degenerate part from M >= 4
#     WFG3           non-degenerate part from M >= 3
#     MaF6           from M >= 7 — our own analysis, not the paper's; below
#
# From 2026-09-22 those sizes use their FULL fronts, built and verified by
# fronts_full.py and shipped in _fronts/ (see _attach_full_fronts at the end of
# this file); the curve samplers below remain for the smaller sizes, where the
# curve IS the whole front — which the construction confirms: at DTLZ5_3D and
# WFG3_2D it finds no point off the curve at all.
#
# The naive repair was tried first (2026-09-09) and does not work: sampling the
# transition space and keeping the nondominated points yields the front OF THE
# SAMPLE — on WFG3 at M = 2, 173 of 300 sampled points off the curve. What
# fronts_full.py adds is the verification: every point must survive an
# independent sample of 10^6, a local search for a dominating point and an
# exact test against every attainable point, which is what removes those
# points; see its header.
#
# DEGENERATE_SUBSET_FROM still names the sizes at which the old curve would be
# a subset, and degenerate_subset_note warns only where a size has no shipped
# full front: WFG3 at six and ten, DTLZ5 at fifteen, DTLZ6 at ten and fifteen,
# MaF6 at fifteen, where a build in pure Python runs for hours (make_fronts
# builds one on request; the six-objective ones took about six hours each,
# DTLZ5_10D under three).
#
# MaF6 is NOT like DTLZ5 here, although it is DTLZ5(I, M) with I = 2: its
# objectives carry (1 + 100 g) where DTLZ5's carry (1 + g), while the angles
# still move with g alone. Leaving the curve then pays off only where an
# objective multiplies many cosines: at g = 2.5 each angle can shrink its
# factor from 0.707 to 0.2225, and 0.2225^k * 251 < 0.707^k needs k >= 5, i.e.
# M >= 7. Measured 2026-09-22: at M = 5 the build finds no point off the curve
# at all, and at M = 8 the front runs to the bound of g like DTLZ5's (f_7 up
# to 244.7). (Ishibuchi et al. 2016 predates MaF, 2017; the old entry
# "MaF6: 4" was carried over from DTLZ5.)
DEGENERATE_SUBSET_FROM = {"DTLZ5": 4, "DTLZ6": 4, "MaF6": 7, "WFG3": 3}

_DEGENERATE_WARNED: set[str] = set()

# Problem keys whose reference front is the full one (fronts_full).
FULL_FRONT: set[str] = set()


def degenerate_subset_note(name: str) -> str | None:
    """The caveat for `name`, or None when its reference front is exact."""
    if name in FULL_FRONT:
        return None
    stem = name.split("_")[0]
    m_from = DEGENERATE_SUBSET_FROM.get(stem)
    if m_from is None:
        return None
    tail = name.rsplit("_", 1)[-1]
    m = int(tail[:-1]) if tail.endswith("D") and tail[:-1].isdigit() else 2
    if m < m_from:
        return None
    return (f"{name}: the reference front is the DEGENERATE part only. "
            f"{stem} has a non-degenerate part from {m_from} objectives "
            f"(Ishibuchi, Masuda & Nojima, IEEE TEC 20(5), 2016), so IGD and "
            f"IGD+ here are measured against a subset of the true front: "
            f"comparable between algorithms run against this same set, not "
            f"with published numbers.")


def _nondominated_mask(F: np.ndarray) -> np.ndarray:
    n = len(F)
    keep = np.ones(n, bool)
    for i in range(n):
        if not keep[i]:
            continue
        dom = np.all(F <= F[i], axis=1) & np.any(F < F[i], axis=1)
        if np.any(dom):
            keep[i] = False
    return keep


def _nondominated(F: np.ndarray) -> np.ndarray:
    """Keep the nondominated points (minimization)."""
    return F[_nondominated_mask(F)]


# ---- a sampled front must be Pareto-optimal, not only unbeaten in the sample
def _strictly_dominated_by(C: np.ndarray, V: np.ndarray, block: int = 256) -> np.ndarray:
    out = np.zeros(len(C), bool)
    for a in range(0, len(C), block):
        c = C[a:a + block, None, :]
        out[a:a + block] = np.any(np.all(V[None] <= c, axis=2) & np.any(V[None] < c, axis=2),
                                  axis=1)
    return out


def _checked(X: np.ndarray, F: np.ndarray, images, seed: int,
             samples: int = 20_000) -> np.ndarray:
    """Mask of the rows of a sampled front that nothing outside the sample beats.

    DTLZ7, WFG1, WFG2 — and MaF7 and MaF11, which share their fronts — and ZDT3
    sample the front as the nondominated images of a grid of position vectors
    X. Where the front is disconnected that keeps grid points that merely no
    other GRID point beats. Measured 2026-09-22, n = 1000 against 100 000
    fresh front points: 150 of WFG2_3D's 416 reference points were dominated
    (by up to 0.38 in an objective spanning 2..6), 256 of WFG2_4D's 700 (1.13),
    32 of WFG2_6D's 1024 (5.2), 140 of WFG2_10D's 844 (11.7), 164 of
    DTLZ7_10D's 1990 (1.5), 7 of DTLZ7_3D's 529 (0.01); WFG1 had none.
    Each row is checked against `samples` fresh position vectors and its own
    neighbours — every position moved alone, both ways, by 0.05 down to 0.001,
    and 32 Gaussian moves of all of them at once — and dropped when any of
    them strictly dominates it. `images(X)` maps positions to objectives. The
    draws come from their own generator, so a front in which nothing is
    dropped is bit-identical to the unchecked one.
    """
    rng = np.random.default_rng(seed)
    d = X.shape[1]
    bad = _strictly_dominated_by(F, images(rng.random((samples, d))))
    moves = []
    for step in (0.05, 0.02, 0.01, 0.005, 0.002, 0.001):
        for j in range(d):
            for sgn in (-1.0, 1.0):
                Q = X.copy()
                Q[:, j] = np.clip(Q[:, j] + sgn * step, 0.0, 1.0)
                moves.append(Q)
    for s in (0.2, 0.1, 0.05, 0.02):
        for _ in range(8):
            moves.append(np.clip(X + rng.normal(0.0, s, X.shape), 0.0, 1.0))
    for Q in moves:
        G = images(Q)
        bad |= np.all(G <= F, axis=1) & np.any(G < F, axis=1)
    return ~bad


def _cached_front(fn):
    """Sample a front once per arguments and process; every caller gets a copy.

    A campaign asks for the reference front on every run, and the check above
    costs seconds.
    """
    import functools
    cached = functools.lru_cache(maxsize=None)(fn)

    @functools.wraps(fn)
    def front(*args):
        return cached(*args).copy()
    return front


def _dtlz7_records(rng, k: int, d: int) -> np.ndarray:
    """k points of DTLZ7's Pareto set in d positions: every coordinate a left
    record of phi(x) = x (1 + sin 3 pi x) (see _pf_dtlz7), drawn uniformly."""
    lo = _dtlz7_record_cells()
    X = lo[rng.integers(0, len(lo), (k, d))] + rng.random((k, d)) * _DTLZ7_CELL
    return np.minimum(X, 1.0)


_DTLZ7_CELL = 1.0 / 200_000


def _dtlz7_record_cells() -> np.ndarray:
    """Left ends of the cells of a 2e5 grid on [0, 1] where phi is a left record."""
    t = np.linspace(0.0, 1.0, 200_001)
    phi = t * (1.0 + np.sin(3.0 * math.pi * t))
    ok = phi >= np.maximum.accumulate(phi)
    return t[:-1][ok[:-1]]


def _img_dtlz7(M: int, X: np.ndarray) -> np.ndarray:
    F = np.empty((len(X), M))
    F[:, :M - 1] = X
    F[:, M - 1] = 2.0 * M - np.sum(X * (1.0 + np.sin(3.0 * math.pi * X)), axis=1)
    return F


def _img_wfg_convex(M: int, X: np.ndarray, last: np.ndarray) -> np.ndarray:
    """f_m = 2m h_m with WFG's convex h_1..h_M-1 and the given h_M."""
    s = np.sin(math.pi * X / 2.0)
    c = np.cos(math.pi * X / 2.0)
    H = np.empty((len(X), M))
    H[:, 0] = np.prod(1.0 - c, axis=1)
    for m in range(1, M - 1):
        H[:, m] = np.prod(1.0 - c[:, :M - 1 - m], axis=1) * (1.0 - s[:, M - 1 - m])
    H[:, M - 1] = last
    return H * (2.0 * np.arange(1, M + 1, dtype=float))


def _img_wfg1(M: int, X: np.ndarray) -> np.ndarray:
    x1 = X[:, 0]
    return _img_wfg_convex(M, X, 1.0 - x1 - np.cos(10.0 * math.pi * x1 + math.pi / 2.0)
                           / (10.0 * math.pi))


def _img_wfg2(M: int, X: np.ndarray) -> np.ndarray:
    x1 = X[:, 0]
    return _img_wfg_convex(M, X, 1.0 - x1 * np.cos(5.0 * math.pi * x1) ** 2)


def _img_zdt3(X: np.ndarray) -> np.ndarray:
    f1 = X[:, 0]
    return np.column_stack([f1, 1.0 - np.sqrt(f1) - f1 * np.sin(10.0 * math.pi * f1)])


_pf_dtlz7 = _cached_front(_pf_dtlz7)                # defined above, before the helpers


# ---- WFG base fronts: f_m = 2m*h_m, with x_M = 0 on the front -------
def _pf_wfg_concave(M: int, n: int) -> np.ndarray:
    """WFG4-9: h is concave, a DTLZ2-like sphere, so f_m = 2m*sphere_m."""
    d = _dd_sphere(M, n)
    scale = 2.0 * np.arange(1, M + 1, dtype=float)
    return d * scale


@_cached_front
def _pf_wfg1(M: int, n: int) -> np.ndarray:
    """WFG1 (Huband et al. 2006, Table XIV): h_{1..M-1} = convex_m and
    h_M = mixed_M with alpha = 1, A = 5, i.e.
        h_M = 1 - x_1 - cos(10 pi x_1 + pi/2) / (10 pi),
    so f_m = 2m * h_m on the front (x_M = 0).
    FIX 2026-09-05: the sampler used convex_M = 1 - sin(pi x_1 / 2) for the
    last objective, which is the WFG2-without-disconnection front, not
    WFG1's mixed convex/concave one; wfg1() itself was always right.
    """
    X = _dd_xset(M, n)
    F = np.empty((len(X), M))
    for r, x in enumerate(X):
        s = np.sin(math.pi * x / 2.0); c = np.cos(math.pi * x / 2.0)
        h = np.empty(M)
        h[0] = float(np.prod(1.0 - c)) if M > 1 else 1.0
        for m in range(1, M - 1):
            h[m] = float(np.prod(1.0 - c[:M - 1 - m])) * (1.0 - s[M - 1 - m])
        x1 = x[0]
        h[M - 1] = 1.0 - x1 - math.cos(10.0 * math.pi * x1 + math.pi / 2.0) / (10.0 * math.pi)
        F[r] = h
    scale = 2.0 * np.arange(1, M + 1, dtype=float)
    F = F * scale
    keep = _nondominated_mask(F)
    X, F = X[keep], F[keep]
    return F[_checked(X, F, lambda Q: _img_wfg1(M, Q), 20260922 + 100 + M)]


@_cached_front
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
    F = F * scale
    keep = _nondominated_mask(F)
    X, F = X[keep], F[keep]
    return F[_checked(X, F, lambda Q: _img_wfg2(M, Q), 20260922 + 200 + M)]


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


@_cached_front
def _pf_zdt3(n: int) -> np.ndarray:
    f1 = np.linspace(0.0, 1.0, 4 * n)
    f2 = 1.0 - np.sqrt(f1) - f1 * np.sin(10.0 * math.pi * f1)
    F = np.column_stack([f1, f2])
    keep = _nondominated_mask(F)
    return F[keep][_checked(f1[keep, None], F[keep], _img_zdt3, 20260922)]


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
    """MaF2 (Cheng et al. 2017, Eq. 3-4). On the front every g_i = 0 and
    theta_i = pi/2 * (x_i/2 + 1/4) with x_i in [0,1], so theta_i only covers
    [pi/8, 3pi/8]: the front is the PART of the unit sphere sum f^2 = 1 whose
    angles lie in that band ("partially concave"). Sampled on a uniform grid
    of x in [0,1]^{M-1} mapped through theta.
    FIX 2026-09-05: the sampler used to return the whole octant."""
    X = _dd_xset(M, n)
    F = np.empty((len(X), M))
    for r, x in enumerate(X):
        th = 0.5 * math.pi * (0.5 * x + 0.25)
        cos = np.cos(th); sin = np.sin(th)
        f = np.empty(M)
        f[0] = float(np.prod(cos)) if M > 1 else 1.0
        for i in range(2, M):
            f[i - 1] = float(np.prod(cos[:M - i])) * sin[M - i]
        if M > 1:
            f[M - 1] = sin[0]
        F[r] = f
    return F


def _pf_maf6(M: int, n: int) -> np.ndarray:
    """MaF6 = DTLZ5(I, M) with I = 2 (Cheng et al. 2017, Eq. 11-13). On the
    front g = 0, theta_i = pi/2 * x_i for i < I (ONE free angle, theta_1) and
    theta_i = pi/4 for i = I..M-1, so the front is the same one-dimensional
    curve as DTLZ5's, whatever M is. (The paper calls it "an I-dimensional
    manifold"; the curve lives in an I-dimensional subspace.)
    FIX 2026-09-05: the sampler used to free TWO angles, producing a 2-D
    surface that is not the front."""
    return _pf_dtlz5(M, n)


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


def _maf9_forbidden_regions(M: int) -> list:
    """The infeasible bow-tie polygons Phi of MaF9 (Cheng et al. 2017), one
    per pair of non-adjacent, non-parallel target lines. Empty for M <= 4."""
    ang = 2.0 * np.pi * np.arange(M) / M
    V = np.column_stack([np.cos(ang), np.sin(ang)])
    out = []
    for i in range(M):
        for n_ in range(i + 2, M):
            if i == 0 and n_ == M - 1:
                continue                          # adjacent through the wrap
            a1, b1 = V[i], V[(i + 1) % M]
            a2, b2 = V[n_], V[(n_ + 1) % M]
            d1 = b1 - a1; d2 = b2 - a2
            det = d1[0] * d2[1] - d1[1] * d2[0]
            if abs(det) < 1e-12:
                continue                          # parallel lines (even M)
            t = ((a2[0] - a1[0]) * d2[1] - (a2[1] - a1[1]) * d2[0]) / det
            O = a1 + t * d1
            # The chain facing O runs from the endpoint of edge i nearer to
            # O to the endpoint of edge n_ nearer to O, along the boundary.
            if np.linalg.norm(O - b1) < np.linalg.norm(O - a1):
                chain = [V[j % M] for j in range(i + 1, n_ + 1)]      # V[i+1..n_]
            else:
                chain = [V[j % M] for j in range(i, n_ - M, -1)]      # V[i], V[i-1], .., V[n_+1]
            refl = [2.0 * O - c for c in chain]
            poly = [chain[0]] + refl + chain[:0:-1]
            out.append(np.asarray(poly, float))
    return out


def _poly_edge_distance(x: np.ndarray, P: np.ndarray) -> float:
    """Distance from x to the nearest edge of polygon P."""
    best = float("inf")
    n = len(P)
    for i in range(n):
        a = P[i]; b = P[(i + 1) % n]
        ab = b - a; L2 = float(ab @ ab)
        t = 0.0 if L2 == 0.0 else float(np.clip(((x - a) @ ab) / L2, 0.0, 1.0))
        d = float(np.linalg.norm(x - (a + t * ab)))
        if d < best:
            best = d
    return best


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
#  ZDT1-6 (M = 2; n = 30 for ZDT1-3, n = 10 for ZDT4 and ZDT6; ZDT5 is binary
#  and not registered)
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

    # Zitzler, Deb & Thiele 2000, Eq. 10: m = 10 for T4 (ZDT1-3 use m = 30).
    # FIX 2026-09-05: this entry used to register 30 variables.
    bnd4 = [(0.0,1.0)] + [(-5.0,5.0)]*9
    PROBLEMS["ZDT4"] = BenchProblem(
        name="ZDT4", n_vars=10, bounds=bnd4, n_obj=2,
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
#  R. Cheng et al.
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

        # ── MaF2 — DTLZ2BZ (Cheng et al. 2017, Eq. 3-4) ─────────────────────
        # theta_i = pi/2 * (x_i/2 + 1/4) for the M-1 position variables, and a
        # SEPARATE distance term g_i per objective, each summing
        # ((x_j/2 + 1/4) - 0.5)^2 over its own chunk of the K distance
        # variables: chunk size c = floor(K/M); g_1..g_{M-1} take c variables
        # each and g_M takes the remainder (Eq. 4, 1-indexed j from
        # M+(i-1)c to M+ic-1, and to D for g_M). With K = 10 and M = 15 the
        # chunk size is 0, so g_1..g_14 are empty sums, as printed.
        # FIX 2026-09-05: this entry used to evaluate plain DTLZ2 (single g,
        # theta = pi/2 * x_i), so the whole "partially concave" character —
        # and the front — was wrong.
        def maf2(x, M=M):
            x = np.asarray(x, dtype=float)
            D = len(x)
            K = D - M + 1
            c = K // M
            th = 0.5 * math.pi * (0.5 * x[:M-1] + 0.25)
            cos = np.cos(th); sin = np.sin(th)
            g = np.empty(M)
            for i in range(M - 1):
                lo = (M - 1) + i * c; hi = lo + c
                seg = x[lo:hi]
                g[i] = float(np.sum((0.5 * seg + 0.25 - 0.5) ** 2)) if hi > lo else 0.0
            seg = x[(M - 1) + (M - 1) * c:]
            g[M-1] = float(np.sum((0.5 * seg + 0.25 - 0.5) ** 2))
            f = np.empty(M)
            f[0] = float(np.prod(cos)) * (1.0 + g[0])
            for i in range(2, M):
                f[i-1] = float(np.prod(cos[:M-i])) * sin[M-i] * (1.0 + g[i-1])
            f[M-1] = sin[0] * (1.0 + g[M-1])
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
            # Eq. 11: the objectives carry (1 + 100 g); Eq. 12's theta uses g
            # itself. FIX 2026-09-05: this used to be (1 + g).
            base = 1.0 + 100.0*g; f = np.empty(M)
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

        # MaF7 is the one MaF problem with K = 20 distance variables
        # (D = M + 19); the others use K = 10. FIX 2026-09-05: it used K = 10.
        n7 = M + 19
        PROBLEMS[f"MaF7_{M}D"] = BenchProblem(
            name=f"MaF7_{M}D", n_vars=n7,
            bounds=[(0.0,1.0)]*n7, n_obj=M,
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
        # Decision space x in [-10000, 10000]^2, as the paper states; the
        # huge box IS the difficulty (a random point starts ~10^4 away from
        # the front). FIX 2026-09-05: this used to register [-2, 2]^2.
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
            bounds=[(-10000.0,10000.0)]*2, n_obj=M,
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
        # Decision space x in [-10000, 10000]^2, as the paper states.
        # Infeasible regions (Cheng et al. 2017, MaF9 text): for M >= 5 some
        # pairs of NON-adjacent target lines meet at a point O outside the
        # polygon, and points near O are nondominated with the interior. The
        # paper removes them by declaring infeasible, for every such pair,
        # the polygon Phi = <A_i, A'_i, A'_{i+1}, ..., A'_n, A_n, ..., A_{i+1}>
        # where A_i..A_n is the boundary chain facing O and A' is the point
        # reflection of A through O — a bow-tie of two lobes touching at O.
        # The paper repairs offenders by resampling; this port cannot move a
        # decision vector from inside evaluate(), so membership of any Phi is
        # reported as a constraint violation (cv = penetration depth), which
        # under the feasibility rules has the same effect: such points never
        # survive. For M = 3 and 4 no such pair exists (the paper: "such
        # areas exist in the problem with five or more objectives"), so those
        # instances are unconstrained.
        # FIX 2026-09-05: this used to register [-2, 2]^2 and to declare the
        # WHOLE exterior of the polygon infeasible at every M, which is a
        # different (much easier) problem.
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

        _phi9 = _maf9_forbidden_regions(M)

        def maf9_cons(x, M=M, _phi=_phi9):
            x = np.asarray(x, dtype=float)[:2]
            cv = 0.0
            for P in _phi:
                if _point_in_poly(x[None, :], P)[0]:
                    cv += _poly_edge_distance(x, P)
            return [cv]

        PROBLEMS[f"MaF9_{M}D"] = BenchProblem(
            name=f"MaF9_{M}D", n_vars=2,
            bounds=[(-10000.0,10000.0)]*2, n_obj=M,
            evaluate=maf9, constraints=(maf9_cons if _phi9 else _no_cons),
            hv_ref_raw=tuple([2.0]*M),
            hv_ref_norm=_ref_norm(M), hv_norm_divisor=_divisor(M),
            ideal=tuple([0.0]*M), nadir=tuple([1.0]*M),
            pop_size=pop, n_gen=ng, K_runs=21, has_cons=bool(_phi9))

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
            _ps = None
            if name in ("DTLZ1", "DTLZ2", "DTLZ3", "DTLZ4"):
                def _ps(n, _M=M, _nv=n_vars): return _psets.dtlz(_M, _nv, n)
            prob_name = f"{name}_{M}D"
            PROBLEMS[prob_name] = BenchProblem(
                name=prob_name, n_vars=n_vars, bounds=bounds, n_obj=M,
                evaluate=_eval, constraints=_no_cons,
                hv_ref_raw=tuple(v*1.1 for v in nadir),
                hv_ref_norm=_ref_norm(M), hv_norm_divisor=_divisor(M),
                ideal=ideal, nadir=nadir,
                pop_size=pop, n_gen=ng, K_runs=21, has_cons=False,
                pareto_front=_pf, pareto_set=_ps)


# =============================================================
#  shiftDTLZ1-4 × the DTLZ sizes: DTLZ1-4 with every distance variable's
#  optimum moved off the centre of the box (dtlz_variants.shift_centres).
#  Front, ideal and nadir are DTLZ's, and so is the reference front.
# =============================================================
def _register_shifted_dtlz():
    for M in (2, 3, 4, 5, 6, 10, 15):
        pop, ng = _budget(M)
        for base in SHIFT_BASES:
            n_vars = dtlz_n_vars(base, M)
            nadir  = dtlz_nadir(base, M)
            ideal  = dtlz_ideal(base, M)
            def _eval(x, _b=base, _M=M): return shift_dtlz(_b, list(x), _M)
            def _pf(n, _g=_DTLZ_PF[base], _M=M): return _g(_M, n)
            def _ps(n, _M=M, _nv=n_vars):
                return _psets.dtlz(_M, _nv, n, shift_centres(_nv - _M + 1))
            prob_name = f"shift{base}_{M}D"
            PROBLEMS[prob_name] = BenchProblem(
                name=prob_name, n_vars=n_vars, bounds=[(0.0, 1.0)] * n_vars, n_obj=M,
                evaluate=_eval, constraints=_no_cons,
                hv_ref_raw=tuple(v * 1.1 for v in nadir),
                hv_ref_norm=_ref_norm(M), hv_norm_divisor=_divisor(M),
                ideal=ideal, nadir=nadir,
                pop_size=pop, n_gen=ng, K_runs=21, has_cons=False,
                pareto_front=_pf, pareto_set=_ps,
                cyclic_vars=tuple(range(M - 1, n_vars)))


# =============================================================
#  ZCAT1-20 × M={2,3,5,10} — Zapotecas-Martínez, Coello Coello, Aguirre &
#  Tanaka, Swarm and Evolutionary Computation 81 (2023) 101350.
#  Registered with the suite's own defaults (Section 5.1): n = 10M, Level 1,
#  a complicated Pareto set, no bias, no imbalance. The other dials — the six
#  difficulty levels, bias, imbalance, the simple PS — are arguments of
#  zcat.evaluate, not separate registry entries, because they multiply 20
#  problems by 24 and none of them change the front.
#  THE REFERENCE FRAME IS THE PAPER'S, NOT THE SAMPLED FRONT'S, and that is a
#  deliberate exception to the convention at the top of this file. Section 4.1
#  states the frame outright — "the ideal and the Nadir points of the test
#  problems with M objectives are z* = (0, 0, ..., 0) and n* = (1², 2², ...,
#  M²)" — deriving it from F ∈ [0,1]^M, so it is exact where a sample is an
#  estimate. Measured, the two agree: the sampled front reaches that bound on
#  every objective of all 20 problems at M = 3 and at M = 10. The agreement is
#  why the exact frame costs nothing here, and the reason to prefer it anyway
#  is that it does not depend on how well the front happened to be sampled —
#  before zcat.pareto_front sampled the corners of the position cube and kept
#  the per-objective extremes, that same sample reached 0.007 of the bound on
#  ZCAT2's first objective at M = 10, and taking it as the nadir would have
#  rescaled every normalized indicator on that problem by 140x. The reference
#  FRONT is still sampled, from the image of alpha, since only Theorem 1
#  characterises it.
# =============================================================
def _register_zcat():
    for M in (2, 3, 5, 10):
        pop, ng = _budget(M)
        for name in _zcat.NAMES:
            sp = _zcat.spec(name, M)
            prob_name = f"{name}_{M}D"
            PROBLEMS[prob_name] = BenchProblem(
                name=prob_name, n_vars=sp["n_vars"], bounds=sp["bounds"], n_obj=M,
                evaluate=sp["evaluate"], constraints=_no_cons,
                hv_ref_raw=tuple(v * 1.1 for v in sp["nadir"]),
                hv_ref_norm=_ref_norm(M), hv_norm_divisor=_divisor(M),
                ideal=sp["ideal"], nadir=sp["nadir"],
                pop_size=pop, n_gen=ng, K_runs=21, has_cons=False,
                pareto_front=sp["pareto_front"],
                pareto_set=(lambda n, _n=name, _M=M: _psets.zcat(_n, _M, n)))


# =============================================================
#  bbob-biobj × n={5,10} — Brockhoff, Auger, Hansen & Tusar, Evolutionary
#  Computation 30(2), 2022. 55 problems, each a pair of the ten bbob functions
#  in bbob.py, at two objectives and 5 or 10 variables.
#
#  THESE ARE THE ONLY PROBLEMS HERE WITH NO REFERENCE FRONT, and that is a
#  property of the suite rather than a gap: the Pareto set of a pair of bbob
#  functions has no closed form, and COCO itself estimates each instance's
#  hypervolume from the accumulated output of many experiments. So
#  pareto_front stays None and IGD/IGD+/GD+ report nothing on them, which is
#  the honest answer; the hypervolume works, because the ideal and the nadir
#  ARE exact — each objective's unique global optimum is known by
#  construction, so the ideal is the pair of optimal values and the nadir is
#  each objective's value at the other's optimum.
#
#  The name carries the variable count in its own segment, `_n05_`, because
#  the registry's `_<k>D` suffix means the OBJECTIVE count and the config's
#  `objectives = [...]` filter matches on it. These are two-objective problems
#  that scale in variables, so the suffix is `_2D` for all of them.
# =============================================================
def _register_bbob_biobj():
    pop, ng = _budget(2)
    for D in (5, 10):
        for fnum in range(1, 56):
            sp = _bbobbiobj.spec(fnum, D, 1)
            ideal, nadir = sp["ideal"], sp["nadir"]
            key = _bbobbiobj.name(fnum, D)
            PROBLEMS[key] = BenchProblem(
                name=key, n_vars=sp["n_vars"], bounds=sp["bounds"], n_obj=2,
                evaluate=sp["evaluate"], constraints=_no_cons,
                hv_ref_raw=tuple(nd + 0.1 * (nd - id_)
                                 for id_, nd in zip(ideal, nadir)),
                hv_ref_norm=_ref_norm(2), hv_norm_divisor=_divisor(2),
                ideal=ideal, nadir=nadir,
                pop_size=pop, n_gen=ng, K_runs=21, has_cons=False,
                pareto_front=None)


def _register_uninformative():
    pop, ng = _budget(2)
    for n in _uninf.SIZES:
        key = _uninf.name(n)
        PROBLEMS[key] = BenchProblem(
            name=key, n_vars=n, bounds=[(0.0, 1.0)] * n, n_obj=2,
            evaluate=_uninf.evaluate_by_x, constraints=_no_cons,
            hv_ref_raw=(1.1, 1.1), hv_ref_norm=_ref_norm(2), hv_norm_divisor=_divisor(2),
            ideal=(0.0, 0.0), nadir=(1.0, 1.0),
            pop_size=pop, n_gen=ng, K_runs=30, has_cons=False,
            pareto_front=None, make_evaluator=_uninf.Evaluator)


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
        def _ps(n, _M=M, _nv=n_vars): return _psets.polygon(_M, _nv, n)
        prob_name = f"Polygon_{M}D"
        PROBLEMS[prob_name] = BenchProblem(
            name=prob_name, n_vars=n_vars, bounds=bounds, n_obj=M,
            evaluate=_eval, constraints=_no_cons,
            hv_ref_raw=tuple(v*1.1 for v in nadir),
            hv_ref_norm=_ref_norm(M), hv_norm_divisor=_divisor(M),
            ideal=ideal, nadir=nadir,
            pop_size=pop, n_gen=ng, K_runs=21, has_cons=False,
            pareto_front=_pf, pareto_set=_ps)


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
_register_shifted_dtlz()
_register_zcat()
_register_bbob_biobj()
_register_uninformative()
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


def _attach_full_fronts():
    """Give the partially degenerate problems their full fronts (fronts_full).

    The reference front becomes the stored, verified set — thinned by taking
    its first n points, since it is stored in DSS order. Nothing is read here,
    only checked for: the file loads on the first pareto_front() call, and the
    frame (which moves: WFG3_3D's first objective now reaches about 3, not 1)
    comes from _refframe_cache.json on get(), where fronts_full.py writes the
    whole verified set's ideal and nadir.
    """
    from . import fronts_full as _ff
    for name, sizes in _ff.SIZES.items():
        for M in sizes:
            key = f"{name}_{M}D"
            if key in PROBLEMS and _ff.has_front(name, M):
                PROBLEMS[key].pareto_front = (lambda n, _n=name, _M=M: _ff.front(_n, _M, n))
                FULL_FRONT.add(key)


_attach_full_fronts()


# ── HV/IGD reference frame = the sampled reference PF (PlatEMO convention) ──
# Computed LAZILY, per problem, on first lookup.
#
# Doing all 216 eagerly at import cost 136 seconds, which makes the package
# unusable as a library: `import mootation.benchmarks` to list names should
# not sample a Pareto front. A shipped cache file makes the common case free;
# a cache miss falls back to sampling that one problem, in memory. Nothing is
# ever written back — a library that writes into its own installation
# directory breaks on any read-only or shared install.
# The shipped file is _refframe_cache.json (FIX 2026-09-05: the code looked
# for reference_frames.json, which never existed, so every lookup fell back
# to sampling the front — 136 s for the whole registry).
_REFRAME_CACHE_FILE = _Path(__file__).with_name("_refframe_cache.json")
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
    note = degenerate_subset_note(name)
    if note and name not in _DEGENERATE_WARNED:
        # Once per problem per process: a caveat that scrolls past on every
        # call is a caveat nobody reads, and this one changes how the number
        # may be used, not whether the run works.
        _DEGENERATE_WARNED.add(name)
        warnings.warn(note, stacklevel=2)
    return p


def families() -> dict:
    """Problem names grouped by family stem: {"DTLZ": [...], "WFG": [...]}."""
    out: Dict[str, List[str]] = {}
    for n in sorted(PROBLEMS):
        stem = n.split("_")[0]
        stem = "".join(c for c in stem if not c.isdigit()) or stem
        out.setdefault(stem, []).append(n)
    return out

# SPDX-License-Identifier: Apache-2.0
# ============================================================================
# Irregular and scaled variants of DTLZ.
#
# IDTLZ1/2   — inverted DTLZ. IDTLZ1 is Jain & Deb 2014 (Part II), Eq. 9:
#              f_i <- 0.5(1+g) - f_i. IDTLZ2, f_i <- (1+g) - f_i, appears in
#              neither NSGA-III paper; it is the analogous construction used
#              by later suites (PlatEMO's IDTLZ2) and is declared as such.
#              The front is a rotated (inverted) simplex or sphere of the
#              same size.
# minusDTLZ  — the minus variants (Ishibuchi et al. 2017): every objective is
#              negated. The front is far larger, and the optimum sits at
#              distance variables of 0 or 1 (g at its maximum, 0.25k for
#              DTLZ2) rather than at 0.5.
# SDTLZ1/2   — scaled DTLZ (Deb & Jain 2014, §V-C and Table VIII): objective
#              i is multiplied by p^(i-1) with a base p that DEPENDS ON M AND
#              ON THE PROBLEM: SDTLZ1 p = 10, 10, 3, 2, 1.2 and SDTLZ2
#              p = 10, 10, 3, 3, 2 for M = 3, 5, 8, 10, 15. The point of the
#              family is to break algorithms that assume comparable objective
#              magnitudes. FIX 2026-09-05: base 10 used to be applied at
#              every M, which at M = 10 scaled f_10 by 10^9 instead of 2^9.
# shiftDTLZ1-4 — DTLZ1-4 with the optimum of every distance variable moved off
#              the centre of the box. DTLZ puts it at exactly x = 1/2, where an
#              operator or an initialisation that drifts toward the middle of
#              the range lands for free. Each distance variable is read through
#              the cyclic shift y = (x - c + 1/2) mod 1, which moves its optimum
#              to x = c. The shifted g is continuous, because every DTLZ g term
#              takes the same value at y = 0 and at y = 1, and it keeps DTLZ's
#              optima (for DTLZ1 and DTLZ3 the two half-basins of the local
#              optimum at the bounds join into one). Position variables, front,
#              ideal and nadir are DTLZ's. c_i = 1/2 +- (0.15 + 0.2 frac(i phi)),
#              + for odd i, phi the golden-ratio conjugate: every optimum is at
#              least 0.15 from the centre and from both bounds, and each
#              distance variable has its own. The family is a control for
#              centre bias written for this library, not a published suite.
#
# The reference fronts are built analytically, with g pinned at the value it
# takes on the front, so that IGD, IGD+ and GD+ are computed against the real
# thing rather than against whatever the run happened to find.
# ============================================================================
from __future__ import annotations
import math
from typing import List
import numpy as np
import itertools

from . import dtlz as _d

# Deb & Jain 2014, Table VIII: the base p of the per-objective factor p^(i-1).
SCALE_BASE = {
    "SDTLZ1": {3: 10.0, 5: 10.0, 8: 3.0, 10: 2.0, 15: 1.2},
    "SDTLZ2": {3: 10.0, 5: 10.0, 8: 3.0, 10: 3.0, 15: 2.0},
}


def scale_base(name: str, M: int) -> float:
    """The scaling base for SDTLZ1/2 at M objectives; M outside Table VIII
    falls back to the nearest tabulated M below it (10 at the low end)."""
    tab = SCALE_BASE[name]
    if M in tab:
        return tab[M]
    below = [m for m in tab if m < M]
    return tab[max(below)] if below else 10.0


# ---- eval ----------------------------------------------------------
def idtlz1(x: List[float], M: int) -> List[float]:
    x = np.asarray(x, float)
    g = _d._g_dtlz1(x[M - 1:])
    f = np.asarray(_d.dtlz1(x, M))
    return (0.5 * (1.0 + g) - f).tolist()


def idtlz2(x: List[float], M: int) -> List[float]:
    x = np.asarray(x, float)
    g = _d._g_dtlz2(x[M - 1:])
    f = np.asarray(_d.dtlz2(x, M))
    return ((1.0 + g) - f).tolist()


def minus_dtlz2(x: List[float], M: int) -> List[float]:
    return (-np.asarray(_d.dtlz2(x, M))).tolist()


def _scale(M: int, name: str = "SDTLZ1") -> np.ndarray:
    return scale_base(name, M) ** np.arange(M, dtype=float)


def sdtlz1(x: List[float], M: int) -> List[float]:
    return (np.asarray(_d.dtlz1(x, M)) * _scale(M, "SDTLZ1")).tolist()


def sdtlz2(x: List[float], M: int) -> List[float]:
    return (np.asarray(_d.dtlz2(x, M)) * _scale(M, "SDTLZ2")).tolist()


# ---- samplers for the base fronts (g = 0) --------------------------
def _simplex(M: int, n: int) -> np.ndarray:
    # FIX 2026-07-09, in step with maf.py::_simplex: at high M the grid
    # degenerated - about 55 points at M=10, up to 120 at M=15, against the
    # 1000 requested - leaving the DTLZ/IDTLZ/SDTLZ/minusDTLZ reference front
    # for IGD+ built from a handful of points. It is topped up to n with a
    # uniform Dirichlet(1,...,1) draw from a fixed seed. IGD+ values at
    # M >= 10 are NOT comparable with numbers produced before this fix.
    p = max(1, int(round(n ** (1.0 / (M - 1)))) ) if M > 1 else 1
    pts = []
    for combo in itertools.product(np.linspace(0, 1, p + 1), repeat=M - 1):
        if sum(combo) <= 1.0 + 1e-9:
            pts.append(list(combo) + [1.0 - sum(combo)])
    W = np.asarray(pts, float)               # sum = 1; the DTLZ1 front is half of this
    if len(W) < n:
        rng = np.random.default_rng(20260709 + 1000 * M + n)
        extra = rng.dirichlet(np.ones(M), size=n - len(W))
        W = np.vstack([W, extra])
    return W


def _sphere_oct(M: int, n: int) -> np.ndarray:
    w = _simplex(M, n)
    nrm = np.linalg.norm(w, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    return w / nrm                            # sum f^2 = 1, one octant


# ---- reference fronts ----------------------------------------------
def pf_idtlz1(M: int, n: int = 1000) -> np.ndarray:
    s = 0.5 * _simplex(M, n)                  # a point of the DTLZ1 front (sum = 0.5)
    return 0.5 - s                            # IDTLZ1: f_i = 0.5 - s_i


def pf_idtlz2(M: int, n: int = 1000) -> np.ndarray:
    d = _sphere_oct(M, n)
    return 1.0 - d                            # IDTLZ2: f_i = 1 - d_i


def pf_minus_dtlz2(M: int, n: int = 1000) -> np.ndarray:
    kd = _d.DTLZ_K["DTLZ2"]
    d = _sphere_oct(M, n)
    return -(1.0 + 0.25 * kd) * d             # g = max = 0.25k


def pf_sdtlz1(M: int, n: int = 1000) -> np.ndarray:
    return (0.5 * _simplex(M, n)) * _scale(M, "SDTLZ1")


def pf_sdtlz2(M: int, n: int = 1000) -> np.ndarray:
    return _sphere_oct(M, n) * _scale(M, "SDTLZ2")


# ---- ideal/nadir + n_vars ------------------------------------------
def _kd(name_base: str) -> int:
    return _d.DTLZ_K[name_base]


# name -> (eval, base_dtlz_for_k, pf, ideal_fn, nadir_fn)
def _spec_idtlz1(M):
    return tuple([0.0] * M), tuple([0.5] * M)


SPECS = {}  # filled in below: name -> dict


def build_specs():
    out = {}
    out["IDTLZ1"] = dict(eval=idtlz1, base="DTLZ1", pf=pf_idtlz1,
                         ideal=lambda M: tuple([0.0] * M),
                         nadir=lambda M: tuple([0.5] * M))
    out["IDTLZ2"] = dict(eval=idtlz2, base="DTLZ2", pf=pf_idtlz2,
                         ideal=lambda M: tuple([0.0] * M),
                         nadir=lambda M: tuple([1.0] * M))
    out["minusDTLZ2"] = dict(eval=minus_dtlz2, base="DTLZ2", pf=pf_minus_dtlz2,
                         ideal=lambda M: tuple([-(1.0 + 0.25 * _kd("DTLZ2"))] * M),
                         nadir=lambda M: tuple([0.0] * M))
    out["SDTLZ1"] = dict(eval=sdtlz1, base="DTLZ1", pf=pf_sdtlz1,
                         ideal=lambda M: tuple([0.0] * M),
                         nadir=lambda M: tuple((0.5 * _scale(M, "SDTLZ1")).tolist()))
    out["SDTLZ2"] = dict(eval=sdtlz2, base="DTLZ2", pf=pf_sdtlz2,
                         ideal=lambda M: tuple([0.0] * M),
                         nadir=lambda M: tuple((_scale(M, "SDTLZ2")).tolist()))
    return out


SPECS = build_specs()


def variant_n_vars(base: str, M: int) -> int:
    return M + _d.DTLZ_K[base] - 1


def hv_ref_raw(ideal: tuple, nadir: tuple) -> tuple:
    # ref = nadir + 0.1*(nadir - ideal), correct for negative objectives too
    return tuple(nd + 0.1 * (nd - id_) for id_, nd in zip(ideal, nadir))


# ---- shiftDTLZ1-4: the distance optimum moved off the centre -------
SHIFT_BASES = ("DTLZ1", "DTLZ2", "DTLZ3", "DTLZ4")
_PHI = (math.sqrt(5.0) - 1.0) / 2.0


def shift_centres(k: int) -> np.ndarray:
    """Where the optimum of each of the k distance variables sits.

    c_i = 1/2 + d_i for odd i and 1/2 - d_i for even i (i = 1..k), with
    d_i = 0.15 + 0.2 frac(i phi).
    """
    i = np.arange(1, k + 1)
    d = 0.15 + 0.2 * np.mod(i * _PHI, 1.0)
    return 0.5 + np.where(i % 2 == 1, d, -d)


def shift_dtlz(name: str, x: List[float], M: int) -> List[float]:
    """DTLZ1-4 with every distance variable read through y = (x - c + 1/2) mod 1."""
    y = np.array(x, dtype=float)
    y[M - 1:] = np.mod(y[M - 1:] - shift_centres(len(y) - M + 1) + 0.5, 1.0)
    return _d.DTLZ_FUNCS[name](y, M)

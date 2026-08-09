# SPDX-License-Identifier: Apache-2.0
# ============================================================================
# Irregular and scaled variants of DTLZ.
#
# IDTLZ1/2   — inverted DTLZ (Jain & Deb 2014, Part II): f_i <- 0.5(1+g) - f_i
#              for DTLZ1, and (1+g) - f_i for DTLZ2 by analogy. The front is a
#              rotated (inverted) simplex or sphere of the same size.
# minusDTLZ  — the minus variants (Ishibuchi et al. 2017): every objective is
#              negated. The front is far larger, and the optimum sits at
#              distance variables of 0 or 1 (g at its maximum, 0.25k for
#              DTLZ2) rather than at 0.5.
# SDTLZ1/2   — scaled DTLZ (Deb & Jain 2014): objective i is multiplied by
#              10^i. Base 10 is Deb and Jain's own choice, and the point of
#              the family is to break algorithms that assume comparable
#              objective magnitudes.
#
# The reference fronts are built analytically, with g pinned at the value it
# takes on the front, so that IGD, IGD+ and GD+ are computed against the real
# thing rather than against whatever the run happened to find.
# ============================================================================
from __future__ import annotations
from typing import List
import numpy as np
import itertools

from . import dtlz as _d

SCALE_BASE = 10.0  # SDTLZ: objective i is scaled by SCALE_BASE^i


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


def _scale(M: int) -> np.ndarray:
    return SCALE_BASE ** np.arange(M, dtype=float)


def sdtlz1(x: List[float], M: int) -> List[float]:
    return (np.asarray(_d.dtlz1(x, M)) * _scale(M)).tolist()


def sdtlz2(x: List[float], M: int) -> List[float]:
    return (np.asarray(_d.dtlz2(x, M)) * _scale(M)).tolist()


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
    return (0.5 * _simplex(M, n)) * _scale(M)


def pf_sdtlz2(M: int, n: int = 1000) -> np.ndarray:
    return _sphere_oct(M, n) * _scale(M)


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
                         nadir=lambda M: tuple((0.5 * _scale(M)).tolist()))
    out["SDTLZ2"] = dict(eval=sdtlz2, base="DTLZ2", pf=pf_sdtlz2,
                         ideal=lambda M: tuple([0.0] * M),
                         nadir=lambda M: tuple((_scale(M)).tolist()))
    return out


SPECS = build_specs()


def variant_n_vars(base: str, M: int) -> int:
    return M + _d.DTLZ_K[base] - 1


def hv_ref_raw(ideal: tuple, nadir: tuple) -> tuple:
    # ref = nadir + 0.1*(nadir - ideal), correct for negative objectives too
    return tuple(nd + 0.1 * (nd - id_) for id_, nd in zip(ideal, nadir))

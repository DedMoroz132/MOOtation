# SPDX-License-Identifier: Apache-2.0
# ============================================================================
# MOP1-7, the imbalanced test problems.
#
# MOP1-5 are bi-objective, MOP6-7 have three objectives. n = 10 variables over
# [0, 1]^n, all minimized. The g function is modified to be multimodal and
# imbalanced, and that is the trap: global indicator-based and decomposition
# methods are the ones it catches, because the imbalance makes a solution that
# scores well on the aggregate far from the front.
#
# Each problem provides: eval(x) -> [f...]; pf(n) -> ndarray (n x M), a sample
# of the TRUE front for IGD, IGD+ and GD+.
# ============================================================================
from __future__ import annotations
from typing import List
import numpy as np

from ._rows import spow

N_VARS = 10


# ---- shared g kernels ----------------------------------------------
# ---- eval, of a batch (task 10 of 2026-10-18) -----------------------
# mop_rows(name, X): the (n, M) objectives of the rows of X, and mopN(x) its
# batch of one row, by the expressions of the point code before, bit for bit;
# a scalar's ** as pow gives it (spow). g_a = sum(-0.9 t^2 + |t|^0.6) (MOP1,
# 5, 6, 7), g_b = sum(|t| / (1 + e^|t|)) (MOP2, 3, 4).
def mop_rows(name: str, X) -> np.ndarray:
    X = np.ascontiguousarray(X, dtype=float).reshape(-1, N_VARS)
    x1 = X[:, 0]
    if name in ("MOP6", "MOP7"):
        x2 = X[:, 1]
        T = X[:, 2:] - (x1 * x2)[:, None]
        g = 2.0 * np.sin(np.pi * x1) * np.sum(-0.9 * T * T + np.abs(T) ** 0.6, axis=1)
        if name == "MOP6":
            return np.column_stack([(1 + g) * x1 * x2, (1 + g) * x1 * (1 - x2), (1 + g) * (1 - x1)])
        c1, c2 = np.cos(0.5 * np.pi * x1), np.cos(0.5 * np.pi * x2)
        s1, s2 = np.sin(0.5 * np.pi * x1), np.sin(0.5 * np.pi * x2)
        return np.column_stack([(1 + g) * c1 * c2, (1 + g) * c1 * s2, (1 + g) * s1])
    T = X[:, 1:] - np.sin(0.5 * np.pi * x1)[:, None]
    if name in ("MOP1", "MOP5"):
        ga = np.sum(-0.9 * T * T + np.abs(T) ** 0.6, axis=1)
        g = (2.0 * np.sin(np.pi * x1) if name == "MOP1" else 2.0 * np.abs(np.cos(np.pi * x1))) * ga
        return np.column_stack([(1 + g) * x1, (1 + g) * (1 - np.sqrt(x1))])
    at = np.abs(T)
    gb = np.sum(at / (1.0 + np.exp(at)), axis=1)
    if name == "MOP3":
        g = 10.0 * np.sin(0.5 * np.pi * x1) * gb
        return np.column_stack([(1 + g) * np.cos(0.5 * np.pi * x1), (1 + g) * np.sin(0.5 * np.pi * x1)])
    g = 10.0 * np.sin(np.pi * x1) * gb
    if name == "MOP2":
        return np.column_stack([(1 + g) * x1, (1 + g) * (1 - x1 * x1)])
    return np.column_stack([(1 + g) * x1,                                       # MOP4
                            (1 + g) * (1 - spow(x1, 0.5) * spow(np.cos(2 * np.pi * x1), 2))])


def _one(name: str):
    def one(x: List[float]) -> List[float]:
        return mop_rows(name, [list(x)])[0].tolist()
    one.__name__ = name.lower()
    return one


mop1, mop2, mop3, mop4, mop5, mop6, mop7 = (_one(f"MOP{i}") for i in range(1, 8))


def _nd_filter(F: np.ndarray) -> np.ndarray:
    # keep the nondominated subset (MOP4 is disconnected).
    # FIX 2026-09-05: the previous version computed the rows that DOMINATE
    # F[i] and dropped THEM, i.e. it returned the dominated part of the MOP4
    # curve as the reference front.
    keep = np.ones(len(F), bool)
    for i in range(len(F)):
        if not keep[i]:
            continue
        dominated_by = np.all(F <= F[i], axis=1) & np.any(F < F[i], axis=1)
        if np.any(dominated_by):
            keep[i] = False
    return F[keep]


def pf_mop1(n: int = 500) -> np.ndarray:
    f1 = np.linspace(0, 1, n)
    return np.column_stack([f1, 1 - np.sqrt(f1)])


pf_mop5 = pf_mop1


def pf_mop2(n: int = 500) -> np.ndarray:
    f1 = np.linspace(0, 1, n)
    return np.column_stack([f1, 1 - f1 * f1])


def pf_mop3(n: int = 500) -> np.ndarray:
    th = np.linspace(0, 0.5 * np.pi, n)
    return np.column_stack([np.cos(th), np.sin(th)])


def pf_mop4(n: int = 500) -> np.ndarray:
    u = np.linspace(0, 1, 20 * n)
    F = np.column_stack([u, 1 - u ** 0.5 * np.cos(2 * np.pi * u) ** 2])
    return _nd_filter(F)


def _simplex_grid(M: int, n: int) -> np.ndarray:
    # ~n points on the unit simplex sum f = 1 (uniform grid over the first M-1)
    import itertools
    p = max(1, int(round(n ** (1.0 / (M - 1)))))
    pts = []
    for combo in itertools.product(np.linspace(0, 1, p + 1), repeat=M - 1):
        if sum(combo) <= 1.0 + 1e-9:
            last = 1.0 - sum(combo)
            pts.append(list(combo) + [last])
    return np.asarray(pts, float)


def pf_mop6(n: int = 1000) -> np.ndarray:
    # f1+f2+f3 = 1, fi >= 0: the full simplex, the image of (x1,x2) in [0,1]^2
    return _simplex_grid(3, n)


def pf_mop7(n: int = 1000) -> np.ndarray:
    # f1^2+f2^2+f3^2 = 1, an octant of the sphere, the image of (x1,x2)
    p = max(2, int(round(np.sqrt(n))))
    a, b = np.meshgrid(np.linspace(0, 1, p), np.linspace(0, 1, p))
    a, b = a.ravel(), b.ravel()
    c1, c2 = np.cos(0.5 * np.pi * a), np.cos(0.5 * np.pi * b)
    s1, s2 = np.sin(0.5 * np.pi * a), np.sin(0.5 * np.pi * b)
    return np.column_stack([c1 * c2, c1 * s2, s1])


# ---- registry ------------------------------------------------------
# name -> (eval_fn, M, pf_sampler)
MOP_SPECS = {
    "MOP1": (mop1, 2, pf_mop1),
    "MOP2": (mop2, 2, pf_mop2),
    "MOP3": (mop3, 2, pf_mop3),
    "MOP4": (mop4, 2, pf_mop4),
    "MOP5": (mop5, 2, pf_mop5),
    "MOP6": (mop6, 3, pf_mop6),
    "MOP7": (mop7, 3, pf_mop7),
}


def mop_nadir(M: int) -> tuple:
    return tuple([1.0] * M)


def mop_ideal(M: int) -> tuple:
    return tuple([0.0] * M)

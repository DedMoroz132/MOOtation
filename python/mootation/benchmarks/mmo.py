# SPDX-License-Identifier: Apache-2.0
"""Multimodal multiobjective problems: SYM-PART, Omni-test and the MMF suite
of the CEC 2019 competition (task 7 of 2026-10-10, supplement 3, §5.4).

J. J. Liang, B. Y. Qu, D. W. Gong, C. T. Yue, "Problem definitions and
evaluation criteria for the CEC 2019 special session on multimodal
multiobjective optimization", technical report, Zhengzhou University, 2019
(source cec2019-mmo-problems_liang2018_report, §1.3, read off the rendered
pages), which collects SYM-PART simple and rotated (Rudolph, Naujoks and
Preuss, EMO 2007), Omni-test (Deb and Tiwari, EJOR 185(3), 2008) and MMF1-15
(Yue, Qu and Liang, IEEE TEVC 22(5), 2018, and new ones), at the parameters
it gives: SYM-PART a = 1, b = 10, c = 8 (rotated by pi/4); Omni-test n = 3;
MMF1_z k = 3; MMF1_e a = e; MMF9-13 n_p = 2; MMF12 q = 4; MMF14, MMF14_a,
MMF15, MMF15_a three objectives in three variables, n_p = 2.

THE PARETO SETS are derived here from the problems' formulas, every global
part, and checked (the tests: on the front to 1e-9; for the two-variable
problems, against the non-dominated points of a fine grid). Where the report's
own description of a set or a front disagrees with its formulas, the formulas
win:
  SYM-PART rotated  the report writes p = x - t (c + 2a, b) here too, with t
          read off the rotated r; its Fig. 3 is the simple problem's sets
          turned by pi/4, which is p = r - t (c + 2a, b), Rudolph et al.'s
          rotation of the whole problem: that is the one here;
  MMF1_e  the set on x1 in [2, 3] is x2 = a^x1 sin(6 pi |x1 - 2| + pi) (the
          report writes 2 pi there; its objective and its Fig. 7 have 6 pi);
  MMF2, MMF3  the second branch carries -cos where the first carries -2 cos,
          so its least value is 2 above the first's: a local set (the
          report's Table I marks both problems as having local sets, its text
          and figures draw both as global); one global part;
  MMF5, MMF6  the search space is x1 in [1, 3] and x2 in [-1, 3] / [-1, 2]
          (the report prints the bounds swapped; its figures and formulas
          agree with these);
  MMF6    its second branch is optimal only where sin(...) + 1 > 1, so that
          part covers the front in pieces;
  MMF8    the front is f2 = sqrt(1 - f1^2) (the report prints 1 - sqrt f1;
          its Fig. 14 is the quarter circle);
  MMF10-13, MMF15, MMF15_a  the least value of g is where g
          takes it, found here numerically, not at the printed 0.2 or
          1 / (2 n_p): the second factor shifts it (MMF10 by 1.2e-5);
  MMF13   t = x2 + sqrt(x3) is at least 0.1 + sqrt(0.1) in the box, so the
          report's global set t = 1/(2 n_p) = 0.25 lies outside it; the
          global set is where g is least within the box (t near 0.75);
  MMF14, 15  the report's examples say m = 2 and draw three objectives; three.
The local sets are not sampled (pareto_set is the global optimum, every part),
nor are global pieces of no area that map onto a curve of the front only:
MMF6's second branch where it is optimal (above), and MMF14_a's g = 1 on the
segments x2 = 1/2, x3 = 0 and x2 in {0, 1}, x3 = 1 of the box's faces, where
x3 - sin(pi x2) / 2 is -1/2 or 1.

Omni-test and SYM-PART simple: values checked against pymoo 0.6.2
(Apache-2.0) at five points each, reference in
tests/data/pymoo-0.6.2_omni-sympart.json.

parts(name): how many parts of the global set each map onto the whole front
(MMF6's pieces of a third and fourth not counted: PARTS_NOTE).

THE C++ SUITE (task 17, supplement 3, stage 3a): with the extension built the
problems and the Pareto sets are computed by include/mootation/suite/mmo.hpp
(benchmarks/native.py); this module stays the reference (BenchProblem.reference)
and what runs without it.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

NAMES = ("SYM-PART_simple", "SYM-PART_rotated", "Omni-test", "MMF1", "MMF1_z", "MMF1_e",
         "MMF2", "MMF3", "MMF4", "MMF5", "MMF6", "MMF7", "MMF8", "MMF9", "MMF10", "MMF11",
         "MMF12", "MMF13", "MMF14", "MMF14_a", "MMF15", "MMF15_a")

A, B, C = 1.0, 10.0, 8.0                        # SYM-PART
OMEGA = 0.25 * math.pi
OMNI_N = 3
K_Z = 3.0                                       # MMF1_z
A_E = math.e                                    # MMF1_e
NP = 2                                          # MMF9-15
Q = 4                                           # MMF12

BOUNDS = {
    "SYM-PART_simple": [(-20.0, 20.0)] * 2,
    "SYM-PART_rotated": [(-20.0, 20.0)] * 2,
    "Omni-test": [(0.0, 6.0)] * OMNI_N,
    "MMF1": [(1.0, 3.0), (-1.0, 1.0)],
    "MMF1_z": [(1.0, 3.0), (-1.0, 1.0)],
    "MMF1_e": [(1.0, 3.0), (-A_E ** 3, A_E ** 3)],
    "MMF2": [(0.0, 1.0), (0.0, 2.0)],
    "MMF3": [(0.0, 1.0), (0.0, 1.5)],
    "MMF4": [(-1.0, 1.0), (0.0, 2.0)],
    "MMF5": [(1.0, 3.0), (-1.0, 3.0)],
    "MMF6": [(1.0, 3.0), (-1.0, 2.0)],
    "MMF7": [(1.0, 3.0), (-1.0, 1.0)],
    "MMF8": [(-math.pi, math.pi), (0.0, 9.0)],
    "MMF9": [(0.1, 1.1)] * 2,
    "MMF10": [(0.1, 1.1)] * 2,
    "MMF11": [(0.1, 1.1)] * 2,
    "MMF12": [(0.0, 1.0)] * 2,
    "MMF13": [(0.1, 1.1)] * 3,
    "MMF14": [(0.0, 1.0)] * 3,
    "MMF14_a": [(0.0, 1.0)] * 3,
    "MMF15": [(0.0, 1.0)] * 3,
    "MMF15_a": [(0.0, 1.0)] * 3,
}


def n_obj(name: str) -> int:
    return 3 if name.startswith(("MMF14", "MMF15")) else 2


# ── the distance-like functions ──────────────────────────────────────────────

def _g9(x):
    return 2.0 - np.sin(NP * np.pi * x) ** 6


def _g10(x):
    return 2.0 - np.exp(-((x - 0.2) / 0.004) ** 2) - 0.8 * np.exp(-((x - 0.6) / 0.4) ** 2)


def _g11(x):                                    # MMF11, 12 and 13's g
    return 2.0 - np.exp(-2.0 * math.log(2.0) * ((x - 0.1) / 0.8) ** 2) * np.sin(NP * np.pi * x) ** 6


def _g14(x):
    return 2.0 - np.sin(NP * np.pi * x) ** 2


def _g15(x):
    return 2.0 - np.exp(-2.0 * math.log(2.0) * ((x - 0.1) / 0.8) ** 2) * np.sin(NP * np.pi * x) ** 2


@lru_cache(maxsize=None)
def _argmin(fn_name: str, lo: float, hi: float) -> float:
    """Where g is least on [lo, hi]: a grid of 200 001 points, then golden
    section around the best to the last bit."""
    fn = {"g9": _g9, "g10": _g10, "g11": _g11, "g14": _g14, "g15": _g15}[fn_name]
    t = np.linspace(lo, hi, 200001)
    k = int(np.argmin(fn(t)))
    a, b = t[max(k - 1, 0)], t[min(k + 1, len(t) - 1)]
    r = (math.sqrt(5.0) - 1.0) / 2.0
    for _ in range(200):
        c, d = b - r * (b - a), a + r * (b - a)
        if fn(c) < fn(d):
            b = d
        else:
            a = c
        if b - a < 1e-16:
            break
    return 0.5 * (a + b)


# ── SYM-PART ────────────────────────────────────────────────────────────────

def _sym_t(v: float, half: float, period: float) -> float:
    """t_i = sgn(t^) min(|t^|, 1), t^ = sgn(v) ceil((|v| - half) / period)."""
    th = math.copysign(1.0, v) * math.ceil((abs(v) - half) / period) if v != 0.0 else 0.0
    return math.copysign(min(abs(th), 1.0), th) if th != 0.0 else 0.0


def _sym_part(r1: float, r2: float) -> list:
    t1 = _sym_t(r1, A + C / 2.0, 2.0 * A + C)
    t2 = _sym_t(r2, B / 2.0, B)
    p1, p2 = r1 - t1 * (C + 2.0 * A), r2 - t2 * B
    return [(p1 + A) ** 2 + p2 ** 2, (p1 - A) ** 2 + p2 ** 2]


def evaluate(name: str, x) -> list:
    x = [float(v) for v in x]
    if name == "SYM-PART_simple":
        return _sym_part(x[0], x[1])
    if name == "SYM-PART_rotated":
        c, s = math.cos(OMEGA), math.sin(OMEGA)
        return _sym_part(c * x[0] - s * x[1], s * x[0] + c * x[1])
    if name == "Omni-test":
        # term by term from 0 (Python's sum of floats is compensated from 3.12)
        s = c = 0.0
        for v in x:
            s = s + math.sin(math.pi * v)
        for v in x:
            c = c + math.cos(math.pi * v)
        return [s, c]
    if name in ("MMF1", "MMF1_z", "MMF1_e", "MMF5", "MMF6", "MMF7"):
        d = abs(x[0] - 2.0)
        if name == "MMF7":
            s = (0.3 * d * d * math.cos(24.0 * math.pi * d + 4.0 * math.pi) + 0.6 * d) \
                * math.sin(6.0 * math.pi * d + math.pi)
            return [d, 1.0 - math.sqrt(d) + (x[1] - s) ** 2]
        s = math.sin(6.0 * math.pi * d + math.pi)
        if name == "MMF1_z" and x[0] < 2.0:
            s = math.sin(2.0 * K_Z * math.pi * d + math.pi)
        elif name == "MMF1_z":
            s = math.sin(2.0 * math.pi * d + math.pi)
        elif name == "MMF1_e" and x[0] >= 2.0:
            s = A_E ** x[0] * s
        y = x[1]
        if name == "MMF5" and y > 1.0:
            y -= 2.0
        elif name == "MMF6" and y > 1.0:
            y -= 1.0
        return [d, 1.0 - math.sqrt(d) + 2.0 * (y - s) ** 2]
    if name in ("MMF2", "MMF3"):
        x1, x2 = x
        if name == "MMF2":
            first = x2 <= 1.0
            shift = 1.0
        else:
            first = x2 <= 0.5 or (0.5 < x2 < 1.0 and 0.25 < x1)
            shift = 0.5
        u = x2 - math.sqrt(x1) - (0.0 if first else shift)
        ccoef = 2.0 if first else 1.0
        return [x1, 1.0 - math.sqrt(x1)
                + 2.0 * (4.0 * u * u - ccoef * math.cos(20.0 * u * math.pi / math.sqrt(2.0)) + 2.0)]
    if name == "MMF4":
        x1, x2 = x
        y = x2 - 1.0 if x2 >= 1.0 else x2
        return [abs(x1), 1.0 - x1 * x1 + 2.0 * (y - math.sin(math.pi * abs(x1))) ** 2]
    if name == "MMF8":
        x1, x2 = x
        sa = math.sin(abs(x1))
        y = x2 - 4.0 if x2 > 4.0 else x2
        return [sa, math.sqrt(1.0 - sa * sa) + 2.0 * (y - sa - abs(x1)) ** 2]
    if name in ("MMF9", "MMF10", "MMF11"):
        g = {"MMF9": _g9, "MMF10": _g10, "MMF11": _g11}[name](x[1])
        return [x[0], float(g) / x[0]]
    if name == "MMF12":
        g = float(_g11(x[1]))
        f1 = x[0]
        h = 1.0 - (f1 / g) ** 2 - (f1 / g) * math.sin(2.0 * math.pi * Q * f1)
        return [f1, g * h]
    if name == "MMF13":
        t = x[1] + math.sqrt(x[2])
        return [x[0], float(_g11(t)) / x[0]]
    if name in ("MMF14", "MMF14_a", "MMF15", "MMF15_a"):
        x1, x2, x3 = x
        if name == "MMF14":
            g = float(_g14(x3))
        elif name == "MMF14_a":
            g = float(_g14(x3 - 0.5 * math.sin(math.pi * x2) + 1.0 / (2 * NP)))
        elif name == "MMF15":
            g = float(_g15(x3))
        else:
            g = float(_g15(x3 - 0.5 * math.sin(math.pi * x2) + 1.0 / (2 * NP)))
        a, b = 0.5 * math.pi * x1, 0.5 * math.pi * x2
        return [math.cos(a) * math.cos(b) * (1.0 + g), math.cos(a) * math.sin(b) * (1.0 + g),
                math.sin(a) * (1.0 + g)]
    raise ValueError(f"unknown multimodal problem {name!r}")


# ── the global Pareto sets ───────────────────────────────────────────────────

def g_star(name: str) -> float:
    """The least value of the problem's g in the box (the front's g)."""
    if name == "MMF9":
        return 1.0
    if name == "MMF10":
        return float(_g10(_argmin("g10", 0.1, 1.1)))
    if name in ("MMF11", "MMF12"):
        return float(_g11(_argmin("g11", 0.1 if name == "MMF11" else 0.0, 1.1 if name == "MMF11" else 1.0)))
    if name == "MMF13":
        return float(_g11(_argmin("g11", 0.1 + math.sqrt(0.1), 1.1 + math.sqrt(1.1))))
    if name in ("MMF14", "MMF14_a"):
        return 1.0
    if name in ("MMF15", "MMF15_a"):
        return float(_g15(_argmin("g15", 0.0, 1.0)))
    raise ValueError(name)


def _even(n: int, k: int) -> list:
    """n points split as evenly as can be over k parts."""
    return [n // k + (1 if i < n % k else 0) for i in range(k)]


def _mmf12_pieces(g: float, n: int = 200001) -> np.ndarray:
    """The f1 of the non-dominated part of f2 = g h(f1, g), f1 in [0, 1]."""
    f1 = np.linspace(0.0, 1.0, n)
    f2 = g * (1.0 - (f1 / g) ** 2 - (f1 / g) * np.sin(2.0 * np.pi * Q * f1))
    best = np.minimum.accumulate(f2)
    keep = np.concatenate([[True], f2[1:] < best[:-1]])
    return f1[keep]


def pareto_set(name: str, n: int) -> np.ndarray:
    """About n points of the global Pareto set, every part."""
    if name in ("SYM-PART_simple", "SYM-PART_rotated"):
        rows = []
        for (t1, t2), m in zip([(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1)], _even(n, 9)):
            p1 = np.linspace(-A, A, max(m, 2))
            r1, r2 = t1 * (C + 2.0 * A) + p1, np.full_like(p1, t2 * B)
            if name == "SYM-PART_rotated":               # x = R^T r
                c, s = math.cos(OMEGA), math.sin(OMEGA)
                r1, r2 = c * r1 + s * r2, -s * r1 + c * r2
            rows.append(np.column_stack([r1, r2]))
        return np.vstack(rows)
    if name == "Omni-test":
        ms = [(a, b, c) for a in range(3) for b in range(3) for c in range(3)]
        rows = []
        for m, k in zip(ms, _even(n, len(ms))):
            u = np.linspace(1.0, 1.5, max(k, 2))
            rows.append(np.column_stack([2.0 * mi + u for mi in m]))
        return np.vstack(rows)
    if name in ("MMF1", "MMF1_z", "MMF1_e", "MMF7"):
        x1 = np.linspace(1.0, 3.0, max(n, 3))
        d = np.abs(x1 - 2.0)
        if name == "MMF7":
            x2 = (0.3 * d * d * np.cos(24.0 * np.pi * d + 4.0 * np.pi) + 0.6 * d) \
                * np.sin(6.0 * np.pi * d + np.pi)
        else:
            x2 = np.sin(6.0 * np.pi * d + np.pi)
            if name == "MMF1_z":
                x2 = np.where(x1 < 2.0, np.sin(2.0 * K_Z * np.pi * d + np.pi), np.sin(2.0 * np.pi * d + np.pi))
            elif name == "MMF1_e":
                x2 = np.where(x1 >= 2.0, A_E ** x1 * x2, x2)
        return np.column_stack([x1, x2])
    if name in ("MMF2", "MMF3"):
        x2 = np.linspace(0.0, 1.0, max(n, 2))
        if name == "MMF3":
            x2 = x2[x2 < 1.0]                           # x2 = 1 is the second branch's
        return np.column_stack([x2 ** 2, x2])
    if name == "MMF4":
        rows = []
        for sign, up in ((1, 0), (-1, 0), (1, 1), (-1, 1)):
            a = np.linspace(0.0, 1.0, max(n // 4, 2))
            x2 = np.sin(np.pi * a) + up
            keep = x2 < 1.0 if up == 0 else x2 >= 1.0
            rows.append(np.column_stack([sign * a[keep], x2[keep]]))
        return np.vstack(rows)
    if name in ("MMF5", "MMF6"):
        x1 = np.linspace(1.0, 3.0, max(n // 2, 3))
        s = np.sin(6.0 * np.pi * np.abs(x1 - 2.0) + np.pi)
        up = s + (2.0 if name == "MMF5" else 1.0)
        keep = up > 1.0
        return np.vstack([np.column_stack([x1, s]), np.column_stack([x1[keep], up[keep]])])
    if name == "MMF8":
        x1 = np.linspace(-math.pi, math.pi, max(n // 2, 3))
        x2 = np.sin(np.abs(x1)) + np.abs(x1)
        keep = x2 + 4.0 > 4.0
        return np.vstack([np.column_stack([x1, x2]), np.column_stack([x1[keep], x2[keep] + 4.0])])
    if name in ("MMF9", "MMF10", "MMF11"):
        x1 = np.linspace(0.1, 1.1, max(n // (NP if name == "MMF9" else 1), 2))
        if name == "MMF9":
            return np.vstack([np.column_stack([x1, np.full_like(x1, (0.5 + i) / NP)]) for i in range(NP)])
        t = _argmin("g10" if name == "MMF10" else "g11", 0.1, 1.1)
        return np.column_stack([x1, np.full_like(x1, t)])
    if name == "MMF12":
        f1 = _mmf12_pieces(g_star(name))
        f1 = f1[np.linspace(0, len(f1) - 1, max(n, 2)).astype(int)]
        return np.column_stack([f1, np.full_like(f1, _argmin("g11", 0.0, 1.0))])
    if name == "MMF13":
        t = _argmin("g11", 0.1 + math.sqrt(0.1), 1.1 + math.sqrt(1.1))
        k = max(int(math.sqrt(n)), 2)
        top = min(1.1, (t - 0.1) ** 2)              # x2 = t - sqrt(x3) >= 0.1
        x1, x3 = (a.ravel() for a in np.meshgrid(np.linspace(0.1, 1.1, k), np.linspace(0.1, top, k)))
        return np.column_stack([x1, np.maximum(t - np.sqrt(x3), 0.1), x3])   # the last ulp
    if name in ("MMF14", "MMF14_a", "MMF15", "MMF15_a"):
        parts = NP if name in ("MMF14", "MMF14_a") else 1
        k = max(int(math.sqrt(n / parts)), 2)
        x1, x2 = (a.ravel() for a in np.meshgrid(np.linspace(0.0, 1.0, k), np.linspace(0.0, 1.0, k)))
        rows = []
        for i in range(parts):
            if name == "MMF14":
                x3 = np.full_like(x1, (0.5 + i) / NP)
            elif name == "MMF14_a":
                x3 = 0.5 * np.sin(np.pi * x2) + i / NP
            elif name == "MMF15":
                x3 = np.full_like(x1, _argmin("g15", 0.0, 1.0))
            else:                                       # t = x3 - 0.5 sin(pi x2) + 1/(2 n_p)
                x3 = 0.5 * np.sin(np.pi * x2) + _argmin("g15", 0.0, 1.0) - 1.0 / (2 * NP)
            keep = (x3 >= 0.0) & (x3 <= 1.0)
            rows.append(np.column_stack([x1[keep], x2[keep], x3[keep]]))
        return np.vstack(rows)
    raise ValueError(name)


def parts(name: str) -> int:
    """The parts of the global Pareto set, each mapping onto the whole front."""
    return {"SYM-PART_simple": 9, "SYM-PART_rotated": 9, "Omni-test": 27, "MMF1": 2, "MMF1_z": 2,
            "MMF1_e": 2, "MMF2": 1, "MMF3": 1, "MMF4": 4, "MMF5": 4, "MMF6": 2, "MMF7": 2,
            "MMF8": 8, "MMF9": 2, "MMF10": 1, "MMF11": 1, "MMF12": 1, "MMF13": 1, "MMF14": 2,
            "MMF14_a": 2, "MMF15": 1, "MMF15_a": 1}[name]


PARTS_NOTE = {"MMF6": "and pieces of a third and fourth that cover the front in part"}


# ── the fronts ───────────────────────────────────────────────────────────────

def front(name: str, n: int) -> np.ndarray:
    """About n points of the global Pareto front."""
    if name in ("SYM-PART_simple", "SYM-PART_rotated"):
        v = np.linspace(0.0, 1.0, max(n, 2))
        return np.column_stack([4 * A * A * v * v, 4 * A * A * (1 - v) ** 2])
    if name == "Omni-test":
        th = np.linspace(math.pi, 1.5 * math.pi, max(n, 2))
        return OMNI_N * np.column_stack([np.sin(th), np.cos(th)])
    if name in ("MMF1", "MMF1_z", "MMF1_e", "MMF2", "MMF3", "MMF5", "MMF6", "MMF7"):
        f1 = np.linspace(0.0, 1.0, max(n, 2))
        return np.column_stack([f1, 1.0 - np.sqrt(f1)])
    if name == "MMF4":
        f1 = np.linspace(0.0, 1.0, max(n, 2))
        return np.column_stack([f1, 1.0 - f1 * f1])
    if name == "MMF8":
        f1 = np.linspace(0.0, 1.0, max(n, 2))
        return np.column_stack([f1, np.sqrt(1.0 - f1 * f1)])
    if name in ("MMF9", "MMF10", "MMF11", "MMF13"):
        f1 = np.linspace(0.1, 1.1, max(n, 2))
        return np.column_stack([f1, g_star(name) / f1])
    if name == "MMF12":
        g = g_star(name)
        f1 = _mmf12_pieces(g)
        f1 = f1[np.linspace(0, len(f1) - 1, max(n, 2)).astype(int)]
        return np.column_stack([f1, g * (1.0 - (f1 / g) ** 2 - (f1 / g) * np.sin(2.0 * np.pi * Q * f1))])
    if name in ("MMF14", "MMF14_a", "MMF15", "MMF15_a"):
        k = max(int(math.sqrt(n)), 2)
        a, b = (v.ravel() for v in np.meshgrid(np.linspace(0, 0.5 * math.pi, k),
                                                np.linspace(0, 0.5 * math.pi, k)))
        r = 1.0 + g_star(name)
        return r * np.column_stack([np.cos(a) * np.cos(b), np.cos(a) * np.sin(b), np.sin(a)])
    raise ValueError(name)


def frame(name: str) -> tuple:
    """(ideal, nadir): the front's corners, exact (front() takes its ends)."""
    F = front(name, 2001 if n_obj(name) == 2 else 2601)
    F = np.where(np.abs(F) < 1e-12, 0.0, F)        # cos(pi / 2)
    return tuple(float(v) for v in F.min(0)), tuple(float(v) for v in F.max(0))


def front_gap(name: str, F) -> np.ndarray:
    """How far each objective vector is from the front's equation, row by row
    (0 on the front): the tests' check of the Pareto sets."""
    F = np.atleast_2d(np.asarray(F, float))
    f1, f2 = F[:, 0], F[:, 1]
    if name in ("SYM-PART_simple", "SYM-PART_rotated"):
        return np.abs(np.sqrt(f1) + np.sqrt(f2) - 2.0 * A)
    if name == "Omni-test":
        return np.abs(np.hypot(f1, f2) - OMNI_N) + np.maximum(F.max(1), 0.0)
    if name in ("MMF1", "MMF1_z", "MMF1_e", "MMF2", "MMF3", "MMF5", "MMF6", "MMF7"):
        return np.abs(f2 - (1.0 - np.sqrt(f1)))
    if name == "MMF4":
        return np.abs(f2 - (1.0 - f1 * f1))
    if name == "MMF8":
        return np.abs(f2 - np.sqrt(1.0 - f1 * f1))
    if name in ("MMF9", "MMF10", "MMF11", "MMF13"):
        return np.abs(f1 * f2 - g_star(name))
    if name == "MMF12":
        g = g_star(name)
        return np.abs(f2 - g * (1.0 - (f1 / g) ** 2 - (f1 / g) * np.sin(2.0 * np.pi * Q * f1)))
    return np.abs(np.sqrt((F ** 2).sum(1)) - (1.0 + g_star(name)))

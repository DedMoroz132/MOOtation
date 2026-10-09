# SPDX-License-Identifier: Apache-2.0
"""Uninformative objectives: a probe of an algorithm's structural bias.

Every objective value is drawn at random and does not depend on x at all, so
no region of the decision space is better than another: where an algorithm's
points end up is a property of the ALGORITHM — of its operators, its repair at
the bounds, its selection — not of the problem. A population piled up at the
bounds or in the centre of the box is a bias the algorithm will carry onto
every problem. (Task 2 of 2026-09-23, A3, and task 6 of 2026-10-09, item 11:
Kudela, van Stein, Bäck & Kononova, "Structural Bias in Multi-objective
Optimisation", GECCO 2026, arXiv 2602.06742; the analysis is `--bias` on the
campaign command line.)

The problems, M = 2, x in [0, 1]^n for n = 2 and 10, after Kudela et al.'s
Table 1 (variant beta where there are three), a = U(0, 1) and z = N(0, 1)
fresh at every evaluation:
  f1   uninformative_n02_2D      [U, U]: the Pareto set is the point (0, 0)
  f2b  uninformative_f2b_n02_2D  [a, 1 - sin(12 pi a^1.5)/4 - sqrt(a/2) + 0.08 z^2]:
                                 a disconnected front
  f3b  uninformative_f3b_n02_2D  [a, 0.1/(10 (a - b/10) + 1)^10 + s_{b+1} + 0.9 z^2],
                                 b = #{i : a > p_i}, p = (0.1, .., 0.9), s = (1, 0.9,
                                 .., 0.1): convex stairs
  f4b  uninformative_f4b_n02_2D  [a, c + 2 z^2], c = -2a^2 below a = 1/2 and
                                 -2(a - 1/2)^2 - 1/2 above: two concave regions
  f5   uninformative_f5_n02_2D   [a, -a]: every point non-dominated
(and _n10_ for n = 10). f4's first objective, g1 = a, is printed in the
paper's Table 1 in its PDF; the task notes the HTML version omits it.

The values are deterministic from (the run's seed, the evaluation's number):
two runs with the same seed see the same stream, whatever they ask for. That
needs a counter, so the problem hands a campaign a fresh evaluator per run
(`BenchProblem.make_evaluator(seed)`). The plain `evaluate`, for a caller
without a seed, draws from x instead — the same x gets the same values —
which is just as uninformative. f1's stream is the one it always had; the
others draw from a domain of their own (`draw`), z by Box-Muller from two of
its uniforms.

WHAT A RUN LEAVES BEHIND. The evaluator counts, over every point the run
evaluated (Kudela's X_A) and over those non-dominated over the whole run
(their X_P), the coordinates in 20 equal bins of [0, 1] and those exactly on a
bound or exactly at 1/2 (`summary()`, which the campaign writes into
meta.json). Counters, not points: f5 makes every one of 30 000 evaluations
non-dominated.
"""

from __future__ import annotations

import bisect
import math
import struct
from typing import List

M = 2
SIZES = (2, 10)
PROBLEMS = ("f1", "f2b", "f3b", "f4b", "f5")
R = {"f2b": 0.08, "f3b": 0.9, "f4b": 2.0}          # Kudela et al., Table 1, variant beta
BINS = 20
_MASK = (1 << 64) - 1


def _splitmix64(z: int) -> int:
    z = (z + 0x9E3779B97F4A7C15) & _MASK
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK
    return z ^ (z >> 31)


def _unit(h: int) -> float:
    """53 high bits of a 64-bit hash as a double in [0, 1)."""
    return (h >> 11) * (1.0 / (1 << 53))


def value(seed: int, k: int, i: int) -> float:
    """Objective i of evaluation k in the f1 run with this seed: U(0, 1)."""
    h = _splitmix64((seed & _MASK) ^ _splitmix64(k * M + i))
    return _unit(h)


def draw(seed: int, k: int, i: int) -> float:
    """Uniform i (0..3) of evaluation k for f2b-f5: U(0, 1), a domain apart
    from f1's."""
    h = _splitmix64((seed & _MASK) ^ _splitmix64((1 << 62) + 4 * k + i))
    return _unit(h)


def objectives(problem: str, a: float, z: float) -> List[float]:
    """Kudela et al.'s [g1, g2] for a = U(0, 1) and z = N(0, 1)."""
    if problem == "f5":
        return [a, -a]
    if problem == "f2b":
        g2 = 1.0 - math.sin(12.0 * math.pi * a ** 1.5) / 4.0 - math.sqrt(a / 2.0)
    elif problem == "f3b":
        b = sum(1 for i in range(1, 10) if a > i / 10.0)
        g2 = 0.1 / (10.0 * (a - b / 10.0) + 1.0) ** 10 + (1.0 - b / 10.0)
    elif problem == "f4b":
        g2 = -2.0 * a * a if a < 0.5 else -2.0 * (a - 0.5) ** 2 - 0.5
    else:
        raise ValueError(f"unknown problem '{problem}'")
    return [a, g2 + R[problem] * z * z]


class Evaluator:
    """One run's objective function: the k-th call returns the k-th draw of
    the stream, whatever x is; it counts where the evaluated points lie."""

    def __init__(self, seed: int, problem: str = "f1"):
        self.seed = int(seed)
        self.problem = problem
        self.k = 0
        self.all = _Counts()
        self.front_f1: list = []          # the non-dominated points so far, by f1 ascending
        self.front_f2: list = []          # (f2 descending along it)
        self.front_x: list = []

    def __call__(self, x) -> List[float]:
        if self.problem == "f1":
            f = [value(self.seed, self.k, i) for i in range(M)]
        else:
            a = draw(self.seed, self.k, 0)
            u1, u2 = draw(self.seed, self.k, 1), draw(self.seed, self.k, 2)
            z = math.sqrt(-2.0 * math.log(1.0 - u1)) * math.cos(2.0 * math.pi * u2)
            f = objectives(self.problem, a, z)
        self.k += 1
        x = [float(v) for v in x]
        self.all.add(x)
        self._front(f, x)
        return f

    def _front(self, f, x) -> None:
        """Keep the non-dominated points of the run (two objectives: sorted by
        f1, f2 falls along the list)."""
        i = bisect.bisect_left(self.front_f1, f[0])
        if i > 0 and self.front_f2[i - 1] <= f[1]:
            return                                            # dominated by a smaller f1
        if i < len(self.front_f1) and self.front_f1[i] == f[0] and self.front_f2[i] <= f[1]:
            return
        j = i
        while j < len(self.front_f1) and self.front_f2[j] >= f[1]:
            j += 1                                            # these it dominates
        self.front_f1[i:j] = [f[0]]
        self.front_f2[i:j] = [f[1]]
        self.front_x[i:j] = [x]

    def summary(self) -> dict:
        """The counts over every evaluated point and over the run's
        non-dominated ones (module docstring)."""
        nd = _Counts()
        for x in self.front_x:
            nd.add(x)
        return {"bins": BINS, "all": self.all.as_dict(), "nondominated": nd.as_dict(),
                "evaluations": self.k}


class _Counts:
    """Coordinates of [0, 1]^n counted in BINS bins, and exactly on 0, 1, 1/2."""

    def __init__(self):
        self.hist = [0] * BINS
        self.points = self.lower = self.upper = self.half = 0

    def add(self, x) -> None:
        self.points += 1
        for v in x:
            self.hist[min(BINS - 1, max(0, int(v * BINS)))] += 1
            self.lower += v == 0.0
            self.upper += v == 1.0
            self.half += v == 0.5

    def as_dict(self) -> dict:
        return {"points": self.points, "hist": self.hist, "at_lower": self.lower,
                "at_upper": self.upper, "at_half": self.half}


def evaluate_by_x(x, problem: str = "f1") -> List[float]:
    """Without a seed: values drawn from x itself, the same for the same x."""
    h = 0
    for v in x:
        h = _splitmix64(h ^ struct.unpack("<Q", struct.pack("<d", float(v)))[0])
    if problem == "f1":
        return [_unit(_splitmix64(h + i + 1)) for i in range(M)]
    a, u1, u2 = (_unit(_splitmix64(h + i + 1)) for i in range(3))
    return objectives(problem, a, math.sqrt(-2.0 * math.log(1.0 - u1))
                      * math.cos(2.0 * math.pi * u2))


def name(n: int, problem: str = "f1") -> str:
    middle = "" if problem == "f1" else f"{problem}_"
    return f"uninformative_{middle}n{n:02d}_{M}D"

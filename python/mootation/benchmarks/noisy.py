# SPDX-License-Identifier: Apache-2.0
"""Noisy copies of benchmark problems: a control written for this library, not
a published suite (task 6 of 2026-10-09, item 7; task 7 of 2026-10-10, §4).
They stand in for the noise of a mesh and a solver behind a real objective.

THE FOUR KINDS. The copy of a problem returns f~_i = f_i + s_i z_i, z_i a
standard normal, independently over the objectives; its name is the base's
with the kind and the level after it, sigma = 0.01 for 1 and 0.05 for 5:

  <base>_n1, _n5  stochastic: s_i = sigma (nadir_i - ideal_i), z new at every
                  evaluation, from (the run's seed, the evaluation's number) —
                  repeats average it away;
  <base>_d1, _d5  deterministic: the same s_i, z from a hash of x, the same in
                  every run and at every repeat — the "wrong but repeatable"
                  value of a solver's tolerance, which repeats do not cure;
  <base>_s1, _s5  a step: the same s_i, z from a hash of the cell
                  floor((x - lb) / h), h = CELL (ub - lb) per variable, CELL =
                  0.01 by default ([campaign] noise_cell): constant inside a
                  cell, a jump at its border — a remeshing;
  <base>_h1, _h5  heteroscedastic: s_i = sigma (f_i(x) - ideal_i), a relative
                  error, as of a solver with a relative tolerance; stochastic,
                  as _n.

ideal and nadir are the base problem's frame. Front, frame, Pareto set and
distance function are the base's.

THE BASES. ZDT1, ZDT4, DTLZ1_3D, DTLZ2_3D and WFG7_3D (task 6); DTLZ2_5D,
WFG4_3D, rcDTLZ2_3D and three real-world problems whose objectives differ in
scale by orders of magnitude (task 7, §4): RE21 (about 10^3 and 10^-2), RE25
(10^-1 and 10^6) and RE34 (10^3, 10 and 10^-1). Every kind at both levels of
every base.

THE STREAMS. As with the uninformative problems (uninformative.py), z of _n and
_h is deterministic from (the run's seed, the evaluation's number): a campaign
hands each run a fresh evaluator (`BenchProblem.make_evaluator(seed)`), and two
runs with the same seed see the same noise. z by Box-Muller from two uniforms
of the splitmix64 hash, in a domain of its own. The plain `evaluate`, for a
caller without a seed, draws _n's and _h's noise from x instead: the same x,
the same values. _d draws from x always, and _s from the cell, whatever the
seed: the hash of the doubles of x, or of the cell's integers, then one pair of
uniforms per objective.

WHAT IS MEASURED. The algorithm sees f~. A campaign measures every indicator,
in `final`, three ways side by side (task 7, §4):
  * recommended — the noise-free f of the answer the algorithm chose by the
    noisy values (BenchProblem.true_objectives, the base's evaluate, called
    again): the indicators without a suffix, as since task 6;
  * oracle (`<metric>_oracle`) — the non-dominated set by the noise-free f of
    every point evaluated, reduced to the answer's size by the archive's rule:
    what the run found, whether or not it recognised it;
  * observed (`<metric>_observed`) — the noisy values of the answer: biased
    towards better, recorded to show by how much.
true_dominated_share is the share of the answer that another of its points
dominates by the true values: the points the noise let through.
"""

from __future__ import annotations

import math
import struct
from typing import Callable, List

from .uninformative import _MASK, _splitmix64, _unit

BASES = ("ZDT1", "ZDT4", "DTLZ1_3D", "DTLZ2_3D", "WFG7_3D", "DTLZ2_5D", "WFG4_3D")
LATE_BASES = ("rcDTLZ2_3D", "RE21", "RE25", "RE34")   # registered after the reference version
# level -> (kind, sigma)
LEVELS = {"n1": ("n", 0.01), "n5": ("n", 0.05), "d1": ("d", 0.01), "d5": ("d", 0.05),
          "s1": ("s", 0.01), "s5": ("s", 0.05), "h1": ("h", 0.01), "h5": ("h", 0.05)}
SIGMA = {level: sigma for level, (_, sigma) in LEVELS.items()}
CELL = 0.01
_DOMAIN = 3 << 61                       # apart from uninformative.py's two
_DOMAIN_X = 5 << 60                     # the hashes of x and of cells


def name(base: str, level: str) -> str:
    return f"{base}_{level}"


def is_noisy(problem: str) -> bool:
    """Is this registry name a noisy copy (`<base>_n1`, `<base>_d5`, ...)?"""
    return "_" in problem and problem.rsplit("_", 1)[1] in LEVELS


def kind_of(problem: str) -> str | None:
    """n, d, s or h for a noisy copy, None for any other name."""
    return LEVELS[problem.rsplit("_", 1)[1]][0] if is_noisy(problem) else None


def base_of(problem: str) -> str:
    return problem.rsplit("_", 1)[0]


def normal(seed: int, k: int, i: int) -> float:
    """z of objective i at evaluation k of the run with this seed: N(0, 1)."""
    u1 = _unit(_splitmix64((seed & _MASK) ^ _splitmix64(_DOMAIN + 4 * k + 2 * i)))
    u2 = _unit(_splitmix64((seed & _MASK) ^ _splitmix64(_DOMAIN + 4 * k + 2 * i + 1)))
    return math.sqrt(-2.0 * math.log(1.0 - u1)) * math.cos(2.0 * math.pi * u2)


def _hash_x(x) -> int:
    """A hash of the doubles of x, bit for bit."""
    h = 0
    for v in x:
        h = _splitmix64(h ^ struct.unpack("<Q", struct.pack("<d", float(v)))[0])
    return h


def _hash_cell(x, lower, width) -> int:
    """A hash of the cell floor((x - lb) / h) that x lies in."""
    h = _DOMAIN_X
    for v, lo, w in zip(x, lower, width):
        h = _splitmix64(h ^ (math.floor((float(v) - lo) / w) & _MASK))
    return h


def normal_of(h: int, i: int) -> float:
    """z of objective i from a hash (of x or of a cell): N(0, 1)."""
    u1, u2 = _unit(_splitmix64(h + 2 * i + 1)), _unit(_splitmix64(h + 2 * i + 2))
    return math.sqrt(-2.0 * math.log(1.0 - u1)) * math.cos(2.0 * math.pi * u2)


class Evaluator:
    """One run's objective function. kind n: the base's values plus the k-th
    draw of the stream at the k-th call, `scale` sigma (nadir - ideal) per
    objective; kind h: the same stream, the scale sigma (f - ideal); kinds d
    and s: the draw of x or of its cell, whatever the seed. `last_true` keeps
    the noise-free values of the last call, for the oracle."""

    def __init__(self, base: Callable, scale, seed: int, kind: str = "n", *, ideal=None,
                 sigma: float = 0.0, lower=None, upper=None, cell: float = CELL):
        self.base = base
        self.scale = [float(s) for s in scale]
        self.seed = int(seed)
        self.kind = kind
        self.ideal = None if ideal is None else [float(v) for v in ideal]
        self.sigma = float(sigma)
        if kind == "s":
            self.lower = [float(v) for v in lower]
            self.width = [cell * (float(b) - float(a)) for a, b in zip(lower, upper)]
        self.cell = cell
        self.k = 0
        self.last_true = None

    def __call__(self, x) -> List[float]:
        f = self.base(list(x))
        self.last_true = list(f)
        if self.kind in ("n", "h"):
            z = [normal(self.seed, self.k, i) for i in range(len(f))]
        else:
            h = _hash_x(x) if self.kind == "d" else _hash_cell(x, self.lower, self.width)
            z = [normal_of(h, i) for i in range(len(f))]
        if self.kind == "h":
            s = [self.sigma * (v - lo) for v, lo in zip(f, self.ideal)]
        else:
            s = self.scale
        self.k += 1
        return [v + si * zi for v, si, zi in zip(f, s, z)]


def evaluate_by_x(base: Callable, scale, x, kind: str = "n", *, ideal=None,
                  sigma: float = 0.0, lower=None, upper=None, cell: float = CELL) -> List[float]:
    """Without a seed: the noise drawn from x itself (_s: from its cell), the
    same for the same x — _n's as since task 6, bit for bit."""
    f = base(list(x))
    if kind == "s":
        width = [cell * (float(b) - float(a)) for a, b in zip(lower, upper)]
        h = _hash_cell(x, [float(v) for v in lower], width)
    else:
        h = _hash_x(x)
    if kind == "n":                     # task 6's arithmetic, (s sqrt) cos, bit for bit
        out = []
        for i, (v, s) in enumerate(zip(f, scale)):
            u1, u2 = _unit(_splitmix64(h + 2 * i + 1)), _unit(_splitmix64(h + 2 * i + 2))
            out.append(v + s * math.sqrt(-2.0 * math.log(1.0 - u1))
                       * math.cos(2.0 * math.pi * u2))
        return out
    s = [sigma * (v - lo) for v, lo in zip(f, ideal)] if kind == "h" else scale
    return [v + si * normal_of(h, i) for i, (v, si) in enumerate(zip(f, s))]

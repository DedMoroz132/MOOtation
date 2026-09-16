# SPDX-License-Identifier: Apache-2.0
# ============================================================================
# bbob-biobj: 55 bi-objective problems, each a pair of the ten bbob functions
# in bbob.py.
# D. Brockhoff, A. Auger, N. Hansen, T. Tusar. "Using Well-Understood
# Single-Objective Functions in Multiobjective Black-Box Optimization Test
# Suites." Evolutionary Computation 30(2):165-193, 2022.
#                                    (source: ReadyMarkdown/evco_a_00298)
#
# THE CONSTRUCTION (Section 5.1). Ten bbob functions are chosen, two from each
# of the five bbob groups, so no group is over-weighted: f1 and f2 (separable),
# f6 and f8 (low or moderate conditioning), f13 and f14 (high conditioning,
# unimodal), f15 and f17 (multimodal with global structure), f20 and f21
# (multimodal with weak global structure). Objective order is assumed not to
# matter, so F = (f_a, f_b) and (f_b, f_a) are the same problem and only one is
# kept, which leaves C(11,2) = 55 pairs INCLUDING the diagonal (f_a paired with
# another instance of itself).
#
# THE ORDER of F1..F55 is the row order of the paper's Figure 1, whose cells
# the markdown conversion left empty: F1..F10 pair f1 with each of the ten,
# F11..F19 pair f2 with each from f2 on, and so forth. That is not a guess. The
# paper names one group's members in the text — "the group of separable
# ill-conditioned functions (F5, F6, F14, F15)" — and under this order those
# four are exactly (f1,f13), (f1,f14), (f2,f13), (f2,f14), i.e. one separable
# with one ill-conditioned function, which is what the sentence says they are.
#
# NO REFERENCE FRONT, SO NO IGD. The Pareto set of a pair of bbob functions has
# no closed form; COCO itself estimates each instance's hypervolume from the
# accumulated results of many experiments. These problems therefore ship with
# pareto_front = None: hypervolume works, IGD and IGD+ correctly report nothing
# rather than a number measured against a front we invented.
#
# READINGS (resolved and declared):
#   BIOBJ-1. Section 2 defines the nadir as "in each objective the worst value
#     obtained by a Pareto-optimal solution", then prints, for the unique-optimum
#     case, z_nadir = (f_a(arg min f_a), f_b(arg min f_b)) — which is the IDEAL
#     point, not the nadir, so the formula contradicts its own sentence. The
#     sentence wins: the two extremes of the Pareto set are the two single
#     optima, and f_a is at its worst over that set at the OTHER objective's
#     optimum, so nadir = (f_a(x_b^opt), f_b(x_a^opt)).
#   BIOBJ-2. The instances are ours, inherited from bbob.py's BBOB-4: the
#     instance-to-parameter generator is COCO's code, not in any paper. The
#     instance NUMBERING below is the paper's (Section 5.1.3) and is applied to
#     our draws, so instance 1 here is not instance 1 in the COCO archives.
# ============================================================================
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from . import bbob

BASE: Tuple[int, ...] = bbob.BASE_IDS          # f1 f2 f6 f8 f13 f14 f15 f17 f20 f21

MIN_X_DISTANCE = 1e-4                          # Section 5.1.3, condition 1
MIN_F_DISTANCE = 1e-1                          # Section 5.1.3, condition 2
_MAX_BUMPS = 100


def pairs() -> Tuple[Tuple[int, int], ...]:
    """The 55 (f_alpha, f_beta) pairs in the paper's F1..F55 order."""
    out = []
    for i in range(len(BASE)):
        for j in range(i, len(BASE)):
            out.append((BASE[i], BASE[j]))
    return tuple(out)


PAIRS = pairs()


def single_instance_ids(k_bi: int) -> Tuple[int, int]:
    """Section 5.1.3: K_alpha = 2*K + 1 and K_beta = K_alpha + 1.

    With the stated historical exceptions for the bi-objective instances 1 and
    2, which carry the single-objective instances (2, 4) and (3, 5).
    """
    if k_bi == 1:
        return (2, 4)
    if k_bi == 2:
        return (3, 5)
    a = 2 * k_bi + 1
    return (a, a + 1)


@dataclass
class BiObjective:
    """One bbob-biobj problem: two bbob instances and the frame they imply."""
    fnum: int                                  # 1..55
    D: int
    iid: int
    alpha: bbob.Instance
    beta: bbob.Instance
    ideal: Tuple[float, float]
    nadir: Tuple[float, float]

    def __call__(self, x) -> List[float]:
        x = np.asarray(x, dtype=float)
        return [self.alpha(x), self.beta(x)]


def problem(fnum: int, D: int, iid: int = 1) -> BiObjective:
    """Build F`fnum` in `D` variables at bi-objective instance `iid`."""
    if not 1 <= fnum <= 55:
        raise ValueError(f"bbob-biobj has F1..F55, not F{fnum}")
    fa, fb = PAIRS[fnum - 1]
    ka, kb = single_instance_ids(iid)
    a = bbob.instance(fa, ka, D)
    # "If the two above conditions are not satisfied ... we increase the
    # instance ID of the second objective successively until both are."
    for _ in range(_MAX_BUMPS):
        b = bbob.instance(fb, kb, D)
        ideal = (a.f_opt, b.f_opt)
        nadir = (a(b.x_opt), b(a.x_opt))            # BIOBJ-1
        if (np.linalg.norm(a.x_opt - b.x_opt) >= MIN_X_DISTANCE
                and math.dist(ideal, nadir) >= MIN_F_DISTANCE):
            return BiObjective(fnum=fnum, D=D, iid=iid, alpha=a, beta=b,
                               ideal=ideal, nadir=nadir)
        kb += 1
    raise RuntimeError(
        f"F{fnum} at D={D}, instance {iid}: no second-objective instance in "
        f"{_MAX_BUMPS} tries separated the optima by {MIN_X_DISTANCE} and the "
        f"ideal from the nadir by {MIN_F_DISTANCE}")


def name(fnum: int, D: int) -> str:
    """The registry key.

    The suite is scalable in the NUMBER OF VARIABLES while always having two
    objectives, which the registry's `_<k>D` suffix does not mean — there it is
    the objective count, and the config's `objectives = [...]` filter matches on
    it. So the dimension goes in its own segment and the suffix stays the
    truthful `_2D`: bbobbiobj07_n10_2D is F7 in ten variables, two objectives.
    """
    return f"bbobbiobj{fnum:02d}_n{D:02d}_2D"


def bounds(D: int) -> List[Tuple[float, float]]:
    return bbob.bounds(D)


def spec(fnum: int, D: int, iid: int = 1) -> dict:
    """Everything the registry needs for one bbob-biobj problem.

    The problem is built ONCE here, not per evaluation: an instance carries two
    rotation matrices and, for f21, 101 peaks, and redrawing those per call
    would cost more than the function.
    """
    p = problem(fnum, D, iid)
    return {
        "n_vars": D,
        "bounds": bounds(D),
        "evaluate": p,
        "ideal": p.ideal,
        "nadir": p.nadir,
        "pareto_front": None,
        "problem": p,
    }


def describe(fnum: int) -> str:
    fa, fb = PAIRS[fnum - 1]
    return f"F{fnum} = (f{fa} {bbob.NAMES[fa]}, f{fb} {bbob.NAMES[fb]})"

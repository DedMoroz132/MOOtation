# SPDX-License-Identifier: Apache-2.0
"""Ablation baselines: variation without selection, and the smallest elitist EA.

  random_selection_ea
      NSGA-II's variation with none of its selection. Each generation draws N
      parent pairs uniformly at random, crosses them with SBX (eta_c = 20,
      p_c = 0.9) and mutates the children polynomially (eta_m = 20, p_m = 1/n)
      — the operators and settings of the library's NSGA-II (Deb et al. 2002,
      §IV-A: "The crossover probability of p_c = 0.9 and a mutation
      probability of p_m = 1/n", with distribution indexes "eta_c = 20 and
      eta_m = 20") — and keeps N of the 2N
      parents and children uniformly at random. What is left is what the
      operators do on their own: a population that drifts. Its answer is its
      last population, as an EA's is; the run archive (archive.csv) still
      holds the best it ever evaluated.
  gsemo
      Global SEMO — the simple evolutionary multi-objective optimizer (Laumanns,
      Thiele & Zitzler, IEEE TEVC 8(2), 2004) with the global mutation of Giel
      (CEC 2003) — carried to real variables. The population is every
      nondominated point found so far; each step mutates one member drawn
      uniformly at random, every variable with probability 1/n (the real-valued
      counterpart of standard bit mutation) by the same polynomial mutation;
      the child enters unless a member weakly dominates it and removes every
      member it weakly dominates. One evaluation a step. The population can
      grow past N, so the answer is it reduced to N by DSS, as for the
      sampling baselines. The carrying-over to real variables is this
      module's, not either paper's, which define the algorithm on bit strings.

The operators follow include/mootation/operators/sbx.hpp and
poly_mutation.hpp formula for formula (Deb & Agrawal's bounded SBX with the
NSGA-II code's per-variable coin and swap; the NSGA-II code's bounded
polynomial mutation), drawn from NumPy's generator rather than the core's.
"""

from __future__ import annotations

import numpy as np

ETA_C, PC, ETA_M = 20.0, 0.9, 20.0


def sbx(p1, p2, lo, hi, rng, eta_c: float = ETA_C, pc: float = PC):
    """Bounded SBX on one pair (sbx.hpp): two children."""
    c1, c2 = p1.copy(), p2.copy()
    if pc < 1.0 and rng.random() > pc:
        return c1, c2
    for j in range(len(p1)):
        if rng.random() > 0.5:
            continue
        if abs(p1[j] - p2[j]) < 1e-14:
            continue
        y1, y2 = min(p1[j], p2[j]), max(p1[j], p2[j])
        dy = y2 - y1
        u = rng.random()
        bl = 1.0 + 2.0 * (y1 - lo[j]) / dy
        al = 2.0 - bl ** -(eta_c + 1.0)
        bq = (u * al) ** (1.0 / (eta_c + 1.0)) if u <= 1.0 / al else \
            (1.0 / (2.0 - u * al)) ** (1.0 / (eta_c + 1.0))
        c1[j] = min(max(0.5 * ((y1 + y2) - bq * dy), lo[j]), hi[j])
        bh = 1.0 + 2.0 * (hi[j] - y2) / dy
        ah = 2.0 - bh ** -(eta_c + 1.0)
        bq = (u * ah) ** (1.0 / (eta_c + 1.0)) if u <= 1.0 / ah else \
            (1.0 / (2.0 - u * ah)) ** (1.0 / (eta_c + 1.0))
        c2[j] = min(max(0.5 * ((y1 + y2) + bq * dy), lo[j]), hi[j])
        if rng.random() < 0.5:
            c1[j], c2[j] = c2[j], c1[j]
    return c1, c2


def polynomial_mutation(x, lo, hi, rng, eta_m: float = ETA_M, pm: float | None = None):
    """Bounded polynomial mutation (poly_mutation.hpp), in place; returns x."""
    n = len(x)
    pm = 1.0 / n if pm is None else pm
    for j in range(n):
        if rng.random() > pm:
            continue
        dx = hi[j] - lo[j]
        if dx < 1e-14:
            continue
        u = rng.random()
        if u < 0.5:
            b = min(max((x[j] - lo[j]) / dx, 0.0), 1.0)
            tmp = 2.0 * u + (1.0 - 2.0 * u) * (1.0 - b) ** (eta_m + 1.0)
            dq = tmp ** (1.0 / (eta_m + 1.0)) - 1.0
        else:
            b = min(max((hi[j] - x[j]) / dx, 0.0), 1.0)
            tmp = 2.0 * (1.0 - u) + (2.0 * u - 1.0) * (1.0 - b) ** (eta_m + 1.0)
            dq = 1.0 - tmp ** (1.0 / (eta_m + 1.0))
        x[j] = min(max(x[j] + dq * dx, lo[j]), hi[j])
    return x


def random_selection_ea(evaluate, lo, hi, *, pop: int, max_evaluations: int, seed: int,
                        on_generation=None, record_every: int = 1):
    """(answer objectives, answer variables, steps) — see the module docstring."""
    rng = np.random.default_rng(seed)
    d = len(lo)
    X = lo + rng.random((pop, d)) * (hi - lo)
    k0 = min(pop, max_evaluations)
    X = X[:k0]
    F = np.array([evaluate(list(x)) for x in X], float)
    spent, gen = k0, 0
    if on_generation is not None:
        on_generation(0, F.tolist())
    while spent < max_evaluations:
        k = min(pop, max_evaluations - spent)
        kids = []
        while len(kids) < k:
            a, b = rng.integers(0, len(X), 2)
            c1, c2 = sbx(X[a], X[b], lo, hi, rng)
            kids.append(polynomial_mutation(c1, lo, hi, rng))
            if len(kids) < k:
                kids.append(polynomial_mutation(c2, lo, hi, rng))
        Xk = np.array(kids)
        Fk = np.array([evaluate(list(x)) for x in Xk], float)
        spent += k
        gen += 1
        Xa, Fa = np.vstack([X, Xk]), np.vstack([F, Fk])
        keep = rng.choice(len(Xa), size=min(pop, len(Xa)), replace=False)
        X, F = Xa[keep], Fa[keep]
        if on_generation is not None and (gen % max(1, record_every) == 0
                                          or spent >= max_evaluations):
            on_generation(gen, F.tolist())
    return F, X, gen


def gsemo(evaluate, lo, hi, *, pop: int, max_evaluations: int, seed: int, select,
          on_generation=None, record_every: int = 1):
    """(answer objectives, answer variables, steps); `select(F, k)` picks the answer rows."""
    rng = np.random.default_rng(seed)
    d = len(lo)
    x0 = lo + rng.random(d) * (hi - lo)
    f0 = np.array(evaluate(list(x0)), float)
    # the population in the first `size` rows of two arrays that double when full
    PX = np.empty((64, d))
    PF = np.empty((64, len(f0)))
    PX[0], PF[0], size = x0, f0, 1
    spent, step = 1, 0

    def answer():
        idx = select(PF[:size], pop)
        return PF[:size][idx], PX[:size][idx]

    if on_generation is not None:
        on_generation(0, answer()[0].tolist())
    while spent < max_evaluations:
        parent = PX[int(rng.integers(0, size))]
        y = polynomial_mutation(parent.copy(), lo, hi, rng)
        fy = np.array(evaluate(list(y)), float)
        spent += 1
        F = PF[:size]
        if not np.any(np.all(F <= fy, axis=1)):             # nothing weakly dominates y
            keep = ~np.all(fy <= F, axis=1)                  # drop what y weakly dominates
            k = int(keep.sum())
            PX[:k], PF[:k] = PX[:size][keep], PF[:size][keep]
            if k == len(PX):
                PX = np.vstack([PX, np.empty_like(PX)])
                PF = np.vstack([PF, np.empty_like(PF)])
            PX[k], PF[k], size = y, fy, k + 1
        # a "generation" is pop evaluations, so records line up with the others
        if spent % pop == 0 or spent >= max_evaluations:
            step += 1
            if on_generation is not None and (step % max(1, record_every) == 0
                                              or spent >= max_evaluations):
                on_generation(step, answer()[0].tolist())
    Fa, Xa = answer()
    return Fa, Xa, step

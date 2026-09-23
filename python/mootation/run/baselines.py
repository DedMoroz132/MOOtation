# SPDX-License-Identifier: Apache-2.0
"""Baselines: what an algorithm has to beat to have done anything at all.

Blind sampling of the box with the run archive (archive.GridArchive) behind
it: the null model of a comparison. Where an algorithm is not clearly ahead of
it, the problem is too easy or the budget too small, which says something about
the benchmark rather than the algorithm. With an archive, what random search
converges to is decided by the archiver, not by how the points are drawn
(Schütze et al. 2008), so the two are built together.

  random_search   uniform points of the box
  sobol_search    a scrambled Sobol sequence (scipy.stats.qmc, Owen-type
                  scrambling seeded per run): lower discrepancy than uniform,
                  which matters at budgets of tens to hundreds of points and
                  not after; needs SciPy

and two ablations (ablations.py), which take one ingredient of an EA away:

  random_selection_ea  NSGA-II's variation (SBX, polynomial mutation) with
                       uniform random survival: the operators without the
                       selection
  gsemo                global SEMO on real variables: the nondominated set,
                       one mutated member per evaluation: selection by
                       dominance alone, no diversity mechanism and no
                       population size

The ANSWER of a sampling baseline is not its last batch but its archive,
reduced to the problem's population size N by the same DSS selection used
everywhere else, so its metrics are comparable with a population's; GSEMO's is
its population reduced the same way, random_selection_ea's its last population.
Each spends exactly the budget: the last batch is cut to what is left. None of
them handles constraints: only the run archive keeps to feasible points.

These are not C++ cores and are not in algorithms.def: nothing about them
needs the core, and every interface that should run them — the campaign — is
Python. They are validated, listed and run alongside the core's list.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace

BASELINES = {
    "random_search": "uniform sampling of the box, archived, N chosen by DSS",
    "sobol_search": "scrambled Sobol sampling (SciPy), archived, N chosen by DSS",
    "random_selection_ea": "ablation: NSGA-II's SBX and polynomial mutation, survival "
                           "uniformly at random",
    "gsemo": "ablation: global SEMO on real variables, the nondominated set, N chosen by DSS",
}


def baseline_names() -> tuple:
    return tuple(BASELINES)


def run_baseline(name: str, evaluate, bounds, *, pop: int, max_evaluations: int,
                 seed: int, archive, on_generation=None, record_every: int = 1):
    """Sample until max_evaluations; returns a result shaped like minimize()'s.

    `evaluate(x)` must feed `archive` itself (the campaign's evaluator does),
    so that every algorithm's archive is filled by the same code.
    """
    import numpy as np

    if name not in BASELINES:
        raise ValueError(f"unknown baseline '{name}'; known: {', '.join(BASELINES)}")
    lo = np.array([b[0] for b in bounds], float)
    hi = np.array([b[1] for b in bounds], float)
    d = len(lo)
    if name in ("random_selection_ea", "gsemo"):
        from . import ablations
        if name == "random_selection_ea":
            F, X, steps = ablations.random_selection_ea(
                evaluate, lo, hi, pop=pop, max_evaluations=max_evaluations, seed=seed,
                on_generation=on_generation, record_every=record_every)
        else:
            from .archive import dss_order
            # the frame DSS normalises by: the problem's, when the archive has it
            ideal, nadir = (archive.frame() if archive is not None
                            and archive.normalization == "problem" else (None, None))

            def select(F, k):
                if len(F) <= k:
                    return np.arange(len(F))
                return dss_order(F, k=k, ideal=ideal, nadir=nadir)
            F, X, steps = ablations.gsemo(
                evaluate, lo, hi, pop=pop, max_evaluations=max_evaluations, seed=seed,
                select=select, on_generation=on_generation, record_every=record_every)
        return SimpleNamespace(objectives=np.asarray(F).tolist(),
                               variables=np.asarray(X).tolist(),
                               ignored=[], active_n=len(F), steps=steps)
    if name == "sobol_search":
        try:
            from scipy.stats import qmc
        except ImportError as e:                    # pragma: no cover - machine-dependent
            raise ImportError("sobol_search needs SciPy: pip install scipy") from e
        engine = qmc.Sobol(d, scramble=True, seed=seed)

        def draw(k):
            with warnings.catch_warnings():         # "balance properties ... power of 2"
                warnings.simplefilter("ignore")
                return engine.random(k)
    else:
        rng = np.random.default_rng(seed)

        def draw(k):
            return rng.random((k, d))

    def answer():
        F, X, _ = archive.points()
        idx = archive.select(pop)
        return F[idx], X[idx]

    spent, gen = 0, 0
    while spent < max_evaluations:
        k = min(pop, max_evaluations - spent)
        for x in lo + draw(k) * (hi - lo):
            evaluate(list(x))
        spent += k
        if on_generation is not None and (gen % max(1, record_every) == 0
                                          or spent >= max_evaluations):
            on_generation(gen, answer()[0].tolist())
        gen += 1
    F, X = answer()
    return SimpleNamespace(objectives=F.tolist(), variables=X.tolist(),
                           ignored=[], active_n=len(F), steps=gen)

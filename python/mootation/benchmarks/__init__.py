# SPDX-License-Identifier: Apache-2.0
"""The standard test suites, as runnable problems.

ZDT, DTLZ, WFG, MaF, the Ishibuchi polygon family, MOP and BT — the sets a
multi-objective paper is expected to report on. Each one arrives as a
`BenchProblem`: bounds, an evaluator, a constraint function, the reference
points a hypervolume needs, and — where a closed form exists — a sampler for
the true Pareto front so IGD and friends can be computed.

This subpackage is the ONE part of `mootation.run` that needs NumPy. The
config, validation, parser, journal and runner layers stay dependency-free on
purpose, so a machine that only drives an external solver never has to install
anything. Import failures are therefore reported as a clear message rather
than a bare ImportError from three modules down.
"""

from __future__ import annotations

try:
    import numpy as _np           # noqa: F401
except ImportError as _e:         # pragma: no cover - depends on the machine
    raise ImportError(
        "mootation.benchmarks needs NumPy (the rest of mootation.run does "
        "not). Install it with `pip install numpy`, or drive an external "
        "evaluator instead — see python/examples/demo.toml."
    ) from _e

from .registry import (          # noqa: E402
    PROBLEMS,
    BenchProblem,
    get,
    names,
    families,
)


def describe():
    """(name, problem) for every entry, WITHOUT resolving reference frames.

    `get()` also fixes a problem's ideal and nadir by sampling its true Pareto
    front, which is the right thing when you are about to compute IGD and the
    wrong thing when you only want to list what exists: doing it for all 216
    takes over two minutes. Listing, filtering and counting go through here.
    """
    return sorted(PROBLEMS.items())


def solve(problem, algorithm="nsga2", *, pop_size=None, n_gen=None,
          seed=0, **knobs):
    """Run one of the suites through the optimizer.

    `problem` is a name ("DTLZ2_3D") or a BenchProblem. `pop_size` and `n_gen`
    default to the values the problem itself carries, which are the ones its
    paper used — a benchmark run with an arbitrary budget compares nothing.

    Returns the binding's Result. This is the bridge between the Python suites
    and the C++ algorithms: 216 problems on one side, 60 algorithms on the
    other, and the objective function crosses the boundary once per individual.
    """
    from .. import minimize

    p = get(problem) if isinstance(problem, str) else problem
    return minimize(
        p.evaluate,
        bounds=p.bounds,
        n_objs=p.n_obj,
        algorithm=algorithm,
        pop_size=p.pop_size if pop_size is None else pop_size,
        n_gen=p.n_gen if n_gen is None else n_gen,
        seed=seed,
        constraints=(p.constraints if p.has_cons else None),
        **knobs,
    )


def igd(problem, objectives, n_ref=1000):
    """Inverted generational distance against the problem's true front.

    Returns None when the problem has no reference front — which is honest;
    reporting a number computed against the run's own output would be a
    self-assessment dressed up as a metric.
    """
    import numpy as np

    p = get(problem) if isinstance(problem, str) else problem
    if not callable(p.pareto_front):
        return None
    ref = np.asarray(p.pareto_front(n_ref), float)
    got = np.asarray(objectives, float)
    if got.size == 0:
        return float("inf")
    d = np.linalg.norm(ref[:, None, :] - got[None, :, :], axis=2)
    return float(d.min(axis=1).mean())


__all__ = ["BenchProblem", "PROBLEMS", "describe", "get", "names", "families",
           "solve", "igd"]

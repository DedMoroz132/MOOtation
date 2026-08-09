# SPDX-License-Identifier: Apache-2.0
"""MOOtation — 60 multi- and many-objective evolutionary algorithms.

The library is C++; this package is the Python face of it.

    import mootation

    def zdt1(x):
        g = 1 + 9 * sum(x[1:]) / (len(x) - 1)
        return [x[0], g * (1 - (x[0] / g) ** 0.5)]

    res = mootation.minimize(zdt1, bounds=[(0, 1)] * 10, n_objs=2,
                             algorithm="nsga2", pop_size=100, n_gen=250)

    for f in res.objectives:
        print(f)

`minimize` is the whole API for the common case. What it wraps stays reachable
— `Problem`, `Config`, `run`, `algorithms` — for the cases it does not cover.

Three further pieces, each independent:

    mootation.benchmarks   the standard suites: ZDT, DTLZ, WFG, MaF, Polygon,
                           MOP, BT — 216 problems. Needs NumPy.
    mootation.run          driving a run from a TOML file, with objectives from
                           external programs. Needs nothing.
    mootation.tui          a terminal interface over that. Needs Textual.

NOTHING IS IMPORTED EAGERLY, and that is load-bearing rather than tidy.
`import mootation.run` necessarily imports this module first, so if this module
imported the compiled extension at the top, the pure-Python layer would stop
working on any machine without a compiler — exactly the machine that most wants
it, the one that only drives an external solver. The attributes below are
resolved on first use (PEP 562), so a missing extension is an error when you
call `minimize`, not when you import a config parser.
"""

from __future__ import annotations

__version__ = "0.1.0"

_CORE_NAMES = ("Config", "ConstraintMode", "Problem", "Result", "algorithms",
               "run_raw")

__all__ = ["minimize", "algorithms", "Problem", "Config", "Result",
           "ConstraintMode", "run_raw",
           "save_population", "load_population", "EvaluationLog"]


def _load_core():
    try:
        from . import _core
    except ImportError as e:
        raise ImportError(
            "the compiled part of MOOtation is not available: " + str(e) +
            "\nInstall the package (`pip install .` from the repository root), "
            "or build it with `cmake -DMOOTATION_BUILD_PYTHON=ON`. "
            "The pure-Python layers — mootation.run, and mootation.benchmarks "
            "for the problem definitions — do not need it."
        ) from e
    return _core


def __getattr__(name: str):
    """Resolve the compiled names, and `minimize`, on first access."""
    if name == "minimize":
        from .minimize import minimize as _m
        globals()["minimize"] = _m
        return _m
    if name == "run_raw":
        # Named run_raw, not run: `mootation.run` is the TOML subpackage, and
        # having an attribute and a submodule fight over one name is a bug
        # waiting for someone to hit it.
        value = getattr(_load_core(), "run")
        globals()["run_raw"] = value
        return value
    if name in ("save_population", "load_population", "EvaluationLog",
                "fit_population"):
        # persistence is pure Python and needs no extension, so it is resolved
        # without touching _core — a machine with no compiler can still read a
        # population file.
        from . import persistence
        value = getattr(persistence, name)
        globals()[name] = value
        return value
    if name in _CORE_NAMES:
        value = getattr(_load_core(), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'mootation' has no attribute '{name}'")


def __dir__():
    return sorted(set(__all__) | {"benchmarks", "run", "tui", "__version__"})

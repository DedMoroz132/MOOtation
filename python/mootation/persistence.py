# SPDX-License-Identifier: Apache-2.0
"""Saving a run's results, and starting the next one from them.

Three files, all optional, all plain CSV with a `#` preamble that
`pandas.read_csv(path, comment="#")` reads directly:

    population    what survived — the answer set
    evaluations   what was computed — every call, including the discards
    (the preamble carries the settings that produced them)

The point of the pair is restarting. Finish a run, keep the population, start
the next one from it, with the same settings or with different ones:

    res = mootation.minimize(f, bounds, 2, save_population="pop.csv")
    ...
    res = mootation.minimize(f, bounds, 2, seed_population="pop.csv",
                             algorithm="moead_de", n_gen=500)

WHAT THIS IS NOT. It is a warm start, not a checkpoint. No RNG position, no
per-algorithm state — only the decision variables and objectives, which is
what every algorithm shares and therefore what lets a population saved by
NSGA-II seed a MOEA/D run. A resumed run does not reproduce what the
uninterrupted one would have done; it starts from the same place.

Seeding costs ZERO function evaluations: the objectives are read from the file
rather than recomputed, which is the whole point when one evaluation is a
solver run.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any, Callable, Sequence

FORMAT_POPULATION = "# mootation population v1"
FORMAT_EVALUATIONS = "# mootation evaluations v1"


# ── Writing ─────────────────────────────────────────────────────────────────

def save_population(result, path: str | os.PathLike, *,
                    meta: dict | None = None, note: str = "") -> None:
    """Write a Result as a population file."""
    p = Path(path)
    if p.parent != Path(""):
        p.parent.mkdir(parents=True, exist_ok=True)

    n_vars = len(result.variables[0]) if result.variables else 0
    n_objs = len(result.objectives[0]) if result.objectives else 0

    with p.open("w", encoding="utf-8", newline="") as fh:
        fh.write(FORMAT_POPULATION + "\n")
        if note:
            # Its own line, without '=': the loader scans comment lines for
            # key=value pairs split on spaces, so a multi-word note stored as
            # `note=...` would lose everything after the first space.
            fh.write(f"# {note}\n")
        if meta:
            fh.write("#" + "".join(f" {k}={v}" for k, v in sorted(meta.items())) + "\n")
        fh.write(f"# n_vars={n_vars} n_bin=0 n_objs={n_objs}\n")

        w = csv.writer(fh)
        w.writerow([f"x{i+1}" for i in range(n_vars)]
                   + [f"f{i+1}" for i in range(n_objs)] + ["cv"])
        # Sorted by the first objective so a two-objective front reads as one.
        order = sorted(range(len(result.objectives)),
                       key=lambda i: result.objectives[i][0]) if n_objs else \
            range(len(result.variables))
        for i in order:
            cv = result.cv[i] if i < len(result.cv) else 0.0
            w.writerow([repr(v) for v in result.variables[i]]
                       + [repr(v) for v in result.objectives[i]] + [repr(cv)])


class EvaluationLog:
    """Records every evaluation to a CSV. Wrap your objective function in it.

        log = EvaluationLog("evals.csv")
        res = mootation.minimize(log.wrap(f), bounds, 2)
        log.close()

    Or let `minimize(log_evaluations="evals.csv")` do both.

    It records what the ALGORITHM asked for, including re-proposals of points
    it has already seen — that is information, not noise: it tells you how much
    of the budget went on repeats.
    """

    def __init__(self, path: str | os.PathLike, *, meta: dict | None = None):
        self.path = Path(path)
        self._meta = meta or {}
        self._fh = None
        self._w = None
        self.rows = 0

    def wrap(self, fn: Callable[[Sequence[float]], Sequence[float]],
             constraints: Callable | None = None):
        def logged(x, _f=fn, _c=constraints):
            f = list(_f(x))
            g = list(_c(x)) if _c is not None else []
            self._append(x, f, g)
            return f
        return logged

    def _append(self, x, f, g) -> None:
        if self._fh is None:
            self._open(len(x), len(f))
        cv = sum(v for v in g if v > 0.0)
        self._w.writerow([self.rows + 1] + [repr(v) for v in x]
                         + [repr(v) for v in f] + [repr(cv)])
        self.rows += 1
        # Flushed per row: an evaluation is expensive enough that the write is
        # free by comparison, and a run killed with Ctrl-C keeps everything.
        self._fh.flush()

    def _open(self, n_vars: int, n_objs: int) -> None:
        if self.path.parent != Path(""):
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", encoding="utf-8", newline="")
        self._fh.write(FORMAT_EVALUATIONS + "\n")
        if self._meta:
            self._fh.write("#" + "".join(f" {k}={v}"
                                         for k, v in sorted(self._meta.items())) + "\n")
        self._fh.write(f"# n_vars={n_vars} n_bin=0 n_objs={n_objs}\n")
        self._w = csv.writer(self._fh)
        self._w.writerow(["eval"] + [f"x{i+1}" for i in range(n_vars)]
                         + [f"f{i+1}" for i in range(n_objs)] + ["cv"])

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# ── Reading ─────────────────────────────────────────────────────────────────

class LoadedPopulation:
    """What `load_population` returns: the rows, plus the settings that made them."""

    __slots__ = ("variables", "objectives", "cv", "meta")

    def __init__(self, variables, objectives, cv, meta):
        self.variables = variables
        self.objectives = objectives
        self.cv = cv
        self.meta = meta

    def __len__(self):
        return len(self.variables)

    def __repr__(self):
        alg = self.meta.get("algorithm", "?")
        return (f"LoadedPopulation({len(self)} individuals, "
                f"{len(self.variables[0]) if self.variables else 0} vars, "
                f"{len(self.objectives[0]) if self.objectives else 0} objs, "
                f"from algorithm={alg})")


def load_population(path: str | os.PathLike) -> LoadedPopulation:
    """Read a population file written by save_population (Python or C++)."""
    p = Path(path)
    meta: dict[str, str] = {}
    n_vars = n_bin = n_objs = None
    header: list[str] | None = None
    variables: list[list[float]] = []
    objectives: list[list[float]] = []
    cvs: list[float] = []

    with p.open("r", encoding="utf-8", newline="") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#"):
                for tok in line[1:].split():
                    if "=" not in tok:
                        continue          # free-text note, not metadata
                    k, v = tok.split("=", 1)
                    if k == "n_vars":
                        n_vars = int(v)
                    elif k == "n_bin":
                        n_bin = int(v)
                    elif k == "n_objs":
                        n_objs = int(v)
                    else:
                        meta[k] = v
                continue

            if header is None:
                header = [h.strip() for h in line.split(",")]
                if n_vars is None or n_objs is None:
                    # No shape preamble: infer from the column names, so a file
                    # trimmed by hand still loads.
                    n_vars = sum(1 for h in header if h.startswith("x"))
                    n_bin = sum(1 for h in header if h.startswith("b"))
                    n_objs = sum(1 for h in header if h.startswith("f"))
                n_bin = n_bin or 0
                continue

            cells = [c.strip() for c in line.split(",")]
            need = n_vars + n_bin + n_objs
            if len(cells) < need:
                raise ValueError(
                    f"{p}: a row has {len(cells)} values, expected at least {need}")
            try:
                c = 0
                x = [float(cells[c + i]) for i in range(n_vars)]
                c += n_vars + n_bin       # binary columns are read past, not kept
                f = [float(cells[c + i]) for i in range(n_objs)]
                c += n_objs
                cv = float(cells[c]) if c < len(cells) else 0.0
            except ValueError as e:
                raise ValueError(f"{p}: a value is not a number: {e}") from e
            variables.append(x)
            objectives.append(f)
            cvs.append(cv)

    if not variables:
        raise ValueError(f"{p}: no rows")
    return LoadedPopulation(variables, objectives, cvs, meta)


def fit_population(pop: LoadedPopulation, pop_size: int,
                   on_size_mismatch: str = "truncate") -> LoadedPopulation:
    """Resize a loaded population to the run that will use it.

    A population saved by one run rarely matches the next run's pop_size, and
    for M2M or the NSGA-III family pop_size is constrained by the algorithm, so
    the mismatch is routine rather than exceptional.

    truncate  keep the first pop_size rows (the file is sorted by f1, so this
              is a contiguous slice of the front, not a random sample)
    pad       cycle the rows until full; duplicates are harmless because the
              first generation's variation separates them
    error     refuse
    """
    have = len(pop)
    if have == pop_size:
        return pop
    if on_size_mismatch == "error":
        raise ValueError(
            f"the seed population has {have} individuals but pop_size is "
            f"{pop_size}; pass on_size_mismatch='truncate' or 'pad'")
    if have > pop_size:
        return LoadedPopulation(pop.variables[:pop_size],
                                pop.objectives[:pop_size],
                                pop.cv[:pop_size], pop.meta)
    if on_size_mismatch == "pad":
        v, o, c = list(pop.variables), list(pop.objectives), list(pop.cv)
        for i in range(have, pop_size):
            src = i % have
            v.append(list(pop.variables[src]))
            o.append(list(pop.objectives[src]))
            c.append(pop.cv[src])
        return LoadedPopulation(v, o, c, pop.meta)
    raise ValueError(
        f"the seed population has {have} individuals but pop_size is {pop_size}")

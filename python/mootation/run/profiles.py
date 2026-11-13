# SPDX-License-Identifier: Apache-2.0
"""Performance profiles of Dolan and Moré (task 26, Г4).

E.D. Dolan, J.J. Moré, "Benchmarking optimization software with performance
profiles", Math. Program. 91:201-213, 2002 (arXiv cs/0102001; source
performance-profiles_dolan2002_arxiv), §2:

  t_{p,s}      the cost of solver s on problem p (any measure: time,
               evaluations, the ERT of a campaign, the cost to 95 % success);
  r_{p,s}      = t_{p,s} / min_s t_{p,s}, the performance ratio; r_M >= every
               ratio, and r_{p,s} = r_M exactly when s does not solve p;
  rho_s(tau)   = |{p : r_{p,s} <= tau}| / n_p, a nondecreasing step function,
               continuous from the right;
  rho_s(1)     the share of problems s wins;
  rho_s*       = lim_{tau -> r_M^-} rho_s(tau), the share s solves.
The choice of r_M does not change any rho_s below r_M (§2), so the functions
here carry no r_M: an unsolved pair has the ratio inf.

READINGS:
  DM-1. A problem no solver solves has no min_s t_{p,s}; it stays in n_p with
    the ratio inf for every solver, so rho_s* counts it as a failure of all
    (the paper keeps every problem, "no need to eliminate any test problems",
    §4, and says nothing of this case).
  DM-2. A cost of 0 has no ratio: t_{p,s} must be > 0 where it is given
    (refused otherwise). None, inf or NaN mean "not solved".
  DM-3. Ties share the win: every solver at the minimum has r = 1, so the
    rho_s(1) of all solvers may add up to more than 1 (as in the paper's
    figures, where they do).
"""

from __future__ import annotations

import math
from typing import Mapping


def _solved(t) -> bool:
    return t is not None and not (isinstance(t, float) and (math.isnan(t) or math.isinf(t)))


def ratios(costs: Mapping[str, Mapping[str, float | None]]) -> dict:
    """{problem: {solver: r_{p,s}}} from {problem: {solver: t_{p,s}}}; inf
    where s does not solve p (DM-1, DM-2). Every problem must name the same
    solvers."""
    solvers = None
    out: dict = {}
    for p in sorted(costs):
        row = costs[p]
        if solvers is None:
            solvers = sorted(row)
        elif sorted(row) != solvers:
            raise ValueError(f"profiles: problem {p!r} names other solvers than the first")
        for s, t in row.items():
            if _solved(t) and not t > 0:
                raise ValueError(f"profiles: cost {t!r} of {s!r} on {p!r} must be > 0 (DM-2)")
        done = [row[s] for s in solvers if _solved(row[s])]
        best = min(done) if done else None
        out[p] = {s: (row[s] / best if best is not None and _solved(row[s]) else math.inf)
                  for s in solvers}
    return out


def profile(r: Mapping[str, Mapping[str, float]], solver: str):
    """rho_s as its breakpoints: [(tau, rho_s(tau))] at each distinct finite
    ratio of `solver`, ascending; rho_s is that value from tau up to the next
    breakpoint and 0 below the first."""
    n = len(r)
    vals = sorted(row[solver] for row in r.values() if math.isfinite(row[solver]))
    out = []
    for i, v in enumerate(vals):
        if i + 1 < len(vals) and vals[i + 1] == v:
            continue
        out.append((v, (i + 1) / n))
    return out


def rho(r: Mapping[str, Mapping[str, float]], solver: str, tau: float) -> float:
    """rho_s(tau) = the share of problems with r_{p,s} <= tau."""
    n = len(r)
    return sum(1 for row in r.values() if row[solver] <= tau) / n if n else 0.0


def summary(costs: Mapping[str, Mapping[str, float | None]],
            taus=(1.0, 2.0, 4.0, 8.0, 16.0, 32.0)) -> dict:
    """Per solver: rho_s(1) (wins, DM-3), rho_s* (solved), rho_s at `taus`,
    and the breakpoints of rho_s; with the ratios themselves."""
    r = ratios(costs)
    solvers = sorted(next(iter(r.values()))) if r else []
    n = len(r)
    per = {}
    for s in solvers:
        per[s] = {"wins": rho(r, s, 1.0),
                  "solved": sum(1 for row in r.values() if math.isfinite(row[s])) / n,
                  "at": {t: rho(r, s, t) for t in taus},
                  "breakpoints": profile(r, s)}
    return {"problems": n, "ratios": r, "solvers": per}

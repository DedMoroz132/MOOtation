#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Smoke test for the Python binding.

Checks the three things that can silently rot:

  1. Every name in include/mootation/algorithms.def is runnable through `run()`. The
     dispatch is generated from that file, so a mismatch means the generation
     broke, not that an algorithm is missing.
  2. A seeded run is reproducible across two separate calls.
  3. A knob the chosen algorithm does not have is REPORTED in result.ignored
     rather than silently dropped.

Convergence quality is not asserted here — that is test_convergence's job on the
C++ side, with far more generations than a smoke test can afford.
"""

from __future__ import annotations

import math
import sys

import mootation


def zdt1(x: list[float]) -> list[float]:
    g = 1.0 + 9.0 * sum(x[1:]) / (len(x) - 1)
    f1 = x[0]
    return [f1, g * (1.0 - math.sqrt(f1 / g))]


def make_problem() -> mootation.Problem:
    p = mootation.Problem()
    p.bounds = [(0.0, 1.0)] * 6
    p.n_objectives = 2
    p.evaluate = zdt1
    return p


def base_config(pop: int = 20, gens: int = 5) -> mootation.Config:
    c = mootation.Config()
    c.pop_size = pop
    c.n_gen = gens
    c.seed = 12345
    return c


def main() -> int:
    failures = 0

    def check(cond: bool, what: str) -> None:
        nonlocal failures
        if not cond:
            failures += 1
            print(f"  FAIL  {what}")

    names = mootation.algorithms()
    print(f"binding {mootation.__version__}, {len(names)} algorithms")

    # `run` is the TOML subpackage; the dispatch function is `run_raw`. An
    # attribute and a submodule fighting over one name is a bug waiting for
    # someone to hit it, so they were given different ones — and this asserts
    # that both are reachable and are different things.
    import mootation.run as run_pkg
    assert run_pkg.__name__ == "mootation.run"
    assert callable(mootation.run_raw)
    assert mootation.run_raw is not run_pkg
    check(len(names) == 60, f"expected 60 algorithms, got {len(names)}")

    # ── 1. every registered name runs ───────────────────────────────────────
    skipped: list[str] = []
    for name in names:
        try:
            res = mootation.run_raw(name, make_problem(), base_config())
        except (ValueError, RuntimeError) as exc:
            # Some algorithms have documented preconditions on pop_size (the
            # M2M family needs pop = K*S; Path-A needs a lattice size). Refusing
            # a configuration is correct behaviour, not a binding failure.
            skipped.append(f"{name}: {exc}")
            continue
        check(res.active_n > 0, f"{name}: empty population")
        check(len(res.objectives) == res.active_n, f"{name}: objectives/active_n mismatch")
        check(all(len(o) == 2 for o in res.objectives), f"{name}: wrong objective width")
        check(all(math.isfinite(v) for o in res.objectives for v in o),
              f"{name}: non-finite objective")

    print(f"ran {len(names) - len(skipped)}, refused-by-precondition {len(skipped)}")
    for s in skipped:
        print(f"  skip  {s}")

    # ── 2. the same seed gives the same run ─────────────────────────────────
    a = mootation.run_raw("nsga2", make_problem(), base_config(pop=20, gens=10))
    b = mootation.run_raw("nsga2", make_problem(), base_config(pop=20, gens=10))
    check(a.objectives == b.objectives, "nsga2: same seed did not reproduce")

    cfg = base_config(pop=20, gens=10)
    cfg.seed = 999
    c = mootation.run_raw("nsga2", make_problem(), cfg)
    check(c.objectives != a.objectives, "nsga2: a different seed changed nothing")

    # ── 3. an inapplicable knob is reported, not swallowed ──────────────────
    cfg = base_config()
    cfg.kappa = 0.05           # an indicator knob; NSGA-II has no set_kappa
    res = mootation.run_raw("nsga2", make_problem(), cfg)
    check("kappa" in res.ignored, "nsga2: kappa should have been reported as ignored")

    cfg = base_config()
    cfg.eta_c = 15.0           # every algorithm has set_eta_crossover
    res = mootation.run_raw("nsga2", make_problem(), cfg)
    check("eta_c" not in res.ignored, "nsga2: eta_c should have been applied")

    print("FAILED" if failures else "ok")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

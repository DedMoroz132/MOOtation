# SPDX-License-Identifier: Apache-2.0
"""Command line: check a run description before spending a day on it.

    python -m mootation.run --check run.toml
    python -m mootation.run --show  run.toml

`--check` is TUI_SPEC.md §7. It is deliberately a mode without a UI: it runs in
CI, it runs over ssh, it runs on a machine where `textual` is not installed. It
is cheap, and it removes the main class of loss — a run budgeted for a day that
dies ten minutes in because a path had a typo in it.

Exit status is 0 when the run can start and 1 when it cannot, so it composes
with `&&` in a submission script.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import ConfigError, load, validate, _platform_key
from .algorithms import algorithm_names
from .knobs import knob_source
from .ledger import Ledger


def _show(cfg) -> None:
    """Print the parsed config: what the file actually said, after resolution.

    This is the Config screen's content without the screen. Resolution matters
    — `run_windows` versus `run`, relative paths against the config's own
    directory — because most surprises are a difference between what the file
    says and what will actually execute.
    """
    base = cfg.source_path.parent if cfg.source_path else Path.cwd()
    print(f"run          {cfg.name}")
    print(f"  config     {cfg.source_path or '<string>'}")
    print(f"  base dir   {base}")
    print(f"  scratch    {cfg.scratch}")
    print(f"  ledger     {cfg.ledger}")
    print(f"  workers    {cfg.workers}")
    print(f"  on_fail    {cfg.on_fail}"
          + (f" (penalty {cfg.penalty:g})" if cfg.on_fail == "penalty" else ""))
    print(f"  resume     {cfg.resume}")
    if cfg.warm_start:
        print(f"  warm start {cfg.warm_start.source} "
              f"[{cfg.warm_start.on_size_mismatch}]")

    print(f"\nproblem      {cfg.kind}")
    if cfg.kind == "external":
        print(f"  n_vars     {cfg.n_vars}")
        print(f"  n_objs     {cfg.n_objs}")
        print(f"  n_cons     {cfg.n_cons}")
        if cfg.input_template:
            print(f"  input      {cfg.input_template} -> {cfg.input_write_to}")
        print(f"  steps      ({len(cfg.steps)}, resolved for {_platform_key()})")
        for st in cfg.steps:
            tmo = f"  timeout {st.timeout:g}s" if st.timeout else ""
            shell = "  shell=true" if st.shell else ""
            print(f"    {st.name:<10} {' '.join(st.argv)}{tmo}{shell}")
        if cfg.output:
            o = cfg.output
            print(f"  output     parser={o.parser}"
                  + (f" from={o.source}" if o.source else ""))
            print(f"    objectives  {', '.join(o.objectives) or '-'}")
            print(f"    constraints {', '.join(o.constraints) or '-'}")

    if cfg.algorithms:
        print(f"\nalgorithms   ({len(cfg.algorithms)})")
        for a in cfg.algorithms:
            p = ("  " + ", ".join(f"{k}={v:g}" for k, v in sorted(a.params.items()))
                 if a.params else "")
            print(f"  {a.name:<14} pop={a.pop:<5} gens={a.gens:<6}{p}")
    if cfg.benchmarks:
        print(f"\nbenchmarks   {cfg.benchmarks}")


def _check(cfg) -> int:
    problems = validate(cfg)
    if not problems:
        n_alg = len(cfg.algorithms)
        total = sum(a.pop * a.gens for a in cfg.algorithms)
        print(f"OK — {n_alg} algorithm(s), "
              f"about {total:,} evaluations at full budget")
        if cfg.resume:
            led = Ledger(
                (cfg.source_path.parent if cfg.source_path else Path.cwd())
                / cfg.ledger.format(name=cfg.name),
                resume=True)
            s = led.stats()
            if s["exists"]:
                print(f"resume — journal has {s['cached']:,} reusable "
                      f"evaluations ({s['bytes'] / 1e6:.1f} MB)")
            led.close()
        return 0

    print(f"{len(problems)} problem(s):\n", file=sys.stderr)
    for p in problems:
        print(f"  - {p}", file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="mootation.run",
        description="Check and inspect a MOOtation run description.")
    ap.add_argument("config", nargs="?", help="path to the TOML run description")
    ap.add_argument("--check", action="store_true",
                    help="validate and exit non-zero if the run cannot start")
    ap.add_argument("--show", action="store_true",
                    help="print the parsed configuration as it resolved")
    ap.add_argument("--algorithms", action="store_true",
                    help="list every algorithm name and exit")
    ap.add_argument("--problems", action="store_true",
                    help="list every benchmark problem and exit (needs NumPy)")
    ap.add_argument("--tui", action="store_true",
                    help="open the terminal interface (needs Textual)")
    args = ap.parse_args(argv)

    if args.algorithms:
        names = algorithm_names()
        for n in names:
            print(n)
        print(f"\n{len(names)} algorithms; knobs read from {knob_source()}",
              file=sys.stderr)
        return 0

    if args.problems:
        try:
            from ..benchmarks import describe
        except ImportError as e:
            print(e, file=sys.stderr)
            return 1
        rows = describe()
        for n, p in rows:
            print(f"{n:<18} M={p.n_obj:<3} n_vars={p.n_vars:<4} "
                  f"pop={p.pop_size:<5} gens={p.n_gen:<6}"
                  f"{'  true PF' if p.pareto_front else ''}")
        print(f"\n{len(rows)} problems", file=sys.stderr)
        return 0

    if not args.config:
        ap.error("a config file is required (or use --algorithms / --problems)")

    try:
        cfg = load(args.config)
    except ConfigError as e:
        print(f"config error — {e}", file=sys.stderr)
        return 1

    if args.tui:
        try:
            from ..tui import run as run_tui
        except ImportError as e:
            print(e, file=sys.stderr)
            return 1
        run_tui(args.config)
        return 0

    if args.show:
        _show(cfg)
        if args.check:
            print()
        # stdout is block-buffered through a pipe while stderr is not, so
        # without this the validator's complaints appear ABOVE the listing they
        # refer to.
        sys.stdout.flush()

    if args.check or not args.show:
        return _check(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

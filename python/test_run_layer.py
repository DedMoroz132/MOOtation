#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Tests for the config / parser / ledger layer.

Plain asserts and a hand-rolled runner: this package has no dependencies, and
requiring pytest to test a zero-dependency package would defeat the point.

    python python/test_mootation_run.py
"""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from mootation.run import ParseError, loads                      # noqa: E402
from mootation.run.algorithms import (                           # noqa: E402
    algorithm_names, check_pop, das_dennis_count, nearest_lattice_sizes)
from mootation.run.config import ConfigError, validate           # noqa: E402
from mootation.run.knobs import knob_names                       # noqa: E402
from mootation.run.ledger import Ledger, Record, hash_x          # noqa: E402
from mootation.run import parsers                                # noqa: E402

_MIN = """
[run]
name = "t"
scratch = "s/{worker:02d}"
ledger = "l.jsonl"

[problem]
kind = "external"
n_vars = 2
n_objs = 2
bounds = [[0.0, 1.0], [0.0, 1.0]]

[[problem.steps]]
name = "solve"
run = ["python", "-c", "pass"]

[problem.output]
parser = "csv"
from = "{scratch}/out.csv"
objectives = ["a", "b"]

[[algorithms]]
name = "nsga2"
pop = 20
gens = 5
"""

TESTS = []


def test(fn):
    TESTS.append(fn)
    return fn


def raises(exc, fn, *a, **kw):
    try:
        fn(*a, **kw)
    except exc as e:
        return e
    raise AssertionError(f"expected {exc.__name__}, nothing was raised")


# ── registry ────────────────────────────────────────────────────────────────

@test
def registry_is_read_from_the_def_file():
    names = algorithm_names()
    assert len(names) == 74, f"expected 74 algorithms, got {len(names)}"
    assert "nsga2" in names and "naemo" in names
    assert len(set(names)) == len(names), "duplicate name in algorithms.def"


@test
def knobs_are_read_from_settings_hpp():
    k = knob_names()
    for expected in ("eta_c", "pc", "theta", "K", "div"):
        assert expected in k, f"knob '{expected}' missing"


# ── population-size rules ───────────────────────────────────────────────────

@test
def das_dennis_counts_match_the_closed_form():
    # m=3: 3, 6, 10, 15, ... triangular numbers; m=2: h+1
    assert [das_dennis_count(3, h) for h in (1, 2, 3, 4)] == [3, 6, 10, 15]
    assert [das_dennis_count(2, h) for h in (1, 5, 99)] == [2, 6, 100]


@test
def exact_lattice_algorithms_reject_a_non_lattice_pop():
    assert check_pop("nsga3", 91, 3) is None            # 91 = C(14,2)
    msg = check_pop("nsga3", 92, 3)
    assert msg and "91" in msg and "105" in msg, msg
    # The paper's own N=92 (nearest multiple of 4 above H=91) is unattainable
    # here by design; the message has to name the attainable values.


@test
def m2m_requires_pop_divisible_by_k():
    assert check_pop("moead_m2m", 100, 3) is None       # default K=10
    assert check_pop("moead_m2m", 289, 3, {"K": 17}) is None
    msg = check_pop("moead_m2m", 91, 3)
    assert msg and "divisible" in msg and "7" in msg, msg


@test
def unconstrained_algorithms_accept_anything():
    for pop in (2, 37, 92, 1000):
        assert check_pop("nsga2", pop, 3) is None
        assert check_pop("spea2", pop, 5) is None


@test
def nearest_lattice_brackets_the_request():
    assert nearest_lattice_sizes(3, 91) == (91, 91)
    below, above = nearest_lattice_sizes(3, 92)
    assert below == 91 and above == 105


# ── config parsing ──────────────────────────────────────────────────────────

@test
def minimal_config_loads():
    cfg = loads(_MIN)
    assert cfg.name == "t"
    assert cfg.n_vars == 2 and cfg.n_objs == 2 and cfg.n_cons == 0
    assert len(cfg.steps) == 1 and cfg.steps[0].argv[0] == "python"
    assert cfg.algorithms[0].name == "nsga2"
    assert cfg.on_fail == "penalty" and cfg.resume is True


@test
def a_missing_required_key_names_itself():
    # scratch/ledger have defaults since the campaign work (a builtin campaign
    # writes neither); `name` and the [problem] table are still required.
    e = raises(ConfigError, loads, "[run]\nscratch='s'\nledger='l'\n")
    assert "name" in str(e), e
    e = raises(ConfigError, loads, "[run]\nname='x'\n")
    assert "problem" in str(e), e


@test
def bad_enums_are_rejected_with_the_alternatives():
    bad = _MIN.replace('ledger = "l.jsonl"', 'ledger = "l.jsonl"\non_fail = "explode"')
    e = raises(ConfigError, loads, bad)
    assert "penalty" in str(e) and "abort" in str(e), e


@test
def a_command_must_be_an_array_not_a_string():
    bad = _MIN.replace('run = ["python", "-c", "pass"]', 'run = "python -c pass"')
    e = raises(ConfigError, loads, bad)
    assert "array" in str(e), e


@test
def a_step_with_no_command_for_this_platform_is_an_error():
    bad = _MIN.replace('run = ["python", "-c", "pass"]',
                       'run_plan9 = ["rc", "solve"]')
    e = raises(ConfigError, loads, bad)
    assert "platform" in str(e), e


@test
def platform_override_wins_over_the_generic_command():
    import mootation.run.config as C
    plat = C._platform_key()
    text = _MIN.replace(
        'run = ["python", "-c", "pass"]',
        f'run = ["generic"]\nrun_{plat} = ["specific", "--flag"]')
    cfg = loads(text)
    assert cfg.steps[0].argv == ["specific", "--flag"], cfg.steps[0].argv


@test
def malformed_toml_says_so():
    e = raises(ConfigError, loads, "[run\nname = ")
    assert "TOML" in str(e), e


# ── validation ──────────────────────────────────────────────────────────────

@test
def validation_collects_every_problem_not_just_the_first():
    text = _MIN.replace("n_vars = 2", "n_vars = 5") \
               .replace('name = "nsga2"', 'name = "no_such_alg"')
    probs = validate(loads(text), base=Path(tempfile.gettempdir()))
    assert len(probs) >= 2, probs
    joined = " ".join(probs)
    assert "bounds" in joined and "no_such_alg" in joined, probs


@test
def bounds_must_be_ordered_and_counted():
    text = _MIN.replace("bounds = [[0.0, 1.0], [0.0, 1.0]]",
                        "bounds = [[1.0, 0.0], [0.0, 1.0]]")
    probs = validate(loads(text), base=Path(tempfile.gettempdir()))
    assert any("strictly less" in p for p in probs), probs


@test
def an_unknown_knob_is_rejected_and_lists_the_known_ones():
    text = _MIN.replace("gens = 5", "gens = 5\nparams = { nonsense = 1 }")
    probs = validate(loads(text), base=Path(tempfile.gettempdir()))
    assert any("nonsense" in p and "eta_c" in p for p in probs), probs


@test
def a_missing_executable_is_caught_before_the_run():
    text = _MIN.replace('run = ["python", "-c", "pass"]',
                        'run = ["definitely-not-installed-xyzzy"]')
    probs = validate(loads(text), base=Path(tempfile.gettempdir()))
    assert any("not found" in p for p in probs), probs


@test
def the_template_placeholder_range_is_checked():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "t.in").write_text("{x[0]} {x[7]}\n", encoding="utf-8")
        text = _MIN.replace(
            "[[problem.steps]]",
            '[problem.input]\ntemplate = "t.in"\nwrite_to = "{scratch}/x"\n\n'
            "[[problem.steps]]")
        probs = validate(loads(text), base=d)
        assert any("x[7]" in p and "n_vars = 2" in p for p in probs), probs


@test
def a_regex_pattern_without_a_capture_group_is_rejected():
    text = _MIN.replace(
        'parser = "csv"\nfrom = "{scratch}/out.csv"\nobjectives = ["a", "b"]',
        'parser = "regex"\nfrom = "{scratch}/log"\nobjectives = ["a", "b"]\n'
        "[problem.output.patterns]\n"
        "a = 'MASS = [0-9.]+'\n"
        "b = 'DRAG = ([0-9.]+)'\n")
    probs = validate(loads(text), base=Path(tempfile.gettempdir()))
    assert any("capturing group" in p for p in probs), probs


@test
def a_clean_config_validates():
    with tempfile.TemporaryDirectory() as d:
        probs = validate(loads(_MIN), base=Path(d))
        assert probs == [], probs


# ── parsers ─────────────────────────────────────────────────────────────────

def _out(**kw):
    from mootation.run.config import Output
    return Output(**kw)


@test
def csv_reads_by_column_name_and_takes_the_last_row():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "out.csv"
        p.write_text("mass,drag,extra\n1,2,9\n3.5,4.5,9\n", encoding="utf-8")
        o = _out(parser="csv", source=str(p), objectives=["mass", "drag"])
        f, g = parsers.parse_output(o, Path(d), lambda s: s)
        assert f == [3.5, 4.5] and g == [], (f, g)


@test
def csv_names_the_columns_it_did_find():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "out.csv"
        p.write_text("mass,drag\n1,2\n", encoding="utf-8")
        o = _out(parser="csv", source=str(p), objectives=["mass", "lift"])
        e = raises(ParseError, parsers.parse_output, o, Path(d), lambda s: s)
        assert "lift" in str(e) and "drag" in str(e), e


@test
def json_digs_through_dotted_paths_and_list_indices():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "out.json"
        p.write_text(json.dumps({"results": {"mass": 1.25,
                                             "aero": {"drag": 0.5}},
                                 "list": [7, 8]}), encoding="utf-8")
        o = _out(parser="json", source=str(p),
                 objectives=["results.mass", "results.aero.drag"],
                 constraints=["list.1"])
        f, g = parsers.parse_output(o, Path(d), lambda s: s)
        assert f == [1.25, 0.5] and g == [8.0], (f, g)


@test
def regex_take_last_is_the_default_because_solvers_print_per_iteration():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "solve.log"
        p.write_text("TOTAL MASS = 5.0\nTOTAL MASS = 4.0\nTOTAL MASS = 3.25\n",
                     encoding="utf-8")
        o = _out(parser="regex", source=str(p), objectives=["mass"],
                 patterns={"mass": r"TOTAL MASS\s*=\s*([0-9.eE+-]+)"})
        f, _ = parsers.parse_output(o, Path(d), lambda s: s)
        assert f == [3.25], f
        o.take = "first"
        f, _ = parsers.parse_output(o, Path(d), lambda s: s)
        assert f == [5.0], f


@test
def a_pattern_that_does_not_match_is_an_evaluation_failure():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "solve.log"
        p.write_text("the solver diverged\n", encoding="utf-8")
        o = _out(parser="regex", source=str(p), objectives=["mass"],
                 patterns={"mass": r"MASS = ([0-9.]+)"})
        e = raises(ParseError, parsers.parse_output, o, Path(d), lambda s: s)
        assert "did not match" in str(e), e


@test
def columns_reads_positionally_from_the_chosen_line():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "fort.7"
        p.write_text("  1  0.0  0.0\n  2  1.5  2.5\n", encoding="utf-8")
        o = _out(parser="columns", source=str(p), objectives=["a", "b"],
                 line=-1, fields=[1, 2])
        f, _ = parsers.parse_output(o, Path(d), lambda s: s)
        assert f == [1.5, 2.5], f


@test
def a_missing_output_file_is_an_evaluation_failure_not_a_crash():
    with tempfile.TemporaryDirectory() as d:
        o = _out(parser="csv", source=str(Path(d) / "never_written.csv"),
                 objectives=["a"])
        e = raises(ParseError, parsers.parse_output, o, Path(d), lambda s: s)
        assert "not produced" in str(e), e


@test
def a_non_numeric_value_is_refused_rather_than_coerced():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "out.csv"
        p.write_text("mass\nNaN\n", encoding="utf-8")
        o = _out(parser="csv", source=str(p), objectives=["mass"])
        e = raises(ParseError, parsers.parse_output, o, Path(d), lambda s: s)
        assert "NaN" in str(e), e


@test
def objectives_and_constraints_come_back_split_in_config_order():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "out.csv"
        p.write_text("m,d,s\n1,2,-0.5\n", encoding="utf-8")
        o = _out(parser="csv", source=str(p), objectives=["m", "d"],
                 constraints=["s"])
        f, g = parsers.parse_output(o, Path(d), lambda s: s)
        assert f == [1.0, 2.0] and g == [-0.5], (f, g)


# ── ledger ──────────────────────────────────────────────────────────────────

@test
def hashing_ignores_float_noise_below_the_quantum():
    a = [0.1, 0.2, 0.3]
    b = [0.1, 0.2, 0.3 + 1e-16]
    assert hash_x(a) == hash_x(b)
    assert hash_x(a) != hash_x([0.1, 0.2, 0.4])


@test
def the_journal_round_trips_and_indexes_for_resume():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "sub" / "ev.jsonl"      # parent is created
        with Ledger(path) as led:
            led.append(Record(gen=0, idx=0, x=[0.1, 0.2], f=[1.0, 2.0], t=0.5))
            led.append(Record(gen=0, idx=1, x=[0.3, 0.4], f=[3.0, 4.0]))
        assert path.is_file()

        reopened = Ledger(path, resume=True)
        assert len(reopened) == 2
        hit = reopened.lookup([0.1, 0.2])
        assert hit is not None and hit.f == [1.0, 2.0]
        assert reopened.lookup([9.9, 9.9]) is None
        reopened.close()


@test
def failed_evaluations_are_journalled_but_never_replayed():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "ev.jsonl"
        with Ledger(path) as led:
            led.append(Record(gen=0, idx=0, x=[1.0], f=[], status="failed"))
        again = Ledger(path, resume=True)
        # The record is on disk — the run is auditable ...
        assert path.read_text(encoding="utf-8").count("failed") == 1
        # ... but a transient solver crash must not become permanent.
        assert again.lookup([1.0]) is None and len(again) == 0
        again.close()


@test
def a_torn_final_line_does_not_cost_the_whole_journal():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "ev.jsonl"
        with Ledger(path) as led:
            led.append(Record(gen=0, idx=0, x=[0.5], f=[1.0]))
        with path.open("a", encoding="utf-8") as fh:
            fh.write('{"gen":1,"idx":1,"x":[0.6],"f":[2.')   # killed mid-write
        led = Ledger(path, resume=True)
        assert len(led) == 1 and led.lookup([0.5]) is not None
        led.close()


@test
def opening_a_journal_creates_nothing_on_disk():
    """--check opens the journal to report resumable work. It must not write.

    A read-only command that leaves directories behind is a command nobody
    trusts to be read-only; this caught exactly that, `--check` on the shipped
    example having quietly created python/examples/results/.
    """
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "nope" / "deeper" / "ev.jsonl"
        led = Ledger(path, resume=True)
        assert led.stats()["exists"] is False
        assert not path.parent.exists(), "opening a journal created directories"
        led.close()
        assert not path.parent.exists()
        # ... but the first append does create them.
        led2 = Ledger(path)
        led2.append(Record(gen=0, idx=0, x=[1.0], f=[1.0]))
        led2.close()
        assert path.is_file()


@test
def resume_false_ignores_an_existing_journal():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "ev.jsonl"
        with Ledger(path) as led:
            led.append(Record(gen=0, idx=0, x=[0.5], f=[1.0]))
        fresh = Ledger(path, resume=False)
        assert len(fresh) == 0
        fresh.close()


# ── runner: real subprocesses ───────────────────────────────────────────────

def _demo_copy(tmp: Path):
    """A private copy of the demo example, so tests never dirty the repo."""
    import shutil
    from mootation.run.config import load
    dst = tmp / "ex"
    shutil.copytree(HERE / "examples", dst)
    cfg = load(dst / "demo.toml")
    return cfg


@test
def an_evaluation_runs_the_real_steps_and_parses_the_result():
    from mootation.run.runner import Evaluator
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _demo_copy(Path(tmp))
        with Evaluator(cfg, worker=3) as ev:
            f, g = ev.evaluate([0.5, 0.1, 0.1, 0.1, 0.1, 0.1], gen=0, idx=0)
            # ZDT1 by hand: g = 1 + 9*0.5/5 = 1.9, f2 = 1.9*(1 - sqrt(0.5/1.9))
            assert abs(f[0] - 0.5) < 1e-12, f
            assert abs(f[1] - 0.925321) < 1e-5, f
            assert abs(g[0] - (1.0 - 6 * 0.9)) < 1e-12, g
            assert ev.scratch.name == "worker_03", ev.scratch


@test
def a_repeated_vector_is_served_from_the_journal():
    from mootation.run.runner import Evaluator
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _demo_copy(Path(tmp))
        with Evaluator(cfg, worker=0) as ev:
            x = [0.3, 0.2, 0.2, 0.2, 0.2, 0.2]
            first, _ = ev.evaluate(x, gen=0, idx=0)
            second, _ = ev.evaluate(x, gen=1, idx=0)
            assert first == second
            # One journal entry, not two: the solver ran once.
            assert len(ev.ledger) == 1, len(ev.ledger)


@test
def a_failing_evaluation_yields_the_penalty_and_never_a_fabricated_value():
    from mootation.run.runner import Evaluator
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _demo_copy(Path(tmp))
        with Evaluator(cfg, worker=0) as ev:
            f, g = ev.evaluate([float("nan")] * 6, gen=0, idx=0)
            assert f == [cfg.penalty] * cfg.n_objs, f
            assert g == [cfg.penalty] * cfg.n_cons, g
            # Journalled as failed, and NOT reusable.
            assert len(ev.ledger) == 0
            assert "failed" in ev.ledger.path.read_text(encoding="utf-8")


@test
def on_fail_abort_stops_the_run_instead_of_scoring_the_point():
    from mootation.run.runner import Evaluator, AbortRun
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _demo_copy(Path(tmp))
        cfg.on_fail = "abort"
        with Evaluator(cfg, worker=0) as ev:
            raises(AbortRun, ev.evaluate, [float("nan")] * 6)


@test
def scratch_is_wiped_between_evaluations():
    from mootation.run.runner import Evaluator
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _demo_copy(Path(tmp))
        with Evaluator(cfg, worker=1) as ev:
            ev.evaluate([0.4, 0.1, 0.1, 0.1, 0.1, 0.1])
            litter = ev.scratch / "stale_from_last_time.vtu"
            litter.write_text("junk", encoding="utf-8")
            ev.evaluate([0.6, 0.1, 0.1, 0.1, 0.1, 0.1])
            # Stale solver output must not survive: otherwise a silently failing
            # step gets "parsed" from the previous evaluation's leftovers.
            assert not litter.exists()


@test
def a_step_that_exits_non_zero_is_a_failure_with_its_output_quoted():
    from mootation.run.runner import Evaluator, EvaluationFailed
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _demo_copy(Path(tmp))
        cfg.steps[0].argv = ["python", "-c",
                             "import sys; print('solver diverged'); sys.exit(3)"]
        with Evaluator(cfg, worker=0) as ev:
            e = raises(EvaluationFailed, ev._run_once, [0.1] * 6)
            assert "exited 3" in str(e) and "diverged" in str(e), e


@test
def a_step_that_overruns_its_timeout_is_a_failure():
    from mootation.run.runner import Evaluator, EvaluationFailed
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _demo_copy(Path(tmp))
        cfg.steps[0].argv = ["python", "-c", "import time; time.sleep(30)"]
        cfg.steps[0].timeout = 1.0
        with Evaluator(cfg, worker=0) as ev:
            e = raises(EvaluationFailed, ev._run_once, [0.1] * 6)
            assert "timeout" in str(e), e


@test
def resume_reuses_a_previous_run_journal():
    from mootation.run.runner import Evaluator
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _demo_copy(Path(tmp))
        x = [0.25, 0.15, 0.15, 0.15, 0.15, 0.15]
        with Evaluator(cfg, worker=0) as ev:
            first, _ = ev.evaluate(x)
        # A brand-new Evaluator, as after a crash and a restart.
        with Evaluator(cfg, worker=0) as ev2:
            assert len(ev2.ledger) == 1, len(ev2.ledger)
            again, _ = ev2.evaluate(x)
            assert again == first


# ── benchmarks (needs NumPy; skipped without it) ────────────────────────────

def _have_numpy():
    try:
        import numpy  # noqa: F401
        return True
    except ImportError:
        return False


@test
def the_benchmark_registry_loads_and_is_not_empty():
    if not _have_numpy():
        print("    (skipped: no NumPy)"); return
    from mootation.benchmarks import names, families
    n = names()
    assert len(n) > 200, len(n)
    fam = families()
    for expected in ("DTLZ", "WFG", "ZDT", "MaF"):
        assert expected in fam, sorted(fam)


@test
def benchmark_problems_evaluate_to_their_published_values():
    """Spot checks against the papers, computed by hand.

    Not a self-consistency check: these are the numbers the definitions must
    produce, so a silent change to a formula shows up here rather than as a
    subtly different convergence plot six months later.
    """
    if not _have_numpy():
        print("    (skipped: no NumPy)"); return
    from mootation.benchmarks import get

    # DTLZ2 at x = 0.5 everywhere: g = 0, so the point sits ON the unit sphere.
    p = get("DTLZ2_3D")
    f = p.evaluate([0.5] * p.n_vars)
    assert abs(sum(v * v for v in f) - 1.0) < 1e-12, f

    # ZDT1 at x = 0.5: f1 = 0.5, g = 1 + 9*14.5/29 = 5.5, f2 = g(1-sqrt(f1/g)).
    import math
    z = get("ZDT1")
    f = z.evaluate([0.5] * z.n_vars)
    g = 1.0 + 9.0 * (0.5 * (z.n_vars - 1)) / (z.n_vars - 1)
    assert abs(f[0] - 0.5) < 1e-12, f
    assert abs(f[1] - g * (1.0 - math.sqrt(0.5 / g))) < 1e-12, f

    # WFG's nadir is 2i by construction.
    w = get("WFG4_3D")
    assert tuple(w.nadir) == (2.0, 4.0, 6.0), w.nadir


@test
def shifted_dtlz_keeps_the_front_and_moves_the_optimum_off_the_centre():
    """shiftDTLZ1-4 reach the DTLZ front at x = c instead of x = 1/2, with no jump at the wrap."""
    if not _have_numpy():
        print("    (skipped: no NumPy)"); return
    import numpy as np
    from mootation.benchmarks import get
    from mootation.benchmarks.dtlz_variants import shift_centres
    c = [float(v) for v in shift_centres(10)]
    assert all(min(abs(v - 0.5), v, 1.0 - v) >= 0.15 for v in c), c
    assert len({round(v, 12) for v in c}) == len(c), c
    pos = [0.3, 0.6]
    for base in ("DTLZ1", "DTLZ2", "DTLZ3", "DTLZ4"):
        d, s = get(f"{base}_3D"), get(f"shift{base}_3D")
        k = d.n_vars - 2
        assert s.n_vars == d.n_vars and tuple(s.nadir) == tuple(d.nadir), base
        on = d.evaluate(pos + [0.5] * k)                          # g = 0: on the front
        here = s.evaluate(pos + [float(v) for v in shift_centres(k)])
        assert max(abs(a - b) for a, b in zip(here, on)) < 1e-12, (base, here, on)
        assert sum(s.evaluate(pos + [0.5] * k)) > sum(on) + 1e-6, base   # the centre is not
        # Where y = (x - c + 1/2) mod 1 wraps, both sides give the same objectives.
        wrap = [(v - 0.5) % 1.0 for v in shift_centres(k)]
        below = s.evaluate(pos + [v - 1e-9 for v in wrap])
        above = s.evaluate(pos + [v + 1e-9 for v in wrap])
        assert max(abs(a - b) for a, b in zip(below, above)) < 1e-6 * max(1.0, max(below)), base
    assert np.allclose(get("shiftDTLZ2_5D").pareto_front(50), get("DTLZ2_5D").pareto_front(50))


# Reference values of other libraries at fixed points (tests/data, task 8 §4a):
# one JSON file per source, with its library, version, license, the command
# that made it (tools/refs/) and the tolerance the comparison takes.
DATA = HERE.parent / "tests" / "data"


def reference_data(name: str) -> dict:
    """tests/data/<name>.json."""
    return json.loads((DATA / f"{name}.json").read_text(encoding="utf-8"))


def assert_close(mine, theirs, tolerance: dict, what) -> None:
    """Every value within the file's tolerance: absolute, or relative to
    max(1, |reference|)."""
    assert len(mine) == len(theirs), (what, len(mine), len(theirs))
    tol = tolerance["value"]
    for a, b in zip(mine, theirs):
        scale = 1.0 if tolerance["kind"] == "absolute" else max(1.0, abs(b))
        assert abs(a - b) <= tol * scale, (what, a, b)


def wfg_at_fixed_points() -> dict:
    """{problem: MOOtation's WFG objectives at the reference file's point}."""
    from mootation.benchmarks import wfg
    out = {}
    for case in reference_data("pymoo-0.6.2_wfg")["cases"]:
        name, M = case["problem"].split("_")[0], int(case["problem"].split("_")[1][:-1])
        out[case["problem"]] = wfg.WFG_FUNCS[name](case["x"], M)
    return out


@test
def wfg_matches_pymoo_with_and_without_numba():
    """WFG1-9 at 2, 3, 5 and 10 objectives give pymoo's values at fixed points
    (tests/data/pymoo-0.6.2_wfg.json): on the path this interpreter takes and,
    in a child process with NUMBA_DISABLE_JIT=1, on the interpreted one. Until
    2026-10-09 the path without numba got f_2 .. f_{M-1} wrong from three
    objectives on."""
    if not _have_numpy():
        print("    (skipped: no NumPy)"); return
    import os
    import subprocess
    from mootation.benchmarks import wfg
    ref = reference_data("pymoo-0.6.2_wfg")
    child = subprocess.run(
        [sys.executable, "-c",
         "import json, test_run_layer as t; from mootation.benchmarks import wfg; "
         "print(json.dumps([wfg.numba_enabled(), t.wfg_at_fixed_points()]))"],
        capture_output=True, text=True, check=True, cwd=str(HERE),
        env=dict(os.environ, NUMBA_DISABLE_JIT="1"))
    numba_there, there = json.loads(child.stdout.strip().splitlines()[-1])
    assert numba_there is False
    paths = {f"this process (numba {'on' if wfg.numba_enabled() else 'off'})":
             wfg_at_fixed_points(), "NUMBA_DISABLE_JIT": there}
    for path, got in paths.items():
        for case in ref["cases"]:
            assert_close(got[case["problem"]], case["f"], ref["tolerance"], (path, case["problem"]))


@test
def zcat_vanishes_on_its_pareto_set_and_keeps_the_papers_frame():
    """On y_II = g(y_I|m) every beta is zero, so f = alpha = i²·F (ZCAT Eq. 1-8)."""
    if not _have_numpy():
        print("    (skipped: no NumPy)"); return
    import numpy as np
    from mootation.benchmarks import zcat
    from mootation.benchmarks.registry import PROBLEMS

    for M in (2, 3):
        scale = np.arange(1, M + 1, dtype=float) ** 2
        n = zcat.n_vars(M)
        assert n == 10 * M, n                                  # Section 5.1
        for name in zcat.NAMES:
            # Straight off PROBLEMS: get() would reframe ideal/nadir from the
            # sampled front, and what is checked here is the analytic frame.
            p = PROBLEMS[f"{name}_{M}D"]
            assert p.n_vars == n and p.bounds[-1] == (-n / 2.0, n / 2.0), name
            assert tuple(p.nadir) == tuple(scale) and set(p.ideal) == {0.0}, name
            y = np.zeros(n)
            y[:M - 1] = 0.3 + 0.1 * np.arange(M - 1)           # ZCAT20 switches m here
            m = zcat.position_count(name, y, M)
            y[m:] = zcat.topology(zcat.G_OF[name], y[:m], m, n)
            assert np.all((y >= 0.0) & (y <= 1.0)), (name, y.min(), y.max())
            x = (y - 0.5) * np.arange(1, n + 1)                # Omega = Π[-i/2, i/2]
            f = np.asarray(p.evaluate(x.tolist()), float)
            alpha = scale * zcat.F_FUNCS[name](y, M)
            assert np.max(np.abs(f - alpha)) <= 1e-6 * M ** 2, (name, M, f, alpha)

    # Every Z is zero only at w = 0 (Section 4.3.1), including the deceptive
    # pair, whose |w|^0.002 term makes that zero unreachable from a rounded x.
    for lv in range(1, 7):
        assert zcat.level_value(lv, np.zeros(4)) == 0.0, lv
        assert zcat.level_value(lv, np.full(4, 0.25)) > 0.0, lv


@test
def zcat_reference_fronts_are_nondominated_and_inside_the_paper_box():
    if not _have_numpy():
        print("    (skipped: no NumPy)"); return
    import numpy as np
    from mootation.benchmarks import zcat
    scale = np.arange(1, 4, dtype=float) ** 2
    for name in ("ZCAT1", "ZCAT6", "ZCAT14", "ZCAT19", "ZCAT20"):
        F = zcat.pareto_front(name, 3, 100)
        assert len(F) > 10, (name, len(F))
        assert np.all(F >= -1e-9) and np.all(F <= scale + 1e-9), (name, F.max(0))
        dom = ((F[:, None, :] <= F[None, :, :]).all(2)
               & (F[:, None, :] < F[None, :, :]).any(2))
        assert not dom.any(), name


@test
def zcat_reference_fronts_survive_a_fresh_sample():
    """Nondominated within its own sample is not Pareto-optimal (zcat._dominated_elsewhere).

    Before the check, 24 of ZCAT11_3D's 1000 reference points and 164 of
    ZCAT11_5D's were dominated by points of a fresh sample, next to the gaps
    of the disconnected front.
    """
    if not _have_numpy():
        print("    (skipped: no NumPy)"); return
    import numpy as np
    from mootation.benchmarks import zcat
    for name, M in (("ZCAT11", 3), ("ZCAT13", 3)):
        F = zcat.pareto_front(name, M, 300)
        P = np.random.default_rng(424242).random((20_000, M - 1))
        V = zcat._images(name, M, P)
        dom = zcat._strictly_dominated_by(F, V)
        assert not dom.any(), (name, M, int(dom.sum()))
        assert zcat.pareto_front(name, M, 300) is not zcat.pareto_front(name, M, 300)


@test
def disconnected_reference_fronts_survive_a_fresh_sample():
    """DTLZ7 and WFG2 sample their fronts as the nondominated images of a grid.

    Before registry._checked, 150 of WFG2_3D's 416 reference points were
    dominated by fresh front points, and DTLZ7's grid put most of its points
    outside the Pareto set (all but one of DTLZ7_15D's 16 384).
    """
    if not _have_numpy():
        print("    (skipped: no NumPy)"); return
    import math
    import numpy as np
    from mootation.benchmarks import registry as R
    rng = np.random.default_rng(424242)
    for key, M, images in (("WFG2_3D", 3, lambda X: R._img_wfg2(3, X)),
                           ("DTLZ7_3D", 3, lambda X: R._img_dtlz7(3, X)),
                           ("DTLZ7_5D", 5, lambda X: R._img_dtlz7(5, X))):
        F = R.PROBLEMS[key].pareto_front(1000)
        dom = R._strictly_dominated_by(F, images(rng.random((20_000, M - 1))))
        assert not dom.any(), (key, int(dom.sum()))
    # DTLZ7's Pareto set is the product of phi's left records: every position
    # of every reference point is one, and the frame is the analytic one
    t = np.linspace(0.0, 1.0, 200_001)
    phi = t * (1.0 + np.sin(3.0 * math.pi * t))
    F = R.PROBLEMS["DTLZ7_10D"].pareto_front(1000)
    x = F[:, :9]
    run_max = np.maximum.accumulate(phi)
    below = run_max[np.clip((x * 200_000).astype(int) - 1, 0, None)]
    assert np.all(x * (1.0 + np.sin(3.0 * math.pi * x)) >= below - 1e-6), "a position is no record"
    p = R.get("DTLZ7_5D")
    assert abs(p.ideal[4] - (10.0 - 4.0 * phi.max())) < 1e-4 and abs(p.nadir[0] - 0.8594) < 1e-3, \
        (p.ideal, p.nadir)


@test
def bbob_functions_reach_their_own_optimum():
    """f(x_opt) = f_opt for all ten bbob base functions.

    Every one of them is <something>(z) + f_opt where the <something> vanishes
    at the optimum, so this one identity catches a wrong transformation order,
    a wrong constant or a misread erratum in any of the ten.
    """
    if not _have_numpy():
        print("    (skipped: no NumPy)"); return
    import numpy as np
    from mootation.benchmarks import bbob
    for fid in bbob.BASE_IDS:
        for D in (2, 5, 10):
            s = bbob.instance(fid, 1, D)
            assert abs(s(s.x_opt) - s.f_opt) <= 1e-9 * max(1.0, abs(s.f_opt)), (fid, D)
            for M in (s.R, s.Q):
                if M is not None:
                    assert np.allclose(M @ M.T, np.eye(D), atol=1e-10), (fid, D)
    # f20 is the one where the report's own text is ambiguous (2|x^opt| against
    # 2|x_hat^opt|): only this value puts the optimum where Schwefel's is.
    s = bbob.instance(20, 1, 5)
    assert abs(2 * abs(float(s.x_opt[0])) - 4.2096874633) < 1e-9, s.x_opt


@test
def bbob_biobj_pairs_and_frame_follow_the_paper():
    """The 55 pairs, the instance numbering, and a frame with no reference front."""
    if not _have_numpy():
        print("    (skipped: no NumPy)"); return
    from mootation.benchmarks import bbob_biobj as bb, igd
    from mootation.benchmarks.registry import PROBLEMS
    assert len(bb.PAIRS) == 55 and len(set(bb.PAIRS)) == 55, len(bb.PAIRS)
    # The paper names this group outright, so it pins the F-numbering: F5, F6,
    # F14 and F15 are the separable x ill-conditioned combinations.
    for f in (5, 6, 14, 15):
        a, b = bb.PAIRS[f - 1]
        assert {a, b} & {1, 2} and {a, b} & {13, 14}, (f, a, b)
    assert bb.single_instance_ids(1) == (2, 4)        # the historical exceptions
    assert bb.single_instance_ids(2) == (3, 5)
    assert bb.single_instance_ids(4) == (9, 10)       # K_a = 2K+1, K_b = K_a+1
    for key in ("bbobbiobj01_n05_2D", "bbobbiobj55_n10_2D"):
        p = PROBLEMS[key]
        assert p.n_obj == 2 and p.pareto_front is None, key
        # the hypervolume is the only indicator here, so the box must be real
        assert all(n > i for i, n in zip(p.ideal, p.nadir)), (key, p.ideal, p.nadir)
    # and the front-based indicators must say nothing rather than invent it
    assert igd("bbobbiobj01_n05_2D", [[1.0, 2.0]]) is None


@test
def full_fronts_are_the_curve_where_the_paper_says_the_curve_is_all():
    """DTLZ5, DTLZ6 and MaF6 at M = 3 and WFG3 at M = 2 have no non-degenerate part.

    Built by the same pipeline as the shipped fronts, at a test-sized budget:
    no point of the verified front may lie off the curve.
    """
    if not _have_numpy():
        print("    (skipped: no NumPy)"); return
    from mootation.benchmarks import fronts_full as FF
    for name, M in (("DTLZ5", 3), ("DTLZ6", 3), ("MaF6", 3), ("WFG3", 2)):
        F, rep = FF.build_front(name, M, n_int=6000, n_face=600, n_verify=30000,
                                n_keep=200, log=lambda s: None)
        assert rep["verified"] > 100, rep
        assert rep["distance_scalar_max_on_front"] == 0.0, (name, M, rep["distance_scalar_max_on_front"])


@test
def the_shipped_full_fronts_leave_the_curve_and_survive_a_fresh_sample():
    if not _have_numpy():
        print("    (skipped: no NumPy)"); return
    import numpy as np
    from mootation.benchmarks import fronts_full as FF, get
    from mootation.benchmarks.registry import FULL_FRONT, degenerate_subset_note
    for key in ("DTLZ5_5D", "DTLZ6_5D", "WFG3_3D", "WFG3_5D"):
        assert key in FULL_FRONT, (key, sorted(FULL_FRONT))
        assert degenerate_subset_note(key) is None, key
        name, M = key.split("_")[0], int(key.split("_")[1][:-1])
        F, rep = FF.load(name, M)
        assert rep["share_off_the_curve"] > 0.0 and rep["verify_sample"] >= 1_000_000, rep
        # an independent sample of the reduced box, from a seed the build never used
        lo, hi = FF.box(name, M)
        V = FF._sample(lo, hi, 60000, 4000, np.random.default_rng(424242))
        assert not FF.dominated_by_any(F, FF.objectives(name, V, M)).any(), key
        # and exactly, on a grid of g (or t_M) that the build did not use: no
        # point beaten by 1e-5 of the range or more (a point beaten by less can
        # sit between two grid values; the one such point met was 9.7e-6)
        span = np.array(rep["nadir"]) - np.array(rep["ideal"])
        exact = (FF.radial_dominated(name, M, F, span=span, tau=1e-5, n_g=10_007)
                 if name in FF.G_MAX else
                 FF.wfg3_dominated(M, F, span=span, tau=1e-5, n_t=10_007))
        assert not exact.any(), (key, int(exact.sum()))
        # the registry uses it: thinning is slicing, the frame is the verified set's
        p = get(key)
        assert np.array_equal(p.pareto_front(500), F[:500]), key
        assert np.allclose(p.nadir, rep["nadir"]) and np.allclose(p.ideal, rep["ideal"]), key


@test
def full_fronts_contain_the_points_the_analysis_says_they_must():
    """(3,1,1) on WFG3_3D (Ishibuchi et al.'s counterexample) and the far corner of DTLZ6."""
    if not _have_numpy():
        print("    (skipped: no NumPy)"); return
    import math
    import numpy as np
    from mootation.benchmarks import fronts_full as FF
    pt = FF.objectives("WFG3", np.array([[1.0, 1.0, 1.0]]), 3)[0]
    assert np.allclose(pt, [3.0, 1.0, 1.0]), pt                 # t = (1, 1, 1) gives it
    F, _ = FF.load("WFG3", 3)
    assert not (np.all(F <= pt, axis=1) & np.any(F < pt, axis=1)).any()
    assert np.min(np.linalg.norm(F - pt, axis=1)) < 0.05, np.min(np.linalg.norm(F - pt, axis=1))
    # DTLZ6 at M = 4, x = (0, 1, 0) and g = 9.9: f_2 = (1+g) sin^2(pi/(4(1+g))) is the
    # smallest any point with f_4 = 0 can have, and it falls with g — so this point
    # is Pareto-optimal and the front reaches g ~ 10, where DTLZ5 stops at 2.5
    g = 9.9
    q = FF.objectives("DTLZ6", np.array([[0.0, 1.0, 0.0, g]]), 4)[0]
    assert abs(q[1] - (1 + g) * math.sin(math.pi / (4 * (1 + g))) ** 2) < 1e-12, q
    F6, rep6 = FF.load("DTLZ6", 4)
    assert not (np.all(F6 <= q, axis=1) & np.any(F6 < q, axis=1)).any()
    assert rep6["nadir"][2] > 10.0 and FF.load("DTLZ5", 4)[1]["nadir"][2] < 4.0, rep6["nadir"]


@test
def the_exact_dominance_tests_see_what_sampling_misses():
    """radial_dominated and wfg3_dominated decide dominance by the whole attainable set.

    A DTLZ5 point with g > 0 and every theta_i strictly inside its range is
    beaten by its own radial projection (the same angles at the smallest g
    that reaches them); the curve is not beaten; neither is WFG3's (3, 1, 1),
    while the same point moved up by 0.01 is.
    """
    if not _have_numpy():
        print("    (skipped: no NumPy)"); return
    import numpy as np
    from mootation.benchmarks import fronts_full as FF
    rng = np.random.default_rng(5)
    U = 0.05 + 0.9 * rng.random((200, 4))
    U[:, 3] = 0.05 + 2.0 * rng.random(200)
    assert FF.radial_dominated("DTLZ5", 4, FF.objectives("DTLZ5", U, 4)).all()
    C = FF.objectives("DTLZ5", FF._curve_u("DTLZ5", 4, np.linspace(0.0, 1.0, 30)), 4)
    assert not FF.radial_dominated("DTLZ5", 4, C).any()
    pt = FF.objectives("WFG3", np.ones((1, 3)), 3)
    assert not FF.wfg3_dominated(3, pt, span=np.ones(3)).any()
    assert FF.wfg3_dominated(3, pt + 0.01, span=np.ones(3)).all()


@test
def no_full_front_point_of_the_dtlz5_family_has_f_M_above_one():
    """Every point with f_M > 1 is dominated by the curve (fronts_full's header).

    At x_1 = 1 that needs cos(pi/2) = 0 exactly: with the rounded 6.1e-17 the
    points with x_1 = 1 and g > 0 looked nondominated and set the nadir of f_M
    to 1 + g_max, 251 for MaF6 at eight objectives.
    """
    if not _have_numpy():
        print("    (skipped: no NumPy)"); return
    import numpy as np
    from mootation.benchmarks import fronts_full as FF
    rng = np.random.default_rng(11)
    for name, M in (("DTLZ5", 4), ("DTLZ6", 5), ("MaF6", 8)):
        lo, hi = FF.box(name, M)
        U = lo + rng.random((4000, M)) * (hi - lo)
        U[:1000, 0] = 1.0                                        # x_1 = 1: the corner case
        F = FF.objectives(name, U, M)
        assert np.all(F[:1000, :M - 1] == 0.0), name               # exactly, not ~1e-17
        far = F[:, M - 1] > 1.0
        assert far.sum() > 1000, (name, far.sum())
        hit, _ = FF.curve_dominated(name, M, F[far])
        assert hit.all(), (name, int((~hit).sum()))
    for key in sorted(FF.SIZES):
        if key == "WFG3":
            continue
        for M in FF.SIZES[key]:
            if FF.has_front(key, M):
                F, rep = FF.load(key, M)
                assert rep["nadir"][M - 1] == 1.0 and F[:, M - 1].max() <= 1.0, (key, M, rep["nadir"])


@test
def every_benchmark_evaluates_without_raising():
    if not _have_numpy():
        print("    (skipped: no NumPy)"); return
    import random
    from mootation.benchmarks.registry import PROBLEMS
    random.seed(7)
    broken = []
    # Straight off PROBLEMS rather than through get(): get() also resolves the
    # reference frame, which samples a Pareto front per problem and would turn
    # a two-second test into a two-minute one. Evaluating needs neither ideal
    # nor nadir.
    for nm, p in PROBLEMS.items():
        x = [lo + random.random() * (hi - lo) for lo, hi in p.bounds]
        try:
            f = p.evaluate(x)
        except Exception as e:
            broken.append((nm, f"{type(e).__name__}: {e}")); continue
        # a failing copy (benchmarks/failing.py) returns NaN in its region, by design
        if len(f) != p.n_obj or (any(v != v for v in f) and p.failing is None):
            broken.append((nm, f"bad output {f[:3]}"))
    assert not broken, broken[:5]


@test
def a_benchmark_section_resolves_families_against_the_real_registry():
    if not _have_numpy():
        print("    (skipped: no NumPy)"); return
    text = """
[run]
name = "b"
scratch = "s/{worker:02d}"
ledger = "l.jsonl"

[problem]
kind = "builtin"

[benchmarks]
families = ["DTLZ"]
objectives = [3]

[[algorithms]]
name = "nsga2"
pop = 40
gens = 5
"""
    cfg = loads(text)
    probs = validate(cfg, base=Path(tempfile.gettempdir()))
    assert probs == [], probs
    assert "DTLZ2_3D" in cfg.benchmark_problems, cfg.benchmark_problems


@test
def a_benchmark_name_that_does_not_exist_is_named_with_near_misses():
    if not _have_numpy():
        print("    (skipped: no NumPy)"); return
    # A plausible spelling is DTLZ2_M3. The registry calls it
    # DTLZ2_3D, and a plausible-looking wrong name is exactly what this catches.
    text = """
[run]
name = "b"
scratch = "s/{worker:02d}"
ledger = "l.jsonl"

[problem]
kind = "builtin"

[benchmarks]
problems = ["DTLZ2_M3"]

[[algorithms]]
name = "nsga2"
pop = 40
gens = 5
"""
    probs = validate(loads(text), base=Path(tempfile.gettempdir()))
    assert any("DTLZ2_M3" in p and "DTLZ2_3D" in p for p in probs), probs


@test
def an_objective_count_a_family_does_not_have_lists_the_ones_it_does():
    if not _have_numpy():
        print("    (skipped: no NumPy)"); return
    text = """
[run]
name = "b"
scratch = "s/{worker:02d}"
ledger = "l.jsonl"

[problem]
kind = "builtin"

[benchmarks]
families = ["WFG"]
objectives = [8]

[[algorithms]]
name = "nsga2"
pop = 40
gens = 5
"""
    probs = validate(loads(text), base=Path(tempfile.gettempdir()))
    assert any("8 objectives" in p and "available sizes" in p for p in probs), probs


@test
def importing_the_benchmarks_writes_nothing_to_disk():
    """A library that writes into its own install directory breaks read-only
    installs. The reference frames are computed lazily and never written back."""
    if not _have_numpy():
        print("    (skipped: no NumPy)"); return
    import mootation.benchmarks.registry as R
    before = {p.name for p in Path(R.__file__).parent.iterdir()}
    R.get("DTLZ2_3D"); R.get("WFG4_3D")
    after = {p.name for p in Path(R.__file__).parent.iterdir()}
    assert after - before <= {"__pycache__"}, after - before


# ── persistence: population and evaluation log ──────────────────────────────
# Pure Python and independent of the compiled extension, so these run on any
# machine — which is the point: reading a saved population must not need a
# compiler.

class _FakeResult:
    """Just the fields save_population reads."""
    def __init__(self, X, F, cv=None):
        self.variables = X
        self.objectives = F
        self.cv = cv if cv is not None else [0.0] * len(X)


@test
def a_population_round_trips_through_the_file():
    from mootation.persistence import load_population, save_population
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "sub" / "pop.csv"          # parent is created
        X = [[0.5, 0.25], [0.1, 0.9], [0.75, 0.05]]
        F = [[0.5, 0.7], [0.1, 0.95], [0.75, 0.3]]
        save_population(_FakeResult(X, F), p,
                        meta={"algorithm": "nsga2", "pop_size": 3})
        back = load_population(p)

        assert len(back) == 3
        # Rows are written sorted by the first objective, so compare as sets.
        assert sorted(map(tuple, back.variables)) == sorted(map(tuple, X))
        assert back.meta["algorithm"] == "nsga2"
        assert back.meta["pop_size"] == "3"


@test
def full_double_precision_survives_the_round_trip():
    from mootation.persistence import load_population, save_population
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "pop.csv"
        # A value that a short %g would mangle. Seeding a run with a rounded
        # decision vector would restart from a DIFFERENT point than the one
        # that was evaluated.
        x = [0.1234567890123456789, 1.0 / 3.0, 1e-17]
        save_population(_FakeResult([x], [[1.0, 2.0]]), p)
        back = load_population(p)
        assert back.variables[0] == x, (back.variables[0], x)


@test
def a_multi_word_note_survives_alongside_metadata():
    from mootation.persistence import load_population, save_population
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "pop.csv"
        save_population(_FakeResult([[0.5]], [[1.0]]), p,
                        meta={"algorithm": "nsga2"},
                        note="run of 2026-08-09 on the cluster")
        text = p.read_text(encoding="utf-8")
        assert "run of 2026-08-09 on the cluster" in text
        back = load_population(p)
        # The note must not leak into metadata: the loader splits comment
        # lines on spaces looking for key=value, and a note stored as
        # `note=...` would lose everything after the first space.
        assert back.meta == {"algorithm": "nsga2"}, back.meta


@test
def a_population_written_by_the_cpp_side_loads():
    from mootation.persistence import load_population
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "pop.csv"
        # Byte-for-byte the shape io/population.hpp writes.
        p.write_text(
            "# mootation population v1\n"
            "# a note with spaces\n"
            "# algorithm=moead_de n_gen=250\n"
            "# n_vars=2 n_bin=0 n_objs=2\n"
            "x1,x2,f1,f2,cv\n"
            "1.000000000000e-01,2.000000000000e-01,3.000000000000e-01,4.000000000000e-01,0.000000000000e+00\n",
            encoding="utf-8")
        back = load_population(p)
        assert len(back) == 1
        assert back.variables[0] == [0.1, 0.2]
        assert back.objectives[0] == [0.3, 0.4]
        assert back.meta["algorithm"] == "moead_de"


@test
def fit_population_truncates_pads_and_refuses():
    from mootation.persistence import fit_population, load_population, save_population
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "pop.csv"
        X = [[float(i)] for i in range(5)]
        save_population(_FakeResult(X, [[float(i), 0.0] for i in range(5)]), p)
        pop = load_population(p)

        assert len(fit_population(pop, 3, "truncate")) == 3
        assert len(fit_population(pop, 8, "pad")) == 8
        assert len(fit_population(pop, 5, "error")) == 5
        raises(ValueError, fit_population, pop, 8, "error")

        # Padding cycles, so every padded row is one of the originals.
        padded = fit_population(pop, 8, "pad")
        originals = {tuple(v) for v in pop.variables}
        assert all(tuple(v) in originals for v in padded.variables)


@test
def a_corrupt_row_is_named_rather_than_silently_dropped():
    from mootation.persistence import load_population
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "pop.csv"
        p.write_text("# mootation population v1\n"
                     "# n_vars=1 n_bin=0 n_objs=1\n"
                     "x1,f1,cv\n"
                     "0.5,not_a_number,0.0\n", encoding="utf-8")
        e = raises(ValueError, load_population, p)
        assert "not a number" in str(e), e


@test
def the_evaluation_log_records_every_call_including_repeats():
    from mootation.persistence import EvaluationLog
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "evals.csv"
        calls = []

        def f(x):
            calls.append(list(x))
            return [x[0], 1.0 - x[0]]

        with EvaluationLog(p, meta={"algorithm": "nsga2"}) as log:
            wrapped = log.wrap(f)
            wrapped([0.5])
            wrapped([0.5])          # the same point again: still recorded
            wrapped([0.25])

        rows = [ln for ln in p.read_text(encoding="utf-8").splitlines()
                if ln and not ln.startswith("#")]
        assert len(rows) == 4, rows          # header + 3 evaluations
        assert rows[0].startswith("eval,")
        assert len(calls) == 3
        assert "algorithm=nsga2" in p.read_text(encoding="utf-8")


@test
def the_evaluation_log_writes_nothing_until_it_is_used():
    from mootation.persistence import EvaluationLog
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "never" / "evals.csv"
        log = EvaluationLog(p)
        log.close()
        # Off means absent: constructing the object must not create a file or
        # a directory, or "disabled logging" would still litter the disk.
        assert not p.exists() and not p.parent.exists()


# ── the shipped examples ────────────────────────────────────────────────────

@test
def the_demo_config_validates_on_any_machine_with_python():
    from mootation.run.config import load
    demo = HERE / "examples" / "demo.toml"
    assert demo.is_file(), demo
    cfg = load(demo)
    # No explicit base: every relative path in a config resolves against the
    # config file's own directory. One rule for templates, step commands and
    # warm-start files alike — a config with two bases is a config nobody can
    # move.
    probs = validate(cfg)
    assert probs == [], probs


# ── campaign ────────────────────────────────────────────────────────────────

_CAMP = """
[run]
name = "camp"

[problem]
kind = "builtin"

[benchmarks]
problems = ["ZDT1", "DTLZ2_3D"]
runs = 3

[campaign]
budget_fe = 2000
record_every = 2
metrics = ["igd", "hv"]

[[algorithms]]
name = "nsga2"
pop = 0
gens = 0

[[algorithms]]
name = "moead_m2m"
pop = 0
gens = 0

[[algorithms]]
name = "nsga3"
pop = 0
gens = 0
"""


@test
def campaign_expands_problems_algorithms_seeds():
    try:
        import numpy  # noqa: F401
    except ImportError:
        return
    from mootation.run import campaign as C
    cfg = loads(_CAMP)
    assert validate(cfg) == [], validate(cfg)
    spec = C.campaign_spec(cfg)
    jobs = C.expand_jobs(cfg, spec)
    assert len(jobs) == 2 * 3 * 3, len(jobs)
    assert [j.index for j in jobs] == list(range(len(jobs)))
    assert {j.seed for j in jobs} == {1, 2, 3}
    z = [j for j in jobs if j.problem == "ZDT1" and j.algorithm == "nsga2"][0]
    assert z.pop == 100 and z.gens == 20, (z.pop, z.gens)          # ceil(2000/100)
    m2m = [j for j in jobs if j.problem == "DTLZ2_3D" and j.algorithm == "moead_m2m"][0]
    assert m2m.pop == 90 and "K=10" in m2m.pop_note, (m2m.pop, m2m.pop_note)
    n3 = [j for j in jobs if j.problem == "DTLZ2_3D" and j.algorithm == "nsga3"][0]
    assert n3.pop == 91 and n3.pop_note == "", (n3.pop, n3.pop_note)


@test
def campaign_fit_pop_rounds_to_what_the_core_accepts():
    from mootation.run.campaign import fit_pop
    assert fit_pop("nsga3", 91, 3, {}) == (91, "")
    assert fit_pop("nsga3", 100, 3, {}) == (100, "")      # two-layer 55 + 45 at M=3
    assert fit_pop("nsga3", 92, 3, {})[0] == 91            # largest lattice size <= 92 at M=3
    assert fit_pop("nsga3", 275, 10, {}) == (275, "")      # two-layer 220 + 55
    assert fit_pop("moead_m2m", 91, 3, {})[0] == 90
    assert fit_pop("moead_m2m", 90, 3, {"K": 7})[0] == 84
    assert fit_pop("nsga2", 77, 3, {}) == (77, "")
    # pop_fit = "nearest" (task 5, run B1): 21 rather than 6 at six objectives
    assert fit_pop("nsga3", 20, 6, {})[0] == 6 and fit_pop("nsga3", 20, 6, {}, "nearest")[0] == 21
    assert fit_pop("nsga3", 50, 6, {}, "nearest")[0] == 56
    assert fit_pop("moead", 20, 3, {}, "nearest")[0] == 21
    assert fit_pop("nsga3", 10, 5, {}, "nearest")[0] == 5            # 5 and 15 as near
    assert fit_pop("moead_m2m", 97, 3, {}, "nearest")[0] == 100


@test
def campaign_rejects_unknown_keys_and_metrics():
    from mootation.run import campaign as C
    from mootation.run.config import ConfigError
    cfg = loads(_CAMP.replace('metrics = ["igd", "hv"]', 'metrics = ["igd", "gd"]'))
    raises(ConfigError, C.campaign_spec, cfg)
    cfg = loads(_CAMP.replace("budget_fe = 2000", "budget_fe = 2000" + chr(10) + "bogus = 1"))
    raises(ConfigError, C.campaign_spec, cfg)
    nl = chr(10)
    cfg = loads(_CAMP.replace('metrics = ["igd", "hv"]',
                              'metrics = ["igd", "hv"]' + nl + 'final_metrics = ["eps", "zzz"]'))
    raises(ConfigError, C.campaign_spec, cfg)
    spec = C.campaign_spec(loads(_CAMP.replace(
        'metrics = ["igd", "hv"]',
        'metrics = ["igd"]' + nl + 'final_metrics = ["igdp_norm", "eps", "hv_h"]')))
    assert spec.metrics == ("igd",) and spec.final_metrics == ("igdp_norm", "eps", "hv_h"), spec


_LADDER = """
[run]
name = "ladder"

[problem]
kind = "builtin"

[benchmarks]
problems = ["DTLZ2_3D"]
runs = 2

[campaign]
budget_fe = 25000
record_grid = "log"
record_at = [10000, 2500, 5000]
ladder = [2500, 10000, 30000]

[[algorithms]]
name = "nsga2"
pop = 0
gens = 0

[[algorithms]]
name = "rvea"
pop = 0
gens = 0
"""


@test
def campaign_record_at_and_ladder_expand_as_documented():
    """Task 3, item 3: exact ladder counts in the log grid, and the rungs of the
    budget-dependent algorithms as separate jobs after all the others."""
    if not _have_numpy():
        print("    (skipped: no NumPy)"); return
    from mootation.run import campaign as C
    from mootation.run.budget import BUDGET_DEPENDENT
    assert "rvea" in BUDGET_DEPENDENT and "nsga2" not in BUDGET_DEPENDENT
    cfg = loads(_LADDER)
    assert validate(cfg) == [], validate(cfg)
    spec = C.campaign_spec(cfg)
    assert spec.record_at == (2500, 5000, 10000) and spec.ladder == (2500, 10000, 30000), spec
    jobs = C.expand_jobs(cfg, spec)
    assert [j.index for j in jobs] == list(range(len(jobs)))
    full, rungs = jobs[:4], jobs[4:]
    assert all(not j.ladder_of for j in full) and len(rungs) == 4, [j.key for j in jobs]
    assert [(j.key, j.evaluations, j.seed) for j in rungs] == [
        ("rvea@2500", 2500, 1), ("rvea@2500", 2500, 2),
        ("rvea@10000", 10000, 1), ("rvea@10000", 10000, 2)]      # 30000 > the full budget
    assert all(j.ladder_of == "rvea" and j.pop == 91 and j.budget == j.evaluations for j in rungs)
    assert rungs[0].rel_dir.as_posix() == "DTLZ2_3D/rvea@2500/run_1"
    nl = chr(10)
    for old, bad in (("record_at = [10000, 2500, 5000]", "record_at = [0]"),
                     ("record_at = [10000, 2500, 5000]", "record_at = 2500"),
                     ("ladder = [2500, 10000, 30000]", 'ladder = ["x"]')):
        raises(ConfigError, C.campaign_spec, loads(_LADDER.replace(old, bad)))
    raises(ConfigError, C.campaign_spec, loads(_LADDER.replace('record_grid = "log"' + nl, "")))
    # without a ladder the job list is the one it always was
    plain = loads(_LADDER.replace("ladder = [2500, 10000, 30000]" + nl, ""))
    assert validate(plain) == []
    assert len(C.expand_jobs(plain, C.campaign_spec(plain))) == 4


@test
def campaign_workers_get_one_blas_thread():
    """single_threaded_blas: 1 where unset, a user's value kept."""
    import os
    from mootation.run import campaign as C
    names = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
    saved = {v: os.environ.pop(v, None) for v in names}
    try:
        os.environ["MKL_NUM_THREADS"] = "4"
        C.single_threaded_blas()
        assert [os.environ[v] for v in names] == ["1", "1", "4"]
    finally:
        for v, old in saved.items():
            if old is None:
                os.environ.pop(v, None)
            else:
                os.environ[v] = old


@test
def ladder_rungs_stand_in_for_budget_dependent_runs_at_a_fraction():
    """--at 0.4 of 25 000 reads rvea's 10 000-evaluation rung, nsga2's trajectory."""
    from mootation.run import campaign as C
    rows = [
        {"problem": "P", "algorithm": "rvea", "seed": 1, "status": "done", "budget_fe": 25000,
         "final": {"igd": 0.1}, "ladder_of": None, "dir": "."},
        {"problem": "P", "algorithm": "nsga2", "seed": 1, "status": "done", "budget_fe": 25000,
         "final": {"igd": 0.2}, "ladder_of": None, "dir": "."},
        {"problem": "P", "algorithm": "rvea@10000", "seed": 1, "status": "done",
         "budget_fe": 10000, "final": {"igd": 0.5}, "ladder_of": "rvea", "dir": "."},
    ]
    assert [r["algorithm"] for r in C.main_rows(rows)] == ["rvea", "nsga2"]
    at = C.rungs_in_place(rows, 0.4)
    assert [r["algorithm"] for r in at] == ["rvea", "nsga2"]
    assert at[0]["final"] == {"igd": 0.5} and at[0]["rung_of"] == 25000
    assert C.value_at(at[0], "igd", 0.4) == 0.5 and at[1] is rows[1]
    assert C.rungs_in_place(rows, None) == C.main_rows(rows)
    assert C.rungs_in_place(rows, 0.2)[0] is rows[0]                 # no rung of 5000


@test
def the_tables_and_cover_read_a_budget_alike():
    """--at and --cover read a run at b evaluations the same way (cover.read_at):
    the first record at or after b of the budget asked for — 2 548, not 2 002,
    for "2 500" at a population of 91, whose budget_fe is 25 025; a run that
    stopped before b (DMS) at its end; a rung found by its nominal budget
    though it spent more; and in the archive scenario the checkpoint of b, also
    through report._value, which used to ignore --at there."""
    from mootation.run import campaign as C
    from mootation.run import cover as V
    from mootation.run import report as R
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        traj = {"nsga2": [(2002, 0.9), (2548, 0.5), (5005, 0.3), (10010, 0.2), (25025, 0.1)],
                "dms": [(90, 0.8), (540, 0.4)]}
        spent = {"nsga2": 25025, "dms": 630}
        for alg, recs in traj.items():
            d = root / "DTLZ2_3D" / alg / "run_1"
            d.mkdir(parents=True)
            (d / "trajectory.jsonl").write_text(
                "".join(json.dumps({"fe": fe, "igdp_norm": v}) + "\n" for fe, v in recs),
                encoding="utf-8")
            final = {"igdp_norm": recs[-1][1] if alg == "nsga2" else 0.35}
            (d / "meta.json").write_text(json.dumps(
                {"problem": "DTLZ2_3D", "algorithm": alg, "seed": 1, "status": "done",
                 "final": final, "final_archive": {"igdp_norm": final["igdp_norm"] / 2},
                 "archive_at": {"2500": {"igdp_norm": 0.25}} if alg == "nsga2" else {},
                 "budget_nominal": 25000, "budget_fe": 25025, "fe": spent[alg], "pop": 91,
                 "has_reference_front": True}), encoding="utf-8")
        d = root / "DTLZ2_3D" / "rvea@2500" / "run_1"              # a rung that spent 2 548
        d.mkdir(parents=True)
        (d / "meta.json").write_text(json.dumps(
            {"problem": "DTLZ2_3D", "algorithm": "rvea@2500", "seed": 1, "status": "done",
             "final": {"igdp_norm": 0.6}, "ladder_of": "rvea", "budget_nominal": 2500,
             "budget_fe": 2548, "fe": 2548}), encoding="utf-8")
        d = root / "DTLZ2_3D" / "rvea" / "run_1"
        d.mkdir(parents=True)
        (d / "meta.json").write_text(json.dumps(
            {"problem": "DTLZ2_3D", "algorithm": "rvea", "seed": 1, "status": "done",
             "final": {"igdp_norm": 0.05}, "budget_dependent": True, "budget_nominal": 25000,
             "budget_fe": 25025, "fe": 25025}), encoding="utf-8")
        rows = C.rungs_in_place(C.scan_results(root), 0.1)
        at = {r["algorithm"]: C.value_at(r, "igdp_norm", 0.1) for r in rows}
        assert at == {"nsga2": 0.5, "dms": 0.35, "rvea": 0.6}, at
        data = V.collect(root, (2500, 25000))
        assert {k[1]: v[2500]["igdp_norm"] for k, v in data["values"].items()} == at
        assert data["values"][("DTLZ2_3D", "nsga2", 1)][25000]["igdp_norm"] == 0.1
        arch = C.rungs_in_place(C.scenario_rows(C.scan_results(root), "archive"), 0.1)
        got = {r["algorithm"]: C.value_at(r, "igdp_norm", 0.1) for r in arch}
        assert got == {"nsga2": 0.25, "dms": 0.175, "rvea": None}, got
        raw = next(r for r in C.scan_results(root) if r["algorithm"] == "nsga2")
        assert R._value(raw, "igdp_norm", 0.1, "archive") == 0.25
        assert V.read_at(30000, 25000, 25025, {"igdp_norm": 1.0})["igdp_norm"] is None


@test
def cover_reads_the_front_flag_from_any_run():
    """collect takes whether a problem has a reference front from the first run
    that says so, in whatever order the file system lists the runs: a run
    without the flag listed first (macOS, CI #93) made it ask the registry,
    which needs NumPy."""
    from mootation.run import cover as V
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        for alg, flag in (("a_old", None), ("b_new", True)):   # listed in this order here
            d = root / "DTLZ2_3D" / alg / "run_1"
            d.mkdir(parents=True)
            meta = {"problem": "DTLZ2_3D", "algorithm": alg, "seed": 1, "status": "done",
                    "final": {"igdp_norm": 0.1}, "budget_nominal": 100, "budget_fe": 100,
                    "fe": 100, "pop": 10}
            if flag is not None:
                meta["has_reference_front"] = flag
            (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
        assert V.collect(root, (100,))["front"] == {"DTLZ2_3D": True}


@test
def cover_finds_the_smallest_sets_exactly_where_greedy_does_not():
    """cover.py on the textbook case: greedy takes 3 sets where 2 suffice."""
    from mootation.run import cover as V
    P = [f"p{i:02d}" for i in range(1, 16)]                         # p15: nobody covers it
    fam = lambda p: "X" if int(p[1:]) <= 7 else "Y"                # noqa: E731
    cov = {"A": set(P[0:7]), "B": set(P[7:14]),
           "C": {P[i] for i in (0, 1, 2, 3, 7, 8, 9, 10)},
           "D": {P[i] for i in (4, 5, 11, 12)}, "E": {P[6], P[13]}}
    sets = V._masks(cov, P)
    universe = 0
    for m in sets.values():
        universe |= m
    assert V.greedy_cover(sets, universe) == ["C", "D", "E"]
    size, sets_found, exact = V.smallest_covers(sets, universe)
    assert (size, sets_found, exact) == (2, [("A", "B")], True), (size, sets_found)
    res = V.analyse(cov, P, fam)
    assert res["nobody"] == ["p15"] and res["n_covered"] == 14 and res["size"] == 2
    c1, c2 = res["curve"]
    assert c1["set"] == ["C"] and abs(c1["share"] - 8 / 14) < 1e-12, c1
    assert c2["set"] == ["A", "B"] and c2["share"] == 1.0 and c2["exact"], c2
    assert c2["per_family"] == {"X": 7, "Y": 7}
    # several smallest sets, all listed
    two = {"a": {1, 2}, "b": {3, 4}, "c": {1, 3}, "d": {2, 4}}
    m2 = V._masks(two, [1, 2, 3, 4])
    assert V.smallest_covers(m2, 0b1111)[1] == [("a", "b"), ("c", "d")]
    assert V.smallest_covers(m2, 0b1111, max_sets=1)[1] == [("a", "b")]


@test
def cover_reads_levels_seeds_and_the_hypervolume_gap():
    """igdp_norm <= (1 + tau) x floor and gdp_norm <= tau with a front, the gap to
    the best hv_h without one, and >= min_seeds seeds for a success."""
    if not _have_numpy():
        print("    (skipped: no NumPy)"); return
    from mootation.run import cover as V
    vals = {}
    for s in range(1, 11):
        vals[("F", "a", s)] = {100: {"igdp_norm": 0.005 if s <= 7 else 0.5, "gdp_norm": 0.001}}
        vals[("F", "b", s)] = {100: {"igdp_norm": 0.005 if s <= 6 else 0.5, "gdp_norm": 0.001}}
        vals[("H", "a", s)] = {100: {"hv_h": 0.995}}
        vals[("H", "b", s)] = {100: {"hv_h": 0.90}}
    data = {"values": vals, "front": {"F": True, "H": False}, "main": [
        {"problem": "H", "algorithm": "b", "seed": 1, "final": {"hv_h": 1.0}}],
        "pop": {("F", "a"): 100, ("F", "b"): 100}}
    best = V.best_known_hv(data)
    assert best == {"H": (1.0, "b", 1)}
    floor = {("F", 100): 0.004}                                  # per problem and population
    hit = V.reached(data, "igdp", (0.5, 0.01), 100, best, floor)   # igdp_norm <= 0.006
    cov = V.successes(hit, 7)
    assert cov == {"a": {"F", "H"}, "b": set()}, cov              # b: 6 seeds; gap 0.1 > 0.01
    assert V.successes(V.reached(data, "igdp", (0.1, 0.01), 100, best, floor), 7)["a"] == {"H"}
    assert V.successes(V.reached(data, "gdp", (0.01, 0.01), 100, best), 7) == {"a": {"F", "H"},
                                                                               "b": {"F"}}
    assert V.successes(hit, 7, seeds=[1] * 10)["b"] == {"F"}       # a resample of seed 1 only


@test
def a_gdp_level_needs_half_an_answer_and_one_run_each_reads_the_seeds_alone():
    """Task 5: a gdp level counts only an answer of at least N/2 points; an
    algorithm whose seeds are all alike is found; 'one run each' covers with
    one seed at a time."""
    if not _have_numpy():
        print("    (skipped: no NumPy)"); return
    from mootation.run import cover as V
    vals = {}
    for s in range(1, 11):
        # d: every seed alike, on F two points of 100 at GD+ 0 (DMS on BT6-BT8)
        vals[("F", "d", s)] = {100: {"gdp_norm": 0.0, "igdp_norm": 0.5, "n": 2}}
        vals[("F", "a", s)] = {100: {"gdp_norm": 0.001, "igdp_norm": 0.5,
                                     "n": 60 if s <= 5 else 40}}
        vals[("G", "d", s)] = {100: {"gdp_norm": 0.001, "igdp_norm": 0.5, "n": 80}}
        vals[("G", "a", s)] = {100: {"gdp_norm": 0.001 if s == 3 else 0.5, "igdp_norm": 0.5,
                                     "n": 80}}
    data = {"values": vals, "front": {"F": True, "G": True}, "main": [],
            "pop": {(p, a): 100 for p in "FG" for a in "da"}}
    assert V.deterministic(vals) == ["d"]
    hit = V.reached(data, "gdp", (0.01, 0.01), 100, {})
    assert not any(hit["F"]["d"].values())                     # two points: not counted
    assert [hit["F"]["a"][s] for s in range(1, 11)] == [True] * 5 + [False] * 5
    assert V.successes(hit, 5) == {"a": {"F"}, "d": {"G"}}
    per = V.one_run(hit, ["F", "G"], list(range(1, 11)), ["d"])
    assert [r[1] for r in per] == [2] * 5 + [1] * 5, per      # covered by someone
    assert [r[2] for r in per] == [2, 2, 1, 2, 2, 1, 1, 1, 1, 1], per   # seed 3: a alone
    assert [r[5]["d"] for r in per] == [1, 1, 0, 1, 1, 1, 1, 1, 1, 1], per
    short = V.short_answers(data, [100])
    assert [(r[0], r[1], r[5], r[6]) for r in short] == [("F", "a", 5, 50), ("F", "d", 10, 2)]
    # the size read at a budget: a final population's n_final, a record's n
    assert V.read_at(100, 100, 100, {"gdp_norm": 0.1, "n_final": 7})["n"] == 7
    assert V.read_at(50, 100, 100, {}, [{"fe": 60, "gdp_norm": 0.2, "n": 9}])["n"] == 9


@test
def the_igdp_floor_is_what_n_points_on_the_front_leave():
    """The floor of DTLZ2: N points DSS takes from the reference itself leave
    igdp_norm above 0.01 at three objectives and above 0.03 at five (the
    absolute levels those populations could never reach), and the floors are
    written once and read back."""
    try:
        import numpy as np  # noqa: F401
    except ImportError:
        return
    from mootation.run import cover as V
    f3, f5 = V.igdp_floors("DTLZ2_3D", 1000, [91]), V.igdp_floors("DTLZ2_5D", 1000, [126])
    (g3, d3), (g5, d5) = f3[91], f5[126]
    # the greedy subset leaves less than DSS's: DSS picks for coverage, not IGD+
    assert 0.01 < g3 < d3 < 0.05 and 0.03 < g5 < d5 < 0.12, (f3, f5)
    assert (f3["rows"], f3["distinct"]) == (1000, 1000), f3
    assert V.igdp_floors("bbobbiobj01_n05_2D", 1000, [100]) is None
    # fewer points (an M2M core's multiple of K) leave a higher floor
    assert V.igdp_floors("DTLZ2_3D", 1000, [90])[90][0] >= g3
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "igdp_floor.csv"
        got = V.floors({"DTLZ2_3D": {91}, "bbobbiobj01_n05_2D": {100}}, 1000, path)
        assert got == {("DTLZ2_3D", 91): {"greedy": g3, "dss": d3, "rows": 1000,
                                          "distinct": 1000}} and path.is_file(), got
        path.write_text(path.read_text(encoding="utf-8").replace(f"{g3:.10g}", "0.5"),
                        encoding="utf-8")
        again = V.floors({"DTLZ2_3D": {91}}, 1000, path)            # read, not recomputed
        assert again[("DTLZ2_3D", 91)]["greedy"] == 0.5, again


@test
def cover_sets_aside_the_problems_with_a_doubtful_reference():
    """DTLZ5, DTLZ6 and MaF6 from four objectives, WFG3, DTLZ1 at five objectives;
    the same families elsewhere, and the rest, are kept."""
    if not _have_numpy():
        print("    (skipped: no NumPy)"); return
    from mootation.run import cover as V
    marked = ["DTLZ5_5D", "DTLZ6_4D", "MaF6_8D", "WFG3_3D", "WFG3_5D", "DTLZ1_5D"]
    kept = ["DTLZ5_3D", "DTLZ6_3D", "DTLZ1_3D", "WFG4_5D", "DTLZ2_5D", "ZDT1", "BT9",
            "bbobbiobj01_n05_2D", "IPolygon_3D", "WFG1_5D", "WFG2_5D", "WFG1_3D"]
    assert all(V.reference_caveat(p) for p in marked), [p for p in marked
                                                        if not V.reference_caveat(p)]
    assert not any(V.reference_caveat(p) for p in kept)
    # against references of version 1: IPolygon, and WFG1/2 where x1's grid aliased
    old = ["IPolygon_3D", "IPolygon_4D", "WFG1_5D", "WFG2_5D"]
    assert all(V.reference_caveat(p, 1) for p in old) and not V.reference_caveat("WFG1_3D", 1)
    # marked, not left out
    lattice = "Das-Dennis lattice topped up with Dirichlet points, coarse from four objectives"
    assert V.reference_marks("DTLZ2_5D", 1000, 1000, 1000) == [lattice]
    assert V.reference_marks("DTLZ2_3D", 1000, 1000, 1000) == []
    assert V.reference_marks("Polygon_4D", 1000, 1000, 1000) == ["front of dimension 2"]
    assert V.reference_marks("DTLZ5_3D", 1000, 1000, 1000) == ["front of dimension 1"]
    assert V.reference_marks("ZDT1", 259, 228, 1000) == ["259 reference rows, not 1000",
                                                         "31 repeated reference rows"]


@test
def the_floor_review_reads_the_best_run_median_and_deciding_seed():
    """floor_review: the best run, the best algorithm's median and the smallest
    min_seeds-th best seed, each against the floor."""
    if not _have_numpy():
        print("    (skipped: no NumPy)"); return
    from mootation.run import cover as V
    main = []
    for s_ in range(1, 11):
        main.append({"problem": "DTLZ2_3D", "algorithm": "a", "seed": s_,
                     "final": {"igdp_norm": 0.02 + 0.001 * s_, "n_final": 91}})
        main.append({"problem": "DTLZ2_3D", "algorithm": "b", "seed": s_,
                     "final": {"igdp_norm": 0.015 if s_ == 1 else 0.05, "n_final": 91}})
    fl = {("DTLZ2_3D", 91): {"greedy": 0.025, "dss": 0.03, "rows": 1000, "distinct": 1000}}
    rows = V.floor_review({"main": main, "scenario": "final"}, fl, 7)
    r = rows[0]
    assert (r["best"], r["best_algorithm"], r["best_seed"]) == (0.015, "b", 1), r
    assert abs(r["best_ratio"] - 0.6) < 1e-12 and r["median_algorithm"] == "a", r
    assert abs(r["kth"] - 0.027) < 1e-12 and r["kth_algorithm"] == "a", r   # a's 7th best seed
    assert r["n_obj"] == 3 and r["pop"] == 91 and r["caveat"] == "", r
    # the population reads the greedy floor; DSS's tax and eta = ratio^(-d) beside it
    assert r["level_floor"] == "greedy" and abs(r["dss_tax"] - 1.2) < 1e-12, r
    assert r["front_dim"] == 2 and abs(r["best_eta"] - 0.6 ** -2) < 1e-9, r
    # the archive scenario, itself a DSS selection, reads the DSS floor
    main_a = [dict(m, final_archive=m["final"]) for m in main]
    ra = V.floor_review({"main": main_a, "scenario": "archive"}, fl, 7)[0]
    assert ra["level_floor"] == "dss" and abs(ra["best_ratio"] - 0.5) < 1e-12, ra


@test
def the_igdp_floor_is_deterministic():
    """The same floor in two calls and in a worker process: the reference is
    sampled from fixed seeds and DSS breaks ties by index, never at random."""
    try:
        import numpy as np  # noqa: F401
    except ImportError:
        return
    from mootation.run import cover as V
    a = V.igdp_floors("DTLZ2_3D", 1000, [91])
    assert a == V.igdp_floors("DTLZ2_3D", 1000, [91])
    with tempfile.TemporaryDirectory() as td:
        got = V.floors({"DTLZ2_3D": {91}, "ZDT1": {100}}, 1000, Path(td) / "f.csv", workers=2)
    z = V.igdp_floors("ZDT1", 1000, [100])
    for key, want in ((("DTLZ2_3D", 91), a[91]), (("ZDT1", 100), z[100])):
        assert (got[key]["greedy"], got[key]["dss"]) == want, (key, got[key], want)


@test
def reference_fronts_of_version_2_have_n_distinct_rows():
    """Reference version 2: no repeated rows and exactly n of them, the front's
    range kept (WFG1 and WFG2 at five objectives no longer the same 1 296
    points, one vertex 216 times; WFG2_3D no longer 259 rows, MOP4 no longer
    6 399), a front that had nothing to fix unchanged."""
    if not _have_numpy():
        print("    (skipped: no NumPy)"); return
    import numpy as np
    from mootation.benchmarks import get
    from mootation.benchmarks.registry import REFERENCE_VERSION
    assert REFERENCE_VERSION >= 2
    for name in ("WFG1_5D", "WFG2_5D", "WFG2_3D", "DTLZ7_3D", "BT5", "MOP4", "MOP7", "ZDT3",
                 "DTLZ2_3D"):
        p = get(name)
        R = np.asarray(p.pareto_front(1000), float)
        assert len(R) == 1000 == len(np.unique(np.round(R, 12), axis=0)), (name, len(R))
        span = np.asarray(p.nadir) - np.asarray(p.ideal)
        assert np.all(R.min(0) >= np.asarray(p.ideal) - 1e-3 * span), name
        assert np.all(R.max(0) <= np.asarray(p.nadir) + 1e-3 * span), name
        assert np.allclose(R.max(0), np.asarray(p.nadir), atol=1e-3 * span.max()), name
    assert not np.allclose(get("WFG1_5D").pareto_front(1000), get("WFG2_5D").pareto_front(1000))
    assert np.array_equal(get("DTLZ2_3D").pareto_front(1000), get("DTLZ2_3D").pareto_front(1000))


@test
def the_ipolygon_polygons_are_far_enough_apart():
    """IPolygon of reference version 2: four polygons, every point inside one
    non-dominated by the points of a grid of [0, 100]^2, the reference and the
    Pareto set too, and the frame exact (0 at a vertex, the longest chord)."""
    if not _have_numpy():
        print("    (skipped: no NumPy)"); return
    import numpy as np
    from mootation.benchmarks import get
    from mootation.benchmarks import polygon_ishibuchi as ip
    assert ip.N_POLY == 4
    g = np.linspace(0.0, 100.0, 201)
    G = np.array([[a, b] for a in g for b in g])
    for M in (3, 4):
        p = get(f"IPolygon_{M}D")
        V = ip._vertices(M)
        F = np.array([p.evaluate(list(x)) for x in G])
        inside = np.zeros(len(G), bool)
        for poly in V:
            inside |= ip._inside(G, poly)
        R = np.asarray(p.pareto_front(300), float)
        S = np.array([p.evaluate(list(x)) for x in p.pareto_set(300)])
        for Q in (F[inside], R, S):
            beaten = [np.any(np.all(F <= q + 1e-9, axis=1) & np.any(F < q - 1e-9, axis=1))
                      for q in Q]
            assert not any(beaten), (M, sum(beaten))
        chord = np.linalg.norm(V[0][:, None] - V[0][None], axis=2).max()
        assert np.allclose(p.ideal, 0.0) and np.allclose(p.nadir, chord), (p.ideal, p.nadir)


@test
def recompute_reference_measures_old_runs_again_and_keeps_their_values():
    """--recompute-reference: a run measured against an older reference of a
    problem whose reference changed gets every reference-dependent value again
    — final, archive, archive checkpoints, trajectory — with the old ones kept
    (meta["reference_v1"], trajectory.ref_v1.jsonl); a problem whose
    definition changed is only counted, to be run again; --cover sets a
    problem with runs measured against the old reference aside."""
    if not _have_numpy() or _core_with("on_generation") is None:
        print("  skip  recompute_reference...: no NumPy or stale _core"); return
    from mootation.benchmarks import registry as REG
    from mootation.run import campaign as C
    from mootation.run import cover as V
    from mootation.run.config import load
    text = """algorithms = [
    { name = "nsga2", pop = 0, gens = 0 },
]
[run]
name = "rr"
[problem]
kind = "builtin"
[benchmarks]
runs = 1
problems = ["ZDT1", "DTLZ2_3D"]
[campaign]
out = "res"
budget_fe = 1500
record_grid = "log"
record_at = [500, 1000, 1500]
metrics = ["igdp_norm", "gdp_norm", "hv_h"]
archive_variables = "selected"
archive_checkpoints = true
snapshots = true
"""
    saved = (REG.REFERENCE_CHANGED, REG.PROBLEM_CHANGED)
    try:
        with tempfile.TemporaryDirectory() as td:
            cfg_path = Path(td) / "c.toml"
            cfg_path.write_text(text, encoding="utf-8")
            cfg = load(cfg_path)
            assert validate(cfg) == [], validate(cfg)
            spec = C.campaign_spec(cfg)
            root = C.out_root(cfg, spec)
            for job in C.expand_jobs(cfg, spec):
                assert C.run_job(job, root, spec, quiet=True) == "done", job
            d = root / "ZDT1" / "nsga2" / "run_1"
            meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
            assert meta["reference_version"] == REG.REFERENCE_VERSION and \
                meta["reference_rows"] == 1000, meta
            T = [json.loads(l) for l in (d / "trajectory.jsonl").read_text().splitlines()]
            # as if measured against an older reference: version 1, values off
            for run in (root / "ZDT1" / "nsga2" / "run_1", root / "DTLZ2_3D" / "nsga2" / "run_1"):
                m = json.loads((run / "meta.json").read_text(encoding="utf-8"))
                m["reference_version"] = 1
                m["final"]["igdp_norm"] *= 2
                m["final_archive"]["igdp_norm"] *= 2
                m["archive_at"]["500"]["igdp_norm"] *= 2
                (run / "meta.json").write_text(json.dumps(m), encoding="utf-8")
                recs = [json.loads(l) for l in (run / "trajectory.jsonl").read_text().splitlines()]
                for rec in recs:
                    rec["igdp_norm"] *= 2
                (run / "trajectory.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs),
                                                      encoding="utf-8")
            REG.REFERENCE_CHANGED = {2: frozenset({"ZDT1", "DTLZ2_3D"})}
            REG.PROBLEM_CHANGED = {2: frozenset({"DTLZ2_3D"})}
            out = V.run(root, taus=(10.0,), floor_taus=(1e6,), budgets=(1500,), min_seeds=1)
            assert "ZDT1" in out and "measured against reference v1" in \
                (root / "_cover" / "cover_excluded.csv").read_text(encoding="utf-8"), out
            assert C.recompute_reference(root) == {"to_run_again": 1, "done": 1}
            m = json.loads((d / "meta.json").read_text(encoding="utf-8"))
            assert m["reference_version"] == REG.REFERENCE_VERSION, m
            assert abs(m["final"]["igdp_norm"] - meta["final"]["igdp_norm"]) <= \
                1e-8 * meta["final"]["igdp_norm"], (m["final"], meta["final"])
            assert m["reference_v1"]["final"]["igdp_norm"] == 2 * meta["final"]["igdp_norm"]
            for got, want in ((m["final_archive"], meta["final_archive"]),
                              (m["archive_at"]["500"], meta["archive_at"]["500"])):
                assert abs(got["igdp_norm"] - want["igdp_norm"]) <= 1e-5 * want["igdp_norm"]
            again = [json.loads(l) for l in (d / "trajectory.jsonl").read_text().splitlines()]
            for t, u in zip(T, again):
                assert abs(u["igdp_norm"] - t["igdp_norm"]) <= 1e-5 * max(1.0, t["igdp_norm"])
            old = [json.loads(l) for l in (d / "trajectory.ref_v1.jsonl").read_text().splitlines()]
            assert old[0]["igdp_norm"] == 2 * T[0]["igdp_norm"], (old[0], T[0])
            assert C.recompute_reference(root) == {"to_run_again": 1, "skipped": 1}
    finally:
        REG.REFERENCE_CHANGED, REG.PROBLEM_CHANGED = saved


@test
def portfolio_shapley_oracle_and_complementarity():
    """The Shapley value of the coverage game by its closed form (each problem
    split among the algorithms covering it) against the definition over every
    order of the players; the oracle, the single best solver, what each covers
    alone, and the complementarity of the pairs."""
    import itertools
    from mootation.run import portfolio as P
    cov = {"A": {1, 2, 3}, "B": {3, 4}, "C": {4}}
    phi = P.shapley(cov)
    assert phi == {"A": 2.5, "B": 1.0, "C": 0.5}, phi
    orders = list(itertools.permutations(sorted(cov)))
    for a in cov:
        total = 0.0
        for order in orders:
            before = set().union(*(cov[b] for b in order[:order.index(a)]))
            total += len(before | cov[a]) - len(before)
        assert abs(total / len(orders) - phi[a]) < 1e-12, (a, total / len(orders))
    o = P.oracle(cov)
    assert (o["oracle"], o["sbs"], o["sbs_covers"]) == (4, "A", 3), o
    assert o["unique"] == {"A": {1, 2}, "B": set(), "C": set()}, o["unique"]
    assert P.complementarity(cov) == {("A", "B"): 2, ("A", "C"): 3, ("B", "A"): 1,
                                      ("B", "C"): 1, ("C", "A"): 1}
    # the oracle gap: the single best on average, and its ratio to the best per problem
    sbs, rows = P.oracle_gap({"P1": {"a": 1.0, "b": 2.0}, "P2": {"a": 4.0, "b": 2.0}},
                             "igdp_norm")
    assert sbs == "a" and rows == [("P1", "a", 1.0, 1.0, 1.0), ("P2", "b", 2.0, 4.0, 2.0)], rows
    sbs, rows = P.oracle_gap({"Q": {"a": 0.9, "b": 1.0}}, "hv_h")
    assert sbs == "b" and rows == [("Q", "b", 1.0, 1.0, 0.0)], rows


@test
def portfolio_runs_on_a_results_tree():
    """--portfolio end to end on a made-up campaign: a problem with a front and
    a bbob-biobj one kept apart, the seed rules, and the matrix."""
    if not _have_numpy():
        print("    (skipped: no NumPy)"); return
    from mootation.run import portfolio as P
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        # moead: every seed alike, and a gdp_norm within the level from 30 points of 91
        values = {("DTLZ2_3D", "nsga2"): {"igdp_norm": 0.02, "gdp_norm": 0.01, "eps_norm": 0.05,
                                          "n_final": 91},
                  ("DTLZ2_3D", "moead"): {"igdp_norm": 0.1, "gdp_norm": 0.05, "eps_norm": 0.3,
                                          "n_final": 30},
                  ("bbobbiobj01_n05_2D", "nsga2"): {"hv_h": 0.9},
                  ("bbobbiobj01_n05_2D", "moead"): {"hv_h": 1.0}}
        for (prob, alg), final in values.items():
            for s in range(1, 11):
                d = root / prob / alg / f"run_{s}"
                d.mkdir(parents=True)
                f = dict(final)
                if alg == "nsga2" and prob == "DTLZ2_3D" and s > 6:
                    f["igdp_norm"] = 0.5                    # 6 seeds reach the level, 4 do not
                (d / "meta.json").write_text(json.dumps(
                    {"problem": prob, "algorithm": alg, "seed": s, "status": "done", "final": f,
                     "budget_nominal": 1000, "budget_fe": 1000, "pop": 91 if prob != "bbobbiobj01_n05_2D" else 100,
                     "has_reference_front": prob == "DTLZ2_3D", "reference_version": 2}),
                    encoding="utf-8")
        text = P.run(root, budgets=(1000,), taus=(0.1,), floor_taus=(1.0,))
        # DTLZ2's optimum is at the centre: no front_interior group (task 6)
        assert "1 problems with a front (0 with the optimum neither" in text, text
        assert "1 without" in text, text
        with (root / "_portfolio" / "portfolio_summary.csv").open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        get = lambda g, c, rule: next(r for r in rows if r["group"] == g and r["criterion"] == c  # noqa: E731
                                      and r["seeds_required"] == str(rule))
        assert get("front", "igdp", 5)["oracle_covers"] == "1"      # nsga2: 6 seeds of 10
        assert get("front", "igdp", 7)["oracle_covers"] == "0"
        assert get("front", "gdp", 10)["single_best"] == "nsga2"
        assert get("bbob", "hv", 10)["oracle_covers"] == "1" and \
            get("bbob", "hv", 10)["problems"] == "1"
        with (root / "_portfolio" / "portfolio_matrix.csv").open(encoding="utf-8") as fh:
            m = {(r["problem"], r["algorithm"]): r for r in csv.DictReader(fh)}
        assert len(m) == 4, m
        r = m[("DTLZ2_3D", "nsga2")]
        assert (r["igdp_level0_seeds"], r["gdp_level0_seeds"], r["hv_level0_seeds"]) == \
            ("6", "10", ""), r
        assert float(r["igdp_norm_7th"]) == 0.5 and float(r["igdp_norm_median"]) == 0.02, r
        b = m[("bbobbiobj01_n05_2D", "nsga2")]
        assert b["hv_level0_seeds"] == "10" and b["igdp_level0_seeds"] == "", b
        for name in ("portfolio_shapley.csv", "portfolio_complementarity.csv",
                     "portfolio_oracle_gap.csv", "portfolio_meta.json", "portfolio_report.txt"):
            assert (root / "_portfolio" / name).is_file(), name
        # task 5: 30 of 91 points do not count for gdp; moead's seeds are one run
        s = m[("DTLZ2_3D", "moead")]
        assert (s["gdp_level0_seeds"], s["size_median"], s["seeds_short"]) == ("0", "30", "10"), s
        assert m[("DTLZ2_3D", "nsga2")]["seeds_short"] == "0"
        assert "deterministic, every seed the same run on every problem: moead" in text, text
        assert get("front", "igdp", "one_run")["oracle_covers"] == "0.6"   # 6 seeds of 10 alone
        with (root / "_portfolio" / "portfolio_shapley.csv").open(encoding="utf-8") as fh:
            det = {r["algorithm"]: r["deterministic"] for r in csv.DictReader(fh)}
        assert det == {"nsga2": "False", "moead": "True"}, det


def _have_scipy():
    try:
        import scipy  # noqa: F401
        return True
    except ImportError:
        return False


@test
def instance_space_pilot_reaches_the_optimum():
    """PILOT: BFGS reaches the optimum of min ||F - B A F||^2 + ||Y - C A F||^2,
    whose loss pilot_optimum_loss gives in closed form; the analytical solution
    of the survey's Algorithm 3 is no better, and equals the optimum and the
    two leading principal components of [F; Y] when Y lies in F's row space;
    a repeated feature (F F^T singular) takes the pseudo-inverse."""
    if not _have_numpy() or not _have_scipy():
        print("  skip  instance_space_pilot...: no NumPy or SciPy"); return
    import numpy as np
    from mootation.run import instance_space as S
    rng = np.random.default_rng(3)
    F = rng.standard_normal((4, 40))
    Y = rng.standard_normal((5, 40)) + 0.5 * (rng.standard_normal((5, 4)) @ F)
    opt = S.pilot_optimum_loss(F, Y)
    num = S.pilot_numerical(F, Y, tries=5, seed=1)
    ana = S.pilot_analytic(F, Y)
    assert abs(num["loss"] - opt) <= 1e-6 * opt, (num["loss"], opt)
    assert ana["loss"] >= opt - 1e-9 * opt, (ana["loss"], opt)
    assert np.allclose(num["Z"], (num["A"] @ F).T) and -1.0 <= num["preservation"] <= 1.0
    Y2 = rng.standard_normal((5, 4)) @ F
    X = np.vstack([F, Y2])
    pca = float((X ** 2).sum() - np.sort(np.linalg.eigvalsh(X @ X.T))[-2:].sum())
    for got in (S.pilot_analytic(F, Y2)["loss"], S.pilot_optimum_loss(F, Y2)):
        assert abs(got - pca) <= 1e-8 * pca, (got, pca)
    F3 = np.vstack([F, F[:1]])
    a3 = S.pilot_analytic(F3, Y)
    assert np.isfinite(a3["loss"]) and a3["loss"] >= S.pilot_optimum_loss(F3, Y) - 1e-6


@test
def instance_space_footprints_and_contradictions():
    """The 2017 paper's Algorithm 1 on a grid: the footprint of the left
    columns is their hull, triangles longer than Delta and impure ones go;
    Algorithm 2 takes a small triangle out where a larger one overlaps it, and
    touching triangles do not overlap."""
    if not _have_numpy() or not _have_scipy():
        print("  skip  instance_space_footprints...: no NumPy or SciPy"); return
    import numpy as np
    from mootation.run import instance_space as S
    g = np.linspace(0.0, 1.0, 11)
    Z = np.array([[x, y] for x in g for y in g]) + 1e-4 * np.random.default_rng(0).standard_normal((121, 2))
    dmax = float(np.sqrt(((Z[:, None] - Z[None]) ** 2).sum(axis=2)).max())
    known = S.footprint(Z, np.ones(len(Z), bool), dmax)
    assert abs(known["area"] - 1.0) < 0.01 and known["inside"] == 121, known["area"]
    left = Z[:, 0] < 0.45
    fp = S.footprint(Z, left, dmax, rho=0.0, purity=0.75)
    assert abs(fp["area"] - 0.4) < 0.01 and fp["good_inside"] == left.sum(), fp["area"]
    assert fp["inside"] == fp["good_inside"]            # nothing impure kept
    # five good problems in a corner, far apart against Delta = 25 % of dmax:
    # no triangle survives
    corner = np.zeros(121, bool)
    corner[[0, 10, 110, 120, 60]] = True
    assert S.footprint(Z, corner, dmax)["triangles"] == []
    # a purity limit of 1 takes out every triangle holding a bad problem
    mixed = left.copy()
    mixed[np.argmin(np.hypot(Z[:, 0] - 0.2, Z[:, 1] - 0.5))] = False
    pure = S.footprint(Z, mixed, dmax, purity=1.0)
    assert pure["area"] < fp["area"] and pure["inside"] == pure["good_inside"]
    P = np.array([[0.0, 0.0], [4.0, 0.0], [0.0, 4.0], [1.0, 1.0], [2.0, 1.0], [1.0, 2.0],
                  [4.0, 4.0]])
    none = np.zeros(len(P), bool)
    big, small = S.summarize(P, none, [(0, 1, 2)]), S.summarize(P, none, [(3, 4, 5)])
    assert S.remove_contradictions(P, big, small) == ([(0, 1, 2)], [])
    assert not S.overlap(P[[1, 2, 6]], P[[0, 1, 2]])     # a shared side only
    assert S.overlap(P[[0, 1, 2]], P[[3, 4, 5]])


@test
def instance_space_features_and_performance():
    """The features read off the properties (0.5 where unknown, the bbob-biobj
    groups as counts); PRELIM's relative performance, the best algorithm the
    first by name on a tie, and its Box-Cox and z-score, a constant algorithm
    dropped; SIFTED's first step keeps a feature that tracks a performance."""
    if not _have_numpy() or not _have_scipy():
        print("  skip  instance_space_features...: no NumPy or SciPy"); return
    import math
    import numpy as np
    from mootation.run import instance_space as S
    names = S.feature_names()
    rows = S.feature_rows(["ZDT1", "DTLZ2_3D", "bbobbiobj01_n05_2D"])
    val = lambda n, j: rows[names.index(n)][j]  # noqa: E731
    assert (val("front=convex", 0), val("front=concave", 1), val("front=convex", 2)) == \
        (1.0, 1.0, 0.5), rows
    assert (val("centre", 1), val("scaled", 2), val("bbob=separable", 2)) == (1.0, 0.5, 2.0)
    assert (val("n_obj", 1), val("log2_n_vars", 2), val("front_dim", 1)) == (3.0, math.log2(5), 2.0)
    med = {"P": {"a": 2.0, "b": 1.0, "c": 1.0}, "Q": {"a": 1.0, "b": 3.0}}
    rel, best = S.relative_performance(med, ["P", "Q"], ["a", "b", "c"], higher=False)
    assert best == ["b", "a"] and rel[0] == [1.0, 0.0] and rel[1] == [0.0, 2.0], (rel, best)
    assert rel[2][0] == 0.0 and math.isnan(rel[2][1])
    rel, best = S.relative_performance({"P": {"a": 0.5, "b": 1.0}}, ["P"], ["a", "b"], higher=True)
    assert best == ["b"] and rel == [[0.5], [0.0]], rel
    rng = np.random.default_rng(2)
    perf = [list(rng.uniform(0.0, 2.0, 50)), [0.3] * 50, list(rng.uniform(0.0, 1.0, 50))]
    keep, Y = S.prelim_performance(perf)
    assert keep == [0, 2] and np.allclose(Y.mean(axis=1), 0.0) and np.allclose(Y.std(axis=1), 1.0)
    F = np.vstack([Y[0] + 0.1 * rng.standard_normal(50), rng.standard_normal(50)])
    F = (F - F.mean(axis=1, keepdims=True)) / F.std(axis=1, keepdims=True)
    sel, R = S.sifted(F, Y)
    assert 0 in sel and R[0, 0] > 0.9, (sel, R)


@test
def instance_space_runs_on_a_results_tree():
    """--instance-space end to end on a made-up campaign: the problems with a
    front and the bbob-biobj ones in spaces of their own, the files written."""
    if not _have_numpy() or not _have_scipy():
        print("  skip  instance_space_runs...: no NumPy or SciPy"); return
    from mootation.run import instance_space as S
    front = ["ZDT1", "ZDT2", "ZDT3", "DTLZ2_3D", "WFG4_3D", "MOP1"]
    bbob = ["bbobbiobj01_n05_2D", "bbobbiobj15_n05_2D", "bbobbiobj40_n10_2D",
            "bbobbiobj55_n10_2D"]
    algs = {"a": 1.0, "b": 2.0, "c": 3.0}
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        for i, prob in enumerate(front + bbob):
            for alg, x in algs.items():
                for s in range(1, 11):
                    d = root / prob / alg / f"run_{s}"
                    d.mkdir(parents=True)
                    v = (x + i) % 3 + 1.0 + 0.01 * s     # a different order per problem
                    final = ({"igdp_norm": 0.01 * v, "gdp_norm": 0.001 * v} if prob in front
                             else {"hv_h": 1.0 / v})
                    (d / "meta.json").write_text(json.dumps(
                        {"problem": prob, "algorithm": alg, "seed": s, "status": "done",
                         "final": final, "budget_nominal": 1000, "budget_fe": 1000,
                         "has_reference_front": prob in front, "reference_version": 2}),
                        encoding="utf-8")
        text = S.run(root, budgets=(1000,), taus=(0.1,), floor_taus=(1.0,), n_ref=200, tries=3,
                     bbob_features="properties")
        assert "front: 6 problems" in text and "bbob: 4 problems" in text, text
        out = root / "_instance_space"
        with (out / "is_coordinates.csv").open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert [r["problem"] for r in rows if r["group"] == "front"] == sorted(front)
        assert [r["problem"] for r in rows if r["group"] == "bbob"] == sorted(bbob)
        zdt1 = next(r for r in rows if r["problem"] == "ZDT1")
        assert zdt1["best_algorithm"] == "c" and zdt1["front=convex"] == "1", zdt1
        with (out / "is_projection.csv").open(encoding="utf-8") as fh:
            proj = list(csv.DictReader(fh))
        assert {r["name"] for r in proj if r["kind"] == "performance"} == set(algs)
        meta = json.loads((out / "is_meta.json").read_text(encoding="utf-8"))
        for g in ("front", "bbob"):
            p = meta["groups"][g]["pilot"]
            assert abs(p["loss"] - p["optimum_loss"]) <= 1e-5 * p["optimum_loss"], p
        for name in ("is_performance.csv", "is_footprints.csv", "is_footprints.json",
                     "is_report.txt"):
            assert (out / name).is_file(), name
        with (out / "is_footprints.csv").open(encoding="utf-8") as fh:
            for r in csv.DictReader(fh):                 # the known region lies in the hull
                assert float(r["area_share_hull"]) <= float(r["area_share"]) + 1e-12, r
                if r["area_share_min"]:                  # the main plane is one of the family
                    assert float(r["area_share_min"]) - 1e-12 <= float(r["area_share"])                         <= float(r["area_share_max"]) + 1e-12, r
        assert meta["calibration"] == "whiten" and meta["weight"] == "blocks", meta
        g = meta["groups"]["front"]
        assert g["pilot"]["method"] == "exact" and len(g["planes"]) == 6, g
        assert abs(g["planes"]["whitened"] - g["pilot"]["preservation"]) < 1e-12
        assert g["planes"]["preserving"] >= max(g["planes"].values()) - 1e-12
        # the bbob-biobj problems' features from a sample, the default
        S.run(root, budgets=(1000,), taus=(0.1,), floor_taus=(1.0,), n_ref=200, tries=0,
              out_dir=root / "sampled")
        with (root / "sampled" / "is_coordinates.csv").open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert all(r["nd_share"] != "" for r in rows if r["group"] == "bbob")
        assert all(r["nd_share"] == "" for r in rows if r["group"] == "front")


@test
def instance_space_plane_calibration_and_sample_features():
    """THE PLANE: pilot_exact reaches the optimum's loss, a repeated feature
    included; every calibration of it keeps the loss and the R^2, the whitened
    plane has the identity covariance and the preserving one the highest
    preservation of the family. FEATURES FROM A SAMPLE on made-up samples: a
    linear trade-off is all non-dominated, its objectives opposed and linear in
    x; two equal objectives agree; PRELIM drops a repeated and a constant row."""
    if not _have_numpy() or not _have_scipy():
        print("  skip  instance_space_plane...: no NumPy or SciPy"); return
    import numpy as np
    from mootation.run import instance_space as S
    rng = np.random.default_rng(5)
    F = rng.standard_normal((5, 60))
    F = (F - F.mean(axis=1, keepdims=True)) / F.std(axis=1, keepdims=True)
    Y = rng.standard_normal((7, 60)) + 0.7 * (rng.standard_normal((7, 5)) @ F)
    for FF in (F, np.vstack([F, F[:1]])):
        e = S.pilot_exact(FF, Y)
        opt = S.pilot_optimum_loss(FF, Y)
        assert abs(e["loss"] - opt) <= 1e-9 * opt, (e["loss"], opt)
        assert np.allclose(e["Z"], (e["A"] @ FF).T)
    fam = S.planes(F, Y, S.pilot_exact(F, Y))
    for k, v in fam.items():
        assert abs(v["loss"] - fam["whitened"]["loss"]) <= 1e-9 * v["loss"], k
        assert np.allclose(v["r2_performance"], fam["whitened"]["r2_performance"]), k
    Zw = fam["whitened"]["Z"]
    assert np.allclose(np.cov(Zw.T, bias=True), np.eye(2), atol=1e-9)
    assert fam["preserving"]["preservation"] >= max(v["preservation"] for v in fam.values()) - 1e-12
    assert fam["main"] is fam["whitened"]
    U = rng.random((400, 3))
    f = S.sample_features(U, np.column_stack([U[:, 0], 1.0 - U[:, 0]]))
    assert (f["nd_share"], f["dom_pairs"], f["lnd_share"]) == (1.0, 0.0, 1.0), f
    assert abs(f["rho_mean"] + 1.0) < 1e-12 and f["lin_r2_min"] > 0.999
    assert abs(f["log_range_ratio"]) < 1e-12, f
    g = S.sample_features(U, np.column_stack([U[:, 1], 2.0 * U[:, 1]]))
    assert abs(g["rho_mean"] - 1.0) < 1e-12 and g["nd_share"] == 1 / 400, g
    assert abs(g["log_range_ratio"] - np.log10(2.0)) < 1e-12, g
    keep, Z = S.prelim_features([list(U[:, 0]), list(U[:, 0]), [1.0] * 400, list(U[:, 1] ** 3)])
    assert keep == [0, 3] and np.allclose(Z.mean(axis=1), 0.0) and np.allclose(Z.std(axis=1), 1.0)


@test
def behaviour_reads_the_descriptors_off_a_trajectory():
    """Time to targets, settling, the convergence slope, converged-then-spreading,
    collapse, stagnation at nd_share ~ 1, duplicates and igdx, on a made-up run;
    and on a problem without a front, stagnation from hv_h."""
    import math
    from mootation.run import behaviour as B
    rec = lambda fe, g, q, rc, nd, dup, x: {"fe": fe, "gdp_norm": g, "igdp_norm": q,  # noqa: E731
                                            "range_cover": rc, "nd_share": nd,
                                            "dup_share": dup, "igdx": x, "cr": 1.0}
    recs = [rec(100, 1.0, 0.8, 0.9, 0.3, 0.0, 0.5), rec(200, 0.5, 0.5, 1.0, 0.8, 0.1, 0.4),
            rec(400, 0.25, 0.3, 1.0, 1.0, 0.05, 0.3), rec(800, 0.0101, 0.1, 0.95, 1.0, 0.2, 0.3),
            rec(1600, 0.01, 0.05, 0.7, 1.0, 0.1, 0.29), rec(3200, 0.01, 0.05, 0.7, 1.0, 0.1, 0.29)]
    d = B.run_descriptors(recs, best_igdp=0.045)
    assert (d["t_gdp_0.1"], d["t_gdp_0.03"], d["t_gdp_0.01"], d["t_gdp_0.003"]) == \
        (800, 800, 1600, None), d
    assert (d["t_igdp_1"], d["t_igdp_0.25"], d["t_igdp_0.1"]) == (1600, 1600, None), d
    assert (d["settle_gdp"], d["settle_igdp"], d["settle_igdx"]) == (800, 1600, 400), d
    assert abs(d["slope"] - math.log10(0.0101) / math.log10(8.0)) < 1e-12, d
    assert abs(d["spread"] - math.log10(2.0)) < 1e-12, d
    assert abs(d["rc_drop"] - 0.3) < 1e-12 and d["stall"] == 0.5, d   # nd 1 from 400, settled 1600
    assert (d["dup_end"], d["dup_max"], d["igdx_end"], d["cr_end"]) == (0.1, 0.2, 0.29, 1.0), d
    h = [{"fe": 100, "hv_h": 0.1, "nd_share": 1.0}, {"fe": 200, "hv_h": 0.5, "nd_share": 1.0},
         {"fe": 400, "hv_h": 0.52, "nd_share": 1.0}, {"fe": 800, "hv_h": 0.53, "nd_share": 1.0}]
    e = B.run_descriptors(h, best_igdp=None)
    assert e["stall"] == 0.75 and e["t_gdp_0.1"] is None and e["slope"] is None, e
    assert B.settle_index([1.0, 0.5, 0.5]) == 1 and B.settle_index([3.0]) == 0
    # medians over seeds: a target never reached counts as infinitely late
    runs = {("P", "a", s): dict(B.run_descriptors(recs, 0.045)) for s in (1, 2, 3)}
    runs[("P", "a", 2)]["t_gdp_0.01"] = None
    runs[("P", "a", 3)]["t_gdp_0.01"] = None
    assert B.medians(runs)[("P", "a")]["t_gdp_0.01"] == math.inf
    runs[("P", "a", 3)]["t_gdp_0.01"] = 400
    assert B.medians(runs)[("P", "a")]["t_gdp_0.01"] == 1600


@test
def behaviour_groups_the_algorithms_by_their_profiles():
    """Shares ranked per problem and descriptor, distance as their mean absolute
    difference, average linkage, and the cut into k groups."""
    import math
    from mootation.run import behaviour as B
    names = B.descriptor_names()
    med = {}
    for a, v in (("x", 1.0), ("y", 2.0), ("z", 2.0)):
        med[("P", a)] = dict.fromkeys(names)
        med[("P", a)]["slope"] = v
        med[("P", a)]["t_gdp_0.01"] = math.inf                  # nobody reached it: left out
    sh = B.shares(med)
    assert (sh["x"][("P", "slope")], sh["y"][("P", "slope")], sh["z"][("P", "slope")]) == \
        (0.0, 0.75, 0.75), sh
    assert ("P", "t_gdp_0.01") not in sh["x"], sh
    assert B.family_of()["random_search"] == "Baseline" and B.family_of()["nsga2"] != "Baseline"
    prof = {"a1": {("p", "u"): 0.0, ("p", "v"): 0.1}, "a2": {("p", "u"): 0.05, ("p", "v"): 0.0},
            "b1": {("p", "u"): 1.0, ("p", "v"): 0.9}, "b2": {("p", "u"): 0.95, ("p", "v"): 1.0}}
    algs = sorted(prof)
    dist = {(a, b): B.distance(prof[a], prof[b]) for i, a in enumerate(algs) for b in algs[i + 1:]}
    assert abs(dist[("a1", "a2")] - 0.075) < 1e-12
    merges = B.average_linkage(algs, dist)
    assert len(merges) == 3 and merges[-1][0] > 0.8, merges
    assert B.cut(algs, merges, 2) == {"a1": 1, "a2": 1, "b1": 2, "b2": 2}
    assert B.cut(algs, merges, 4) == {"a1": 1, "a2": 2, "b1": 3, "b2": 4}


@test
def behaviour_runs_on_a_results_tree():
    """--behaviour end to end on four made-up algorithms of two families: the
    files, the commit, and the two behaviours falling into different groups."""
    from mootation.run import behaviour as B
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        for alg, fast in (("nsga2", True), ("spea2", True), ("moead", False), ("moead_de", False)):
            for s in (1, 2, 3):
                d = root / "DTLZ2_3D" / alg / f"run_{s}"
                d.mkdir(parents=True)
                g = [0.5, 0.01, 0.01, 0.01] if fast else [0.5, 0.4, 0.2, 0.05]
                recs = [{"fe": fe, "gdp_norm": v, "igdp_norm": 2 * v, "range_cover": 1.0,
                         "nd_share": 1.0, "dup_share": 0.0 if fast else 0.3}
                        for fe, v in zip((100, 1000, 5000, 10000), g)]
                (d / "trajectory.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs),
                                                    encoding="utf-8")
                (d / "meta.json").write_text(json.dumps(
                    {"problem": "DTLZ2_3D", "algorithm": alg, "seed": s, "status": "done",
                     "final": {"igdp_norm": 2 * g[-1]}}), encoding="utf-8")
        text = B.run(root)
        out = root / "_behaviour"
        for f in ("behaviour_meta.json", "behaviour_runs.csv", "behaviour_medians.csv",
                  "behaviour_profile.csv", "behaviour_merges.csv", "behaviour_report.txt"):
            assert (out / f).is_file(), f
        meta = json.loads((out / "behaviour_meta.json").read_text(encoding="utf-8"))
        assert "commit" in meta["analysis"] and meta["settle"] == B.SETTLE
        assert "12 runs, 1 problems, 4 algorithms" in text, text
        with (out / "behaviour_profile.csv").open(newline="", encoding="utf-8") as fh:
            group = {r["algorithm"]: r["group"] for r in csv.DictReader(fh)}
        assert group["nsga2"] == group["spea2"] != group["moead"] == group["moead_de"], group


@test
def metrics_agree_with_closed_forms():
    try:
        import numpy as np
    except ImportError:
        return
    from mootation.run import metrics as M
    f1 = np.linspace(0, 1, 401)
    F = np.column_stack([f1, 1 - np.sqrt(f1)])                    # the ZDT1 front
    assert M.igd(F, F) == 0.0 and M.igd_plus(F, F) == 0.0
    hv, method = M.hypervolume(F, [0, 0], [1, 1])
    # exact: (0.1 + 2/3 + 0.11) / 1.21 = 0.7245, minus the staircase gap of a 401-point sample
    assert method == "exact" and abs(hv - 0.7245) < 0.003, (hv, method)
    F3 = np.array([[0.2, 0.3, 0.4], [0.5, 0.1, 0.6], [0.7, 0.7, 0.1]])
    v3, _ = M.hypervolume(F3, [0, 0, 0], [1, 1, 1])
    v_mc = M._hv_mc(F3, np.full(3, 1.1), np.zeros(3), 200000, 1) / 1.1 ** 3
    assert abs(v3 - v_mc) < 0.01, (v3, v_mc)
    dom = M.nondominated(np.array([[1, 1], [0.5, 2], [2, 0.5], [0.6, 2.1]]))
    assert len(dom) == 3
    # Copies of a row must not multiply the work. Equal rows do not dominate
    # each other, so they used to survive the filter and the WFG recursion
    # branched on every copy: a MOEA/D-AM2M population with 11 distinct rows
    # out of 126 kept a campaign job busy for more than ten minutes.
    # Six distinct rows on the simplex (mutually nondominated whatever the
    # draw) twelve times over: about 100 000 recursive calls and ten seconds
    # with the copies kept, milliseconds without.
    import time
    base = np.random.default_rng(3).dirichlet(np.ones(5), size=6)
    copies = np.repeat(base, 12, axis=0)
    assert len(M.nondominated(copies)) == 6
    t0 = time.perf_counter()
    v_rep, method = M.hypervolume(copies, [0] * 5, [1] * 5)
    assert method == "exact" and time.perf_counter() - t0 < 1.0
    v_one, _ = M.hypervolume(base, [0] * 5, [1] * 5)
    assert abs(v_rep - v_one) < 1e-12, (v_rep, v_one)
    # additive epsilon, and the scale-free variants
    R = F                                                           # the ZDT1 front sample
    assert abs(M.eps_plus(R, R)) < 1e-12
    assert abs(M.eps_plus(R + 0.1, R) - 0.1) < 1e-12                 # 0.1 behind everywhere
    wide = np.array([1.0, 100.0])
    out = M.compute(R * wide, ref_front=R * wide, ideal=[0, 0], nadir=[1, 100],
                    which=("igdp_norm", "eps_norm"))
    assert out["igdp_norm"] == 0.0 and abs(out["eps_norm"]) < 1e-12, out
    behind = M.compute((R + [0.0, 0.1]) * wide, ref_front=R * wide, ideal=[0, 0],
                       nadir=[1, 100], which=("eps", "eps_norm", "igdp", "igdp_norm"))
    assert abs(behind["eps"] - 10.0) < 1e-9 and abs(behind["eps_norm"] - 0.1) < 1e-12, behind
    # normalised, the 100x axis is a unit again: the same as IGD+ of the unscaled offset
    assert abs(behind["igdp_norm"] - M.igd_plus(R + [0.0, 0.1], R)) < 1e-12, behind
    assert (M.lattice_h(2, 100), M.lattice_h(3, 91), M.lattice_h(5, 126)) == (99, 12, 5)
    hh = M.compute(R, ideal=[0, 0], nadir=[1, 1], which=("hv", "hv_h"), pop=100)
    assert abs(hh["hv_h_ref"] - (1 + 1 / 99)) < 1e-12 and hh["hv_h"] != hh["hv"], hh
    assert M.compute(R, ideal=[0, 0], nadir=[1, 1], which=("hv_h",))["hv_h"] is None


@test
def new_indicators_agree_with_closed_forms():
    """GD+, roi_dist, range_cover, nd_share, dup_share on sets whose values are known."""
    try:
        import numpy as np
    except ImportError:
        return
    from mootation.run import metrics as M
    f1 = np.linspace(0, 1, 201)
    R = np.column_stack([f1, 1 - f1])                               # a linear front
    assert M.gd_plus(R, R) == 0.0
    assert abs(M.gd_plus(R + 0.1, R) - 0.1 * np.sqrt(2)) < 1e-12     # every point 0.1 behind in both
    # GD+ averages over the SET: one converged point scores 0 however little it covers
    assert M.gd_plus(R[:1], R) == 0.0 and M.igd_plus(R[:1], R) > 0.3
    # roi_dist: zero inside the box, the normalized overshoot outside it
    assert M.roi_dist(R, [0, 0], [1, 1]) == 0.0
    assert abs(M.roi_dist(np.array([[2.0, 3.0], [1.5, 1.0]]), [0, 0], [1, 2]) - 0.5) < 1e-12
    # range_cover: per objective, clipped to the box; the worst objective is the score
    each = M.range_cover_each(np.array([[0.0, 0.5], [0.5, 0.6], [2.0, 0.7]]), [0, 0], [1, 1])
    assert np.allclose(each, [1.0, 0.2]), each
    out = M.compute(np.array([[0.0, 0.5], [0.5, 0.6]]), ideal=[0, 0], nadir=[1, 1],
                    which=("range_cover", "nd_share", "dup_share", "roi_dist", "gdp"))
    assert abs(out["range_cover"] - 0.1) < 1e-12 and out["range_cover_each"] == [0.5, 0.1], out
    assert out["gdp"] is None                                        # no reference front: nothing
    # nd_share / dup_share
    F = np.array([[1, 1], [1, 1], [0.5, 2], [2, 2], [2, 0.5]], float)
    assert abs(M.nd_share(F) - 0.8) < 1e-12                          # (2,2) is dominated; copies are not
    assert abs(M.dup_share(F) - 0.2) < 1e-12


@test
def the_log_grid_is_the_same_at_every_budget():
    from mootation.run.campaign import log_grid
    g10, g25 = log_grid(10_000, 10), log_grid(25_000, 10)
    assert g25[:len(g10)] == g10 and g10[-1] <= 10_000 < g25[len(g10)], (g10[-3:], g25[:len(g10) + 1])
    assert g10 == sorted(set(g10)) and g10[0] == 1
    assert [v for v in g10 if 100 <= v <= 1000] == [100, 126, 158, 200, 251, 316, 398, 501,
                                                    631, 794, 1000]


@test
def dss_keeps_the_extremes_and_its_order_is_incremental():
    try:
        import numpy as np
    except ImportError:
        return
    from mootation.run.archive import dss_order
    t = np.linspace(0, np.pi / 2, 400)
    F = np.column_stack([np.cos(t), np.sin(t)])                     # a quarter circle
    order = dss_order(F)
    assert sorted(order.tolist()) == list(range(len(F)))           # a permutation
    assert set(order[:2].tolist()) == {int(np.argmin(F[:, 0])), int(np.argmin(F[:, 1]))}
    # the first n of the full order ARE the n-point selection
    assert np.array_equal(dss_order(F, k=25), order[:25])

    # Farthest-first on the IGD+ distance: each next point is exactly the
    # candidate the prefix covers worst. (Not uniform in angle — d+ counts only
    # the coordinate in which the selected point is worse.)
    def cover(prefix):
        S = F[prefix]
        d = np.sqrt((np.maximum(S[:, None, :] - F[None, :, :], 0.0) ** 2).sum(axis=2))
        return d.min(axis=0)
    for k in (2, 5, 12, 24):
        c = cover(order[:k])
        assert abs(c[order[k]] - c.max()) < 1e-12, (k, c[order[k]], c.max())
    # and it beats chance at what it is for: IGD+ of the selection against the set
    from mootation.run.metrics import igd_plus
    mine = igd_plus(F[order[:25]], F)
    rng = np.random.default_rng(1)
    assert all(mine < igd_plus(F[rng.choice(len(F), 25, replace=False)], F) for _ in range(20))


@test
def dss_sets_almost_dominated_points_aside():
    """Points that buy the best value of one objective with a much worse value of
    another are neither seeds nor picks while other points are left, and do not
    stretch the frame the others are normalized in (archive.py, 2026-09-27)."""
    try:
        import numpy as np
    except ImportError:
        return
    from mootation.run.archive import DRS_GAIN, DRS_LOSS, dss_order, normalized, set_aside
    rng = np.random.default_rng(7)
    D = rng.random((50, 3))
    front = D / np.linalg.norm(D, axis=1, keepdims=True)            # 50 points of DTLZ2's front
    bad = []
    for j in range(3):                     # each objective's best, beaten by 1e-4, worse by 0.8
        p = front[np.argmin(front[:, j])].copy()
        p[j] -= 1e-4
        p[(j + 1) % 3] += 0.8
        bad.append(p)
    F = np.vstack([front, bad])
    drs = set(range(len(front), len(F)))
    for frame in ((None, None), ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))):
        order = dss_order(F, ideal=frame[0], nadir=frame[1]).tolist()
        assert not drs & set(order[:len(front)]), (frame, order[:len(front)])
        assert set(order[len(front):]) == drs and sorted(order) == list(range(len(F)))
        for k in (3, 10, 50):                                       # never as a seed or a pick
            assert not drs & set(dss_order(F, k=k, ideal=frame[0], nadir=frame[1]).tolist())
    G, kept = set_aside(F)
    assert kept.tolist() == [True] * len(front) + [False] * 3
    assert np.array_equal(G[:len(front)], normalized(front))        # the frame is the front's
    # the rule's size: a gain of DRS_GAIN or less and a loss of more than DRS_LOSS
    assert set_aside([[0.0, 1.0], [DRS_GAIN, 1.0 - 2 * DRS_LOSS], [1.0, 0.0]])[1].tolist() \
        == [False, True, True]
    assert set_aside([[0.0, 1.0], [2 * DRS_GAIN, 1.0 - 2 * DRS_LOSS], [1.0, 0.0]])[1].all()
    # a steep but ordinary front keeps its ends: ZDT1's f2 = 1 - sqrt(f1)
    x = np.linspace(0.0, 1.0, 200)
    Z = np.column_stack([x, 1.0 - np.sqrt(x)])
    assert set_aside(Z)[1].all()
    assert set(dss_order(Z, k=2).tolist()) == {0, 199}


@test
def dss_in_a_given_frame_asks_lazily_and_orders_the_same():
    """With the problem's frame DSS asks "set aside?" only of the points it is
    about to take; the order must be the one checking every point first gives."""
    try:
        import numpy as np
    except ImportError:
        return
    from mootation.run.archive import dss_order, set_aside

    def eager(F, k, lo, hi):
        G, kept = set_aside(F, lo, hi)
        n = len(G)
        order, taken, cover = [], np.zeros(n, bool), np.full(n, np.inf)
        for j in range(G.shape[1]):
            if len(order) >= k:
                break
            i = int(np.argmin(np.where(kept, G[:, j], np.inf)))
            if not taken[i]:
                taken[i] = True
                order.append(i)
                np.minimum(cover, np.sqrt((np.maximum(G[i] - G, 0.0) ** 2).sum(axis=1)), out=cover)
        while len(order) < k:
            pool = kept & ~taken if (kept & ~taken).any() else ~taken
            i = int(np.argmax(np.where(pool, cover, -1.0)))
            taken[i] = True
            order.append(i)
            np.minimum(cover, np.sqrt((np.maximum(G[i] - G, 0.0) ** 2).sum(axis=1)), out=cover)
        return order

    rng = np.random.default_rng(11)
    for m in (2, 3, 5):
        D = rng.random((120, m))
        F = np.vstack([D / np.linalg.norm(D, axis=1, keepdims=True),     # a front
                       rng.random((30, m)) * 1.5])                       # dominated points
        for j in range(m):                                              # and some DRS
            p = F[np.argmin(F[:120, j])].copy()
            p[j] -= 1e-4
            p[(j + 1) % m] += 0.6
            F = np.vstack([F, p])
        lo, hi = np.zeros(m), np.full(m, 1.5)
        for k in (m, 20, 60, len(F)):
            assert dss_order(F, k=k, ideal=lo, nadir=hi).tolist() == eager(F, k, lo, hi), (m, k)


@test
def the_grid_archive_keeps_the_front_prunes_neighbours_and_protects_extremes():
    try:
        import numpy as np
    except ImportError:
        return
    from mootation.run.archive import GridArchive
    a = GridArchive(2, 1, ideal=[0, 0], nadir=[1, 1], delta=0.1)
    assert a.add([0.5, 0.5], [0.0]) and not a.add([0.6, 0.6], [0.0])   # dominated
    assert not a.add([0.5, 0.5], [0.0])                                  # a duplicate
    assert a.add([0.4, 0.4], [0.0]) and len(a) == 1                      # dominates, replaces
    # two nondominated neighbours in one cell: the one nearer the lower corner stays
    a.add([0.0, 1.0], [0.0]); a.add([1.0, 0.0], [0.0])                  # the two extremes
    a.add([0.31, 0.49], [0.0]); a.add([0.38, 0.42], [0.0])
    F, X, ext = a.points()
    assert len(F) == len({tuple(np.floor(f / 0.1).astype(int)) for f in F[~ext]}) + ext.sum()
    # extremes live outside the grid: three mutually nondominated points in ONE
    # cell (step 0.5) all survive, because two of them are extremes
    b = GridArchive(2, 0, ideal=[0, 0], nadir=[1, 1], delta=0.5)
    for f in ([0.0, 0.4], [0.2, 0.05], [0.05, 0.3]):
        assert b.add(f)
    F, _, ext = b.points()
    assert len(b) == 3 and int(ext.sum()) == 2, (F, ext)
    # without a frame the grid follows the archive's own estimate (re-gridded
    # when it drifts by a quarter of its span, so the cells end up near 0.01)
    c = GridArchive(2, 0, delta=0.01)
    for t in np.linspace(0, 1, 500):
        c.add([t, 1 - t])
    assert c.normalization == "archive" and 80 <= len(c) <= 160, len(c)


@test
def baselines_spend_the_budget_exactly_and_answer_with_the_archive():
    try:
        import numpy as np
    except ImportError:
        return
    from mootation.run.archive import GridArchive
    from mootation.run.baselines import run_baseline
    from mootation.benchmarks import get
    p = get("ZDT1")
    for name in ("random_search", "sobol_search", "lhs_search", "halton_search",
                 "sobol_owen_search"):
        if name == "sobol_search":
            try:
                import scipy  # noqa: F401
            except ImportError:
                continue
        budget = 234 if name == "lhs_search" else 1234        # a Latin hypercube costs more
        runs = []
        for _ in range(2):
            arc = GridArchive(2, p.n_vars, ideal=p.ideal, nadir=p.nadir)
            spent, seen = [0], []

            def ev(x, arc=arc, spent=spent, seen=seen):
                spent[0] += 1
                seen.append(x)
                f = p.evaluate(x)
                arc.add(f, x)
                return f
            res = run_baseline(name, ev, p.bounds, pop=100, max_evaluations=budget, seed=7,
                               archive=arc)
            assert spent[0] == budget, (name, spent[0])
            assert len(res.objectives) == min(100, len(arc)), (name, len(res.objectives))
            runs.append(np.asarray(res.objectives))
        assert np.array_equal(runs[0], runs[1]), name                   # a seed is a seed
        if name == "lhs_search":                       # one point in each of the n cells
            cells = np.sort(np.floor(np.asarray(seen) * budget).astype(int), axis=0)
            assert (cells == np.arange(budget)[:, None]).all()


@test
def a_maximin_latin_hypercube_reaches_the_papers_catalog():
    """designs.maximin_lhs finds the maximin designs of Morris & Mitchell 1995,
    Table 2(A) (Euclidean): n = 5, k = 2: d1 = 0.5590 (squared, in levels: 5),
    J1 = 4; n = 9, k = 2: d1 = 0.3953 (10), J1 = 12; every column a
    permutation of the levels, and a design the same for a seed."""
    try:
        import numpy as np
    except ImportError:
        return
    from mootation import designs as D
    for n, k, d1, j1 in ((5, 2, 5, 4), (9, 2, 10, 12)):
        L = D.maximin_lhs(n, k, np.random.default_rng(3))
        assert (np.sort(L, axis=0) == np.arange(n)[:, None]).all(), L
        key = D.maximin_order(D.squared_distances(L))
        assert (-key[0], key[1]) == (d1, j1), (n, k, key[:4])
    X = D.design("lhs", 30, [(0.0, 1.0), (-2.0, 2.0)], 11)
    assert np.array_equal(X, D.design("lhs", 30, [(0.0, 1.0), (-2.0, 2.0)], 11))
    assert X[:, 1].min() >= -2.0 and X[:, 1].max() <= 2.0 and not (X == 0.5).any()


@test
def a_randomized_halton_plan_is_stratified_seeded_and_off_the_grid():
    """designs.randomized_halton (task 6, item 12.5): at n = b^m the coordinate
    of base b has one point in each of its b^m intervals; no coordinate is
    exactly 0 or 1/2 over 10 seeds of 1 000 points; a seed is a plan and
    another seed another plan; the first n points of a longer plan are the
    plan of n; its centred L2 discrepancy, over 20 seeds, is below uniform
    points' at d = 2 and 10."""
    try:
        import numpy as np
    except ImportError:
        return
    from mootation import designs as D
    assert D.primes(6) == [2, 3, 5, 7, 11, 13] and D.primes(50)[-1] == 229
    assert D.halton_digits(2) == 53                    # 1 - 2^-53 < 1 == 1 - 2^-54
    for j, b, m in ((0, 2, 9), (1, 3, 5), (2, 5, 3)):
        x = D.randomized_halton(b ** m, 3, 7)[:, j]
        assert len(np.unique(np.floor(x * b ** m))) == b ** m, (b, m)
    for seed in range(10):
        X = D.randomized_halton(1000, 10, seed)
        assert not ((X == 0.0) | (X == 0.5)).any() and X.min() > 0.0 and X.max() < 1.0, seed
    A = D.design("halton_rand", 300, [(0.0, 1.0), (-2.0, 2.0)], 11)
    assert np.array_equal(A, D.design("halton_rand", 300, [(0.0, 1.0), (-2.0, 2.0)], 11))
    assert not np.array_equal(A, D.design("halton_rand", 300, [(0.0, 1.0), (-2.0, 2.0)], 12))
    assert A[:, 1].min() >= -2.0 and A[:, 1].max() <= 2.0
    B = D.randomized_halton(800, 5, 3)
    assert np.array_equal(B[:500], D.randomized_halton(500, 5, 3))
    assert np.array_equal(B[500:], D.randomized_halton(300, 5, 3, start=500))
    if _core_with("max_evaluations") is not None:     # init = "halton_rand": its first points
        from mootation import minimize
        seen = []

        def f(x):
            seen.append(list(x))
            return [x[0], 1.0 - x[0] ** 0.5 + sum(x[1:])]
        r = minimize(f, bounds=[(0.0, 1.0)] * 3, n_objs=2, algorithm="nsga2", pop_size=20,
                     n_gen=3, seed=1, init="halton_rand")
        assert r.evaluations == 60 and len(r.objectives) == 20, r.evaluations
        assert np.allclose(seen[:20], D.design("halton_rand", 20, [(0.0, 1.0)] * 3, 1))
    try:
        from scipy.stats import qmc
    except ImportError:
        return
    for d in (2, 10):
        h = np.mean([qmc.discrepancy(D.randomized_halton(256, d, s)) for s in range(20)])
        u = np.mean([qmc.discrepancy(np.random.default_rng(s).random((256, d)))
                     for s in range(20)])
        assert h < u, (d, h, u)


@test
def an_owen_scrambled_sobol_plan_is_stratified_seeded_and_off_the_grid():
    """designs.sobol_owen (task 6, items 12.2 and 12.5): unscrambled, the
    points are Joe and Kuo's example and SciPy's Sobol(scramble=False); the
    Laine-Karras permutation of 0 under Burley's seed is his Table 1's; at
    n = 2^m each coordinate has one point in each of its 2^m intervals; no
    coordinate exactly 0 or 1/2 over 10 seeds of 1 000 points; a seed is a
    plan, another seed another; the plan extends; its centred L2 discrepancy
    over 20 seeds is below uniform points' at d = 2 and 10."""
    try:
        import numpy as np
    except ImportError:
        return
    from mootation import designs as D
    # "./sobol 10 3 new-joe-kuo-6.21201" on Joe and Kuo's page
    page = [[0, 0, 0], [0.5, 0.5, 0.5], [0.75, 0.25, 0.25], [0.25, 0.75, 0.75],
            [0.375, 0.375, 0.625], [0.875, 0.875, 0.125], [0.625, 0.125, 0.875],
            [0.125, 0.625, 0.375], [0.1875, 0.3125, 0.9375], [0.6875, 0.8125, 0.4375]]
    assert (D.sobol_integers(10, 3) / 2.0 ** 32).tolist() == page
    assert int(D.laine_karras(np.array([0], dtype=np.uint64), 0x552553bc)[0]) == 0x71b1c2ac
    x = np.arange(1000, dtype=np.uint64) * np.uint64(4294967)
    assert np.array_equal(D._reverse_bits(D._reverse_bits(x)), x)
    for m, d in ((10, 5), (7, 30)):
        U = D.sobol_owen(2 ** m, d, 7)
        for j in range(d):
            assert len(np.unique(np.floor(U[:, j] * 2 ** m))) == 2 ** m, (m, j)
    for seed in range(10):
        X = D.sobol_owen(1000, 10, seed)
        assert not ((X == 0.0) | (X == 0.5)).any() and X.min() > 0.0 and X.max() < 1.0, seed
    A = D.design("sobol_owen", 300, [(0.0, 1.0), (-2.0, 2.0)], 11)
    assert np.array_equal(A, D.design("sobol_owen", 300, [(0.0, 1.0), (-2.0, 2.0)], 11))
    assert not np.array_equal(A, D.design("sobol_owen", 300, [(0.0, 1.0), (-2.0, 2.0)], 12))
    B = D.sobol_owen(800, 5, 3)
    assert np.array_equal(B[:500], D.sobol_owen(500, 5, 3))
    assert np.array_equal(B[500:], D.sobol_owen(300, 5, 3, start=500))
    try:
        from scipy.stats import qmc
    except ImportError:
        return
    assert np.array_equal(D.sobol_integers(1024, 40) / 2.0 ** 32,
                          qmc.Sobol(40, scramble=False).random(1024))
    for d in (2, 10):
        h = np.mean([qmc.discrepancy(D.sobol_owen(256, d, s)) for s in range(20)])
        u = np.mean([qmc.discrepancy(np.random.default_rng(s).random((256, d)))
                     for s in range(20)])
        assert h < u, (d, h, u)


@test
def the_older_designs_have_not_moved():
    """The sobol and lhs designs and the draws of random_search, sobol_search
    and lhs_search are what they were before the randomized Halton came (task
    6, item 12.5; lhs's since its power became multiplications, 2026-10-11):
    rows 0, 5 and 11 of 12 points at seed 5, to 1e-12."""
    try:
        import numpy as np
    except ImportError:
        return
    from mootation import designs as D
    from mootation.run.archive import GridArchive
    from mootation.run.baselines import run_baseline
    box = [(0.0, 1.0), (-2.0, 2.0), (0.0, 10.0)]
    was = {
        "random_search": [[0.80500292374538, 1.23176315894597, 5.15325561042142],
                          [0.844231037608741, -0.430381342660887, 4.93023018731743],
                          [0.436667052175653, -1.18698865554174, 3.24942644575606]],
        "lhs_search": [[0.0729064959603389, 0.813428951982356, 5.22444211878804],
                       [0.311445991338462, -1.05261810004373, 7.85666928530813],
                       [0.875990488419677, 0.512934655283951, 1.91795590171167]],
        "sobol_search": [[0.63135180901736, -1.05258386209607, 5.24911201559007],
                         [0.149958958849311, 0.586466815322638, 6.30971090868115],
                         [0.965077580884099, 1.46423656120896, 7.00347781181335]],
    }
    try:
        import scipy  # noqa: F401
    except ImportError:
        del was["sobol_search"]
    for name, rows in was.items():
        seen, arc = [], GridArchive(2, 3)

        def ev(x, seen=seen, arc=arc):
            seen.append(x)
            arc.add([x[0], 1.0 - x[0]], x)
            return [x[0], 1.0 - x[0]]
        run_baseline(name, ev, box, pop=12, max_evaluations=12, seed=5, archive=arc)
        assert np.allclose(np.array(seen)[[0, 5, 11]], rows, rtol=0, atol=1e-12), name
        if name != "random_search":                    # the design is the baseline's draw
            X = D.design(name.split("_")[0], 12, box, 5)
            assert np.allclose(X[[0, 5, 11]], rows, rtol=0, atol=1e-12), name


@test
def results_record_the_revision_they_were_made_with():
    from mootation.run.provenance import revision
    r = revision()
    assert set(r) == {"commit", "dirty", "source", "core_commit", "core_dirty"}, r
    if r["source"] == "git":
        assert len(r["commit"]) == 40 and int(r["commit"], 16) >= 0, r
        assert isinstance(r["dirty"], bool), r


@test
def a_campaign_writes_the_log_grid_the_archive_and_snapshots():
    """The 2026-09-22 recording options end to end, on two tiny problems."""
    try:
        import numpy as np
        import mootation._core as _core
    except ImportError:
        return
    if not hasattr(_core.Config(), "max_evaluations"):
        print("  skip  a_campaign_writes_the_log_grid...: stale _core"); return
    from mootation.run import campaign as C
    from mootation.run.config import load
    text = """algorithms = [
    { name = "nsga2", pop = 0, gens = 0 },
    { name = "random_search", pop = 0, gens = 0 },
]
[run]
name = "rec"
[problem]
kind = "builtin"
[benchmarks]
runs = 1
problems = ["ZDT1", "DTLZ2_5D"]
[campaign]
out = "res"
budget_fe = 600
record_grid = "log"
trajectory_hv_max_m = 3
metrics = ["igdp", "gdp", "hv_h", "roi_dist", "range_cover", "nd_share", "dup_share"]
final_metrics = ["igdp", "hv", "roi_dist"]
snapshots = ["ZDT1"]
"""
    with tempfile.TemporaryDirectory() as td:
        cfg_path = Path(td) / "c.toml"
        cfg_path.write_text(text, encoding="utf-8")
        cfg = load(cfg_path)
        assert validate(cfg) == [], validate(cfg)
        spec = C.campaign_spec(cfg)
        root = C.out_root(cfg, spec)
        for job in C.expand_jobs(cfg, spec):
            assert C.run_job(job, root, spec, quiet=True) == "done", job
        for prob, pop in (("ZDT1", 100), ("DTLZ2_5D", 126)):
            for alg in ("nsga2", "random_search"):
                d = root / prob / alg / "run_1"
                meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
                T = [json.loads(l) for l in (d / "trajectory.jsonl").read_text().splitlines()]
                grid = C.log_grid(meta["budget_fe"], 10)
                # every record after the first sits at or past a grid count it was waiting for
                for t in T[1:-1]:
                    assert any(g <= t["fe"] < g + pop for g in grid), (prob, alg, t["fe"])
                hv = [t["hv_h"] for t in T]
                assert (all(v is None for v in hv) if prob == "DTLZ2_5D"
                        else all(v is not None for v in hv)), (prob, hv)
                rows = (d / "archive.csv").read_text().splitlines()
                assert len(rows) - 2 == meta["archive"]["size"] > 0, (prob, alg)
                assert (d / "snapshots.npz").exists() == (prob == "ZDT1"), (prob, alg)
                assert meta["revision"]["commit"], meta
                assert isinstance(meta["numba"], bool), meta     # which WFG path ran
                if alg == "random_search":
                    assert meta["fe"] == meta["budget_fe"], meta    # exactly, no overshoot


@test
def campaign_results_round_trip():
    """Run two tiny jobs for real and read them back through scan/compare."""
    try:
        import numpy  # noqa: F401
        import mootation._core as _core
    except ImportError:
        return
    if not hasattr(_core.Config(), "on_generation"):
        # An extension built before the observer existed (a stale .pyd for
        # this interpreter): the campaign cannot record trajectories with it.
        print("  skip  campaign_results_round_trip: _core has no on_generation "
              "(rebuild the extension for this interpreter)")
        return
    from mootation.run import campaign as C
    with tempfile.TemporaryDirectory() as td:
        cfg_path = Path(td) / "c.toml"
        # Integer knobs ride along on purpose. The config layer used to turn
        # every parameter into a float, and pybind11 3 refuses 20.0 for an
        # int, so every job with T or nr failed — the shipped campaign's
        # MOEA/D-DE among them.
        text = (_CAMP.replace("runs = 3", "runs = 1")
                .replace("budget_fe = 2000", "budget_fe = 300")
                .replace('name = "nsga2"\npop = 0\ngens = 0',
                         'name = "nsga2"\npop = 0\ngens = 0\nparams = { T = 20, delta = 0.9 }'))
        assert "params = { T = 20" in text
        cfg_path.write_text(text, encoding="utf-8")
        from mootation.run.config import load
        cfg = load(cfg_path)
        assert validate(cfg) == []
        spec = C.campaign_spec(cfg)
        jobs = C.expand_jobs(cfg, spec)
        root = C.out_root(cfg, spec)
        z = [j for j in jobs if j.problem == "ZDT1" and j.algorithm == "nsga2"][0]
        assert C.run_job(z, root, spec, quiet=True) == "done"
        assert C.run_job(z, root, spec, quiet=True) == "done"      # resume: skipped, still done
        meta = json.loads((root / z.rel_dir / "meta.json").read_text(encoding="utf-8"))
        assert meta["status"] == "done" and meta["fe"] >= 300, meta
        assert meta["params"] == {"T": 20, "delta": 0.9}, meta["params"]
        assert isinstance(meta["params"]["T"], int) and meta["mootation"], meta
        # ... and a whole float passed straight to minimize() means the integer
        from mootation import minimize

        def zdt(x):
            return [x[0], 1.0 - x[0] ** 0.5]

        kw = dict(bounds=[(0.0, 1.0)] * 3, n_objs=2, algorithm="moead_de",
                  pop_size=10, n_gen=2, seed=1)
        assert minimize(zdt, T=5.0, nr=2.0, **kw).active_n == 10
        assert "T" in str(raises(TypeError, minimize, zdt, T=5.5, **kw))
        # The budget is in evaluations whatever a core spends per step: a step
        # of NIMMO is one offspring, of MOEA/D-DRA a fifth of the population.
        for name, per_step in (("nsga2", 20), ("nimmo", 1), ("moead_dra", 4)):
            spent = [0]

            def counted(x, spent=spent):
                spent[0] += 1
                return zdt(x)

            r = minimize(counted, bounds=[(0.0, 1.0)] * 3, n_objs=2, algorithm=name,
                         pop_size=20, n_gen=1, seed=1, max_evaluations=200)
            assert 200 <= spent[0] < 200 + per_step, (name, spent[0])
            assert r.evaluations == spent[0], (name, r.evaluations, spent[0])
        traj = C.read_trajectory(root / z.rel_dir)
        assert traj and traj[0]["gen"] == 0 and traj[-1]["fe"] >= z.pop * z.gens
        assert all("igd" in t and "hv" in t for t in traj)
        rows = C.scan_results(root)
        assert len(rows) == 1 and rows[0]["problem"] == "ZDT1"
        table = C.compare_table(rows, "igd")
        assert "ZDT1" in table and "nsga2" in table["ZDT1"]
        out = root / "cmp.csv"
        C.write_compare_csv(table, out, "igd")
        assert out.read_text(encoding="utf-8").startswith("problem,")
        ranks = C.rank_table(rows, "igd")
        assert [a for a, _ in ranks["algorithms"]] == ["nsga2"] and ranks["groups"] == []
        C.write_rank_csv(ranks, root / "ranks.csv")
        assert (root / "ranks.csv").read_text(encoding="utf-8").startswith("algorithm,mean_rank")
        # anytime: at the whole budget the trajectory's last record is the final value
        row = C.scan_results(root)[0]
        assert abs(C.value_at(row, "igd", 1.0) - row["final"]["igd"]) < 1e-12, row
        assert C.value_at(row, "igd", 0.5) is not None
        assert C.rank_table([row], "igd", 0.5)["at"] == 0.5
        # indicators added after the run, from final.csv, without rerunning it
        assert C.recompute_final(root, ["eps", "igdp_norm", "hv_h"]) == {"done": 1}
        fin = json.loads((root / z.rel_dir / "meta.json").read_text(encoding="utf-8"))["final"]
        assert all(k in fin for k in ("eps", "igdp_norm", "hv_h", "igd", "hv")), fin
        script = C.emit_slurm(cfg, 2)
        text = script.read_text(encoding="utf-8")
        assert "--array=0-1" in text and "--shard ${SLURM_ARRAY_TASK_ID}/2" in text
        assert (root / "jobs.txt").read_text(encoding="utf-8").count("--job") == len(jobs)


@test
def campaign_rank_table():
    """Mean ranks: medians ranked per problem, ties averaged, grouped by family and M."""
    from mootation.run import campaign as C

    def row(problem, alg, igd, hv, m, status="done"):
        return {"problem": problem, "algorithm": alg, "seed": 1, "status": status,
                "final": {"igd": igd, "hv": hv}, "n_objs": m}

    rows = [
        row("ZDT1", "a", 0.1, 0.9, 2), row("ZDT1", "b", 0.2, 0.8, 2),
        row("ZDT1", "c", 0.3, 0.7, 2),
        row("DTLZ2_3D", "a", 0.5, 0.4, 3), row("DTLZ2_3D", "b", 0.5, 0.4, 3),
        row("DTLZ2_3D", "c", 0.1, 0.6, 3),
        row("DTLZ1_3D", "a", 0.2, 0.5, 3), row("DTLZ1_3D", "b", 0.3, 0.4, 3),
        row("DTLZ1_3D", "c", 0.0, 1.0, 3, status="failed"),   # not finished: not ranked
        row("DTLZ3_3D", "a", 0.1, 0.3, 3), row("DTLZ3_3D", "b", 0.1, 0.3, 3),
        row("DTLZ3_3D", "c", 0.2, 0.2, 3),
    ]

    def close(x, y):
        return abs(x - y) < 1e-12

    # hv mirrors igd, so higher-is-better must produce exactly the same ranks
    for metric in ("igd", "hv"):
        r = C.rank_table(rows, metric)
        assert r["n_problems"] == 4 and r["groups"] == ["DTLZ", "ZDT", "M=2", "M=3"], r
        assert [a for a, _ in r["algorithms"]] == ["a", "b", "c"], (metric, r["algorithms"])
        e = dict(r["algorithms"])
        assert close(e["a"]["all"][0], 1.5) and e["a"]["all"][1] == 4          # 1, 2.5, 1, 1.5
        assert close(e["b"]["all"][0], 2.0) and e["b"]["all"][1] == 4          # 2, 2.5, 2, 1.5
        assert close(e["c"]["all"][0], 7 / 3) and e["c"]["all"][1] == 3        # absent on DTLZ1
        assert (e["a"]["wins"], e["b"]["wins"], e["c"]["wins"]) == (3, 1, 1)   # shared best counts
        assert close(e["a"]["DTLZ"][0], 5 / 3) and close(e["c"]["ZDT"][0], 3.0)
        assert close(e["c"]["M=3"][0], 2.0) and close(e["b"]["M=2"][0], 2.0)

    # one family at one objective count: no group columns to show
    assert C.rank_table([x for x in rows if x["problem"].startswith("DTLZ")], "igd")["groups"] == []
    assert C.format_ranks(C.rank_table(rows, "igd")).splitlines()[2].startswith("a ")
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "ranks.csv"
        C.write_rank_csv(C.rank_table(rows, "igd"), p)
        lines = p.read_text(encoding="utf-8").splitlines()
        assert lines[0].startswith("algorithm,mean_rank,problems,wins,DTLZ_mean_rank,DTLZ_problems")
        assert lines[1].startswith("a,1.5,4,3,"), lines[1]
        assert lines[0].endswith(",short_answers,empty_runs") and lines[1].endswith(",0,0"), \
            lines[:2]
    # task 5: an answer of fewer than N/2 points is counted beside the rank
    short = [dict(x, pop=10, final=dict(x["final"], n_final=3 if x["algorithm"] == "b" else 8))
             for x in rows]
    e = dict(C.rank_table(short, "igd")["algorithms"])
    assert (e["a"]["short"], e["b"]["short"], e["c"]["short"]) == (0, 4, 0), e
    assert "short" in C.format_ranks(C.rank_table(short, "igd")).splitlines()[1]


@test
def campaign_write_survives_an_open_reader():
    """meta.json is replaced while another handle has it open, as the TUI does."""
    import threading
    from mootation.run import campaign as C
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "meta.json"
        p.write_text('{"status": "running"}', encoding="utf-8")
        reader = p.open("r", encoding="utf-8")
        release = threading.Timer(0.3, reader.close)
        release.start()
        try:
            C._write_json(p, {"status": "done"})
        finally:
            release.join()
            reader.close()
        assert json.loads(p.read_text(encoding="utf-8"))["status"] == "done"
        # a job that cannot even be set up costs that job, not the pool
        assert C._pool_worker((str(Path(td) / "missing.toml"), 7, False)) == (7, "failed")


@test
def campaign_control_files():
    """_workers.txt steers a running campaign; _runner.json says whether one is alive."""
    import time
    from mootation.run import campaign as C
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "results"
        assert C.read_workers(root, 5) == 5                     # nothing written yet
        C.write_workers(root, 8)
        assert C.read_workers(root, 5) == 8
        (root / C.WORKERS_FILE).write_text("eight", encoding="utf-8")
        assert C.read_workers(root, 5) == 5                     # garbage falls back
        C.write_workers(root, -3)
        assert C.read_workers(root, 5) == 0                     # 0 means drain
        assert C.runner_state(root) is None
        beat = {"state": "running", "updated": time.time(), "active": 2, "target": 2}
        C._write_json(root / C.RUNNER_FILE, beat)
        assert C.runner_state(root)["alive"] is True
        C._write_json(root / C.RUNNER_FILE, dict(beat, updated=time.time() - 60))
        assert C.runner_state(root)["alive"] is False           # killed without a word
        C._write_json(root / C.RUNNER_FILE, dict(beat, state="finished"))
        assert C.runner_state(root)["alive"] is False


@test
def campaign_worker_pool():
    """The pool runs every job, and a target of 0 drains without running any."""
    try:
        import numpy  # noqa: F401
        from mootation import _core
    except ImportError:
        return
    if not hasattr(_core.Config(), "max_evaluations"):
        print("  skip  campaign_worker_pool: _core has no max_evaluations "
              "(rebuild the extension for this interpreter)")
        return
    from mootation.run import campaign as C
    from mootation.run.config import load
    with tempfile.TemporaryDirectory() as td:
        for name, workers, expect_done in (("pool", 2, 3), ("drained", 0, 0)):
            cfg_path = Path(td) / f"{name}.toml"
            cfg_path.write_text(
                _CAMP.replace('name = "camp"', f'name = "{name}"')
                .replace('problems = ["ZDT1", "DTLZ2_3D"]', 'problems = ["ZDT1"]')
                .replace("budget_fe = 2000", "budget_fe = 200")
                .split("[[algorithms]]")[0] + '[[algorithms]]\nname = "nsga2"\npop = 20\ngens = 0\n',
                encoding="utf-8")
            cfg = load(cfg_path)
            assert validate(cfg) == [], validate(cfg)
            counts = C.run_campaign(cfg, workers=workers)
            root = C.out_root(cfg, C.campaign_spec(cfg))
            assert counts["done"] == expect_done and counts["pending"] == 3 - expect_done, counts
            state = C.runner_state(root)
            assert state and state["alive"] is False, state
            assert state["state"] == ("finished" if workers else "stopped"), state
            assert C.read_workers(root, -1) == workers


def _worker_that_drops_a_job(cfg_path, force, conn):
    """A pool worker that takes one job and dies without running it."""
    import os
    from mootation.run import campaign as C
    flag = str(Path(cfg_path).with_suffix(".dropped"))
    try:
        os.close(os.open(flag, os.O_CREAT | os.O_EXCL | os.O_WRONLY))   # the first worker only
    except FileExistsError:
        return C._worker_main(cfg_path, force, conn)
    conn.send(("ready", os.getpid()))
    conn.recv()
    os._exit(3)


@test
def campaign_pool_fails_only_the_job_whose_worker_died_holding_it():
    """A worker that dies holding a job fails that job, and the pool runs the rest."""
    try:
        import numpy  # noqa: F401
        from mootation import _core
    except ImportError:
        return
    if not hasattr(_core.Config(), "max_evaluations"):
        print("  skip  campaign_pool_fails_only_the_job_whose_worker_died_holding_it: _core has "
              "no max_evaluations (rebuild the extension for this interpreter)")
        return
    import threading
    from mootation.run import campaign as C
    from mootation.run.config import load
    with tempfile.TemporaryDirectory() as td:
        cfg_path = Path(td) / "drop.toml"
        cfg_path.write_text(
            _CAMP.replace('name = "camp"', 'name = "drop"')
            .replace('problems = ["ZDT1", "DTLZ2_3D"]', 'problems = ["ZDT1"]')
            .replace("budget_fe = 2000", "budget_fe = 200")
            .split("[[algorithms]]")[0] + '[[algorithms]]\nname = "nsga2"\npop = 20\ngens = 0\n',
            encoding="utf-8")
        cfg = load(cfg_path)
        assert validate(cfg) == [], validate(cfg)
        spec = C.campaign_spec(cfg)
        jobs = C.expand_jobs(cfg, spec)
        root = C.out_root(cfg, spec)
        root.mkdir(parents=True, exist_ok=True)
        got = {}
        run = threading.Thread(daemon=True, target=lambda: got.update(C._run_dynamic(
            str(cfg_path), jobs, root, workers=2, force=False, label="drop",
            worker=_worker_that_drops_a_job)))
        run.start()
        run.join(timeout=120)
        assert not run.is_alive(), "the pool is still waiting for the job its worker dropped"
        assert len(jobs) == 3 and got["done"] == 2 and got["failed"] == 1, got
        assert got["pending"] == 0, got


@test
def algorithm_families_cover_the_registry():
    """The families in algorithms.def's section comments hold every algorithm once."""
    from mootation.run.algorithms import algorithm_families
    fams = algorithm_families()
    names = [a for _, members in fams for a in members]
    assert names == list(algorithm_names()), "families must list the registry in order"
    assert len(fams) == 11 and fams[0][1][0] == "nsga2", fams
    assert dict(fams)["Direct search"] == ("dms", "dmulti_mads", "multiglods"), fams
    assert fams[-2:] == (("Particle swarm", ("smpso",)), ("Simulated annealing", ("mosa",))), fams
    assert all(name and "─" not in name for name, _ in fams), fams



# ── analysis layer: variants, Pareto sets, hypervolume, reports, ablations ──

def _core_with(attr: str):
    """The compiled extension if it has `attr`, else None (a stale or absent build)."""
    try:
        from mootation import _core
    except ImportError:
        return None
    return _core if hasattr(_core, attr) or hasattr(_core.Config(), attr) else None


@test
def variant_labels_and_evaluation_budgets_parse_and_validate():
    """[[algorithms]] label = ... files a variant apart; evaluations = ... budgets it."""
    cfg = loads(_MIN.replace('[[algorithms]]\nname = "nsga2"\npop = 20\ngens = 5\n', """
[[algorithms]]
name = "r2ibea"
pop = 20
gens = 5

[[algorithms]]
name = "r2ibea"
label = "r2ibea_norm"
pop = 20
evaluations = 1000
params = { normalize = true }
"""))
    a, b = cfg.algorithms
    assert (a.key, a.budget) == ("r2ibea", 100), (a.key, a.budget)
    assert (b.key, b.budget, b.gens) == ("r2ibea_norm", 1000, 0), (b.key, b.budget, b.gens)
    assert b.params == {"normalize": True} and isinstance(b.params["normalize"], bool)
    assert not [p for p in validate(cfg) if "algorithms" in p], validate(cfg)
    same = loads(_MIN.replace('[[algorithms]]\nname = "nsga2"\npop = 20\ngens = 5\n', """
[[algorithms]]
name = "nsga2"
pop = 20
gens = 5

[[algorithms]]
name = "nsga2"
pop = 20
gens = 9
"""))
    assert any("label" in p for p in validate(same)), validate(same)
    bad = loads(_MIN.replace('name = "nsga2"', 'name = "nsga2"\nlabel = "a/b"'))
    assert any("label" in p for p in validate(bad)), validate(bad)
    assert "normalize" in knob_names()


@test
def pareto_set_samples_land_on_the_front():
    """Every Pareto-set sample evaluates onto the front (Polygon, DTLZ1-4, shiftDTLZ, ZCAT)."""
    if not _have_numpy():
        print("    (skipped: no NumPy)"); return
    import numpy as np
    from mootation.benchmarks import get
    for key in ("DTLZ1_3D", "DTLZ2_5D", "DTLZ3_3D", "DTLZ4_5D",
                "shiftDTLZ1_3D", "shiftDTLZ2_5D", "shiftDTLZ3_5D", "shiftDTLZ4_3D"):
        p = get(key)
        X = np.asarray(p.pareto_set(300), float)
        F = np.array([p.evaluate(list(x)) for x in X])
        on = F.sum(1) - 0.5 if "DTLZ1" in key else (F ** 2).sum(1) - 1.0
        assert np.abs(on).max() < 1e-9, (key, np.abs(on).max())
        lo = np.array([b[0] for b in p.bounds]); hi = np.array([b[1] for b in p.bounds])
        assert np.all((X >= lo) & (X <= hi)), key
    assert get("shiftDTLZ2_3D").cyclic_vars and not get("DTLZ2_3D").cyclic_vars
    p = get("Polygon_4D")
    X = np.asarray(p.pareto_set(400), float)
    F = np.array([p.evaluate(list(x)) for x in X])
    R = np.asarray(p.pareto_front(2000), float)
    dom = np.array([bool(np.any(np.all(R <= f, 1) & np.any(R < f, 1))) for f in F])
    assert not dom.any(), int(dom.sum())
    from mootation.benchmarks import zcat
    for key in ("ZCAT11_3D", "ZCAT2_5D"):
        p = get(key)
        X = np.asarray(p.pareto_set(200), float)
        F = np.array([p.evaluate(list(x)) for x in X])
        name, M = key.split("_")[0], int(key.split("_")[1][:-1])
        y = X / np.arange(1, X.shape[1] + 1) + 0.5
        alpha = np.array([np.arange(1, M + 1) ** 2 * zcat.F_FUNCS[name](r, M) for r in y])
        assert np.abs(F - alpha).max() < 1e-12, key          # beta = 0: on the front


@test
def pareto_sets_of_zdt_dtlz5_7_and_the_dtlz_variants_land_on_their_fronts():
    """The samplers added 2026-10-09 (pareto_sets.py): ZDT1-4 and 6 on g = 1,
    DTLZ5/6 at three objectives and the IDTLZ/SDTLZ on g = 0, DTLZ7 on
    f_M = 2M - sum f_i (1 + sin 3 pi f_i), minusDTLZ2 on g = 0.25 k."""
    if not _have_numpy():
        print("    (skipped: no NumPy)"); return
    import numpy as np
    from mootation.benchmarks import get
    from mootation.benchmarks.dtlz_variants import scale_base

    def front_error(key, F):
        M = F.shape[1]
        if key.startswith("ZDT"):
            f1, f2 = F[:, 0], F[:, 1]
            want = {"ZDT1": 1 - np.sqrt(f1), "ZDT2": 1 - f1 ** 2,
                    "ZDT3": 1 - np.sqrt(f1) - f1 * np.sin(10 * np.pi * f1),
                    "ZDT4": 1 - np.sqrt(f1), "ZDT6": 1 - f1 ** 2}[key]
            return np.abs(f2 - want).max()
        if key.startswith("DTLZ7"):
            P = F[:, :M - 1]
            return np.abs(F[:, -1] - (2 * M - np.sum(P * (1 + np.sin(3 * np.pi * P)), 1))).max()
        if key.startswith("minusDTLZ2"):
            return np.abs(np.sqrt((F ** 2).sum(1)) - (1 + 0.25 * 10)).max()
        if key.startswith(("IDTLZ1", "SDTLZ1")):
            G = 0.5 - F if key.startswith("IDTLZ1") else F / scale_base("SDTLZ1", M) ** np.arange(M)
            return np.abs(G.sum(1) - 0.5).max()
        if key.startswith("IDTLZ2"):
            return np.abs(((1 - F) ** 2).sum(1) - 1).max()
        if key.startswith("SDTLZ2"):
            return np.abs(((F / scale_base("SDTLZ2", M) ** np.arange(M)) ** 2).sum(1) - 1).max()
        return np.abs((F ** 2).sum(1) - 1).max()               # DTLZ5, DTLZ6: the sphere

    for key in ("ZDT1", "ZDT2", "ZDT3", "ZDT4", "ZDT6", "DTLZ5_3D", "DTLZ6_3D", "DTLZ6_2D",
                "DTLZ7_3D", "DTLZ7_5D", "IDTLZ1_3D", "IDTLZ2_5D", "SDTLZ1_3D", "SDTLZ2_5D",
                "minusDTLZ2_3D"):
        p = get(key)
        X = np.asarray(p.pareto_set(200), float)
        lo = np.array([b[0] for b in p.bounds]); hi = np.array([b[1] for b in p.bounds])
        assert np.all((X >= lo) & (X <= hi)), key
        F = np.array([p.evaluate(list(x)) for x in X])
        assert front_error(key, F) < 1e-9, (key, front_error(key, F))
    assert get("DTLZ6_4D").pareto_set is None and get("DTLZ5_5D").pareto_set is None


@test
def shifted_zdt_and_dtlz67_move_the_optimum_off_the_bound():
    """shiftZDT1-4, shiftZDT6, shiftDTLZ6/7 (task 6, shifted.py): the base
    problem at c = 0, h(c) = 0, the Pareto set on the base's front and at least
    0.15 from the bounds and the middle, one optimum per variable for
    shiftZDT4 away from the multiples of 1/2, the base's front and frame."""
    if not _have_numpy():
        print("    (skipped: no NumPy)"); return
    import numpy as np
    from mootation.benchmarks import get
    from mootation.benchmarks import shifted as S
    from mootation.benchmarks.dtlz import DTLZ_FUNCS
    from mootation.benchmarks.registry import FULL_FRONT
    rng = np.random.default_rng(3)
    for base in S.ZDT_BASES:
        b, s = get(base), get("shift" + base)
        lo = np.array([q[0] for q in b.bounds]); hi = np.array([q[1] for q in b.bounds])
        c = S.zdt_centres(base, b.n_vars)
        assert np.all(np.abs(S.zdt_input(base, np.concatenate([[0.3], c]), c)[1:]) == 0.0)
        for x in lo + rng.random((30, b.n_vars)) * (hi - lo):
            at_zero = S.zdt_input(base, x, np.zeros(b.n_vars - 1))       # c = 0: the base
            assert b.evaluate(list(at_zero)) == b.evaluate(list(x)), base
        assert np.array_equal(s.pareto_front(300), b.pareto_front(300)), base
        assert tuple(s.ideal) == tuple(b.ideal) and tuple(s.nadir) == tuple(b.nadir), base
        assert s.cyclic_vars == () and s.bounds == b.bounds, base
    for M in (2, 3, 5):
        for base in S.DTLZ_BASES:
            b = get(f"{base}_{M}D")
            x = rng.random(b.n_vars)
            zero = S.dtlz_input(x, M, np.zeros(b.n_vars - M + 1))
            assert DTLZ_FUNCS[base](list(zero), M) == b.evaluate(list(x)), (base, M)
    assert np.array_equal(get("shiftDTLZ6_5D").pareto_front(500), get("DTLZ6_5D").pareto_front(500))
    assert "shiftDTLZ6_5D" in FULL_FRONT and get("shiftDTLZ6_5D").pareto_set is None
    for key in ("shiftZDT1", "shiftZDT3", "shiftZDT4", "shiftZDT6", "shiftDTLZ6_3D",
                "shiftDTLZ7_3D", "shiftDTLZ7_5D"):
        p, b = get(key), get(key[5:])
        X = np.asarray(p.pareto_set(200), float)
        k = 1 if "ZDT" in key else p.n_obj - 1
        lo = np.array([q[0] for q in p.bounds])[k:]; hi = np.array([q[1] for q in p.bounds])[k:]
        U = (X[:, k:] - lo) / (hi - lo)
        assert np.min(np.minimum(np.minimum(U, 1 - U), np.abs(U - 0.5))) >= 0.15, key
        base_x = np.asarray(b.pareto_set(200), float)                    # the same positions
        F = np.array([p.evaluate(list(x)) for x in X])
        G = np.array([b.evaluate(list(x)) for x in base_x])
        assert np.abs(F - G).max() < 1e-12, key
    # shiftZDT4: x = c the only minimum along every variable, c off the 1/2 grid
    p, c = get("shiftZDT4"), S.zdt_centres("ZDT4", 10)
    assert np.min(np.abs(2 * c - np.round(2 * c))) / 2 > 0.02, c
    grid = np.linspace(-5.0, 5.0, 40001)
    for j in range(9):
        x = np.tile(np.concatenate([[0.4], c]), (len(grid), 1))
        x[:, j + 1] = grid
        f2 = np.array([p.evaluate(list(r))[1] for r in x[::10]])
        best = grid[::10][f2.argmin()]
        assert abs(best - c[j]) <= 2.5e-3 and np.sum(f2 <= f2.min() + 1e-12) == 1, (j, best, c[j])


@test
def boundary_and_centre_follow_the_pareto_sets():
    """properties' boundary agrees with what every Pareto-set sample shows
    (properties.measured, 99 % of the distance variables within 1e-9 of a
    bound), and so does centre where the table has a value; interior() is
    neither."""
    if not _have_numpy():
        print("    (skipped: no NumPy)"); return
    from mootation.benchmarks import names
    from mootation.benchmarks.properties import interior, measured, properties
    from mootation.benchmarks.registry import PROBLEMS
    checked = 0
    for name in names():
        if PROBLEMS[name].pareto_set is None or properties(name) is None:
            continue
        if name.startswith("ZCAT") and name not in ("ZCAT1_3D", "ZCAT11_3D", "ZCAT14_3D"):
            continue                                       # seconds each: three of them
        m, row = measured(name, 100), properties(name)
        if m is None:                                      # a failing copy, its set all failed
            assert PROBLEMS[name].failing is not None and row["boundary"] is None, name
            continue
        assert row["boundary"] is m["boundary"], (name, m, row["boundary"])
        assert row["centre"] is None or row["centre"] is m["centre"], (name, m, row["centre"])
        checked += 1
    assert checked > 100, checked
    assert [properties(n)["boundary"] for n in ("ZDT1", "ZDT4", "DTLZ7_3D", "WFG4_3D")] == \
        [True, False, True, False]
    assert interior("shiftZDT1") and interior("WFG4_3D") and interior("ZCAT1_3D")
    assert not interior("ZDT1") and not interior("DTLZ2_3D") and not interior("MaF1_3D")


@test
def convergence_indicators_without_coverage_vanish_on_the_pareto_set():
    """gdp_min, g_min, g_median and ps_dmin (task 6, item 8): g_min = g_median =
    ps_dmin = 0 on a Pareto-set sample, gdp_min below the reference's
    resolution there and flagged so; off the set g_min > 0; no distance
    function where g is no distance to the Pareto set."""
    if not _have_numpy():
        print("    (skipped: no NumPy)"); return
    import numpy as np
    from mootation.benchmarks import get
    from mootation.run import metrics as M
    which = ("gdp_min", "g_min", "g_median", "ps_dmin")
    for key in ("ZDT1", "ZDT4", "shiftZDT6", "DTLZ1_3D", "DTLZ7_3D", "shiftDTLZ2_3D",
                "shiftDTLZ7_3D", "IDTLZ1_3D", "wideDTLZ3_3D", "Polygon_3D"):
        p = get(key)
        X = np.asarray(p.pareto_set(60), float)
        F = np.array([p.evaluate(list(x)) for x in X])
        kw = dict(ref_front=np.asarray(p.pareto_front(500), float), ideal=p.ideal,
                  nadir=p.nadir, which=which, pareto_set=X, bounds=p.bounds,
                  cyclic=p.cyclic_vars, distance_fn=p.distance_fn)
        on = M.compute(F, X=X, **kw)
        assert on["ps_dmin"] == 0.0, (key, on)
        assert on["gdp_min"] < on["gdp_min_ref_step"] and on["gdp_min_below_step"], (key, on)
        if key == "Polygon_3D":                              # no g of its own
            assert p.distance_fn is None and on["g_min"] is None, on
            continue
        assert abs(on["g_min"]) < 1e-12 and abs(on["g_median"]) < 1e-12, (key, on)
        lo = np.array([b[0] for b in p.bounds]); hi = np.array([b[1] for b in p.bounds])
        k = 1 if "ZDT" in key else p.n_obj - 1
        Y = X.copy()
        Y[:, k:] = np.clip(Y[:, k:] + 0.05 * (hi - lo)[k:], lo[k:], hi[k:])
        off = M.compute(np.array([p.evaluate(list(y)) for y in Y]), X=Y, **kw)
        assert off["g_min"] > 1e-6 and off["ps_dmin"] > 1e-3, (key, off)
    assert all(get(k).distance_fn is None for k in ("DTLZ5_5D", "DTLZ6_4D", "shiftDTLZ6_5D",
                                                    "WFG4_3D", "ZCAT1_3D"))


@test
def decision_space_indicators_agree_with_closed_forms():
    if not _have_numpy():
        print("    (skipped: no NumPy)"); return
    import numpy as np
    from mootation.run import metrics as M
    b = [(0.0, 1.0), (0.0, 1.0)]
    ps = np.array([[0.0, 0.5], [1.0, 0.5]])
    assert M.igdx(ps, ps, bounds=b) == 0.0
    assert abs(M.igdx(np.array([[0.0, 0.5]]), ps, bounds=b) - 0.5) < 1e-12
    # the wrap: 0.98 and 0.02 are 0.04 apart on a cyclic variable, 0.96 on another
    assert abs(M.igdx(np.array([[0.0, 0.98]]), np.array([[0.0, 0.02]]), bounds=b,
                      cyclic=(1,)) - 0.04) < 1e-12
    assert abs(M.igdx(np.array([[0.0, 0.98]]), np.array([[0.0, 0.02]]), bounds=b)
               - 0.96) < 1e-12
    # bounds normalise: the same set on [0, 10] reads the same
    assert abs(M.igdx(np.array([[0.0, 5.0]]), np.array([[0.0, 5.0], [10.0, 5.0]]),
                      bounds=[(0.0, 10.0)] * 2) - 0.5) < 1e-12
    assert abs(M.cover_rate(np.array([[0.0, 0.5], [0.5, 0.5]]), ps) - 0.5 ** 0.5) < 1e-12
    assert M.cover_rate(np.array([[2.0, 2.0]]), ps) == 0.0
    assert abs(M.pairwise_distance(np.array([[0.0, 0.0], [1.0, 0.0]]), bounds=b) - 1.0) < 1e-12
    out = M.compute(np.array([[0.1, 0.9]]), which=["igdx", "cr", "pdist"])
    assert out == {"igdx": None, "cr": None, "pdist": None}, out


@test
def compiled_hypervolume_agrees_with_the_python_recursion():
    """_core.hypervolume (WFG in C++) and hv_covered against metrics' own Python."""
    if not _have_numpy():
        print("    (skipped: no NumPy)"); return
    core = _core_with("hv_covered")
    if core is None:
        print("  skip  compiled_hypervolume...: _core without hypervolume"); return
    import numpy as np
    from mootation.run import metrics as M
    rng = np.random.default_rng(7)
    for m in (2, 3, 4, 5):
        for _ in range(8):
            P = rng.random((int(rng.integers(1, 40)), m))
            P /= np.linalg.norm(P, axis=1, keepdims=True)
            P = M.nondominated(P)
            ref = np.full(m, 1.1)
            a, b = core.hypervolume(P, ref), M._hv_wfg(P, ref)
            assert abs(a - b) <= 1e-12 * max(1.0, b), (m, a, b)
    P = rng.random((30, 4))
    S = rng.random((5000, 4)) * 1.1
    brute = int(np.any(np.all(P[None, :, :] <= S[:, None, :], axis=2), axis=1).sum())
    assert core.hv_covered(P, S) == brute
    # the estimate does not depend on who counted: same samples, same count
    F = rng.random((60, 6)) * 0.9
    v1, how = M.hypervolume(F, np.zeros(6), np.ones(6), exact_max_m=5, mc_samples=30_000)
    saved = core.hv_covered
    try:
        del core.hv_covered
        v2, _ = M.hypervolume(F, np.zeros(6), np.ones(6), exact_max_m=5, mc_samples=30_000)
    finally:
        core.hv_covered = saved
    assert how == "mc" and v1 == v2, (how, v1, v2)
    exact, _ = M.hypervolume(F, np.zeros(6), np.ones(6), exact_max_m=6)
    assert abs(v1 - exact) < 0.02, (v1, exact)


@test
def trajectory_hypervolume_by_monte_carlo_above_the_exact_limit():
    """trajectory_hv_mc_samples fills the hypervolume in where it would be null."""
    if not _have_numpy() or _core_with("max_evaluations") is None:
        print("  skip  trajectory_hypervolume_by_monte_carlo...: no NumPy or stale _core")
        return
    from mootation.run import campaign as C
    from mootation.run.config import load
    text = """algorithms = [ { name = "nsga2", pop = 0, gens = 0 } ]
[run]
name = "mc"
[problem]
kind = "builtin"
[benchmarks]
runs = 1
problems = ["DTLZ2_5D"]
[campaign]
out = "res"
budget_fe = 400
record_grid = "log"
trajectory_hv_max_m = 3
trajectory_hv_mc_samples = 4000
metrics = ["igdp", "hv_h"]
final_metrics = ["hv_h"]
"""
    with tempfile.TemporaryDirectory() as td:
        cfg_path = Path(td) / "c.toml"
        cfg_path.write_text(text, encoding="utf-8")
        cfg = load(cfg_path)
        assert validate(cfg) == [], validate(cfg)
        spec = C.campaign_spec(cfg)
        assert (spec.trajectory_hv_mc_samples, spec.hv_exact_max_m) == (4000, 5)
        root = C.out_root(cfg, spec)
        (job,) = C.expand_jobs(cfg, spec)
        assert C.run_job(job, root, spec, quiet=True) == "done"
        d = root / job.rel_dir
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        T = [json.loads(l) for l in (d / "trajectory.jsonl").read_text().splitlines()]
        assert all(t["hv_h"] is not None and t["hv_method"] == "mc" for t in T), T[-1]
        assert meta["trajectory_hv"].startswith("monte-carlo") and not meta["trajectory_hv_skipped"]
        assert meta["final"]["hv_method"] == "exact", meta["final"]
        assert abs(T[-1]["hv_h"] - meta["final"]["hv_h"]) < 0.05, (T[-1], meta["final"])
    bad = text.replace("trajectory_hv_mc_samples = 4000", "trajectory_hv_mc_samples = -1")
    raises(ConfigError, C.campaign_spec, loads(bad))


@test
def statistics_match_hand_computed_cases():
    from mootation.run import stats as ST
    # the effect size, the correction and the ranks are plain Python ...
    assert ST.a12([1, 2], [3, 4]) == 1.0 and ST.a12([1, 3], [2, 3]) == 0.625
    assert ST.a12([1, 2], [3, 4], lower_better=False) == 0.0
    assert all(abs(x - y) < 1e-15 for x, y in zip(ST.holm([0.01, 0.04, 0.03]),
                                                   [0.03, 0.06, 0.06]))
    assert ST.average_ranks([3.0, 1.0, 3.0]) == [2.5, 1.0, 2.5]
    assert ST.mark(0.01, "A") == "+" and ST.mark(0.01, "B") == "-" and ST.mark(0.2, "A") == "="
    # ... the exact null distributions and the bootstrap are NumPy
    if not _have_numpy():
        print("    (exact tests and bootstrap skipped: no NumPy)"); return
    six = ST.wilcoxon_signed_rank([1, 2, 3, 4, 5, 6], [2, 4, 6, 8, 10, 12])
    assert six["better"] == "A" and six["w_plus"] == 21 and abs(six["p"] - 2 / 64) < 1e-12
    assert ST.wilcoxon_signed_rank([1, 2], [1, 2])["p"] == 1.0          # zeros dropped
    rs = ST.rank_sum([1, 2, 3], [4, 5, 6])
    assert rs["better"] == "A" and abs(rs["p"] - 2 / 20) < 1e-12, rs
    hb = ST.rank_sum([1, 2, 3], [4, 5, 6], lower_better=False)
    assert hb["better"] == "B" and abs(hb["p"] - 0.1) < 1e-12, hb
    tied = ST.rank_sum([1, 1, 2], [1, 2, 2])
    assert 0.0 < tied["p"] <= 1.0 and tied["better"] == "A", tied
    ci = ST.bootstrap_mean_ranks({"p1": {"a": 1, "b": 2}, "p2": {"a": 1, "b": 2}}, n_boot=50)
    assert ci == {"a": (1.0, 1.0), "b": (2.0, 2.0)}, ci


@test
def every_campaign_problem_has_a_property_row():
    """properties.py covers the problem lists of campaign_all and of stage 3
    (MOP, BT, Polygon, IPolygon); the groups read key=value."""
    if not _have_numpy():
        print("    (skipped: no NumPy)"); return
    from mootation.benchmarks.properties import KEYS, label, properties
    from mootation.run.config import load
    for camp, least in (("campaign_all.toml", 100), ("campaign_stage3.toml", 225)):
        cfg = load(HERE / "examples" / camp)
        assert validate(cfg) == [], (camp, validate(cfg)[:3])
        names = cfg.benchmark_problems
        assert len(names) >= least, (camp, len(names))
        missing = [n for n in names if properties(n) is None]
        assert not missing, (camp, missing[:10])
        assert all(set(properties(n)) == set(KEYS) for n in names), camp
    assert properties("DTLZ5_3D")["front"] == "degenerate"
    assert properties("DTLZ5_5D")["front"] == "degenerate+mixed"
    assert properties("WFG3_3D")["front"] == "degenerate+mixed"
    assert properties("shiftDTLZ2_3D")["centre"] is False and properties("DTLZ2_3D")["centre"]
    assert properties("SDTLZ1_3D")["scaled"] is True
    assert label("WFG5_3D", "deceptive") == "deceptive=yes"
    assert label("ZCAT1_3D", "front") == "front=?"
    # the keys beyond Huband's
    assert properties("DTLZ5_3D")["degenerate"] is True and not properties("DTLZ2_3D")["degenerate"]
    assert properties("ZCAT1_3D")["degenerate"] is None
    assert properties("MOP4")["disconnected"] is True and properties("BT5")["disconnected"] is True
    assert properties("MOP1")["linkage"] and properties("ZCAT1_3D")["linkage"]
    assert [properties(f"BT{i}")["linkage"] for i in range(1, 10)] == \
        [False] * 5 + [True] * 3 + [False]
    assert properties("DTLZ2_3D")["linkage"] is False and properties("WFG4_3D")["linkage"] is None
    assert properties("BT8")["multimodal"] and not properties("BT7")["multimodal"]
    assert properties("Polygon_3D")["front"] == "linear"
    assert properties("Polygon_4D")["degenerate"] is True
    assert properties("IPolygon_3D")["multimodal"] is True
    # the instance of reference version 2: four polygons far enough apart
    assert properties("IPolygon_3D")["front"] == "linear" and properties("IPolygon_4D")["degenerate"]
    assert [properties(n)["front_dim"] for n in ("ZDT1", "DTLZ2_3D", "DTLZ2_5D", "DTLZ5_3D",
                                                 "DTLZ5_5D", "ZCAT14_5D", "Polygon_6D",
                                                 "bbobbiobj01_n05_2D")] == \
        [1, 2, 4, 1, None, 1, 2, 1]
    assert (properties("ZDT1")["n_obj"], properties("ZDT1")["n_vars"]) == (2, 30)
    assert properties("DTLZ2_5D")["n_obj"] == 5
    assert properties("bbobbiobj01_n05_2D")["bbob_groups"] == "separable+separable"
    assert properties("bbobbiobj55_n10_2D")["bbob_groups"] == "weakly-structured+weakly-structured"
    assert properties("ZDT1")["bbob_groups"] is None
    assert label("BT7", "linkage") == "linkage=yes"


@test
def postprocessing_runtimes_gaps_and_budget_marks():
    from mootation.run import postprocess as PP
    recs = [{"fe": 100, "igdp": 1.0}, {"fe": 200, "igdp": 0.5}, {"fe": 400, "igdp": 0.1}]
    assert PP.runtime_to_target(recs, "igdp", 0.3) == 400                       # step
    assert abs(PP.runtime_to_target(recs, "igdp", 0.3, interpolation="linear") - 300) < 1e-9
    assert PP.runtime_to_target(recs, "igdp", 0.01) is None
    hv = [{"fe": 100, "hv": 0.1}, {"fe": 300, "hv": 0.5}]
    assert PP.runtime_to_target(hv, "hv", 0.3, interpolation="linear") == 200.0
    rows = [{"problem": "P", "algorithm": a, "status": "done", "final": {"hv": v}}
            for a, v in (("x", 0.0), ("x", 0.4), ("y", 0.5))]
    assert PP.zero_share(rows) == {"P": {"x": 0.5, "y": 0.0}}
    assert PP.gap_to_best(rows, "hv") == {"P": {"x": [0.5, 0.09999999999999998], "y": [0.0]}}
    assert len(PP.BUDGET_SCHEDULED) == 18 and PP.budget_note("mosa") == "*"
    assert PP.budget_note("nsga2") == "" and set(PP.BUDGET_SCHEDULED) <= set(algorithm_names())
    ecdf = PP.runtime_ecdf([recs], "igdp", [0.3, 0.01], grid=[150, 450])
    assert ecdf["share"] == [0.0, 0.5] and ecdf["pairs"] == 2, ecdf


@test
def the_report_tests_rank_and_group_a_toy_campaign():
    """reference_report, rank_intervals, property_ranks on hand-made rows."""
    if not _have_numpy():
        print("    (skipped: no NumPy)"); return
    from mootation.run import report as R
    rows = []
    for p in ("ZDT1", "ZDT2", "ZDT4", "DTLZ2_3D", "WFG4_3D", "WFG5_3D"):
        for s in range(5):
            rows.append({"problem": p, "algorithm": "good", "status": "done",
                         "final": {"igdp": 0.01 + 0.001 * s}, "dir": "."})
            rows.append({"problem": p, "algorithm": "bad", "status": "done",
                         "final": {"igdp": 0.5 + 0.01 * s}, "dir": ".",
                         "final_archive": {"igdp": 0.001 * s}})
            rows.append({"problem": p, "algorithm": "same", "status": "done",
                         "final": {"igdp": 0.01 + 0.001 * s}, "dir": "."})
    rep = R.reference_report(rows, "igdp", "good")
    by = {e["algorithm"]: e for e in rep["algorithms"]}
    assert by["bad"]["losses"] == 6 and by["bad"]["wins"] == 0, by["bad"]
    assert by["same"]["ties"] == 6 and by["same"]["mark"] == "=", by["same"]
    assert by["bad"]["a12_median"] == 0.0
    # six problems all one way: the exact signed-rank p is 2/2^6
    assert abs(by["bad"]["signed_rank"]["p"] - 2 / 64) < 1e-12
    ci = R.rank_intervals(rows, "igdp", n_boot=200)
    assert ci["bad"][0] == 3.0 and ci["good"][1] == 1.5, ci
    g = R.property_ranks(rows, "igdp", "multimodal")
    assert set(g) == {"multimodal=yes", "multimodal=no"}, g
    assert g["multimodal=yes"]["bad"] == (3.0, 3), g        # ZDT4, WFG4, WFG5
    assert "bad" in R.format_reference(rep)
    assert "multimodal" in R.format_property_ranks(g, "igdp", "multimodal")
    # the archive scenario reads final_archive, where "bad" is the best
    arch = R.values(rows, "igdp", scenario="archive")
    assert set(arch["ZDT1"]) == {"bad"} and arch["ZDT1"]["bad"][0] == 0.0, arch["ZDT1"]
    raises(ValueError, R.property_ranks, rows, "igdp", "colour")


@test
def the_reference_report_warns_when_no_mark_is_possible_and_sizes_its_effects():
    """--reference (task 6, item 10): at 5 seeds and 110 problems Holm leaves no
    mark but '=' possible (110 > 0.025 C(10, 5) = 6.3), and the report says so;
    A12's class on shifts of known size; Benjamini-Hochberg, the sign test and
    the Hodges-Lehmann estimate on values worked by hand; igdp_norm compared
    across problems by the logarithm, its HL as a ratio."""
    if not _have_numpy():
        print("    (skipped: no NumPy)"); return
    import math
    from mootation.run import report as R
    from mootation.run import stats as ST

    def rows_of(per_problem):
        out = []
        for p, by_alg in per_problem.items():
            for alg, vals in by_alg.items():
                out += [{"problem": p, "algorithm": alg, "status": "done", "dir": ".",
                         "final": {"igdp_norm": v}} for v in vals]
        return out

    ref = [1.0, 2.0, 3.0, 4.0, 5.0]
    many = rows_of({f"P{i}": {"ref": ref, "x": [v + 10.0 for v in ref]} for i in range(110)})
    rep = R.reference_report(many, "igdp_norm", "ref")
    assert rep["smallest_p"] == 2 / math.comb(10, 5) and 110 > 0.025 * math.comb(10, 5)
    assert not rep["labels_possible"] and rep["algorithms"][0]["ties"] == 110, rep
    assert rep["algorithms"][0]["ties_large"] == 110
    assert "WARNING: no mark but '='" in " ".join(R.format_reference(rep).split())
    assert "0/0/0/110" in R.format_reference(rep)                 # A12 classes: all large
    assert R.reference_report(many[:6 * 10], "igdp_norm", "ref")["labels_possible"]
    assert R.reference_report(many, "igdp_norm", "ref", adjust="bh")["labels_possible"]
    raises(ValueError, R.reference_report, many, "igdp_norm", "ref", adjust="bonferroni")

    # A12 of a against the reference, lower better: 15/25 = 0.6 (small),
    # 17/25 = 0.68 (medium), 0.5 (negligible), 0 (large, worse)
    shifts = {"small": [v - 0.5 for v in ref], "medium": [v - 1.0 for v in ref],
              "negligible": list(ref), "large": [v + 10.0 for v in ref]}
    rep = R.reference_report(rows_of({"P": dict(shifts, ref=ref)}), "igdp_norm", "ref")
    for e in rep["algorithms"]:
        got = e["per_problem"]["P"]
        assert got["a12_class"] == e["algorithm"], (e["algorithm"], got)
    a12 = {e["algorithm"]: e["per_problem"]["P"]["a12"] for e in rep["algorithms"]}
    assert (a12["small"], a12["medium"], a12["negligible"], a12["large"]) == (0.6, 0.68, 0.5, 0.0)

    adj = ST.benjamini_hochberg([0.01, 0.04, 0.03, 0.2])
    assert all(abs(x - y) < 1e-12 for x, y in zip(adj, [0.04, 0.16 / 3, 0.16 / 3, 0.2])), adj
    assert ST.sign_test([1.0] * 10, [2.0] * 10)["p"] == 2 / 2 ** 10
    assert ST.sign_test([1.0, 3.0, 2.0], [2.0, 2.0, 2.0]) == {"n": 2, "wins": 1, "losses": 1,
                                                             "p": 1.0}
    assert ST.hodges_lehmann([1.0, 2.0, 3.0]) == 2.0          # Walsh: 1, 1.5, 2, 2, 2.5, 3
    half = rows_of({f"P{i}": {"ref": [2.0 * (i + 1)] * 3, "x": [1.0 * (i + 1)] * 3}
                    for i in range(8)})
    e = R.reference_report(half, "igdp_norm", "ref")["algorithms"][0]
    assert abs(e["hl_ratio"] - 0.5) < 1e-12 and e["sign"]["wins"] == 8, e
    text = " ".join(R.format_reference(R.reference_report(half, "igdp_norm", "ref")).split())
    assert "HL ratio" in text and "not a rating" in text


@test
def the_ablation_baselines_spend_the_budget_and_behave():
    if not _have_numpy():
        print("    (skipped: no NumPy)"); return
    import numpy as np
    from mootation.run import ablations as A
    from mootation.run.baselines import BASELINES
    assert {"random_selection_ea", "gsemo"} <= set(BASELINES)
    lo, hi = np.zeros(5), np.ones(5)
    calls = []

    def f(x):
        calls.append(x)
        return [x[0], 1.0 - x[0] ** 0.5 + sum(x[1:])]
    F, X, gens = A.random_selection_ea(f, lo, hi, pop=10, max_evaluations=95, seed=1)
    assert len(calls) == 95 and len(F) == 10 and np.all((X >= 0) & (X <= 1))
    calls.clear()

    def pick(Fa, k):
        return np.arange(min(k, len(Fa)))
    F, X, steps = A.gsemo(f, lo, hi, pop=10, max_evaluations=300, seed=1, select=pick)
    assert len(calls) == 300 and len(F) <= 10
    dom = (np.all(F[:, None] <= F[None], 2) & np.any(F[:, None] < F[None], 2)).any(0)
    assert not dom.any()                                   # the population is nondominated
    # the operators keep children in the box however close the parents sit to it
    rng = np.random.default_rng(3)
    for _ in range(200):
        a, b = rng.random(5), rng.random(5)
        c1, c2 = A.sbx(a, b, lo, hi, rng)
        m = A.polynomial_mutation(c1.copy(), lo, hi, rng, pm=1.0)
        assert np.all((c1 >= 0) & (c1 <= 1) & (c2 >= 0) & (c2 <= 1) & (m >= 0) & (m <= 1))


@test
def archive_scenario_recompute_and_report_flags_end_to_end():
    """final_archive, --recompute --scenario archive, and every report flag, on a tiny campaign."""
    if not _have_numpy() or _core_with("max_evaluations") is None:
        print("  skip  archive_scenario...: no NumPy or stale _core"); return
    import contextlib
    import io
    from mootation.run import campaign as C
    from mootation.run.config import load
    text = """algorithms = [
    { name = "nsga2", pop = 0, gens = 0 },
    { name = "rvea", pop = 0, gens = 0 },
    { name = "gsemo", pop = 0, gens = 0 },
    { name = "random_selection_ea", pop = 0, gens = 0 },
]
[run]
name = "rep"
[problem]
kind = "builtin"
[benchmarks]
runs = 2
problems = ["WFG4_2D", "DTLZ2_3D"]
[campaign]
out = "res"
budget_fe = 500
record_grid = "log"
metrics = ["igdp", "igdp_norm", "hv"]
final_metrics = ["igdp", "hv", "igdx", "pdist"]
"""
    with tempfile.TemporaryDirectory() as td:
        cfg_path = Path(td) / "c.toml"
        cfg_path.write_text(text, encoding="utf-8")
        cfg = load(cfg_path)
        assert validate(cfg) == [], validate(cfg)
        spec = C.campaign_spec(cfg)
        root = C.out_root(cfg, spec)
        for job in C.expand_jobs(cfg, spec):
            assert C.run_job(job, root, spec, quiet=True) == "done", job
        d = root / "DTLZ2_3D" / "nsga2" / "run_1"
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        fa = meta["final_archive"]
        assert 0 < fa["n"] <= 91 and fa["igdx"] is not None and fa["pdist"] > 0, fa
        assert meta["final"]["igdx"] is not None and meta["archive"]["frame"], meta["final"]
        assert json.loads((root / "WFG4_2D" / "gsemo" / "run_1" / "meta.json").read_text(
            encoding="utf-8"))["final"]["igdx"] is None             # WFG4_2D has no Pareto set
        # recomputing from archive.csv reproduces what the run wrote (up to the
        # ten digits the csv keeps)
        for key in ("final_archive", "final"):
            meta[key] = {}
        (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
        for scen in ("archive", "final"):
            counts = C.recompute_final(root, ["igdp", "hv", "igdx"], scenario=scen)
            assert counts == {"done": 16}, (scen, counts)
        again = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        assert again["final_archive"]["n"] == fa["n"], again["final_archive"]
        for k in ("igdp", "hv", "igdx"):
            assert abs(again["final_archive"][k] - fa[k]) < 1e-8, (k, again["final_archive"])
        rows = C.scan_results(root)
        assert all(r["final_archive"] for r in rows), [r["dir"] for r in rows if not r["final_archive"]]

        def cli(*argv):
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = C.main([str(cfg_path), *argv])
            return code, out.getvalue()
        code, txt = cli("--ranks", "igdp", "--ci")
        assert code == 0 and "bootstrap" in txt and "rvea*" in txt, txt
        code, txt = cli("--ranks", "hv", "--reference", "nsga2", "--scenario", "archive")
        assert code == 0 and "(archive) against nsga2" in txt, txt
        code, txt = cli("--ranks", "hv", "--reference", "nsga2", "--adjust", "bh")
        assert code == 0 and "Benjamini-Hochberg" in " ".join(txt.split()), txt
        code, txt = cli("--ranks", "igdp", "--by", "front")
        assert code == 0 and "grouped by front" in txt, txt
        code, txt = cli("--compare", "igdx")
        assert code == 0 and "DTLZ2_3D" in txt and "WFG4_2D" not in txt, txt
        for argv in (("--gap", "igdp"), ("--zero-share",),
                     ("--ecdf", "igdp_norm", "--interpolation", "linear")):
            code, txt = cli(*argv)
            assert code == 0 and txt.strip(), (argv, txt)
        assert cli("--ecdf", "hv", "--scenario", "archive")[0] == 1
        assert cli("--ci")[0] == 1 and cli("--gap", "nonsense")[0] == 1
        assert cli("--ranks", "igdp", "--reference", "nobody")[0] == 1


@test
def archive_variables_selected_keeps_the_same_archive_scenario():
    """archive_variables = "selected": the objectives of every archive point in
    archive.csv.gz, the selected points with their variables in
    final_archive.csv.gz; the archive scenario and --recompute give what "all"
    gives."""
    if not _have_numpy() or _core_with("max_evaluations") is None:
        print("  skip  archive_variables...: no NumPy or stale _core"); return
    import gzip
    from mootation.run import campaign as C
    from mootation.run.config import ConfigError, load
    text = """algorithms = [
    { name = "nsga2", pop = 0, gens = 0 },
    { name = "random_search", pop = 0, gens = 0 },
]
[run]
name = "sel"
[problem]
kind = "builtin"
[benchmarks]
runs = 1
problems = ["ZDT1", "DTLZ2_3D"]
[campaign]
out = "res_{keep}"
budget_fe = 500
record_grid = "log"
metrics = ["igdp"]
final_metrics = ["igdp", "hv", "igdx", "pdist"]
archive_variables = "{keep}"
"""
    with tempfile.TemporaryDirectory() as td:
        roots = {}
        for keep in ("all", "selected"):
            cfg_path = Path(td) / f"{keep}.toml"
            cfg_path.write_text(text.replace("{keep}", keep), encoding="utf-8")
            cfg = load(cfg_path)
            assert validate(cfg) == [], validate(cfg)
            spec = C.campaign_spec(cfg)
            roots[keep] = C.out_root(cfg, spec)
            for job in C.expand_jobs(cfg, spec):
                assert C.run_job(job, roots[keep], spec, quiet=True) == "done", job
        for prob, m, n in (("ZDT1", 2, 30), ("DTLZ2_3D", 3, 12)):
            for alg in ("nsga2", "random_search"):
                a = roots["all"] / prob / alg / "run_1"
                s = roots["selected"] / prob / alg / "run_1"
                ma = json.loads((a / "meta.json").read_text(encoding="utf-8"))
                ms = json.loads((s / "meta.json").read_text(encoding="utf-8"))
                assert (a / "archive.csv").exists() and not (a / "final_archive.csv.gz").exists()
                assert not (s / "archive.csv").exists(), (prob, alg)
                assert ms["archive"]["file"] == "archive.csv.gz", ms["archive"]
                with gzip.open(s / "archive.csv.gz", "rt", encoding="utf-8") as fh:
                    rows = [l.strip() for l in fh if l.strip() and not l.startswith("#")][1:]
                assert len(rows) == ms["archive"]["size"] > 0, (prob, alg)
                assert all(len(r.split(",")) == m + 1 for r in rows), rows[0]   # f, extreme
                F, X = C._read_points(s / "final_archive.csv.gz", m, n)
                assert len(F) == ms["final_archive"]["n"] and X.shape == (len(F), n), X.shape
                # the same run, the same archive, the same selection
                Fa, _ = C._read_points(a / "archive.csv", m, n)
                Fs, _ = C._read_points(s / "archive.csv.gz", m, 0)
                assert Fa.tolist() == Fs.tolist(), (prob, alg)
                for key in ("final", "final_archive"):
                    assert ma[key] == ms[key], (prob, alg, key)
        # --recompute --scenario archive reads the kept selection
        d = roots["selected"] / "DTLZ2_3D" / "nsga2" / "run_1"
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        fa, meta["final_archive"] = meta["final_archive"], {}
        (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
        counts = C.recompute_final(roots["selected"], ["igdp", "hv", "igdx"], scenario="archive")
        assert counts == {"done": 4}, counts
        again = json.loads((d / "meta.json").read_text(encoding="utf-8"))["final_archive"]
        assert again["n"] == fa["n"], again
        for k in ("igdp", "hv", "igdx"):
            assert abs(again[k] - fa[k]) < 1e-8, (k, again, fa)
        bad = Path(td) / "bad.toml"
        bad.write_text(text.replace("{keep}", "some"), encoding="utf-8")
        try:
            C.campaign_spec(load(bad))
            raise AssertionError("archive_variables = \"some\" accepted")
        except ConfigError:
            pass


@test
def answer_select_igdp_self_and_the_oracle():
    """answer_select = "igdp_self" (task 6, item 13): the lazy greedy chooses
    what the plain one does; on a dense front its N points cover the archive
    thinned by DSS better than DSS's N; a campaign records the choice and
    --recompute --scenario archive repeats it; --oracle measures every run's
    archive against the reference, igdp_self's value the run's own answer and
    the whole archive a bound for both."""
    if not _have_numpy() or _core_with("max_evaluations") is None:
        print("  skip  answer_select_igdp_self...: no NumPy or stale _core"); return
    import numpy as np
    from mootation.run import archive as A, campaign as C, oracle as O
    from mootation.run.config import ConfigError, load
    from mootation.run.metrics import igd_plus

    def plain(D, k):
        chosen = [int(np.argmin(D.mean(axis=1)))]
        cur = D[chosen[0]].copy()
        while len(chosen) < k:
            gain = np.maximum(cur[None, :] - D, 0.0).mean(axis=1)
            gain[chosen] = -1.0
            c = int(np.argmax(gain))
            chosen.append(c)
            cur = np.minimum(cur, D[c])
        return chosen

    for seed in range(4):
        rng = np.random.default_rng(seed)
        C_ = rng.random((300, 3))
        C_[150:170] = C_[:20]                                   # ties
        D = A.dplus_matrix(C_, rng.random((80, 3)))
        assert A.greedy_igdp(D, 40).tolist() == plain(D, 40), seed
    t = np.sort(np.random.default_rng(9).random(3000))
    F = np.column_stack([t, 1.0 - np.sqrt(t)])                  # ZDT1's front
    proxy = F[A.dss_order(F, k=A.SELF_REFERENCE, ideal=[0, 0], nadir=[1, 1])]
    mine = A.igdp_self_order(F, 50, ideal=[0, 0], nadir=[1, 1])
    assert len(set(mine.tolist())) == 50
    dss = A.dss_order(F, k=50, ideal=[0, 0], nadir=[1, 1])
    assert igd_plus(F[mine], proxy) < igd_plus(F[dss], proxy)
    assert A.igdp_self_order(F[:30], 50).tolist() == list(range(30))
    text = """algorithms = [{ name = "nsga2", pop = 0, gens = 0 }]
[run]
name = "self"
[problem]
kind = "builtin"
[benchmarks]
runs = 1
problems = ["DTLZ2_3D"]
[campaign]
out = "res"
budget_fe = 3000
metrics = ["igdp"]
final_metrics = ["igdp_norm", "hv"]
answer_select = "{how}"
"""
    with tempfile.TemporaryDirectory() as td:
        cfg_path = Path(td) / "c.toml"
        cfg_path.write_text(text.replace("{how}", "igdp_self"), encoding="utf-8")
        cfg = load(cfg_path)
        assert validate(cfg) == [], validate(cfg)
        spec = C.campaign_spec(cfg)
        assert spec.answer_select == "igdp_self"
        root = C.out_root(cfg, spec)
        for job in C.expand_jobs(cfg, spec):
            assert C.run_job(job, root, spec, quiet=True) == "done", job
        d = root / "DTLZ2_3D" / "nsga2" / "run_1"
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        assert meta["archive"]["answer_select"] == "igdp_self", meta["archive"]
        assert meta["archive"]["size"] > 91 and meta["final_archive"]["n"] == 91, meta["archive"]
        kept = meta["final_archive"]
        meta["final_archive"] = {}
        (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
        assert C.recompute_final(root, ["igdp_norm", "hv"], scenario="archive") == {"done": 1}
        again = json.loads((d / "meta.json").read_text(encoding="utf-8"))["final_archive"]
        assert abs(again["igdp_norm"] - kept["igdp_norm"]) <= 1e-8, (again, kept)  # 10 digits
        out = O.run(root)
        assert "1 runs measured" in out, out
        with (root / "_oracle" / "oracle.csv").open(encoding="utf-8") as fh:
            row = next(csv.DictReader(fh))
        v = {k: float(row[k]) for k in O.COLUMNS if k.startswith("igdp_norm")}
        assert int(row["n"]) == 91 and int(row["archive_size"]) == meta["archive"]["size"]
        assert abs(v["igdp_norm_igdp_self"] - kept["igdp_norm"]) <= 1e-8, (v, kept)
        assert v["igdp_norm_archive_whole"] <= min(v["igdp_norm_oracle"], v["igdp_norm_archive"])
        cfg_path.write_text(text.replace("{how}", "best"), encoding="utf-8")
        try:
            C.campaign_spec(load(cfg_path))
            raise AssertionError("answer_select = \"best\" accepted")
        except ConfigError:
            pass


@test
def campaign_problems_flag_restricts_every_mode_to_a_subset():
    """--problems: --list, the run, --force, the tables and --recompute see only those problems."""
    if not _have_numpy() or _core_with("max_evaluations") is None:
        print("  skip  campaign_problems_flag...: no NumPy or stale _core"); return
    import contextlib
    import io
    from mootation.run import campaign as C
    text = """algorithms = [ { name = "random_search", pop = 0, gens = 0 } ]
[run]
name = "sub"
[problem]
kind = "builtin"
[benchmarks]
runs = 1
problems = ["ZDT1", "ZDT2", "ZDT3"]
[campaign]
out = "res"
budget_fe = 300
metrics = ["igdp"]
"""
    with tempfile.TemporaryDirectory() as td:
        cfg_path = Path(td) / "c.toml"
        cfg_path.write_text(text, encoding="utf-8")
        root = Path(td) / "res"      # one seed: one job a problem, run in-process

        def cli(*argv):
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = C.main([str(cfg_path), *argv])
            return code, out.getvalue()

        def runs():
            return {m.parents[2].name: m.stat().st_mtime_ns
                    for m in root.glob("*/*/run_*/meta.json")}
        code, txt = cli("--list", "--problems", "ZDT3, ZDT1")
        assert code == 0 and "ZDT2" not in txt and txt.count("random_search") == 2, txt
        assert cli("--problems", "ZDT3")[0] == 0
        first = runs()
        assert set(first) == {"ZDT3"}, first
        assert cli("--problems", "ZDT3")[0] == 0 and runs() == first     # done: skipped
        assert cli("--problems", "ZDT3", "--force")[0] == 0
        assert runs()["ZDT3"] != first["ZDT3"]                          # rerun
        assert cli("--problems", "ZDT1")[0] == 0 and set(runs()) == {"ZDT1", "ZDT3"}
        code, txt = cli("--ranks", "igdp", "--problems", "ZDT3")
        assert code == 0 and "over 1 problem(s)" in txt, txt
        assert cli("--recompute", "igdp,roi_dist", "--problems", "ZDT1")[0] == 0
        meta = json.loads(next((root / "ZDT1").glob("*/run_1/meta.json")).read_text(
            encoding="utf-8"))
        other = json.loads(next((root / "ZDT3").glob("*/run_1/meta.json")).read_text(
            encoding="utf-8"))
        assert "roi_dist" in meta["final"] and "roi_dist" not in other["final"]
        assert cli("--problems", "ZDT9")[0] == 1
        assert cli("--problems", "ZDT1", "--job", "0")[0] == 1
        assert cli("--problems", "ZDT1", "--emit-slurm", "2")[0] == 1


# ── task 2, step 1: bound repair, operator statistics, tau90, n_final ───────

@test
def text_knobs_take_their_words_and_refuse_others():
    """bound_repair is a word from settings.hpp's text_knobs(); anything else is refused."""
    from mootation.run.knobs import text_knobs
    assert "reflect" in text_knobs()["bound_repair"], text_knobs()
    assert "bound_repair" in knob_names()
    tail = 'gens = 5\n'
    cfg = loads(_MIN.replace(tail, tail + 'params = { bound_repair = "reflect" }\n'))
    assert cfg.algorithms[0].params == {"bound_repair": "reflect"}, cfg.algorithms[0].params
    assert not [q for q in validate(cfg) if "bound_repair" in q or "params" in q], validate(cfg)
    bad = loads(_MIN.replace(tail, tail + 'params = { bound_repair = "clamp" }\n'))
    assert any("bound_repair" in q and "clamp" in q for q in validate(bad)), validate(bad)
    raises(ConfigError, loads, _MIN.replace(tail, tail + 'params = { bound_repair = 2 }\n'))
    raises(ConfigError, loads, _MIN.replace(tail, tail + 'params = { eta_c = "big" }\n'))


@test
def tau90_is_a_quantile_of_the_igd_plus_distances():
    if not _have_numpy():
        print("    (skipped: no NumPy)"); return
    import numpy as np
    from mootation.run import metrics as M
    ref = np.array([[0.0, 1.0], [0.25, 0.75], [0.5, 0.5], [0.75, 0.25], [1.0, 0.0]])
    box = dict(ref_front=ref, ideal=[0.0, 0.0], nadir=[1.0, 1.0])
    out = M.compute(ref, which=["tau90", "n_final"], **box)
    assert out["tau90"] == 0.0 and out["n_final"] == 5, out
    assert all(v == 1.0 for v in out["coverage_curve"].values()), out
    # every reference point's nearest dominated offset is the 0.03 shift
    out = M.compute(ref + [0.03, 0.0], which=["tau90", "igdp_norm"], **box)
    assert abs(out["tau90"] - 0.03) < 1e-12 and abs(out["igdp_norm"] - 0.03) < 1e-12, out
    assert out["coverage_curve"] == {"0.01": 0.0, "0.02": 0.0, "0.05": 1.0, "0.1": 1.0,
                                     "0.2": 1.0}, out["coverage_curve"]
    # one reference point far from the set moves the maximum, not the 0.9 quantile
    ref20 = np.column_stack([np.linspace(0, 1, 20), 1 - np.linspace(0, 1, 20)])
    ref21 = np.vstack([ref20, [[-0.5, -0.5]]])
    d = M.dplus_distances(ref20, ref21)
    assert d.max() > 0.5 and M.tau_quantile(ref20, ref21) == 0.0, (d.max(), M.tau_quantile(ref20, ref21))
    # weakly Pareto-compliant: a set that weakly dominates another is never worse
    rng = np.random.default_rng(5)
    for _ in range(20):
        B = rng.random((8, 2))
        Ab = np.minimum(B, B + rng.normal(0, 0.1, B.shape))
        assert M.tau_quantile(Ab, ref20) <= M.tau_quantile(B, ref20) + 1e-12
    assert M.compute(np.zeros((0, 2)), which=["n_final", "tau90"], **box) == {
        "n_final": 0, "tau90": None}


@test
def gap_max_is_the_largest_hole_and_nn_cv_the_evenness():
    """Task 3, item 6: gap_max = max over the reference of d+(z, A) in igdp_norm's
    frame; nn_cv = CV of the nearest-neighbour distances among the non-dominated."""
    if not _have_numpy():
        print("    (skipped: no NumPy)"); return
    import numpy as np
    from mootation.run import metrics as M
    ref = np.column_stack([np.linspace(0, 1, 21), 1 - np.linspace(0, 1, 21)])
    box = dict(ref_front=ref, ideal=[0.0, 0.0], nadir=[2.0, 2.0])
    out = M.compute(ref, which=["gap_max", "nn_cv"], **box)
    assert out["gap_max"] == 0.0 and abs(out["nn_cv"]) < 1e-9, out       # the front, evenly
    # the left half missing: the hole is the far end, in the normalised frame (/2)
    A = ref[10:]
    out = M.compute(A, which=["gap_max", "tau90", "igdp_norm"], **box)
    assert abs(out["gap_max"] - 0.25) < 1e-12, out                       # d+ to (0.5, 0.5) from (0, 1)
    assert out["gap_max"] >= out["tau90"] >= 0.0 and out["gap_max"] >= out["igdp_norm"]
    assert out["gap_max"] == M.dplus_distances((A - 0) / 2, ref / 2).max()
    # a clump raises nn_cv; a dominated point and a copy do not count
    clumped = np.vstack([ref[:3], ref[[10, 20]]])
    assert M.compute(clumped, which=["nn_cv"], **box)["nn_cv"] > 0.5
    even = ref[::5]
    noisy = np.vstack([even, even[2] + 0.3, even[1]])
    assert abs(M.nn_cv(noisy, [0, 0], [2, 2]) - M.nn_cv(even, [0, 0], [2, 2])) < 1e-12
    assert M.nn_cv(ref[:1], [0, 0], [2, 2]) is None
    assert M.compute(A, which=["gap_max"], ideal=[0, 0], nadir=[2, 2]) == {"gap_max": None}
    from mootation.run.metric_names import DESCRIPTIONS, METRIC_NAMES, NEEDS_FRONT
    assert {"gap_max", "nn_cv"} <= set(METRIC_NAMES) and "gap_max" in NEEDS_FRONT
    assert "nn_cv" not in NEEDS_FRONT and {"gap_max", "nn_cv"} <= set(DESCRIPTIONS)


@test
def de_and_the_portfolio_record_what_they_do():
    """Task 7, §2: crossover = de_pbest with SHADE and = portfolio in a
    campaign — the SHADE memory and the CR and F drawn and kept along the
    trajectory, every operator's share of the children and of them kept;
    a TOML list for `portfolio`; the knobs after the per-knob setters reach
    the binding through Config.params / Config.words, and their lists agree
    with settings.hpp; mutation_scale 0 is refused."""
    if not _have_numpy() or _core_with("params") is None:
        print("  skip  de_and_the_portfolio...: no NumPy or stale _core"); return
    from mootation.run import campaign as C
    from mootation.run import knobs as K
    from mootation.run.config import load
    from mootation.run.metric_names import DE_STATS, is_table_name
    import mootation
    assert callable(mootation.minimize)                # loads the module behind the function
    Mz = sys.modules["mootation.minimize"]
    assert set(Mz.PARAM_KNOBS) <= set(K.knob_names()), set(Mz.PARAM_KNOBS) - set(K.knob_names())
    assert set(Mz.WORD_KNOBS) <= set(K.text_knobs()), set(Mz.WORD_KNOBS) - set(K.text_knobs())
    assert K.LIST_KNOBS == ("portfolio",) and K.accepts("portfolio", ["sbx", "de"])
    assert not K.accepts("portfolio", ["sbx", "nope"]) and K.accepts("portfolio", "sbx, pcx".replace(" ", ""))
    text = """algorithms = [
    { name = "nsga2", label = "shade", pop = 20, gens = 0, params = { crossover = "de_pbest", de_adapt = "shade", mutation = "none" } },
    { name = "sms_emoa", label = "pf", pop = 20, gens = 0, params = { crossover = "portfolio", portfolio = ["sbx", "de", "pcx"], portfolio_credit = "borg" } },
]
[run]
name = "de"
[problem]
kind = "builtin"
[benchmarks]
runs = 1
problems = ["ZDT1"]
[campaign]
out = "res"
budget_fe = 1200
record_grid = "log"
metrics = ["igdp_norm"]
final_metrics = ["igdp_norm"]
operator_stats = true
"""
    with tempfile.TemporaryDirectory() as td:
        cfg_path = Path(td) / "c.toml"
        cfg_path.write_text(text, encoding="utf-8")
        cfg = load(cfg_path)
        assert validate(cfg) == [], validate(cfg)
        spec = C.campaign_spec(cfg)
        root = C.out_root(cfg, spec)
        for job in C.expand_jobs(cfg, spec):
            assert C.run_job(job, root, spec, quiet=True) == "done", job
        d = root / "ZDT1" / "shade" / "run_1"
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        T = [json.loads(l) for l in (d / "trajectory.jsonl").read_text().splitlines()]
        assert {o["operator"] for o in meta["operators"]} == {"de_pbest", "none"}, meta["operators"]
        assert all(k in meta["final"] for k in DE_STATS), sorted(meta["final"])
        assert 0.0 <= T[-1]["de_mu_cr_mean"] <= 1.0 and T[-1]["de_mu_f_mean"] > 0.0, T[-1]
        assert all(is_table_name(k) for k in DE_STATS)
        d = root / "ZDT1" / "pf" / "run_1"
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        shares = {k: v for k, v in meta["final"].items() if k.startswith("portfolio_share_")}
        assert set(shares) == {"portfolio_share_sbx", "portfolio_share_de", "portfolio_share_pcx"}, shares
        assert abs(sum(shares.values()) - 1.0) < 1e-12 and is_table_name("portfolio_kept_de")
        bad = Path(td) / "bad.toml"
        bad.write_text(text.replace('["sbx", "de", "pcx"]', '["sbx", "nope"]'), encoding="utf-8")
        assert validate(load(bad)), "a portfolio word that is no crossover passed"
    p = __import__("mootation.benchmarks", fromlist=["get"]).get("ZDT1")
    r = Mz.minimize(p.evaluate, p.bounds, 2, algorithm="moead", pop_size=20, max_evaluations=200,
                    seed=1, de_adapt="shade", ref_delta=0.5)
    assert set(r.ignored) == {"de_adapt", "ref_delta"}, r.ignored
    try:
        Mz.minimize(p.evaluate, p.bounds, 2, algorithm="nsga2", pop_size=20, max_evaluations=200,
                    mutation="gaussian", mutation_scale=0.0)
        raise AssertionError("mutation_scale = 0 accepted")
    except ValueError as e:
        assert "mutation = \"none\"" in str(e), e


@test
def mosa_spends_its_budget_and_records_its_state():
    """Task 7, §1: mosa spends exactly its budget, the same seed gives the same
    run, and it is budget-dependent; its answer is at most N points, mutually
    non-dominated, inside the box. In a campaign with operator_stats the
    temperature and the step scale fall along the trajectory, and sa_energy =
    tch with sa_step = adaptive records the chains' weights and the steps'
    probabilities."""
    if not _have_numpy() or _core_with("params") is None:
        print("  skip  mosa_spends...: no NumPy or stale _core"); return
    import numpy as np
    import mootation
    from mootation.run import campaign as C
    from mootation.run.budget import BUDGET_DEPENDENT
    from mootation.run.config import load
    from mootation.run.metric_names import SA_STATS, is_table_name
    assert callable(mootation.minimize)                # loads the module behind the function
    Mz = sys.modules["mootation.minimize"]
    assert "mosa" in BUDGET_DEPENDENT
    assert all(is_table_name(k) for k in SA_STATS) and is_table_name("sa_w_sd_3")
    p = __import__("mootation.benchmarks", fromlist=["get"]).get("ZDT1")
    fe = 0

    def ev(x):
        nonlocal fe
        fe += 1
        return p.evaluate(x)
    a = Mz.minimize(ev, p.bounds, 2, algorithm="mosa", pop_size=30, max_evaluations=777, seed=5)
    b = Mz.minimize(p.evaluate, p.bounds, 2, algorithm="mosa", pop_size=30, max_evaluations=777,
                    seed=5)
    assert fe == 777 and a.evaluations == 777, (fe, a.evaluations)
    assert a.objectives == b.objectives and a.variables == b.variables
    F, X = np.asarray(a.objectives), np.asarray(a.variables)
    assert 0 < len(F) <= 30, len(F)
    lb, ub = np.array(p.bounds, float).T
    assert np.all(X >= lb) and np.all(X <= ub)
    assert not any((np.all(F <= f, axis=1) & np.any(F < f, axis=1)).any() for f in F)
    text = """algorithms = [
    { name = "mosa", label = "cmb", pop = 20, gens = 0 },
    { name = "mosa", label = "tch", pop = 20, gens = 0, params = { sa_energy = "tch", sa_step = "adaptive", sa_chains = 4 } },
]
[run]
name = "sa"
[problem]
kind = "builtin"
[benchmarks]
runs = 1
problems = ["DTLZ2_3D"]
[campaign]
out = "res"
budget_fe = 1500
record_grid = "log"
metrics = ["igdp_norm"]
final_metrics = ["igdp_norm"]
operator_stats = true
"""
    with tempfile.TemporaryDirectory() as td:
        cfg_path = Path(td) / "c.toml"
        cfg_path.write_text(text, encoding="utf-8")
        cfg = load(cfg_path)
        assert validate(cfg) == [], validate(cfg)
        spec = C.campaign_spec(cfg)
        root = C.out_root(cfg, spec)
        for job in C.expand_jobs(cfg, spec):
            assert C.run_job(job, root, spec, quiet=True) == "done", job
        for label in ("cmb", "tch"):
            d = root / "DTLZ2_3D" / label / "run_1"
            meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
            T = [json.loads(l) for l in (d / "trajectory.jsonl").read_text().splitlines()]
            temps, scales = [r["sa_T"] for r in T], [r["sa_s"] for r in T]
            assert len(T) > 3 and temps[-1] < temps[0] and scales[-1] < scales[0], (temps, scales)
            assert all(u >= v for u, v in zip(temps, temps[1:]))
            assert 0 < meta["final"]["sa_archive_size"] and "sa_reseeds" in meta["final"]
        w = {k: v for k, v in T[-1].items() if k.startswith(("sa_w_", "sa_p_"))}
        assert set(w) == {"sa_w_mean_1", "sa_w_mean_2", "sa_w_mean_3", "sa_w_sd_1", "sa_w_sd_2",
                          "sa_w_sd_3", "sa_p_laplace", "sa_p_de", "sa_p_sbx"}, sorted(w)
        assert abs(w["sa_w_mean_1"] + w["sa_w_mean_2"] + w["sa_w_mean_3"] - 1.0) < 1e-9
        assert abs(w["sa_p_laplace"] + w["sa_p_de"] + w["sa_p_sbx"] - 1.0) < 1e-9


@test
def operator_statistics_leave_the_run_alone_and_describe_it():
    """[campaign] operator_stats: the same populations, the records and meta it adds."""
    if not _have_numpy() or _core_with("operator_stats") is None:
        print("  skip  operator_statistics...: no NumPy or stale _core"); return
    import contextlib
    import io
    from mootation.run import campaign as C
    from mootation.run.config import load
    text = """algorithms = [
    { name = "nsga2", pop = 0, gens = 0 },
    { name = "moead_de", pop = 0, gens = 0 },
    { name = "moead_de", label = "de_reflect", pop = 0, gens = 0, params = { bound_repair = "reflect" } },
    { name = "random_search", pop = 0, gens = 0 },
]
[run]
name = "ops"
[problem]
kind = "builtin"
[benchmarks]
runs = 1
problems = ["DTLZ2_3D"]
[campaign]
out = "OUT"
budget_fe = 600
record_grid = "log"
metrics = ["igdp_norm"]
final_metrics = ["igdp_norm", "tau90", "n_final"]
operator_stats = STATS
"""
    with tempfile.TemporaryDirectory() as td:
        roots = {}
        for flag in ("true", "false"):
            cfg_path = Path(td) / f"c_{flag}.toml"
            cfg_path.write_text(text.replace("OUT", f"res_{flag}").replace("STATS", flag),
                                encoding="utf-8")
            cfg = load(cfg_path)
            assert validate(cfg) == [], validate(cfg)
            spec = C.campaign_spec(cfg)
            roots[flag] = C.out_root(cfg, spec)
            for job in C.expand_jobs(cfg, spec):
                assert C.run_job(job, roots[flag], spec, quiet=True) == "done", job
        for alg in ("nsga2", "moead_de", "de_reflect"):
            a, b = (roots[f] / "DTLZ2_3D" / alg / "run_1" for f in ("true", "false"))
            assert (a / "final.csv").read_text() == (b / "final.csv").read_text(), alg
            meta = json.loads((a / "meta.json").read_text(encoding="utf-8"))
            T = [json.loads(l) for l in (a / "trajectory.jsonl").read_text().splitlines()]
            assert T[0]["offspring"] == 0 and T[0]["survival_share"] is None, T[0]
            for key in ("oob_share", "survival_share", "offspring_nd_share", "step_mean"):
                assert 0.0 <= T[-1][key] <= (1.0 if key != "step_mean" else 2.0), (alg, key, T[-1])
                assert key in meta["final"], (alg, key)
            ops = {o["operator"]: o["bound_repair"] for o in meta["operators"]}
            if alg == "nsga2":
                assert ops == {"sbx": "none", "polynomial": "none"}, ops
                assert meta["final"]["oob_share"] == 0.0
            else:
                assert ops["box repair"] == ("reflect" if alg == "de_reflect" else "random"), ops
                assert meta["final"]["oob_share"] > 0.0, meta["final"]
            assert meta["final"]["n_final"] == 91 and meta["final"]["tau90"] > 0
            off = json.loads((b / "meta.json").read_text(encoding="utf-8"))
            assert "survival_share" not in off["final"] and off["operators"], off["final"]
        base = json.loads((roots["true"] / "DTLZ2_3D" / "random_search" / "run_1" /
                           "meta.json").read_text(encoding="utf-8"))
        assert base["operator_stats"] is False and "operators" not in base
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            code = C.main([str(Path(td) / "c_true.toml"), "--recompute", "survival_share"])
        assert code == 1 and "operator_stats" in out.getvalue(), out.getvalue()
        with contextlib.redirect_stdout(out):
            code = C.main([str(Path(td) / "c_true.toml"), "--compare", "survival_share"])
        assert code == 0


# ── task 2, step 2: structural bias ─────────────────────────────────────────

@test
def uninformative_values_come_from_the_seed_and_the_count_not_from_x():
    if not _have_numpy():                 # the benchmarks package needs NumPy
        print("    (skipped: no NumPy)"); return
    from mootation.benchmarks.uninformative import Evaluator, evaluate_by_x, name, value
    a, b = Evaluator(11), Evaluator(11)
    fa = [a([0.1, 0.2]) for _ in range(50)]
    fb = [b([0.9, 0.7]) for _ in range(50)]
    assert fa == fb, "the same seed gives the same stream whatever x is"
    assert fa != [Evaluator(12)([0.1, 0.2]) for _ in range(50)]
    assert len(set(map(tuple, fa))) == 50                 # the count moves it on
    vals = [value(3, k, i) for k in range(20000) for i in range(2)]
    assert all(0.0 <= v < 1.0 for v in vals)
    assert abs(sum(vals) / len(vals) - 0.5) < 0.01        # U(0, 1): mean 1/2
    assert evaluate_by_x([0.3, 0.4]) == evaluate_by_x([0.3, 0.4])
    assert name(2) == "uninformative_n02_2D" and name(10) == "uninformative_n10_2D"


@test
def kudelas_problems_and_where_the_points_fell():
    """f2b-f5 (task 6, item 11, Kudela et al. 2026, Table 1, variant beta) from
    the seed and the count; the evaluator's counts of every point and of the
    non-dominated ones: uniform points give binsizes 1 +- 0.1 and no exact
    hits, clipped points atoms on the bounds; on f5 every point is
    non-dominated."""
    if not _have_numpy():
        print("    (skipped: no NumPy)"); return
    import numpy as np
    from mootation.benchmarks import get, names
    from mootation.benchmarks import uninformative as U
    from mootation.run import report as R
    assert len([n for n in names() if n.startswith("uninformative")]) == 10
    assert U.name(10, "f3b") == "uninformative_f3b_n10_2D"
    rng = np.random.default_rng(9)
    X = rng.random((20000, 2))
    for problem in ("f2b", "f3b", "f4b", "f5"):
        a, b = U.Evaluator(7, problem), U.Evaluator(7, problem)
        fa = [a(x) for x in X[:200]]
        assert fa == [b([0.5, 0.5]) for _ in range(200)], problem     # x never matters
        assert all(np.isfinite(f).all() for f in fa), problem
        assert fa != [U.Evaluator(8, problem)([0.1, 0.1]) for _ in range(200)], problem
        g1 = np.array([f[0] for f in fa])
        assert 0.0 <= g1.min() and g1.max() < 1.0, problem             # g1 = a = U(0, 1)
    assert all(f[1] == -f[0] for f in (U.Evaluator(3, "f5")([0.2, 0.3]) for _ in range(5)))
    assert U.objectives("f4b", 0.25, 0.0) == [0.25, -0.125]
    assert abs(U.objectives("f3b", 0.35, 0.0)[1] - (0.1 / 1.5 ** 10 + 0.7)) < 1e-15
    e = get("uninformative_f2b_n02_2D").make_evaluator(1)
    for x in X:
        e(x)
    s = e.summary()
    assert s["evaluations"] == 20000 and s["all"]["points"] == 20000
    m = R._bin_measures(s["all"]["hist"], s["all"]["at_lower"], s["all"]["at_upper"],
                        s["all"]["at_half"])
    assert np.all(np.abs(np.array(m["binsizes"]) - 1.0) < 0.1), m["binsizes"]
    assert m["at_bound"] == 0.0 and m["at_half"] == 0.0, m
    assert 0 < s["nondominated"]["points"] < 2000, s["nondominated"]
    e = U.Evaluator(2, "f5")
    for x in np.clip(0.5 + 0.6 * rng.standard_normal((3000, 2)), 0.0, 1.0):
        e(x)
    s = e.summary()
    assert s["nondominated"]["points"] == 3000, s["nondominated"]["points"]  # every one
    assert s["all"]["at_lower"] + s["all"]["at_upper"] > 1000, s["all"]


@test
def the_chi_square_tail_and_the_bias_shares_are_right():
    if not _have_numpy():
        print("    (skipped: no NumPy)"); return
    import numpy as np
    from mootation.run import report as R
    # P(chi2_9 >= 16.919) = 0.05, P(chi2_9 >= 21.666) = 0.01, P(chi2_2 >= x) = exp(-x/2)
    assert abs(R._chi2_sf(16.918978, 9) - 0.05) < 1e-6
    assert abs(R._chi2_sf(21.665994, 9) - 0.01) < 1e-6
    for x in (0.1, 1.0, 5.0, 40.0):
        assert abs(R._chi2_sf(x, 2) - np.exp(-x / 2)) < 1e-12, x
    assert R._chi2_sf(0.0, 9) == 1.0
    rng = np.random.default_rng(4)
    with tempfile.TemporaryDirectory() as td:
        rows = []
        for alg, maker in (("flat", lambda: rng.random((400, 2))),
                           ("centre", lambda: 0.45 + 0.1 * rng.random((400, 2))),
                           ("clipped", lambda: np.clip(0.5 + 0.6 * rng.standard_normal((400, 2)),
                                                       0.0, 1.0))):
            for seed in (1, 2, 3, 4, 5):
                d = Path(td) / alg / str(seed)
                d.mkdir(parents=True)
                X = maker()
                F = rng.random((400, 2))
                with (d / "final.csv").open("w", encoding="utf-8") as fh:
                    fh.write("# final\nf1,f2,x1,x2\n")
                    for f, x in zip(F, X):
                        fh.write(",".join(f"{v:.10g}" for v in (*f, *x)) + "\n")
                rows.append({"problem": "uninformative_n02_2D", "algorithm": alg, "seed": seed,
                             "status": "done", "dir": d})
        rep = R.structural_bias(rows)
        flat, centre, clipped = (rep[("uninformative_n02_2D", a)]
                                 for a in ("flat", "centre", "clipped"))
        L = flat["L"]                               # 4 000 coordinates, 200 a bin
        assert abs(L["bound"] - 1) < 0.25 and abs(L["centre"] - 1) < 0.25, L
        assert L["at_bound"] == 0.0 and L["at_half"] == 0.0 and flat["class"] == "none", flat
        assert flat["chi2_p"] > 1e-3 and flat["runs"] == 5 and 0.8 < flat["cei"] < 1.2, flat
        assert flat["P"] is None and flat["A"] is None      # no meta.json counts here
        assert centre["class"] == "centre" and centre["L"]["bound"] == 0.0, centre
        assert centre["cei"] < 0.5, centre                  # clustered in the middle
        assert clipped["class"] == "bound" and clipped["L"]["at_bound"] > 0.3, clipped
        text = R.format_bias(rep)
        assert all(a in text for a in ("flat", "centre", "clipped")) and "bound" in text


@test
def a_campaign_hands_each_run_its_own_uninformative_stream():
    """make_evaluator(seed): two campaigns with the same seed run the same, x never matters."""
    if not _have_numpy() or _core_with("operator_stats") is None:
        print("  skip  a_campaign_hands...: no NumPy or stale _core"); return
    from mootation.run import campaign as C
    from mootation.run.config import load
    text = """algorithms = [ { name = "nsga2", pop = 0, gens = 0 } ]
[run]
name = "u"
[problem]
kind = "builtin"
[benchmarks]
runs = 2
problems = ["uninformative_n10_2D"]
[campaign]
out = "OUT"
budget_fe = 400
metrics = ["nd_share"]
final_metrics = ["n_final"]
archive = false
"""
    with tempfile.TemporaryDirectory() as td:
        finals = {}
        for tag in ("a", "b"):
            cfg_path = Path(td) / f"{tag}.toml"
            cfg_path.write_text(text.replace("OUT", f"res_{tag}"), encoding="utf-8")
            cfg = load(cfg_path)
            assert validate(cfg) == [], validate(cfg)
            spec = C.campaign_spec(cfg)
            root = C.out_root(cfg, spec)
            for job in C.expand_jobs(cfg, spec):
                assert C.run_job(job, root, spec, quiet=True) == "done", job
            finals[tag] = [(root / "uninformative_n10_2D" / "nsga2" / f"run_{s}" / "final.csv")
                           .read_text() for s in (1, 2)]
        assert finals["a"] == finals["b"], "the same seed gives the same run"
        assert finals["a"][0] != finals["a"][1], "another seed another stream"
        # the counts --bias reads (task 6, item 11): every evaluated point, and
        # the run's non-dominated ones
        meta = json.loads((root / "uninformative_n10_2D" / "nsga2" / "run_1" / "meta.json")
                          .read_text(encoding="utf-8"))
        sb = meta["structural_bias"]
        assert sb["evaluations"] == meta["fe"] == sb["all"]["points"], (sb["evaluations"], meta["fe"])
        assert sum(sb["all"]["hist"]) == 10 * sb["all"]["points"], sb["all"]
        assert 0 < sb["nondominated"]["points"] < sb["all"]["points"], sb["nondominated"]


# ── task 6, item 5: C-DTLZ and the indicators of a constrained problem ──────

@test
def c_dtlz_matches_pymoo():
    """C1-DTLZ1, C1-DTLZ3, C2-DTLZ2, C2-ConvexDTLZ2, C3-DTLZ1 and C3-DTLZ4 at
    three and five objectives give pymoo's objectives and constraints at five
    points each (tests/data/pymoo-0.6.2_c-dtlz.json) (task 6, item 5.3)."""
    if not _have_numpy():
        print("    (skipped: no NumPy)"); return
    from mootation.benchmarks import get, names
    ref = reference_data("pymoo-0.6.2_c-dtlz")
    assert sorted({c["problem"] for c in ref["cases"]}) == \
        sorted(n for n in names() if n[:3] in ("C1-", "C2-", "C3-"))
    for case in ref["cases"]:
        p = get(case["problem"])
        assert p.has_cons, case["problem"]
        assert_close(p.evaluate(case["x"]), case["f"], ref["tolerance"], case["problem"])
        assert_close(p.constraints(case["x"]), case["g"], ref["tolerance"], case["problem"])


@test
def c_dtlz_fronts_are_feasible_and_the_constraints_point_the_right_way():
    """Every reference front is feasible (to 1e-12: C3's lies on its
    constraints) and mutually non-dominated, its extremes the registered
    frame; the Pareto sets are feasible with g = 0. The sign: C1-DTLZ1 is
    infeasible just above its band and feasible just below, C3-DTLZ1 and
    C3-DTLZ4 are feasible off the unconstrained front and infeasible on it
    (task 6, item 5.3)."""
    if not _have_numpy():
        print("    (skipped: no NumPy)"); return
    import math
    import numpy as np
    from mootation.benchmarks import c_dtlz, get, names
    from mootation.benchmarks.registry import PROBLEMS, _strictly_dominated_by
    keys = [n for n in names() if n[:3] in ("C1-", "C2-", "C3-")]
    for key in keys:
        name, p = key.split("_")[0], get(key)
        F = p.pareto_front(300)
        worst = max(max(c_dtlz.constraints(name, f, p.n_obj)) for f in F)
        assert worst <= 1e-12, (key, worst)
        assert not _strictly_dominated_by(F, F).any(), key
        assert np.allclose(F.min(0), p.ideal, atol=1e-12), (key, F.min(0))
        assert np.allclose(F.max(0), p.nadir, atol=1e-12), (key, F.max(0))
        if p.pareto_set is not None:
            for x in p.pareto_set(40):
                assert max(p.constraints(list(x))) <= 0.0 and p.distance_fn(list(x)) == 0.0, key
    approximate = {n for n in keys if not PROBLEMS[n].front_exact}
    assert approximate == {"C2-ConvexDTLZ2_3D", "C2-ConvexDTLZ2_5D"}, approximate
    assert [get(k).pareto_set is None for k in ("C2-DTLZ2_3D", "C2-ConvexDTLZ2_3D",
                                               "C3-DTLZ1_3D", "C3-DTLZ4_5D")] == \
        [False, True, True, True]

    def dtlz1_x(positions, g, k):
        """DTLZ1's x with these positions and g(x) = g: one distance variable
        at 0.5 + d, where g = 100 (1 + d^2 - cos(20 pi d)) grows with d."""
        lo, hi = 0.0, 0.05
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            if 100.0 * (1.0 + mid * mid - math.cos(20.0 * math.pi * mid)) < g:
                lo = mid
            else:
                hi = mid
        return list(positions) + [0.5 + hi] + [0.5] * (k - 1)

    def feasible(key, x):
        return max(get(key).constraints(x)) <= 0.0

    # C1-DTLZ1 at (0, 0, f_3): f_3 = 0.5 (1 + g), the band's top f_3 = 0.6 at g = 0.2
    above, below = dtlz1_x([0, 0], 0.2 * (1 + 1e-6), 5), dtlz1_x([0, 0], 0.2 * (1 - 1e-6), 5)
    assert abs(get("C1-DTLZ1_3D").evaluate(above)[2] - 0.6) < 1e-6
    assert not feasible("C1-DTLZ1_3D", above) and feasible("C1-DTLZ1_3D", below)
    # C3-DTLZ1 at (f_1, 0, 0), f_1 = 0.5 (1 + g): feasible from g = 1, so at
    # g = 1.5 (sum f = 1.25, off DTLZ1's front) and not on that front (g = 0)
    off, on = dtlz1_x([1, 1], 1.5, 5), [1, 1] + [0.5] * 5
    assert abs(sum(get("C3-DTLZ1_3D").evaluate(off)) - 1.25) < 1e-9
    assert feasible("C3-DTLZ1_3D", off) and not feasible("C3-DTLZ1_3D", on)
    # C3-DTLZ4 at (1 + g, 0, 0): feasible from g = 1; g = sum (x - 1/2)^2 = 1.21
    off = [0, 0, 1, 1, 1, 1, 0.5 + math.sqrt(0.21)]
    assert abs(get("C3-DTLZ4_3D").evaluate(off)[0] - 2.21) < 1e-9
    assert feasible("C3-DTLZ4_3D", off) and not feasible("C3-DTLZ4_3D", [0, 0] + [0.5] * 5)


@test
def mw_matches_pymoo():
    """MW1-14 (MW4, MW8 and MW14 at three and five objectives) give pymoo's
    objectives and constraints at five points each, the constraints in the
    order of the paper's equations (tests/data/pymoo-0.6.2_mw.json) (task 6,
    item 5)."""
    if not _have_numpy():
        print("    (skipped: no NumPy)"); return
    from mootation.benchmarks import get, names
    ref = reference_data("pymoo-0.6.2_mw")
    assert sorted({c["problem"] for c in ref["cases"]}) == \
        sorted(n for n in names() if n.startswith("MW"))
    for case in ref["cases"]:
        p = get(case["problem"])
        assert p.has_cons, case["problem"]
        assert_close(p.evaluate(case["x"]), case["f"], ref["tolerance"], case["problem"])
        assert_close(p.constraints(case["x"]), case["g"], ref["tolerance"], case["problem"])


@test
def mw_fronts_are_feasible_non_dominated_and_framed():
    """Every MW reference front is feasible (to 1e-12: much of it lies on
    the constraints) and mutually non-dominated, its extremes the registered
    frame (mw.frame, to 1e-9), MW11's isolated point (1, 1) on it; the Pareto
    sets, where the front lies at g = 1, are feasible, at g = 1 and on the
    front. 10^4 random points of every box, with x1 on its bound for a tenth
    (reading MW-3), give finite values (task 6, item 5)."""
    if not _have_numpy():
        print("    (skipped: no NumPy)"); return
    import numpy as np
    from mootation.benchmarks import get, mw, names
    from mootation.benchmarks.registry import _strictly_dominated_by
    for key in [n for n in names() if n.startswith("MW")]:
        name, p = key.split("_")[0], get(key)
        F = p.pareto_front(300)
        assert len(F) == 300, (key, len(F))
        assert mw.constraints_f(name, F).max() <= 1e-12, key
        assert not _strictly_dominated_by(F, F).any(), key
        assert np.allclose(F.min(0), p.ideal, atol=1e-9), (key, F.min(0))
        assert np.allclose(F.max(0), p.nadir, atol=1e-9), (key, F.max(0))
        assert (p.pareto_set is not None) == (name in mw.ON_G1), key
        if p.pareto_set is not None:
            X = p.pareto_set(40)
            FX = np.array([p.evaluate(list(x)) for x in X])
            assert all(max(p.constraints(list(x))) <= 1e-12 for x in X), key
            assert all(abs(p.distance_fn(list(x))) <= 1e-12 for x in X), key
            assert not _strictly_dominated_by(FX, p.pareto_front(1000) + 1e-9).any(), key
        rng = np.random.default_rng(len(key))
        R = mw.upper(name) * rng.random((10_000, p.n_vars))
        R[:1000, 0] = mw.upper(name)
        for x in R:
            assert np.isfinite(p.evaluate(list(x))).all() and np.isfinite(p.constraints(list(x))).all(), key
    F = get("MW11").pareto_front(1000)
    assert np.min(np.abs(F - [1.0, 1.0]).max(axis=1)) == 0.0


@test
def a_constrained_answer_is_measured_on_its_feasible_part():
    """metrics.compute with constraints: the indicators of the feasible rows
    and feas_share; no feasible row, no value (None, n_final 0). The tables
    count such an empty run, rank it as the worst value and say so (task 6,
    item 5.2)."""
    if not _have_numpy():
        print("    (skipped: no NumPy)"); return
    import math
    import numpy as np
    from mootation.run import campaign as C
    from mootation.run import metrics as M

    def cons(x):                                   # feasible when x_0 <= 0.5
        return [x[0] - 0.5]

    X = np.array([[0.2, 0.0], [0.4, 0.0], [0.9, 0.0], [0.7, 0.0]])
    F = np.array([[0.2, 0.8], [0.4, 0.6], [0.0, 0.0], [0.7, 0.3]])
    ref = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
    kw = dict(ref_front=ref, ideal=(0.0, 0.0), nadir=(1.0, 1.0), which=("igdp", "n_final"))
    got = M.compute(F, X=X, constraints=cons, **kw)
    assert got["feas_share"] == 0.5 and got["n_final"] == 2, got
    assert got["igdp"] == M.compute(F[:2], **kw)["igdp"]
    none = M.compute(F[2:], X=X[2:], constraints=cons, **kw)
    assert none == {"feas_share": 0.0, "igdp": None, "n_final": 0}, none
    assert "variables" in str(raises(ValueError, M.compute, F, constraints=cons, **kw))

    def row(problem, alg, seed, v):
        final = ({"igdp_norm": None, "hv": None, "feas_share": 0.0} if v is None else
                 {"igdp_norm": v, "hv": 1.0 - v, "feas_share": 1.0})
        return {"problem": problem, "algorithm": alg, "seed": seed, "status": "done",
                "final": final, "n_objs": 3}

    rows = [row("P1", "a", s, 0.1) for s in (1, 2, 3)] + \
           [row("P1", "b", 1, 0.05), row("P1", "b", 2, None), row("P1", "b", 3, None)] + \
           [row("P2", "a", s, 0.3) for s in (1, 2, 3)] + \
           [row("P2", "b", s, 0.2) for s in (1, 2)] + [row("P2", "b", 3, None)]
    for metric in ("igdp_norm", "hv"):
        empty: dict = {}
        table = C.compare_table(rows, metric, empty=empty)
        assert empty == {("P1", "b"): 2, ("P2", "b"): 1}, empty
        assert table["P1"]["b"][3] == 3 and not math.isfinite(table["P1"]["b"][0]), table
        assert math.isfinite(table["P2"]["b"][0]), table          # one of three: the median holds
        e = dict(C.rank_table(rows, metric)["algorithms"])
        assert e["b"]["empty"] == 3 and e["a"]["empty"] == 0, e
        assert e["b"]["all"][0] == 1.5 and e["a"]["all"][0] == 1.5, e   # last on P1, first on P2
    text = C.format_ranks(C.rank_table(rows, "igdp_norm"))
    assert "empty" in text.splitlines()[1] and "no feasible point" in text, text
    # the report's tests and groups count them the same way (report.values)
    from mootation.run import report as R
    empty: dict = {}
    vals = R.values(rows, "igdp_norm", empty=empty)
    assert empty == {"b": 3} and vals["P1"]["b"].count(math.inf) == 2, (empty, vals)
    assert R.values(rows, "hv")["P1"]["b"].count(-math.inf) == 2
    rep = R.reference_report(rows, "igdp_norm", "a")
    e = rep["algorithms"][0]
    assert e["empty"] == 3 and rep["empty_reference"] == 0, e
    assert e["per_problem"]["P1"]["a12"] < 0.5, e["per_problem"]         # worse there
    assert "empty runs" in R.format_reference(rep)
    ci_empty: dict = {}
    R.rank_intervals(rows, "igdp_norm", n_boot=50, empty=ci_empty)
    assert ci_empty == {"b": 3}
    by_empty: dict = {}
    g = R.property_ranks(rows, "igdp_norm", "front", empty=by_empty)
    assert by_empty == {"b": 3} and g["front=?"]["b"] == (1.5, 2), g
    assert "empty" in R.format_property_ranks(g, "igdp_norm", "front", by_empty).splitlines()[1]


@test
def a_campaign_on_a_constrained_problem_records_feasibility():
    """A run on C-DTLZ records feas_share wherever it measures, fe_first_feas
    in final, and its indicators only where the answer has a feasible point;
    the archive keeps feasible points only (task 6, item 5.2)."""
    if not _have_numpy() or _core_with("operator_stats") is None:
        print("  skip  a_campaign_on_a_constrained...: no NumPy or stale _core"); return
    from mootation.run import campaign as C
    from mootation.run.config import load
    text = """algorithms = [ { name = "nsga2", pop = 0, gens = 0 } ]
[run]
name = "c"
[problem]
kind = "builtin"
[benchmarks]
runs = 1
problems = ["C1-DTLZ3_3D", "C1-DTLZ1_3D"]
[campaign]
out = "res"
budget_fe = 400
metrics = ["igdp_norm"]
final_metrics = ["igdp_norm", "n_final"]
archive = true
"""
    with tempfile.TemporaryDirectory() as td:
        cfg_path = Path(td) / "c.toml"
        cfg_path.write_text(text, encoding="utf-8")
        cfg = load(cfg_path)
        assert validate(cfg) == [], validate(cfg)
        spec = C.campaign_spec(cfg)
        root = C.out_root(cfg, spec)
        for job in C.expand_jobs(cfg, spec):
            assert C.run_job(job, root, spec, quiet=True) == "done", job
        for problem in ("C1-DTLZ3_3D", "C1-DTLZ1_3D"):
            run = root / problem / "nsga2" / "run_1"
            meta = json.loads((run / "meta.json").read_text(encoding="utf-8"))
            final, first = meta["final"], meta["final"]["fe_first_feas"]
            assert (first is None) == (final["feas_share"] == 0.0), (problem, final)
            assert (final["igdp_norm"] is None) == (final["feas_share"] == 0.0), (problem, final)
            assert meta["final_archive"]["feas_share"] == (0.0 if first is None else 1.0), meta
            traj = C.read_trajectory(run)
            assert all("feas_share" in t for t in traj), traj[0]
        # outside the shell 16 <= sum f^2 <= 81 every DTLZ3 point is feasible
        meta = json.loads((root / "C1-DTLZ3_3D" / "nsga2" / "run_1" / "meta.json")
                          .read_text(encoding="utf-8"))
        assert meta["final"]["fe_first_feas"] == 1, meta["final"]
        assert C.recompute_final(root, ["igdp_norm"]) == {"done": 2}


# ── task 6, item 7: noisy copies ────────────────────────────────────────────

@test
def a_noisy_copy_adds_the_noise_of_its_seed_and_nothing_else():
    """noisy.py: sigma = 0 gives the base problem; one seed, one stream, another
    seed another; over 10^5 evaluations the noise has the asked-for standard
    deviation to 1 % and mean 0; a copy is its base in everything but the
    values (task 6, item 7)."""
    if not _have_numpy():
        print("    (skipped: no NumPy)"); return
    import numpy as np
    from mootation.benchmarks import families, get, noisy
    from mootation.benchmarks.properties import properties
    from mootation.run.campaign import problem_family
    want = sorted(f"{b}_{lv}" for b in noisy.BASES + noisy.LATE_BASES for lv in noisy.LEVELS)
    assert sorted(families()["noisy"]) == want
    assert all(problem_family(n) == "noisy" for n in want)
    base = get("DTLZ2_3D").evaluate
    x = [0.2, 0.7] + [0.4] * 10
    assert noisy.Evaluator(base, [0.0, 0.0, 0.0], 5)(x) == base(x)
    a, b, c = (noisy.Evaluator(base, [0.05] * 3, s) for s in (5, 5, 6))
    run_a, run_b, run_c = ([e(x) for _ in range(50)] for e in (a, b, c))
    assert run_a == run_b and run_a != run_c and len({tuple(f) for f in run_a}) == 50
    zero = noisy.Evaluator(lambda x: [0.0, 0.0], [0.01, 0.05], 11)
    Z = np.array([zero(x) for _ in range(100_000)])
    for i, s in enumerate((0.01, 0.05)):
        assert abs(Z[:, i].std() / s - 1.0) < 0.01, (i, Z[:, i].std())
        assert abs(Z[:, i].mean()) < 0.02 * s, (i, Z[:, i].mean())
    assert abs(np.corrcoef(Z[:, 0], Z[:, 1])[0, 1]) < 0.01
    for name in want:
        p, q = get(name), get(noisy.base_of(name))
        assert (p.n_vars, p.bounds, p.ideal, p.nadir) == (q.n_vars, q.bounds, q.ideal, q.nadir)
        assert np.array_equal(p.pareto_front(200), q.pareto_front(200)), name
        assert p.true_objectives is q.evaluate and properties(name) == properties(q.name)
        assert (p.distance_fn is None) == (q.distance_fn is None), name
        sigma = noisy.SIGMA[name.rsplit("_", 1)[1]]
        assert p.make_evaluator(1).scale == [sigma * (h - l) for l, h in zip(q.ideal, q.nadir)]
        mid = [(lo + hi) / 2 for lo, hi in p.bounds]
        assert p.evaluate(mid) == p.evaluate(mid) != q.evaluate(mid), name


@test
def deterministic_step_and_relative_noise_and_the_three_report_modes():
    """Task 7, §4: _d gives the same value for the same x in every run and at
    every repeat; _s the same inside a cell and another across its border, the
    cell a share of the range ([campaign] noise_cell); _h's spread is sigma
    (f - ideal), none at the ideal; sigma = 0 is the base problem for every
    kind. A campaign records every final indicator _oracle and _observed
    beside it, and on ZDT1_n5 the observed IGD+ is on average better than the
    recommended one: the noise flatters the answer the algorithm kept."""
    if not _have_numpy() or _core_with("operator_stats") is None:
        print("  skip  deterministic_step...: no NumPy or stale _core"); return
    import numpy as np
    from mootation.benchmarks import get, noisy
    from mootation.run import campaign as C
    from mootation.run.config import load
    from mootation.run.metric_names import is_table_name
    base = get("DTLZ2_3D").evaluate
    x = [0.425, 0.705] + [0.405] * 10
    for kind in "ndsh":
        e = noisy.Evaluator(base, [0.0] * 3, 5, kind, ideal=[0.0] * 3, sigma=0.0,
                            lower=[0.0] * 12, upper=[1.0] * 12)
        assert e(x) == base(x) and e.last_true == base(x), kind
    p = get("DTLZ2_3D_d5")
    e1, e2 = p.make_evaluator(1), p.make_evaluator(2)
    assert [e1(x) for _ in range(3)] == [e2(x)] * 3 == [p.evaluate(x)] * 3 != [base(x)] * 3
    p = get("DTLZ2_3D_s5")
    e = p.make_evaluator(1)
    near = list(x)
    near[0] = 0.428                                  # the same cell of width 0.01
    far = list(x)
    far[0] = 0.435                                   # the next cell
    a, b, c = e(x), e(near), e(far)
    assert np.allclose(np.subtract(a, b), np.subtract(base(x), base(near)), atol=1e-15)
    assert not np.allclose(np.subtract(a, c), np.subtract(base(x), base(far)), atol=1e-6)
    wide = p.make_evaluator(1, cell=0.1)             # 0.425 and 0.435 share a cell of 0.1
    a, c = wide(x), wide(far)
    assert np.allclose(np.subtract(a, c), np.subtract(base(x), base(far)), atol=1e-15)
    rel = noisy.Evaluator(lambda x: [1.0, 2.0, 0.0], [9.0] * 3, 3, "h", ideal=[0.0] * 3,
                          sigma=0.05)
    Z = np.array([rel(x) for _ in range(20_000)])
    assert abs(Z[:, 0].std() / 0.05 - 1) < 0.03 and abs(Z[:, 1].std() / 0.10 - 1) < 0.03
    assert np.all(Z[:, 2] == 0.0)                    # at the ideal: no noise
    assert all(is_table_name(m) for m in ("igdp_norm_oracle", "igdp_norm_observed",
                                          "hv_oracle", "true_dominated_share_observed"))
    text = """algorithms = [ { name = "nsga2", pop = 20, gens = 0 } ]
[run]
name = "nz"
[problem]
kind = "builtin"
[benchmarks]
runs = 6
problems = ["ZDT1_n5", "ZDT1_s5"]
[campaign]
out = "res"
budget_fe = 600
metrics = ["igdp_norm"]
final_metrics = ["igdp_norm"]
noise_cell = 0.1
"""
    with tempfile.TemporaryDirectory() as td:
        cfg_path = Path(td) / "nz.toml"
        cfg_path.write_text(text, encoding="utf-8")
        cfg = load(cfg_path)
        assert validate(cfg) == [], validate(cfg)
        spec = C.campaign_spec(cfg)
        root = C.out_root(cfg, spec)
        for job in C.expand_jobs(cfg, spec):
            assert C.run_job(job, root, spec, quiet=True) == "done", job
        fin = [json.loads((root / "ZDT1_n5" / "nsga2" / f"run_{k}" / "meta.json").read_text(
            encoding="utf-8"))["final"] for k in range(1, 7)]
        rec = [f["igdp_norm"] for f in fin]
        obs = [f["igdp_norm_observed"] for f in fin]
        orc = [f["igdp_norm_oracle"] for f in fin]
        assert np.mean(obs) < np.mean(rec), (obs, rec)
        assert all(np.isfinite(orc)) and np.mean(orc) <= np.mean(rec), (orc, rec)
        meta = json.loads((root / "ZDT1_s5" / "nsga2" / "run_1" / "meta.json").read_text(
            encoding="utf-8"))
        assert (meta["noise"]["kind"], meta["noise"]["cell"]) == ("s", 0.1), meta["noise"]
        bad = Path(td) / "bad.toml"
        bad.write_text(text.replace("noise_cell = 0.1", "noise_cell = 0"), encoding="utf-8")
        try:
            C.campaign_spec(load(bad))
            raise AssertionError("noise_cell = 0 accepted")
        except C.ConfigError:
            pass


@test
def a_campaign_on_a_noisy_copy_measures_the_noise_free_values():
    """The algorithm sees the noisy values; the indicators, the files and the
    snapshots hold the noise-free ones at the same x, final.csv the noisy ones
    too (f_seen), and true_dominated_share goes beside them (task 6, item 7)."""
    if not _have_numpy() or _core_with("operator_stats") is None:
        print("  skip  a_campaign_on_a_noisy...: no NumPy or stale _core"); return
    import numpy as np
    from mootation.benchmarks import get
    from mootation.run import campaign as C
    from mootation.run import metrics as M
    from mootation.run.config import load
    text = """algorithms = [ { name = "nsga2", pop = 0, gens = 0 } ]
[run]
name = "n"
[problem]
kind = "builtin"
[benchmarks]
runs = 1
problems = ["ZDT1_n5"]
[campaign]
out = "res"
budget_fe = 400
metrics = ["igdp"]
final_metrics = ["igdp"]
archive = true
snapshots = true
snapshot_variables = "all"
"""
    with tempfile.TemporaryDirectory() as td:
        cfg_path = Path(td) / "n.toml"
        cfg_path.write_text(text, encoding="utf-8")
        cfg = load(cfg_path)
        assert validate(cfg) == [], validate(cfg)
        spec = C.campaign_spec(cfg)
        root = C.out_root(cfg, spec)
        for job in C.expand_jobs(cfg, spec):
            assert C.run_job(job, root, spec, quiet=True) == "done", job
        run = root / "ZDT1_n5" / "nsga2" / "run_1"
        meta = json.loads((run / "meta.json").read_text(encoding="utf-8"))
        assert meta["noise"]["scale"] == [0.05, 0.05], meta["noise"]
        lines = [r for r in (run / "final.csv").read_text(encoding="utf-8").splitlines()
                 if not r.startswith("#")]
        assert lines[0].endswith(",f1_seen,f2_seen"), lines[0]
        A = np.array([[float(v) for v in r.split(",")] for r in lines[1:]])
        base = get("ZDT1")
        F, X, seen = A[:, :2], A[:, 2:2 + base.n_vars], A[:, 2 + base.n_vars:]
        T = np.array([base.evaluate(list(x)) for x in X])
        assert np.allclose(F, T, rtol=1e-8, atol=1e-9) and not np.allclose(seen, T)
        ref = np.asarray(base.pareto_front(1000), float)
        assert abs(meta["final"]["igdp"] - M.igd_plus(T, ref)) < 1e-6, meta["final"]
        assert 0.0 <= meta["final"]["true_dominated_share"] <= 1.0
        assert "true_dominated_share" in meta["final_archive"], meta["final_archive"]
        AF, AX = C._read_points(run / "archive.csv", 2, base.n_vars)
        TA = np.array([base.evaluate(list(x)) for x in AX])
        assert np.allclose(AF, TA, rtol=1e-8, atol=1e-9)           # the archive: noise-free
        traj = C.read_trajectory(run)
        assert all("true_dominated_share" in t for t in traj), traj[-1]
        with np.load(run / "snapshots.npz") as z:
            last = z["F"][-z["n"][-1]:].astype(float)
            XL = z["X"][-z["n"][-1]:].astype(float)
        TL = np.array([base.evaluate(list(x)) for x in XL])
        assert np.allclose(last, TL, rtol=1e-5, atol=1e-5)          # float32 snapshots


# ── task 6, item 9: dense references at four objectives and more ───────────

@test
def a_dense_reference_is_a_third_of_the_populations_step():
    """registry.dense_reference: the first size whose step is at most a third
    of the population's (5 000 rows at DTLZ2_4D, 10 000 at DTLZ2_5D), the
    largest where none is (DTLZ1_10D); none at three objectives or fifteen.
    At three objectives a dense sample moves IGD+ by less than the step of the
    usual one (task 6, item 9)."""
    if not _have_numpy():
        print("    (skipped: no NumPy)"); return
    import numpy as np
    from mootation.benchmarks import get
    from mootation.benchmarks.registry import DENSE_SIZES, dense_reference, has_dense_reference
    from mootation.run import metrics as M
    from mootation.run.metric_names import base_name, is_table_name
    for name, rows in (("DTLZ2_4D", 5000), ("DTLZ2_5D", 10_000)):
        F, info = dense_reference(name, get(name).pop_size)
        assert len(F) == info["rows"] == rows, (name, info)
        assert info["step"] <= info["population_step"] / 3.0, info
        assert np.allclose(M.reference_step(F), info["step"])        # the frame is [0, 1]
    assert [has_dense_reference(n) for n in ("DTLZ2_3D", "DTLZ2_15D", "WFG4_6D", "ZDT1",
                                             "MaF7_5D")] == [False, False, True, False, False]
    p = get("DTLZ2_3D")
    old = np.asarray(p.pareto_front(1000), float)
    new = np.asarray(p.pareto_front(DENSE_SIZES[-1]), float)
    A = 1.02 * np.asarray(p.pareto_front(91), float)
    assert abs(M.igd_plus(A, new) - M.igd_plus(A, old)) <= M.reference_step(old)
    assert base_name("igdp_norm_ref_v3") == "igdp_norm" and base_name("hv_ref_v3") == "hv_ref_v3"
    assert is_table_name("eps_norm_ref_v3") and not is_table_name("hv_ref_v3")


@test
def a_campaign_measures_against_the_dense_reference_and_recompute_adds_it():
    """[campaign] dense_reference (on by default) records "<indicator>_ref_v3" beside
    the front indicators — final, archive, trajectory — and the reference's
    rows and step; on a campaign without it --recompute-reference adds the
    same values from the saved answers and leaves the old ones as they were;
    the tables read the new columns (task 6, item 9)."""
    if not _have_numpy() or _core_with("operator_stats") is None:
        print("  skip  a_campaign_measures_against_the_dense...: no NumPy or stale _core"); return
    from mootation.run import campaign as C
    from mootation.run.config import load
    text = """algorithms = [ { name = "nsga2", pop = 0, gens = 0 } ]
[run]
name = "d"
[problem]
kind = "builtin"
[benchmarks]
runs = 1
problems = ["DTLZ2_4D"]
[campaign]
out = "OUT"
budget_fe = 500
metrics = ["igdp_norm"]
final_metrics = ["igdp_norm", "hv"]
archive = true
snapshots = true
DENSE
"""
    with tempfile.TemporaryDirectory() as td:
        metas = {}
        for tag, dense in (("with", ""), ("without", "dense_reference = false")):
            cfg_path = Path(td) / f"{tag}.toml"
            cfg_path.write_text(text.replace("OUT", f"res_{tag}").replace("DENSE", dense),
                                encoding="utf-8")
            cfg = load(cfg_path)
            assert validate(cfg) == [], validate(cfg)
            spec = C.campaign_spec(cfg)
            root = C.out_root(cfg, spec)
            for job in C.expand_jobs(cfg, spec):
                assert C.run_job(job, root, spec, quiet=True) == "done", job
            run = root / "DTLZ2_4D" / "nsga2" / "run_1"
            metas[tag] = (root, run, json.loads((run / "meta.json").read_text(encoding="utf-8")))
        root, run, meta = metas["with"]
        assert meta["reference_v3"]["rows"] == 5000, meta["reference_v3"]
        assert "igdp_norm_ref_v3" in meta["final"] and "hv_ref_v3" not in meta["final"]
        assert "igdp_norm_ref_v3" in meta["final_archive"]
        assert all("igdp_norm_ref_v3" in t for t in C.read_trajectory(run))
        root2, run2, before = metas["without"]
        assert before["reference_v3"] is None and "igdp_norm_ref_v3" not in before["final"]
        assert C.recompute_dense(root2) == {"done": 1}
        after = json.loads((run2 / "meta.json").read_text(encoding="utf-8"))
        assert after["reference_v3"] == meta["reference_v3"], after["reference_v3"]
        for part in ("final", "final_archive"):
            assert {k: v for k, v in after[part].items() if not k.endswith("_ref_v3")} == \
                before[part], part                                  # the old values stay
            assert abs(after[part]["igdp_norm_ref_v3"] - meta[part]["igdp_norm_ref_v3"]) < 1e-9
        assert all("igdp_norm_ref_v3" in t for t in C.read_trajectory(run2))
        assert C.recompute_dense(root2) == {"skipped": 1}
        table = C.compare_table(C.scan_results(root2), "igdp_norm_ref_v3")
        assert table["DTLZ2_4D"]["nsga2"][3] == 1, table



@test
def a_campaign_measures_against_its_own_reference_and_ranks_approximate_fronts_apart():
    """--pooled-reference: per problem, the non-dominated points of all the
    campaign's runs (feasible ones), with the runs that went into it, the code
    that built it and its sha256; every run measured against it as
    <indicator>_pooled, the old values untouched, and again only when the pool
    changes. The rank table sets the problems with an approximate reference
    front apart, and the properties say which those are and how their fronts
    were built (owner's decision of 2026-10-11, item 4)."""
    if not _have_numpy() or _core_with("operator_stats") is None:
        print("  skip  a_campaign_measures_against_its_own_reference...: no NumPy or stale _core")
        return
    import numpy as np
    from mootation.benchmarks.properties import properties
    from mootation.run import campaign as C
    from mootation.run.config import load
    from mootation.run.metrics import feasible_mask, nondominated_mask
    from mootation.benchmarks import get
    text = """algorithms = [ { name = "nsga2", pop = 0, gens = 0 }, { name = "spea2", pop = 0, gens = 0 } ]
[run]
name = "p"
[problem]
kind = "builtin"
[benchmarks]
runs = 2
problems = ["ZDT1", "C2-ConvexDTLZ2_3D"]
[campaign]
out = "OUT"
budget_fe = 2000
metrics = ["igdp_norm"]
final_metrics = ["igdp_norm", "igd", "hv"]
archive = true
"""
    with tempfile.TemporaryDirectory() as td:
        cfg_path = Path(td) / "p.toml"
        cfg_path.write_text(text.replace("OUT", "res"), encoding="utf-8")
        cfg = load(cfg_path)
        assert validate(cfg) == [], validate(cfg)
        spec = C.campaign_spec(cfg)
        root = C.out_root(cfg, spec)
        for job in C.expand_jobs(cfg, spec):
            assert C.run_job(job, root, spec, quiet=True) == "done", job
        before = {m: json.loads(m.read_text(encoding="utf-8"))
                  for m in root.glob("*/*/run_*/meta.json")}
        assert C.pooled_reference(root) == {"pools": 2, "done": 8}
        for prob in ("ZDT1", "C2-ConvexDTLZ2_3D"):
            ref, info = C.pooled_reference_of(root, prob)
            p = get(prob)
            assert info["n_runs"] == 4 and len(ref) == info["rows"] and nondominated_mask(ref).all()
            assert sorted(r["run"] for r in info["runs"]) ==                 ["nsga2/run_1", "nsga2/run_2", "spea2/run_1", "spea2/run_2"], info["runs"]
            assert info["analysis"]["mootation"] and info["ideal"] == ref.min(0).tolist()
            for m in root.glob(f"{prob}/*/run_*/meta.json"):
                meta = json.loads(m.read_text(encoding="utf-8"))
                assert meta["reference_pooled"]["sha256"] == info["sha256"]
                old = before[m]["final"]
                assert {k: v for k, v in meta["final"].items() if not k.endswith("_pooled")} == old
                assert "igdp_norm_pooled" in meta["final"] and "hv_pooled" not in meta["final"]
                F, X = C._read_points(m.parent / "final.csv", p.n_obj, p.n_vars)
                if p.has_cons:
                    F = F[feasible_mask(X, p.constraints)]
                # every feasible answer point is weakly dominated by a point of the pool
                assert all((ref <= f + 1e-12).all(axis=1).any() for f in F), prob
        assert C.pooled_reference(root) == {"pools": 2, "skipped": 8}   # the same pool
        rows = C.scan_results(root)
        assert C.compare_table(rows, "igdp_norm_pooled")["ZDT1"]["nsga2"][3] == 2
        for metric in ("igdp_norm", "igdp_norm_pooled", "hv"):
            ranks = C.rank_table(rows, metric)
            assert ranks["groups"][-1] == "approx_front" and ranks["n_approx_front"] == 1, ranks
            assert dict(ranks["algorithms"])["nsga2"]["all"][1] == 1           # ZDT1 alone
            assert "approx_front = " in C.format_ranks(ranks)
        assert "approx_front" not in C.rank_table(rows, "n_final")["groups"]
    assert properties("ZDT1")["front_ref"] == "exact" and properties("ZDT1")["front_built"] is None
    row = properties("C2-ConvexDTLZ2_3D")
    assert row["front_ref"] == "approximate" and "c_dtlz.py" in row["front_built"]
    assert "rwmop.build_front" in properties("RCM01")["front_built"]
    assert properties("RCM28")["front_ref"] == "exact"
    assert properties("bbobbiobj01_n05_2D")["front_ref"] is None

# ── task 6, item 1: rotated DTLZ and WFG ────────────────────────────────────

# sha256 (16 hex digits) of every rotated problem's matrix (Q, Lambda Q, I or
# Lambda), rounded to 1e-6 (LAPACK's QR differs across platforms in the last
# bits, which 1e-12 let through), and of every rotation of the registered
# bbob-biobj problems: numpy does not promise its normal stream across
# versions, and a changed draw must fail here, not move results silently.
ROTATION_HASHES = {
    "c0DTLZ2_10D": "f4051c1a6bd36392", "c0DTLZ2_3D": "f4051c1a6bd36392",
    "c0DTLZ2_5D": "f4051c1a6bd36392", "c0DTLZ2_6D": "f4051c1a6bd36392",
    "c0DTLZ5_3D": "f4051c1a6bd36392", "c0DTLZ5_5D": "f4051c1a6bd36392",
    "c0MaF1_3D": "f4051c1a6bd36392", "c0MaF1_5D": "f4051c1a6bd36392",
    "c0MaF5_3D": "f4051c1a6bd36392", "c0MaF5_5D": "f4051c1a6bd36392",
    "c0MaF6_3D": "f4051c1a6bd36392", "c0MaF6_5D": "f4051c1a6bd36392",
    "c0Polygon_3D": "87d1e133d3eac299", "c0Polygon_5D": "87d1e133d3eac299",
    "r0DTLZ1_3D": "1a8475b42f5782e9", "r0DTLZ1_5D": "1a8475b42f5782e9",
    "r0DTLZ3_3D": "a3c81aeea2e1ab58", "r0DTLZ3_5D": "a3c81aeea2e1ab58",
    "r0MaF2_3D": "a3c81aeea2e1ab58", "r0MaF2_5D": "a3c81aeea2e1ab58",
    "r0MaF3_3D": "a3c81aeea2e1ab58", "r0MaF3_5D": "a3c81aeea2e1ab58",
    "r0MaF4_3D": "a3c81aeea2e1ab58", "r0MaF4_5D": "a3c81aeea2e1ab58",
    "rDTLZ1_3D": "c769f25996708c5a", "rDTLZ1_3D_i1": "c769f25996708c5a",
    "rDTLZ1_3D_i2": "a9f2a2a2e5042517", "rDTLZ1_3D_i3": "e2f9ae2b7505a433",
    "rDTLZ1_5D": "df51d5e160d49f5f", "rDTLZ1_5D_i1": "df51d5e160d49f5f",
    "rDTLZ1_5D_i2": "00f1a132af5fc1df", "rDTLZ1_5D_i3": "ec8a13a3df065c4a",
    "rDTLZ2_3D": "1306b1d7346e87e1", "rDTLZ2_5D": "57abfdd53219d0f0",
    "rDTLZ3_3D": "46f9352bc92d74c0", "rDTLZ3_3D_i1": "46f9352bc92d74c0",
    "rDTLZ3_3D_i2": "89eff7769875428c", "rDTLZ3_3D_i3": "74c473ec4bcbccfb",
    "rDTLZ3_5D": "a39edc31ef18a0f5", "rDTLZ3_5D_i1": "a39edc31ef18a0f5",
    "rDTLZ3_5D_i2": "9495a3bd0bf320cb", "rDTLZ3_5D_i3": "8119c9815f4645c7",
    "rDTLZ4_3D": "d41c9c09a417f819", "rDTLZ4_5D": "9277e86cb3d337a7",
    "rMaF2_3D": "7ec409913cff6a1a", "rMaF2_3D_i1": "7ec409913cff6a1a",
    "rMaF2_3D_i2": "5273ad959500024b", "rMaF2_3D_i3": "eef6f109538ae14e",
    "rMaF2_5D": "8e889631d7e89aea", "rMaF2_5D_i1": "8e889631d7e89aea",
    "rMaF2_5D_i2": "eac07f5d36d3936c", "rMaF2_5D_i3": "2374eca6fc05b910",
    "rMaF3_3D": "56f6ae3c2e81ff1f", "rMaF3_3D_i1": "56f6ae3c2e81ff1f",
    "rMaF3_3D_i2": "2ff9f224fc8c96ad", "rMaF3_3D_i3": "a8ffab49a4bb75c6",
    "rMaF3_5D": "836939cedc20a18a", "rMaF3_5D_i1": "836939cedc20a18a",
    "rMaF3_5D_i2": "15a0b736c2769c5a", "rMaF3_5D_i3": "efe831236fae70f5",
    "rMaF4_3D": "112b9ab1ddcdae9f", "rMaF4_3D_i1": "112b9ab1ddcdae9f",
    "rMaF4_3D_i2": "b7ce173d2d310c82", "rMaF4_3D_i3": "798999d8220770fb",
    "rMaF4_5D": "907fd6d05363afba", "rMaF4_5D_i1": "907fd6d05363afba",
    "rMaF4_5D_i2": "efaff16249e9364b", "rMaF4_5D_i3": "20f5fb05e2a5f277",
    "rWFG1_3D": "69018c08b9388d4c", "rWFG1_3D_i1": "69018c08b9388d4c",
    "rWFG1_3D_i2": "0ea94560e5c90bbf", "rWFG1_3D_i3": "482c0c6e34d8415f",
    "rWFG1_5D": "93b88ffe8c6e4314", "rWFG1_5D_i1": "93b88ffe8c6e4314",
    "rWFG1_5D_i2": "82d6a9f14d38eb8a", "rWFG1_5D_i3": "58f383ac9a0e0033",
    "rWFG2_3D": "5514f9911a27b405", "rWFG2_3D_i1": "5514f9911a27b405",
    "rWFG2_3D_i2": "f9e47fa590b21a3e", "rWFG2_3D_i3": "502c9083527cf994",
    "rWFG2_5D": "caf06525033344e6", "rWFG2_5D_i1": "caf06525033344e6",
    "rWFG2_5D_i2": "058b11027594107e", "rWFG2_5D_i3": "d1c0013b2ce63bd6",
    "rWFG3_3D": "2541763a3cfaa279", "rWFG3_3D_i1": "2541763a3cfaa279",
    "rWFG3_3D_i2": "d33f09082d1e158f", "rWFG3_3D_i3": "d86f7d6498190fb2",
    "rWFG3_5D": "efc08973d5c64c40", "rWFG3_5D_i1": "efc08973d5c64c40",
    "rWFG3_5D_i2": "c82c1812c3cb4d9f", "rWFG3_5D_i3": "631cdcc70ba05f8d",
    "rWFG4_3D": "fbcdf21c15e602b8", "rWFG4_3D_i1": "fbcdf21c15e602b8",
    "rWFG4_3D_i2": "3d4f8f050e0c4ed8", "rWFG4_3D_i3": "67df01e884d95f22",
    "rWFG4_5D": "74af38dc612a7f79", "rWFG4_5D_i1": "74af38dc612a7f79",
    "rWFG4_5D_i2": "591a6f82888340dc", "rWFG4_5D_i3": "f8d3e175c673e190",
    "rWFG5_3D": "ffad98d28186f3c1", "rWFG5_3D_i1": "ffad98d28186f3c1",
    "rWFG5_3D_i2": "89701693fd2cb6a8", "rWFG5_3D_i3": "8c0c184481fdfe3b",
    "rWFG5_5D": "76b3eb3d1941f01c", "rWFG5_5D_i1": "76b3eb3d1941f01c",
    "rWFG5_5D_i2": "7a4f58255ede61ec", "rWFG5_5D_i3": "a309ddf00ef9f996",
    "rWFG6_3D": "1ac3f6901e681c62", "rWFG6_3D_i1": "1ac3f6901e681c62",
    "rWFG6_3D_i2": "7a052eb48b8b1962", "rWFG6_3D_i3": "18bc676c1dbf976a",
    "rWFG6_5D": "efe988f86e5ee278", "rWFG6_5D_i1": "efe988f86e5ee278",
    "rWFG6_5D_i2": "bd2a88f2975979e3", "rWFG6_5D_i3": "5bba1813ce5a22d4",
    "rWFG7_3D": "7d9473afb6844b39", "rWFG7_3D_i1": "7d9473afb6844b39",
    "rWFG7_3D_i2": "5ba6c043318911f0", "rWFG7_3D_i3": "c97690fe04f780c1",
    "rWFG7_5D": "291af541076ce4c7", "rWFG7_5D_i1": "291af541076ce4c7",
    "rWFG7_5D_i2": "e13d16953024f10b", "rWFG7_5D_i3": "4f09fedfe30cfdb3",
    "rWFG8_3D": "b2cc51c3b257d079", "rWFG8_3D_i1": "b2cc51c3b257d079",
    "rWFG8_3D_i2": "ba1b44d9f9f4a1a8", "rWFG8_3D_i3": "eff75ae15a0afada",
    "rWFG8_5D": "6f893527ce83e06d", "rWFG8_5D_i1": "6f893527ce83e06d",
    "rWFG8_5D_i2": "39b84ee79af5b67a", "rWFG8_5D_i3": "bfd12fa21a4e7314",
    "rWFG9_3D": "7f17170806d2982c", "rWFG9_3D_i1": "7f17170806d2982c",
    "rWFG9_3D_i2": "4cb3be2068b9009e", "rWFG9_3D_i3": "ab13d82c37016272",
    "rWFG9_5D": "d6007b40b92b198d", "rWFG9_5D_i1": "d6007b40b92b198d",
    "rWFG9_5D_i2": "a9d5c182c891e51f", "rWFG9_5D_i3": "33730a4cf8262bc2",
    "rZCAT10_3D": "3ffda6a38bebffb9", "rZCAT10_3D_i1": "3ffda6a38bebffb9",
    "rZCAT10_3D_i2": "18038fed28ccea98", "rZCAT10_3D_i3": "afe4d85142e377de",
    "rZCAT10_5D": "d52e06d453b3daa6", "rZCAT10_5D_i1": "d52e06d453b3daa6",
    "rZCAT10_5D_i2": "61c6088cf9c61b66", "rZCAT10_5D_i3": "1d25c3e9fe6ffba7",
    "rZCAT11_3D": "aca4beacb2a27145", "rZCAT11_3D_i1": "aca4beacb2a27145",
    "rZCAT11_3D_i2": "46af50c92194a2f4", "rZCAT11_3D_i3": "1cd986d3a1512e21",
    "rZCAT11_5D": "76c892b546a7ab67", "rZCAT11_5D_i1": "76c892b546a7ab67",
    "rZCAT11_5D_i2": "6c2be4bf460684e5", "rZCAT11_5D_i3": "171a41c4bf52637a",
    "rZCAT12_3D": "4ce01fbab04be6ba", "rZCAT12_3D_i1": "4ce01fbab04be6ba",
    "rZCAT12_3D_i2": "c482e663133fea8b", "rZCAT12_3D_i3": "5a76115b8884c74b",
    "rZCAT12_5D": "702802ce30655f8d", "rZCAT12_5D_i1": "702802ce30655f8d",
    "rZCAT12_5D_i2": "05db1fb86e757993", "rZCAT12_5D_i3": "b37713b178e58186",
    "rZCAT13_3D": "59d66fc291617bc7", "rZCAT13_3D_i1": "59d66fc291617bc7",
    "rZCAT13_3D_i2": "f1688babc9a3acf5", "rZCAT13_3D_i3": "70d5038d8b5c9536",
    "rZCAT13_5D": "2f25fc051d7ebe9e", "rZCAT13_5D_i1": "2f25fc051d7ebe9e",
    "rZCAT13_5D_i2": "0a1ec353364a4ac8", "rZCAT13_5D_i3": "1eed55f857072b3c",
    "rZCAT14_3D": "c750c3c92d83e401", "rZCAT14_3D_i1": "c750c3c92d83e401",
    "rZCAT14_3D_i2": "8b820a64ba449886", "rZCAT14_3D_i3": "a25f6ed5a6cc7e00",
    "rZCAT14_5D": "55e50b7a8a583b87", "rZCAT14_5D_i1": "55e50b7a8a583b87",
    "rZCAT14_5D_i2": "3195f319fe4443fa", "rZCAT14_5D_i3": "83050e865b3b6280",
    "rZCAT15_3D": "3f6cad419890b538", "rZCAT15_3D_i1": "3f6cad419890b538",
    "rZCAT15_3D_i2": "e1f6e97163f984e1", "rZCAT15_3D_i3": "aeb19728a22ee058",
    "rZCAT15_5D": "97049c1fe7a26906", "rZCAT15_5D_i1": "97049c1fe7a26906",
    "rZCAT15_5D_i2": "4b0126a8d7e6630d", "rZCAT15_5D_i3": "d4f6cc50e4378d0b",
    "rZCAT16_3D": "3e646cc6c336ebcb", "rZCAT16_3D_i1": "3e646cc6c336ebcb",
    "rZCAT16_3D_i2": "d126184f963ea71e", "rZCAT16_3D_i3": "666118eec8218aa3",
    "rZCAT16_5D": "cbb7d94872593ade", "rZCAT16_5D_i1": "cbb7d94872593ade",
    "rZCAT16_5D_i2": "6d02317b0ddde707", "rZCAT16_5D_i3": "ea09bd1d2c7811fa",
    "rZCAT17_3D": "ad22544a2999e044", "rZCAT17_3D_i1": "ad22544a2999e044",
    "rZCAT17_3D_i2": "4c3e1532a4dc7669", "rZCAT17_3D_i3": "0ac64f0cde4276ef",
    "rZCAT17_5D": "5ac91feb98e0de88", "rZCAT17_5D_i1": "5ac91feb98e0de88",
    "rZCAT17_5D_i2": "69379086930d2e16", "rZCAT17_5D_i3": "5c13603168376bae",
    "rZCAT18_3D": "6b0082a4e3753fc1", "rZCAT18_3D_i1": "6b0082a4e3753fc1",
    "rZCAT18_3D_i2": "30e97beae33f51af", "rZCAT18_3D_i3": "c7445632d489bc80",
    "rZCAT18_5D": "b161fe8f8521f943", "rZCAT18_5D_i1": "b161fe8f8521f943",
    "rZCAT18_5D_i2": "7e2cad44ba233bd0", "rZCAT18_5D_i3": "3d496e2f89ea609c",
    "rZCAT19_3D": "118086c54befab2c", "rZCAT19_3D_i1": "118086c54befab2c",
    "rZCAT19_3D_i2": "45a3f3f93d9d6e7b", "rZCAT19_3D_i3": "0055016d12e88f24",
    "rZCAT19_5D": "77fb24f140f63fac", "rZCAT19_5D_i1": "77fb24f140f63fac",
    "rZCAT19_5D_i2": "8ec69c709c4fa980", "rZCAT19_5D_i3": "42160f0dca33504a",
    "rZCAT1_3D": "7d86254f36020314", "rZCAT1_3D_i1": "7d86254f36020314",
    "rZCAT1_3D_i2": "bc3e2cc36aedb7f0", "rZCAT1_3D_i3": "86d57d83ee1ed4c1",
    "rZCAT1_5D": "9f5ce729a7765143", "rZCAT1_5D_i1": "9f5ce729a7765143",
    "rZCAT1_5D_i2": "a2f30e1348b85ac4", "rZCAT1_5D_i3": "455dc809288e161f",
    "rZCAT20_3D": "3b49b305731a856a", "rZCAT20_3D_i1": "3b49b305731a856a",
    "rZCAT20_3D_i2": "ae5c253766dc0faa", "rZCAT20_3D_i3": "1678c7e7f1996f3e",
    "rZCAT20_5D": "ae4cbf31e8ace847", "rZCAT20_5D_i1": "ae4cbf31e8ace847",
    "rZCAT20_5D_i2": "c506390d9b600c0f", "rZCAT20_5D_i3": "2d776701cae333bf",
    "rZCAT2_3D": "95f97efcf958147a", "rZCAT2_3D_i1": "95f97efcf958147a",
    "rZCAT2_3D_i2": "99a7ccbeeae7aa7b", "rZCAT2_3D_i3": "6317d8d859b4e6ed",
    "rZCAT2_5D": "89043338c1178292", "rZCAT2_5D_i1": "89043338c1178292",
    "rZCAT2_5D_i2": "1aa56c17c455607f", "rZCAT2_5D_i3": "f5d7c4b4c8500338",
    "rZCAT3_3D": "505768c3c34f3b60", "rZCAT3_3D_i1": "505768c3c34f3b60",
    "rZCAT3_3D_i2": "556f4fb61322f948", "rZCAT3_3D_i3": "027c4e59839a9339",
    "rZCAT3_5D": "a89460f62a3d511c", "rZCAT3_5D_i1": "a89460f62a3d511c",
    "rZCAT3_5D_i2": "61905d7cd39adf14", "rZCAT3_5D_i3": "aad84af1e8b705a3",
    "rZCAT4_3D": "03894ce12a39040c", "rZCAT4_3D_i1": "03894ce12a39040c",
    "rZCAT4_3D_i2": "7c28b2c882266723", "rZCAT4_3D_i3": "bcb47b292398e6d0",
    "rZCAT4_5D": "9859521cfc1ba068", "rZCAT4_5D_i1": "9859521cfc1ba068",
    "rZCAT4_5D_i2": "9435177e8b4bbab6", "rZCAT4_5D_i3": "bc8a18ced05070a3",
    "rZCAT5_3D": "6e42da07f8a1dffc", "rZCAT5_3D_i1": "6e42da07f8a1dffc",
    "rZCAT5_3D_i2": "7395159142142d21", "rZCAT5_3D_i3": "1c22f69bfc708e35",
    "rZCAT5_5D": "e82afcad0cbb522d", "rZCAT5_5D_i1": "e82afcad0cbb522d",
    "rZCAT5_5D_i2": "8d748dc9697480e2", "rZCAT5_5D_i3": "5295ee36d2dfaf3f",
    "rZCAT6_3D": "43aaaf3578deb4f0", "rZCAT6_3D_i1": "43aaaf3578deb4f0",
    "rZCAT6_3D_i2": "058f8a9b943dcb43", "rZCAT6_3D_i3": "ee37adeb12dc46f5",
    "rZCAT6_5D": "980041fd98e428a8", "rZCAT6_5D_i1": "980041fd98e428a8",
    "rZCAT6_5D_i2": "88fc32d2a0f88455", "rZCAT6_5D_i3": "cc294252feb55282",
    "rZCAT7_3D": "7cecca8f79e9d539", "rZCAT7_3D_i1": "7cecca8f79e9d539",
    "rZCAT7_3D_i2": "46d8d8e43933e152", "rZCAT7_3D_i3": "3ee41c08efd8d696",
    "rZCAT7_5D": "e4e57de4c900efb5", "rZCAT7_5D_i1": "e4e57de4c900efb5",
    "rZCAT7_5D_i2": "88f67659d5afe924", "rZCAT7_5D_i3": "9798f88fe2e2c96c",
    "rZCAT8_3D": "dc9ffb47dd1e4255", "rZCAT8_3D_i1": "dc9ffb47dd1e4255",
    "rZCAT8_3D_i2": "9bf7c15075140e46", "rZCAT8_3D_i3": "ca7b3ada0fbdf075",
    "rZCAT8_5D": "87979898e941397f", "rZCAT8_5D_i1": "87979898e941397f",
    "rZCAT8_5D_i2": "85fc9565103b6bc2", "rZCAT8_5D_i3": "c07d54b437b413cb",
    "rZCAT9_3D": "0a44ffb65eccb638", "rZCAT9_3D_i1": "0a44ffb65eccb638",
    "rZCAT9_3D_i2": "5ea59ebfa886d2aa", "rZCAT9_3D_i3": "4d4ca949d25f39f2",
    "rZCAT9_5D": "afbff255cf86ff4e", "rZCAT9_5D_i1": "afbff255cf86ff4e",
    "rZCAT9_5D_i2": "39e0a88cb03e4bb1", "rZCAT9_5D_i3": "9d7d174b128684d6",
    "rZDT1": "6def5ac9d528646e", "rZDT1_i1": "6def5ac9d528646e",
    "rZDT1_i2": "2cefd442de6b073c", "rZDT1_i3": "0f53dd16f05d544c",
    "rZDT2": "7d66ba66072173ae", "rZDT2_i1": "7d66ba66072173ae",
    "rZDT2_i2": "335545f8df56b00a", "rZDT2_i3": "58238761e1ea1a27",
    "rZDT3": "558f12ff3ba89312", "rZDT3_i1": "558f12ff3ba89312",
    "rZDT3_i2": "d2ca7fb43c99b423", "rZDT3_i3": "3700044d0d17db7b",
    "rZDT4": "10e70f50d53139bd", "rZDT4_i1": "10e70f50d53139bd",
    "rZDT4_i2": "b8534060fdec75da", "rZDT4_i3": "9b99a423e0f6bc23",
    "rZDT6": "b5dbffb60df1c58b", "rZDT6_i1": "b5dbffb60df1c58b",
    "rZDT6_i2": "be3ceb3412ae6718", "rZDT6_i3": "ddf2c6638fc05278",
    "rcDTLZ2_10D": "05b30240ae53382b", "rcDTLZ2_10D_i1": "05b30240ae53382b",
    "rcDTLZ2_10D_i2": "ec16666f9f4ad5e1", "rcDTLZ2_10D_i3": "73b4d89e9f0c3a22",
    "rcDTLZ2_3D": "83d2091bdb342835", "rcDTLZ2_3D_i1": "83d2091bdb342835",
    "rcDTLZ2_3D_i2": "1ae6065c7c7f2573", "rcDTLZ2_3D_i3": "ddfd93c2c5835162",
    "rcDTLZ2_5D": "9aff00c4856a01f9", "rcDTLZ2_5D_i1": "9aff00c4856a01f9",
    "rcDTLZ2_5D_i2": "6ed280d8a41550cf", "rcDTLZ2_5D_i3": "59271f759909a3e9",
    "rcDTLZ2_6D": "8a55d7a79f21a5ba", "rcDTLZ2_6D_i1": "8a55d7a79f21a5ba",
    "rcDTLZ2_6D_i2": "be670aaa0387d54b", "rcDTLZ2_6D_i3": "9cc7fb95f84dc5e9",
    "rcDTLZ5_3D": "c6f4cd5da2e2b575", "rcDTLZ5_3D_i1": "c6f4cd5da2e2b575",
    "rcDTLZ5_3D_i2": "3e1604d1549b43d8", "rcDTLZ5_3D_i3": "cab9f8ba7ed8cca0",
    "rcDTLZ5_5D": "8e9b87e4198749d0", "rcDTLZ5_5D_i1": "8e9b87e4198749d0",
    "rcDTLZ5_5D_i2": "5f3b402f5efcf2a2", "rcDTLZ5_5D_i3": "26ca32cc167a3d66",
    "rcMaF1_3D": "8b84960f55c9f6e6", "rcMaF1_3D_i1": "8b84960f55c9f6e6",
    "rcMaF1_3D_i2": "37bfd354c76eea45", "rcMaF1_3D_i3": "c3abf368ca279fa0",
    "rcMaF1_5D": "e23ef1816bb772d5", "rcMaF1_5D_i1": "e23ef1816bb772d5",
    "rcMaF1_5D_i2": "a2bd2b70ae90a359", "rcMaF1_5D_i3": "d54b946edbae35d2",
    "rcMaF5_3D": "941ad658d2ada1c4", "rcMaF5_3D_i1": "941ad658d2ada1c4",
    "rcMaF5_3D_i2": "b8b6bcb4aaab4c90", "rcMaF5_3D_i3": "2da0f230eb8609b9",
    "rcMaF5_5D": "c1b8fb58662777b2", "rcMaF5_5D_i1": "c1b8fb58662777b2",
    "rcMaF5_5D_i2": "397fe4bc0f9c3ca8", "rcMaF5_5D_i3": "56f1fce51224962e",
    "rcMaF6_3D": "eb3ba5522643ce41", "rcMaF6_3D_i1": "eb3ba5522643ce41",
    "rcMaF6_3D_i2": "1d20c3554c20d0cf", "rcMaF6_3D_i3": "91761afab4ea232e",
    "rcMaF6_5D": "f58ce8d5e8a2ed7b", "rcMaF6_5D_i1": "f58ce8d5e8a2ed7b",
    "rcMaF6_5D_i2": "e947cb033b3dd2fe", "rcMaF6_5D_i3": "3d0486d11eae70e8",
    "rcPolygon_3D": "9b50a7b8bba1a19e", "rcPolygon_3D_i1": "9b50a7b8bba1a19e",
    "rcPolygon_3D_i2": "49fd710fd90f1e19", "rcPolygon_3D_i3": "e34b25e5082c011f",
    "rcPolygon_5D": "dc606239c2f29f92", "rcPolygon_5D_i1": "dc606239c2f29f92",
    "rcPolygon_5D_i2": "44616c652c36ea78", "rcPolygon_5D_i3": "162a2e8e6a9b6276",
}
BBOB_ROTATION_HASH = "65c0c29ca35cb30a"


def _matrix_hash(A, digest=None) -> str:
    import hashlib
    import numpy as np
    d = digest or hashlib.sha256()
    d.update((np.round(np.asarray(A, float), 6) + 0.0).tobytes())
    return d.hexdigest()[:16]


@test
def the_rotations_are_rotations_and_numpy_still_draws_them():
    """Every Q of rotated.py is orthogonal (1e-12), the conditioned ones are
    Lambda Q with the stated Lambda, the controls I or Lambda; the hash of every
    matrix, and of bbob-biobj's rotations, is the one recorded (task 6, item
    1.3; the rZDT, the instances and M = 6 and 10, task 7, §3; the other
    families, §7, ZCAT19 and ZCAT20 with a matrix per size of the distance
    block). Instance 1 is the problem without the suffix, 2 and 3 are other
    matrices."""
    if not _have_numpy():
        print("    (skipped: no NumPy)"); return
    import hashlib
    import numpy as np
    from mootation.benchmarks import bbob_biobj as BB
    from mootation.benchmarks import names
    from mootation.benchmarks import rotated as R
    from mootation.benchmarks.registry import ROTATED
    from mootation.benchmarks.noisy import is_noisy
    assert sorted(ROTATED) == sorted(ROTATION_HASHES) == sorted(
        n for n in names() if n.split("_")[0] in R.SPECS and not is_noisy(n))
    for name, r in ROTATED.items():
        family, rotate, condition = R.SPECS[name.split("_")[0]]
        digest = hashlib.sha256()
        for A in r.matrices():
            k = len(A)
            Q = R.rotation(family, r.M, k, r.instance) if rotate else np.eye(k)
            assert np.allclose(Q.T @ Q, np.eye(k), atol=1e-12), name
            L = R.conditioning(k) if condition else np.ones(k)
            assert np.array_equal(A, L[:, None] * Q), name
            h = _matrix_hash(A, digest)
        assert h == ROTATION_HASHES[name], (name, h)
    assert [len(A) for A in ROTATED["rZCAT19_3D"].matrices()] == [28, 29]
    assert R.conditioning(10)[[0, -1]].tolist() == [1.0, 1e3 ** 0.5]   # cond(Lambda^2) = 1e3
    for name in ROTATED:
        if name.endswith("_i1"):
            plain = name[:-3]
            assert ROTATION_HASHES[name] == ROTATION_HASHES[plain], name
            assert len({ROTATION_HASHES[plain], ROTATION_HASHES[plain + "_i2"],
                        ROTATION_HASHES[plain + "_i3"]}) == 3, plain
    d = hashlib.sha256()
    for n in sorted(names()):
        if n.startswith("bbobbiobj") and not n.startswith("bbobbiobjmixint"):   # same matrices
            p = BB.problem(int(n[9:11]), int(n[13:15]), 1)
            for A in (p.alpha.R, p.alpha.Q, p.beta.R, p.beta.Q):
                if A is not None:
                    d.update((np.round(A, 6) + 0.0).tobytes())
    assert d.hexdigest()[:16] == BBOB_ROTATION_HASH, d.hexdigest()[:16]


@test
def a_rotated_problem_is_its_base_where_its_map_is_the_identity():
    """At Q = I and c = 1/2 rDTLZ and rMaF are DTLZ and MaF, at Q = I rWFG and
    rZCAT are WFG and ZCAT, at Q = I and c = 0 rZDT and rcPolygon are ZDT and
    Polygon; rDTLZ2 and rDTLZ4 are their Q = I construction (the isotropic g),
    rDTLZ1, rDTLZ3, rcDTLZ2, rcDTLZ5, rMaF and rcMaF are not, nor rZDT its
    shiftZDT; the Pareto set lands on the front (rWFG8 and rWFG9: on the
    sphere, their optimum computed in order); uniform points give finite
    values; no point dominates the front; an instance other than the first is
    another function; the properties (task 6, item 1.3 and supplement 2; task
    7, §3 and §7). The instances are the same construction: their matrices
    are checked above, their values here only against instance 1."""
    if not _have_numpy():
        print("    (skipped: no NumPy)"); return
    import numpy as np
    from mootation.benchmarks import families, get
    from mootation.benchmarks import rotated as R
    from mootation.benchmarks.properties import properties
    from mootation.benchmarks.registry import ROTATED
    rng = np.random.default_rng(11)

    def box(p, n):
        lo = np.array([b[0] for b in p.bounds])
        hi = np.array([b[1] for b in p.bounds])
        return lo + (hi - lo) * rng.random((n, p.n_vars))

    task6 = ("DTLZ1", "DTLZ2", "DTLZ3", "DTLZ4", "WFG4", "WFG5", "WFG6", "WFG7")
    for name, r in ROTATED.items():
        if "_i" in name:
            continue
        p, base = get(name), get(r.base)
        ident = R.Rotated(name.split("_")[0], r.M, p.n_vars)
        ident.A = np.eye(len(r.A))
        ident._matrices = {len(A): np.eye(len(A)) for A in r.matrices()}
        if r.zdt or r.kind == "polygon":
            ident.optimum = np.zeros(len(r.A))
            ident.spread = np.ones(len(r.A))
        elif r.kind in ("dtlz", "maf"):
            ident.optimum = np.full(len(r.A), 0.5)
        many = r.family in task6 or r.zdt         # task 7's new kinds: fewer points
        X = box(p, 2000 if many else 300)
        assert max(np.abs(np.subtract(ident.evaluate(x), base.evaluate(list(x)))).max()
                   for x in X) < 1e-12, name
        F = np.array([p.evaluate(list(x)) for x in box(p, 10_000 if many else 1000)])
        assert np.all(np.isfinite(F)), name
        if r.kind == "zcat" and name not in ("rZCAT1_3D", "rZCAT14_3D", "rZCAT19_3D"):
            continue                   # ZCAT's Pareto-set sampler takes seconds a problem
        S = np.asarray(p.pareto_set(100), float)
        FS = np.array([p.evaluate(list(x)) for x in S])
        if r.wfg or r.kind == "zcat":  # the base's own values at the same point
            assert np.abs(FS - [base.evaluate(list(x)) for x in S]).max() < 1e-9, name
            if r.family in ("WFG8", "WFG9"):         # on the sphere: u* is the optimum
                H = FS / (2.0 * np.arange(1, r.M + 1))
                assert np.abs((H ** 2).sum(axis=1) - 1.0).max() < 1e-9, name
        elif r.kind in ("maf", "polygon"):      # the base at its own optimum
            S0 = S.copy()
            S0[:, r.first:] = 0.5 if r.kind == "maf" else 0.0
            assert np.abs(FS - [base.evaluate(list(x)) for x in S0]).max() < 1e-9, name
        elif r.zdt:                    # the base's own values on its own Pareto set
            FB = [base.evaluate(list(x)) for x in base.pareto_set(100)]
            assert np.abs(FS - FB).max() < 1e-9, name
            assert max(p.distance_fn(list(x)) for x in S) < 1e-12, name
        else:
            G = FS if r.family == "DTLZ1" else FS ** 2
            assert np.abs(G.sum(axis=1) - (0.5 if r.family == "DTLZ1" else 1.0)).max() < 1e-9
            if p.distance_fn is not None or r.family != "DTLZ5":
                assert max(p.distance_fn(list(x)) for x in S) < 1e-9, name
        row = properties(name)
        assert p.cyclic_vars == () and (row is None) == (r.kind == "maf"), name
        assert row is None or (row["centre"], row["boundary"]) == (False, False), name
    for M in R.SIZES:
        X = box(get(f"DTLZ1_{M}D"), 2000)
        for stem, other, same in (("rDTLZ2", None, True), ("rDTLZ4", None, True),
                                  ("rDTLZ1", "r0DTLZ1", False), ("rDTLZ3", "r0DTLZ3", False),
                                  ("rcDTLZ2", "c0DTLZ2", False)):
            r = ROTATED[R.name(stem, M)]
            if other is None:
                q = R.Rotated(stem, M, r.n_vars)
                q.A = np.eye(len(r.optimum))
            else:
                q = ROTATED[R.name(other, M)]
            Y = box(get(R.name(stem, M)), 2000)
            d = max(np.abs(np.subtract(r.evaluate(x), q.evaluate(x))).max() for x in Y)
            assert (d < 1e-12) == same, (stem, M, d)
        for stem, other in (("rcDTLZ5", "c0DTLZ5"), ("rcMaF1", "c0MaF1"), ("rMaF2", "r0MaF2"),
                            ("rMaF3", "r0MaF3"), ("rMaF4", "r0MaF4"), ("rcMaF5", "c0MaF5"),
                            ("rcMaF6", "c0MaF6"), ("rcPolygon", "c0Polygon"),
                            ("rWFG8", None), ("rZCAT1", None)):
            p = get(R.name(stem, M))
            q = get(R.name(other, M) if other else ROTATED[R.name(stem, M)].base)
            Y = box(p, 200)
            d = max(np.abs(np.subtract(p.evaluate(list(x)), q.evaluate(list(x)))).max()
                    for x in Y)
            assert d > 1e-6, (stem, M, d)
    for stem in ("rZDT1", "rZDT2", "rZDT3", "rZDT4", "rZDT6"):
        p, q = get(stem), get("shift" + stem[1:])
        Y = box(p, 300)
        assert max(np.abs(np.subtract(p.evaluate(list(x)), q.evaluate(list(x)))).max()
                   for x in Y) > 1e-3, stem
    for name in ("rZDT4", "rDTLZ1_3D", "rcDTLZ2_10D", "rWFG7_5D", "rWFG8_3D", "rcMaF1_5D",
                 "rZCAT19_3D", "rcPolygon_3D"):
        p = get(name)
        Y = box(p, 300)
        for k, same in ((1, True), (2, False), (3, False)):
            d = max(np.abs(np.subtract(p.evaluate(list(x)), get(f"{name}_i{k}").evaluate(
                list(x)))).max() for x in Y)
            assert (d < 1e-12) == same, (name, k, d)
    for name in ("rDTLZ1_3D", "rcDTLZ2_3D", "rWFG4_3D", "rWFG7_5D", "rZDT1", "rZDT4",
                 "rZDT6", "rcDTLZ2_10D", "rWFG1_3D", "rWFG8_3D", "rWFG9_5D", "rcDTLZ5_3D",
                 "rcMaF1_3D", "rMaF2_3D", "rMaF3_3D", "rcPolygon_3D"):
        p, r = get(name), ROTATED[name]
        Z = np.asarray(p.pareto_front(2000), float)
        near = np.asarray(p.pareto_set(300), float)
        lo = np.array([b[0] for b in p.bounds])
        hi = np.array([b[1] for b in p.bounds])
        near[:, r.first:] += 1e-3 * (hi - lo)[r.first:] * rng.standard_normal(
            (len(near), p.n_vars - r.first))
        X = np.vstack([box(p, 500), np.clip(near, lo, hi)])
        F = np.array([p.evaluate(list(x)) for x in X])
        D = (F[:, None, :] - Z[None, :, :]) / (Z.max(axis=0) - Z.min(axis=0))
        assert not np.all(D < -1e-9, axis=2).any(), name     # better than the front everywhere
    assert properties("rDTLZ1_3D")["separable"] is False
    assert properties("rDTLZ2_3D")["separable"] is True          # its g ignores the rotation
    assert properties("r0DTLZ1_3D")["separable"] is True
    assert properties("rZDT1")["separable"] is False and properties("rZDT4")["multimodal"]
    assert properties("rDTLZ1_3D_i2")["n_obj"] == 3 and properties("rcDTLZ2_10D")["front_dim"] == 9
    assert properties("rWFG8_3D")["separable"] is False and properties("rcMaF1_3D") is None
    assert properties("rcDTLZ5_5D")["front"] == "degenerate+mixed"
    assert properties("rZCAT1_3D")["linkage"] and properties("rcPolygon_3D")["separable"] is False
    fam = families()
    assert sorted(fam["r0DTLZ"]) == ["r0DTLZ1_3D", "r0DTLZ1_5D", "r0DTLZ3_3D", "r0DTLZ3_5D"]
    counts = {k: len(fam[k]) for k in ("rDTLZ", "rWFG", "c0DTLZ", "rcDTLZ", "rZDT", "rMaF",
                                       "rcMaF", "r0MaF", "c0MaF", "rZCAT", "rcPolygon",
                                       "c0Polygon")}
    assert counts == {"rDTLZ": 20, "rWFG": 72, "c0DTLZ": 6, "rcDTLZ": 24, "rZDT": 20,
                      "rMaF": 24, "rcMaF": 24, "r0MaF": 6, "c0MaF": 6, "rZCAT": 160,
                      "rcPolygon": 8, "c0Polygon": 2}, counts


# ── task 6, item 6: the RE and CRE real-world problems ─────────────────────

@test
def re_problems_match_reproblems():
    """RE21-91 and CRE21-51 give the objectives (and the CRE their constraint
    violations) of ryojitanabe/reproblems' Python version at five points each
    (tests/data/reproblems-7876b4e_re.json); RE91 with its four normal draws
    fixed (task 6, item 6)."""
    if not _have_numpy():
        print("    (skipped: no NumPy)"); return
    from mootation.benchmarks import get, names
    from mootation.benchmarks import re as RE
    from mootation.benchmarks.noisy import is_noisy
    ref = reference_data("reproblems-7876b4e_re")
    assert sorted({c["problem"] for c in ref["cases"]}) == \
        sorted(n for n in names() if (n[:2] == "RE" or n[:3] == "CRE") and not is_noisy(n)
               and not n.endswith("c"))                               # RE22c, RE33c: ours
    for case in ref["cases"]:
        name, x = case["problem"], case["x"]
        p = get(name)
        if name == "RE91":
            mine = RE.re91(x, case["normal_draws"]).tolist()
        else:
            mine = p.evaluate(x)
        assert_close(mine, case["f"], ref["tolerance"], name)
        if "g" in case:
            assert_close(p.constraints(x), case["g"], ref["tolerance"], name)


@test
def re_fronts_frames_and_the_kept_defects():
    """The reference fronts are mutually non-dominated, CRE's the feasible
    points of RE's; the frames are the repository's files; 10^4 uniform points
    give finite values; RE22 keeps the original's "3,10"; RE91 repeats with
    its seed; RE and CRE are ranked apart and are engineering problems
    (task 6, item 6)."""
    if not _have_numpy():
        print("    (skipped: no NumPy)"); return
    import numpy as np
    from mootation.benchmarks import get
    from mootation.benchmarks import re as RE
    from mootation.benchmarks.properties import properties
    from mootation.benchmarks.registry import _strictly_dominated_by
    from mootation.run import campaign as C
    rng = np.random.default_rng(12)
    for name in sorted({c["problem"] for c in reference_data("reproblems-7876b4e_re")["cases"]}):
        p = get(name)
        F = np.asarray(p.pareto_front(1000), float)
        assert len(F) and not _strictly_dominated_by(F, F).any(), name
        assert p.front_exact is False and properties(name)["family_kind"] == "engineering"
        if name in RE.CRE:
            base = RE.CRE[name][0]
            G = RE.reference(base)
            feasible = {tuple(r) for r in G[G[:, -1] == 0.0][:, :-1]}
            assert all(tuple(r) in feasible for r in F), name
            ideal, nadir = RE.frame(base)
            assert (p.ideal, p.nadir) == (ideal[:-1], nadir[:-1]), name
        else:
            assert len(F) == len(RE.reference(name)) and (p.ideal, p.nadir) == RE.frame(name)
        lo = np.array([b[0] for b in p.bounds])
        hi = np.array([b[1] for b in p.bounds])
        X = lo + (hi - lo) * rng.random((10_000, p.n_vars))
        assert all(np.all(np.isfinite(p.evaluate(list(x)))) for x in X), name
        if p.has_cons:
            assert all(np.all(np.isfinite(p.constraints(list(x)))) for x in X[:2000]), name
    assert 3.0 in RE.RE22_VALUES and 10.0 in RE.RE22_VALUES and 3.1 not in RE.RE22_VALUES
    a, b = get("RE91").make_evaluator(4), get("RE91").make_evaluator(4)
    x = [1.0, 0.9, 1.0, 1.0, 1.75, 0.8, 0.8]
    run = [a(x) for _ in range(5)]
    assert run == [b(x) for _ in range(5)] and run[0] != run[1]
    assert get("RE91").evaluate(x) == get("RE91").evaluate(x)
    assert properties("ZDT1")["family_kind"] == "synthetic"

    def row(problem, alg, v):
        return {"problem": problem, "algorithm": alg, "seed": 1, "status": "done",
                "final": {"igdp_norm": v}, "n_objs": 2}

    rows = [row("ZDT1", "a", 0.1), row("ZDT1", "b", 0.2), row("RE21", "a", 0.3),
            row("RE21", "b", 0.1)]
    r = C.rank_table(rows, "igdp_norm")
    e = dict(r["algorithms"])
    assert "engineering" in r["groups"] and e["a"]["all"] == (1.0, 1), e
    assert e["a"]["engineering"] == (2.0, 1) and e["b"]["engineering"] == (1.0, 1), e
    text = C.format_ranks(r)
    assert "ranked apart" in text and "approximation" in text, text
    assert "engineering" not in C.rank_table(rows[2:], "igdp_norm")["groups"]


# ── task 2, step 3: operators and switchable variation ──────────────────────

@test
def operator_words_are_knobs_of_the_run_layer():
    from mootation.run.knobs import text_knobs
    words = text_knobs()
    assert set(words["crossover"]) == {"sbx", "uniform", "blx_alpha", "spx", "rex", "undx",
                                       "pcx", "de", "de_pbest", "portfolio", "none"}, words
    assert {"polynomial", "gaussian", "cauchy", "uniform_reset", "mixture",
            "mixture_cauchy", "none"} == set(words["mutation"]), words
    for k in ("mutation_scale", "mixture_q", "blx_alpha", "crossover", "mutation"):
        assert k in knob_names(), k
    tail = 'gens = 5\n'
    ok = loads(_MIN.replace(tail, tail + 'params = { crossover = "blx_alpha", mutation = '
                                          '"gaussian", mutation_scale = 0.05, blx_alpha = 0.3 }\n'))
    assert not [q for q in validate(ok) if "params" in q or "crossover" in q or "mutation" in q]
    bad = loads(_MIN.replace(tail, tail + 'params = { mutation = "levy" }\n'))
    assert any("mutation" in q and "levy" in q for q in validate(bad)), validate(bad)


@test
def switchable_operators_run_inside_the_box_and_keep_the_defaults():
    """nsga2, ibea_eplus, spea2_sde, agemoea, sms_emoa take crossover/mutation; the defaults are SBX + PM."""
    if not _have_numpy() or _core_with("operator_stats") is None:
        print("  skip  switchable_operators...: no NumPy or stale _core"); return
    import numpy as np
    from mootation import minimize
    from mootation.benchmarks import get
    p = get("ZDT4")                         # x_1 in [0, 1], the rest in [-5, 5]
    lo = np.array([b[0] for b in p.bounds]); hi = np.array([b[1] for b in p.bounds])
    kw = dict(bounds=p.bounds, n_objs=2, pop_size=100, max_evaluations=1500, seed=4)
    for alg in ("nsga2", "ibea_eplus", "spea2_sde", "agemoea", "sms_emoa"):
        plain = minimize(p.evaluate, algorithm=alg, **kw)
        same = minimize(p.evaluate, algorithm=alg, crossover="sbx", mutation="polynomial", **kw)
        assert np.array_equal(np.array(plain.objectives), np.array(same.objectives)), alg
        assert dict(plain.operators) == {"sbx": "none", "polynomial": "none"}, plain.operators
        for extra in ({"crossover": "blx_alpha", "blx_alpha": 1.0, "bound_repair": "reflect"},
                      {"crossover": "uniform", "mutation": "gaussian", "mutation_scale": 0.3},
                      {"mutation": "mixture_cauchy", "mixture_q": 0.5, "bound_repair": "random"},
                      {"mutation": "uniform_reset"},
                      # the multi-parent ones: 11 parents for spx and rex at n = 10
                      {"crossover": "spx"}, {"crossover": "rex"},
                      {"crossover": "undx", "bound_repair": "midpoint"},
                      {"crossover": "pcx", "bound_repair": "clip"}):
            r = minimize(p.evaluate, algorithm=alg, **kw, **extra)
            assert not r.ignored, (alg, extra, r.ignored)
            X = np.array(r.variables)
            assert np.all((X >= lo) & (X <= hi)), (alg, extra)
            ops = dict(r.operators)
            name = extra.get("crossover", "sbx")
            assert name in ops, (alg, extra, r.operators)
    try:                                    # a multi-parent child cannot redraw one variable
        minimize(p.evaluate, algorithm="nsga2", crossover="spx", bound_repair="resample", **kw)
    except ValueError as e:
        assert "resample" in str(e), e
    else:
        raise AssertionError("spx with bound_repair = resample must be refused")
    r = minimize(p.evaluate, algorithm="moead_de", mutation="gaussian", **kw)
    assert dict(r.operators)["gaussian"] == "repaired by the caller", r.operators
    r = minimize(p.evaluate, algorithm="spea2", crossover="uniform", **kw)
    assert r.ignored == ["crossover"], r.ignored
    # sbx_var_prob: for the run only, and ignored by a core without SBX
    plain = minimize(p.evaluate, algorithm="nsga2", **kw)
    every = minimize(p.evaluate, algorithm="nsga2", sbx_var_prob=1.0, **kw)
    again = minimize(p.evaluate, algorithm="nsga2", **kw)
    assert not every.ignored and every.objectives != plain.objectives
    assert again.objectives == plain.objectives, "the toggle is put back after the run"
    r = minimize(p.evaluate, algorithm="moead_de", sbx_var_prob=1.0, **kw)
    assert r.ignored == ["sbx_var_prob"], r.ignored


# ── task 2, step 5: DMS and crowding in the decision space ──────────────────

@test
def dms_and_crowding_space_are_knobs_of_the_run_layer():
    from mootation.run.knobs import text_knobs
    words = text_knobs()
    assert list(words["crowding_space"]) == ["objectives", "decision"], words
    assert list(words["dms_init"]) == ["line", "single", "random"], words
    assert list(words["dms_poll"]) == ["first", "random"], words
    assert "dms" in algorithm_names()
    tail = 'gens = 5\n'
    bad = loads(_MIN.replace(tail, tail + 'params = { crowding_space = "genotype" }\n'))
    assert any("crowding_space" in q and "genotype" in q for q in validate(bad)), validate(bad)


@test
def cpp_dss_selects_what_the_run_layer_selects():
    """dss.hpp (DMS's answer) and archive.dss_order agree index for index, ties included."""
    if not _have_numpy() or _core_with("dss_order") is None:
        print("  skip  cpp_dss_selects...: no NumPy or stale _core"); return
    import numpy as np
    from mootation import _core
    from mootation.run.archive import dss_order
    rng = np.random.default_rng(3)
    for m in (2, 3, 5):
        F = rng.random((60, m))
        F[10] = F[20]                               # a duplicate
        F[30, 0] = F[:, 0].min()                    # a tie for the best of f1
        F = np.round(F, 2)                          # ties in the distances too
        for k in (1, m, 17, 60, 80):
            assert list(_core.dss_order(F.tolist(), k)) == dss_order(F, k).tolist(), (m, k)
    assert list(_core.dss_order([], 5)) == []
    # the almost-dominated points set aside, the same ones in both
    D = rng.random((40, 3))
    F = D / np.linalg.norm(D, axis=1, keepdims=True)
    bad = []
    for j in range(3):
        p = F[np.argmin(F[:, j])].copy()
        p[j] -= 1e-4
        p[(j + 1) % 3] += 0.8
        bad.append(p)
    F = np.vstack([F, bad])
    for k in (3, 20, 40, 43):
        assert list(_core.dss_order(F.tolist(), k)) == dss_order(F, k).tolist(), k
    assert not {40, 41, 42} & set(_core.dss_order(F.tolist(), 40))


@test
def dms_is_deterministic_stops_by_itself_and_answers_with_a_nondominated_set():
    if not _have_numpy() or _core_with("dms_init") is None:
        print("  skip  dms_...: no NumPy or stale _core"); return
    import numpy as np
    from mootation import minimize
    from mootation.run.metrics import nondominated

    def sch(x):                                     # Schaffer's, on [-5, 5]^2 here
        return [x[0] ** 2 + x[1] ** 2, (x[0] - 2) ** 2 + (x[1] - 2) ** 2]

    kw = dict(bounds=[(-5.0, 5.0)] * 2, n_objs=2, algorithm="dms", pop_size=30)
    a = minimize(sch, max_evaluations=200000, seed=1, **kw)
    b = minimize(sch, max_evaluations=200000, seed=99, **kw)
    assert a.objectives == b.objectives, "no randomness: the seed changes nothing"
    assert a.evaluations < 200000, "every step below 1e-3: the run ends before the budget"
    F = np.array(a.objectives)
    assert len(F) == 30 and len(nondominated(F)) == 30, len(F)
    # The Pareto set is x1 = x2 in [0, 2]; the smallest step polled is 1e-3 of
    # the range, 0.01 here, and a point a few such steps off the line is not
    # dominated by anything the list holds.
    X = np.array(a.variables)
    assert np.all(np.abs(X[:, 0] - X[:, 1]) < 0.05) and np.all((X > -0.05) & (X < 2.05)), X
    # early on the two initial lists differ; on this symmetric problem both
    # reach the same list by 2000 evaluations
    single = minimize(sch, max_evaluations=20, dms_init="single", **kw)
    line = minimize(sch, max_evaluations=20, **kw)
    assert single.variables != line.variables       # (the objectives are symmetric)
    assert not single.ignored and "dms_init" in minimize(sch, max_evaluations=100, algorithm="nsga2",
                                                         bounds=kw["bounds"], n_objs=2,
                                                         dms_init="single").ignored


@test
def dms_can_start_off_the_grid_and_poll_at_random():
    """DMS-6 (task 5, run A1): dms_init = random and dms_poll = random make the
    seed matter — the same seed repeats the run, another changes it — while the
    defaults stay one run for every seed. The line's grid puts coordinates on
    exactly 0.5, DTLZ1's optimum; a random start does not."""
    if not _have_numpy() or _core_with("dms_poll") is None:
        print("  skip  dms_can_start...: no NumPy or stale _core"); return
    import numpy as np
    from mootation import minimize
    from mootation.benchmarks import get
    p = get("DTLZ1_3D")
    kw = dict(bounds=p.bounds, n_objs=3, algorithm="dms", pop_size=20, max_evaluations=600)
    rnd = dict(dms_init="random", dms_poll="random")
    a = minimize(p.evaluate, seed=1, **rnd, **kw)
    b = minimize(p.evaluate, seed=1, **rnd, **kw)
    c = minimize(p.evaluate, seed=2, **rnd, **kw)
    assert not a.ignored and a.objectives == b.objectives and a.objectives != c.objectives
    d = minimize(p.evaluate, seed=1, **kw)
    assert d.objectives == minimize(p.evaluate, seed=2, **kw).objectives
    assert minimize(p.evaluate, seed=1, dms_poll="random", **kw).objectives != d.objectives
    assert np.any(np.array(d.variables) == 0.5) and not np.any(np.array(a.variables) == 0.5)
    try:
        minimize(p.evaluate, seed=1, dms_poll="last", **kw)
    except ValueError as exc:
        assert "first or random" in str(exc), exc
    else:
        raise AssertionError("an unknown dms_poll must be refused")


@test
def clipping_is_counted_where_emyo_c_hcca_and_maoea_iamd_clip():
    """Task 5, item 2: EMyO/C's offspring, HCCA's DE/rand/1 and MaOEA-IAMD's
    noise used to be clipped past the out-of-box counts (the first two by
    hand, the last by the store); now each is in the operator log, repaired
    before the store, and counted. EMyO/C's knobs for run A2 (bound_repair,
    step_share, F, F_spread) change its run; its defaults are the paper's."""
    if not _have_numpy() or _core_with("step_share") is None:
        print("  skip  clipping_is_counted...: no NumPy or stale _core"); return
    from mootation import minimize
    from mootation.benchmarks import get
    p = get("DTLZ3_3D")
    kw = dict(bounds=p.bounds, n_objs=3, pop_size=20, max_evaluations=400, seed=3,
              operator_stats=True)
    for alg, op in (("emyo_c", "emyo_c_difference"), ("hcca", "hcca_de_rand_1"),
                    ("maoea_iamd", "maoea_iamd_gaussian")):
        r = minimize(p.evaluate, algorithm=alg, **kw)
        ops = dict(r.operators)
        assert ops.get(op) == "clip" and "vault" not in ops, (alg, r.operators)
        if alg != "maoea_iamd":                 # its noise is 1 % of the range
            assert dict(r.operator_totals)["oob_share"] > 0, (alg, r.operator_totals)
    plain = minimize(p.evaluate, algorithm="emyo_c", **kw)
    knobs = minimize(p.evaluate, algorithm="emyo_c", bound_repair="reflect", step_share=0.45,
                     F=0.5, F_spread=0.1, **kw)
    assert not knobs.ignored and dict(knobs.operators)["emyo_c_difference"] == "reflect"
    assert knobs.objectives != plain.objectives
    same = minimize(p.evaluate, algorithm="emyo_c", step_share=0.5, F=1.0, F_spread=0.0, **kw)
    assert same.objectives == plain.objectives, "the paper's values are the defaults"
    assert minimize(p.evaluate, algorithm="nsga2", step_share=0.45, **kw).ignored == ["step_share"]


@test
def crowding_in_the_decision_space_is_off_by_default_and_changes_the_run_when_on():
    if not _have_numpy() or _core_with("crowding_space") is None:
        print("  skip  crowding_space...: no NumPy or stale _core"); return
    from mootation import minimize
    from mootation.benchmarks import get
    p = get("ZDT1")
    kw = dict(bounds=p.bounds, n_objs=2, pop_size=40, max_evaluations=2000, seed=5)
    plain = minimize(p.evaluate, algorithm="nsga2", **kw)
    same = minimize(p.evaluate, algorithm="nsga2", crowding_space="objectives", **kw)
    dec = minimize(p.evaluate, algorithm="nsga2", crowding_space="decision", **kw)
    assert plain.objectives == same.objectives
    assert not dec.ignored and dec.objectives != plain.objectives
    r = minimize(p.evaluate, algorithm="spea2", crowding_space="decision", **kw)
    assert r.ignored == ["crowding_space"], r.ignored
    try:
        minimize(p.evaluate, algorithm="nsga2", crowding_space="genotype", **kw)
    except ValueError as exc:
        assert "objectives or decision" in str(exc), exc
    else:
        raise AssertionError("an unknown crowding_space must be refused")


# ── task 2, D4-D7: set comparisons and two indicators ───────────────────────

@test
def discrete_r2_magnitude_chamfer_and_eps_mult_are_what_they_say():
    if not _have_numpy():
        print("    (skipped: no NumPy)"); return
    import numpy as np
    from mootation.run import metrics as M
    W = M.das_dennis(3, 12)
    assert W.shape == (91, 3) and np.allclose(W.sum(1), 1.0)
    assert M.das_dennis(2, 99).shape == (100, 2) and M.das_dennis(5, 5).shape == (126, 5)
    box = dict(ideal=[0.0, 0.0], nadir=[1.0, 1.0])
    # R2: 0 at the ideal point; one point at the nadir costs max(w, 1 - w) per weight
    assert M.r2_discrete([[0.0, 0.0]], h=99, **box) == 0.0
    w = np.arange(100) / 99.0
    assert abs(M.r2_discrete([[1.0, 1.0]], h=99, **box) - np.maximum(w, 1 - w).mean()) < 1e-12
    A = np.array([[0.2, 0.8], [0.8, 0.2]])
    assert M.r2_discrete(np.vstack([A, [[0.5, 0.5]]]), h=20, **box) <= M.r2_discrete(A, h=20, **box)
    out = M.compute(A, which=["r2"], pop=100, **box)
    assert out["r2_h"] == 99 and out["r2"] == M.r2_discrete(A, h=99, **box), out
    # magnitude: one box is the product of (1 + L/2) (Remark 2.6); two boxes at
    # M = 2 are 1 + (X + Y)/2 + HV/4 (Theorem 2.5); nothing inside the anchor, 0
    assert abs(M.magnitude([[0.0, 0.0]], ref_scale=1.1, **box) - 1.55 ** 2) < 1e-12
    assert abs(M.magnitude([[0.0, 0.0, 0.0]], ideal=[0] * 3, nadir=[1] * 3, ref_scale=1.0)
               - 1.5 ** 3) < 1e-12
    two = M.magnitude([[0.0, 0.5], [0.5, 0.0]], ref_scale=1.0, **box)
    assert abs(two - (1 + (1 + 1) / 2 + 0.75 / 4)) < 1e-12, two
    assert M.magnitude([[2.0, 2.0]], ref_scale=1.1, **box) == 0.0
    # a point ON the anchor's face: no hypervolume, but a positive magnitude
    edge = M.magnitude([[1.0, 0.0]], ref_scale=1.0, **box)
    assert abs(edge - 1.5) < 1e-12 and M.hypervolume(np.array([[1.0, 0.0]]), [0, 0], [1, 1],
                                                       ref_scale=1.0)[0] == 0.0
    # chamfer: 0 for the same set, symmetric, in normalised units
    b = [(0.0, 2.0)]
    assert M.chamfer([[0.5]], [[0.5]], bounds=b) == 0.0
    assert M.chamfer([[0.0]], [[1.0]], bounds=b) == 0.5
    X, Y = np.random.default_rng(1).random((7, 1)) * 2, np.random.default_rng(2).random((4, 1)) * 2
    assert abs(M.chamfer(X, Y, bounds=b) - M.chamfer(Y, X, bounds=b)) < 1e-15
    # multiplicative epsilon: half of B is better than B; zeros match only zeros
    B = np.array([[1.0, 4.0], [2.0, 2.0]])
    assert M.eps_mult(B, B) == 1.0
    assert M.eps_mult(B / 2, B) == 0.5 and M.eps_mult(B, B / 2) == 2.0
    assert M.eps_mult([[0.0, 1.0]], [[0.0, 2.0]]) == 0.5
    assert M.eps_mult([[1.0, 1.0]], [[0.0, 2.0]]) == float("inf")
    assert M.eps_mult([[-1.0, 1.0]], B) is None
    assert M.eps_mult(B * 3.0, B) == 3.0 and M.eps_mult(B * [1, 7], B * [1, 7]) == 1.0


@test
def seed_distance_eps_table_and_magnitude_read_a_finished_campaign():
    # run_job imports minimize, hence the extension, even for the Python baselines
    if not _have_numpy() or _core_with("Config") is None:
        print("  skip  seed_distance_eps_table...: no NumPy or no _core"); return
    import contextlib
    import io
    from mootation.run import campaign as C
    from mootation.run.config import load
    text = """algorithms = [
    { name = "random_search", pop = 0, gens = 0 },
    { name = "gsemo", pop = 0, gens = 0 },
]
[run]
name = "sets"
[problem]
kind = "builtin"
[benchmarks]
runs = 2
problems = ["ZDT1", "DTLZ2_3D"]
[campaign]
out = "res"
budget_fe = 300
metrics = ["igdp_norm"]
final_metrics = ["igdp_norm", "hv_h", "n_final"]
archive = false
"""
    with tempfile.TemporaryDirectory() as td:
        cfg_path = Path(td) / "c.toml"
        cfg_path.write_text(text, encoding="utf-8")
        cfg = load(cfg_path)
        assert validate(cfg) == [], validate(cfg)
        spec = C.campaign_spec(cfg)
        root = C.out_root(cfg, spec)
        for job in C.expand_jobs(cfg, spec):
            assert C.run_job(job, root, spec, quiet=True) == "done", job
        for flag, want in (("--seed-distance", "chamfer"), ("--eps-table", "multiplicative"),
                           ("--magnitude", "Kendall")):
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = C.main([str(cfg_path), flag])
            text_out = out.getvalue()
            assert code == 0 and want in text_out and "ZDT1" in text_out, (flag, text_out)
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            code = C.main([str(cfg_path), "--eps-table", "--scenario", "archive"])
        assert code == 1 and "final populations" in out.getvalue(), out.getvalue()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            assert C.main([str(cfg_path), "--recompute", "r2"]) == 0
        meta = json.loads((root / "DTLZ2_3D" / "gsemo" / "run_1" / "meta.json")
                          .read_text(encoding="utf-8"))
        assert meta["final"]["r2"] > 0 and meta["final"]["r2_h"] == 12, meta["final"]
        from mootation.run import report as R
        rows = C.scenario_rows(C.scan_results(root), "final")
        rep = R.seed_distance(rows)
        assert set(rep) == {(p, a) for p in ("ZDT1", "DTLZ2_3D") for a in ("random_search", "gsemo")}
        assert all(e["runs"] == 2 and e["pairs"] == 1 and e["chamfer"] > 0 for e in rep.values())
        mag = R.magnitude_experiment(rows)
        assert set(mag["per_problem"]) == {"ZDT1", "DTLZ2_3D"}, mag


# ── task 2, C3: GDE3, SMS-EMOA, MO-CMA-ES ───────────────────────────────────

@test
def hypervolume_contributions_match_the_definition():
    """hv_contribution.hpp: S(F) − S(F without p), exactly at M = 2, 3, 4; MC close."""
    if not _have_numpy() or _core_with("hv_contributions") is None:
        print("  skip  hypervolume_contributions...: no NumPy or stale _core"); return
    import numpy as np
    from mootation import _core
    rng = np.random.default_rng(7)
    for m in (2, 3, 4):
        # a non-dominated set: points on the simplex, with one copy
        P = rng.random((25, m))
        P = P / P.sum(1, keepdims=True)
        P[5] = P[6]
        r = [1.2] * m
        exact = np.array(_core.hv_contributions(P.tolist(), r))
        total = _core.hypervolume(P, r)
        brute = [total - _core.hypervolume(np.delete(P, i, 0), r) for i in range(len(P))]
        assert np.allclose(exact, brute, rtol=1e-9, atol=1e-12), (m, exact - brute)
        assert exact[5] == 0.0 and exact[6] == 0.0, "a copy has no region of its own"
        # MC: each point in its own bounding box, which at M = 4 can be far
        # larger than the region (measured: up to 11 % off per point at 20 000
        # samples, the sum within 1 %)
        mc = np.array(_core.hv_contributions(P.tolist(), r, 20000, 3))
        big = exact > 1e-4
        assert np.all(np.abs(mc[big] - exact[big]) <= 0.25 * exact[big]), (m, mc, exact)
        assert abs(mc.sum() - exact.sum()) <= 0.03 * exact.sum(), (m, mc.sum(), exact.sum())
        assert mc[5] == 0.0 and mc[6] == 0.0


@test
def gde3_smsemoa_mocmaes_run_inside_the_box_and_take_their_knobs():
    if not _have_numpy() or _core_with("hv_contributions") is None:
        print("  skip  gde3_smsemoa_mocmaes...: no NumPy or stale _core"); return
    import numpy as np
    from mootation import minimize
    from mootation.benchmarks import get
    p = get("ZDT4")                         # x_1 in [0, 1], the rest in [-5, 5]
    lo = np.array([b[0] for b in p.bounds]); hi = np.array([b[1] for b in p.bounds])
    kw = dict(bounds=p.bounds, n_objs=2, pop_size=20, max_evaluations=600)
    for alg in ("gde3", "sms_emoa", "mo_cma_es"):
        a = minimize(p.evaluate, algorithm=alg, seed=3, **kw)
        b = minimize(p.evaluate, algorithm=alg, seed=3, **kw)
        c = minimize(p.evaluate, algorithm=alg, seed=4, **kw)
        assert a.objectives == b.objectives and a.objectives != c.objectives, alg
        X = np.array(a.variables)
        assert len(X) == 20 and np.all((X >= lo) & (X <= hi)), alg
    # GDE3: F and CR are its knobs, bound_repair its DE's
    r = minimize(p.evaluate, algorithm="gde3", seed=3, F=0.5, CR=1.0, bound_repair="clip", **kw)
    assert not r.ignored and dict(r.operators) == {"de_rand_1_bin": "clip"}, (r.ignored, r.operators)
    assert r.objectives != minimize(p.evaluate, algorithm="gde3", seed=3, **kw).objectives
    # SMS-EMOA: SBX with eta_c 15 by default; MO-CMA-ES has no variation knobs
    r = minimize(p.evaluate, algorithm="sms_emoa", seed=3, eta_c=20.0, mutation="gaussian", **kw)
    assert not r.ignored, r.ignored
    r = minimize(p.evaluate, algorithm="mo_cma_es", seed=3, eta_c=20.0, **kw)
    assert r.ignored == ["eta_c"], r.ignored


@test
def invariance_check_tells_the_two_known_cases_apart():
    """--invariance-check (task 7, §5.4): NSGA-II runs the same trajectory, bit
    for bit, on DTLZ2_3D and on its objectives times 2^10 — its crowding
    distance is normalised; SMS-EMOA at three objectives does not — its
    reference point nad + 1 is absolute. The shift's outcome from the
    rank-sum and Holm: the same trajectories are invariant, distributions that
    cannot be told apart rounding, others not invariant. A campaign writes the
    table under _invariance/."""
    if not _have_numpy() or _core_with("params") is None:
        print("  skip  invariance_check...: no NumPy or stale _core"); return
    from mootation.run import campaign as C
    from mootation.run import invariance as I
    from mootation.run.config import load
    nsga2 = I.check("nsga2", "DTLZ2_3D", pop=40, budget=1200, shift=False)
    sms = I.check("sms_emoa", "DTLZ2_3D", pop=40, budget=1200, shift=False)
    assert (nsga2["scale"], nsga2["scale_first_diff"]) == ("invariant", None), nsga2
    assert sms["scale"] == "not invariant" and sms["scale_first_diff"] > 40, sms
    assert I.first_difference(*[__import__("numpy").zeros((3, 2))] * 2) is None
    rows = [{"shift_parted": 0, "igdp_plain": [1, 2, 3], "igdp_shifted": [1, 2, 3]},
            {"shift_parted": 3, "igdp_plain": [1.0, 2.0, 3.0], "igdp_shifted": [1.5, 2.5, 3.5]},
            {"shift_parted": 9, "igdp_plain": list(range(10)),
             "igdp_shifted": list(range(100, 110))}]
    I.classify_shift(rows)
    assert [r["shift"] for r in rows] == ["invariant", "rounding", "not invariant"], rows
    text = """algorithms = [ { name = "nsga2", pop = 20, gens = 0 }, { name = "hype", pop = 20, gens = 0 } ]
[run]
name = "inv"
[problem]
kind = "builtin"
[benchmarks]
runs = 1
problems = ["ZDT1"]
[campaign]
out = "res"
budget_fe = 300
"""
    with tempfile.TemporaryDirectory() as td:
        cfg_path = Path(td) / "inv.toml"
        cfg_path.write_text(text, encoding="utf-8")
        cfg = load(cfg_path)
        assert validate(cfg) == [], validate(cfg)
        spec = C.campaign_spec(cfg)
        root = C.out_root(cfg, spec)
        out = I.run_campaign(cfg, spec, root, seeds=3)
        assert "_invariance" in out
        lines = (root / "_invariance" / "invariance.csv").read_text(encoding="utf-8").splitlines()
        assert len(lines) == 3 and lines[0].startswith("problem,label,algorithm"), lines


@test
def archive_checkpoints_are_taken_after_exactly_b_evaluations():
    """[campaign] archive_checkpoints at record_at = [100, 300, 1000] (task 7,
    §5.5): the archive scenario after exactly 100, 300 and 1 000 evaluations,
    the first two inside a generation of 91 (at 100 the initial population is
    not even evaluated yet), not at the first record past them."""
    if not _have_numpy() or _core_with("params") is None:
        print("  skip  archive_checkpoints...: no NumPy or stale _core"); return
    from mootation.run import campaign as C
    from mootation.run.config import load
    text = """algorithms = [ { name = "nsga2", pop = 0, gens = 0 } ]
[run]
name = "ck"
[problem]
kind = "builtin"
[benchmarks]
runs = 1
problems = ["DTLZ2_3D"]
[campaign]
out = "res"
budget_fe = 1500
record_grid = "log"
record_at = [100, 300, 1000]
metrics = ["igdp_norm"]
archive_checkpoints = true
"""
    with tempfile.TemporaryDirectory() as td:
        cfg_path = Path(td) / "ck.toml"
        cfg_path.write_text(text, encoding="utf-8")
        cfg = load(cfg_path)
        assert validate(cfg) == [], validate(cfg)
        spec = C.campaign_spec(cfg)
        root = C.out_root(cfg, spec)
        for job in C.expand_jobs(cfg, spec):
            assert C.run_job(job, root, spec, quiet=True) == "done", job
        d = root / "DTLZ2_3D" / "nsga2" / "run_1"
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        at = meta["archive_at"]
        assert sorted(at, key=int) == ["100", "300", "1000"], at
        assert all(at[b]["fe"] == int(b) and 0 < at[b]["n"] <= 91 for b in at), at
        assert at["100"]["igdp_norm"] >= at["1000"]["igdp_norm"], at


@test
def coords_are_knobs_and_the_diagnostics_find_the_distance_block():
    """coords (task 7, supplement 1): a knob of the switchable algorithms and of
    mosa, refused words refused, ignored elsewhere. On rcDTLZ2_3D started from
    its Pareto set, pca_nd's minor subspace is the distance block: principal
    angles near 0; a campaign with coords_diag records them and the spectrum."""
    if not _have_numpy() or _core_with("coords_state") is None:
        print("  skip  coords_are_knobs...: no NumPy or stale _core"); return
    import numpy as np
    import mootation
    from mootation import _core
    from mootation.benchmarks import get
    from mootation.persistence import LoadedPopulation
    from mootation.run import campaign as C
    from mootation.run import coords_diag as D
    from mootation.run.config import load
    from mootation.run.metric_names import COORDS_STATS, is_table_name
    assert callable(mootation.minimize)
    Mz = sys.modules["mootation.minimize"]
    assert all(is_table_name(k) for k in COORDS_STATS)
    p = get("ZDT1")
    for alg in ("nsga2", "ibea_eplus", "spea2_sde", "agemoea", "sms_emoa", "mosa"):
        r = Mz.minimize(p.evaluate, p.bounds, 2, algorithm=alg, pop_size=20, max_evaluations=400,
                        seed=1, coords="pca", coords_warmup=0)
        assert not set(r.ignored) & {"coords", "coords_warmup"}, (alg, r.ignored)
    r = Mz.minimize(p.evaluate, p.bounds, 2, algorithm="moead", pop_size=20, max_evaluations=200,
                    seed=1, coords="pca")
    assert "coords" in r.ignored
    try:
        Mz.minimize(p.evaluate, p.bounds, 2, algorithm="nsga2", pop_size=20, max_evaluations=200,
                    coords="diagonal")
        raise AssertionError("coords = diagonal accepted")
    except ValueError:
        pass
    q = get("rcDTLZ2_3D")
    S = np.asarray(q.pareto_set(91), float)
    seed = LoadedPopulation([list(x) for x in S], [q.evaluate(list(x)) for x in S],
                            [0.0] * len(S), {})
    states = []

    def watch(gen, F):
        states.append(dict(_core.coords_state()))
    Mz.minimize(q.evaluate, q.bounds, 3, algorithm="nsga2", pop_size=91, max_evaluations=182,
                seed=1, coords="pca_nd", coords_warmup=0, seed_population=seed,
                on_generation=watch, record_every=1)
    st = next(s for s in states if s["active"])
    assert D.distance_start("rcDTLZ2_3D") == 2 and D.distance_start("rWFG4_3D") == 4
    assert D.distance_start("bbobbiobj01_n05_2D") is None
    amax, amean = D.principal_angles(st["R"], 2)
    assert amax < 1e-4, (amax, amean)
    rot = D.principal_angles(np.eye(12)[:, ::-1], 2)          # the block's complement
    assert rot[0] > 89.9, rot
    text = """algorithms = [ { name = "nsga2", pop = 20, gens = 0, params = { coords = "pca", coords_warmup = 2 } } ]
[run]
name = "co"
[problem]
kind = "builtin"
[benchmarks]
runs = 1
problems = ["rDTLZ1_3D"]
[campaign]
out = "res"
budget_fe = 600
metrics = ["igdp_norm"]
coords_diag = true
"""
    with tempfile.TemporaryDirectory() as td:
        cfg_path = Path(td) / "co.toml"
        cfg_path.write_text(text, encoding="utf-8")
        cfg = load(cfg_path)
        assert validate(cfg) == [], validate(cfg)
        spec = C.campaign_spec(cfg)
        root = C.out_root(cfg, spec)
        for job in C.expand_jobs(cfg, spec):
            assert C.run_job(job, root, spec, quiet=True) == "done", job
        d = root / "rDTLZ1_3D" / "nsga2" / "run_1"
        T = [json.loads(l) for l in (d / "trajectory.jsonl").read_text().splitlines()]
        late = [t for t in T if "coords_angle_max" in t]
        assert late and all(0.0 <= t["coords_angle_max"] <= 90.0 and t["coords_ratio"] >= 1.0
                            for t in late), T[-1]
        assert "coords_angle_max" not in T[0], T[0]                # warm-up: R = I


@test
def stochastic_survival_and_the_inner_archive_are_knobs():
    """Task 7, supplement 2: survival, spu_frac, inner_archive, inner_archive_cap,
    archive_reuse_p, bin_crossover and mating reach their algorithms and no
    other; the result holds the inner archive, mutually non-dominated, and
    operator_stats the share of the parents drawn from it."""
    if not _have_numpy() or _core_with("coords_state") is None:
        print("  skip  stochastic_survival...: no NumPy or stale _core"); return
    import numpy as np
    import mootation
    from mootation.benchmarks import get
    assert callable(mootation.minimize)
    Mz = sys.modules["mootation.minimize"]
    p = get("ZDT1")
    for alg in ("sms_emoa", "nsga2", "nsga3"):
        pop = 20 if alg != "nsga3" else 21
        r = Mz.minimize(p.evaluate, p.bounds, 2, algorithm=alg, pop_size=pop,
                        max_evaluations=600, seed=1, survival="stochastic", inner_archive="nd",
                        archive_reuse_p=0.5, operator_stats=True)
        assert not set(r.ignored) & {"survival", "inner_archive", "archive_reuse_p"}, (alg, r.ignored)
        A = np.asarray(r.inner_archive)
        assert len(A) > 0 and len(r.inner_archive_variables) == len(A), alg
        assert not any((np.all(A <= a, axis=1) & np.any(A < a, axis=1)).any() for a in A), alg
        share = r.operator_totals["archive_parent_share"]
        assert 0.3 < share < 0.7, (alg, share)
    r = Mz.minimize(p.evaluate, p.bounds, 2, algorithm="ibea_eplus", pop_size=20,
                    max_evaluations=400, seed=1, inner_archive="nd", inner_archive_cap=10,
                    survival="stochastic")
    assert 0 < len(r.inner_archive) <= 10 and r.ignored == ["survival"], r.ignored
    r = Mz.minimize(p.evaluate, p.bounds, 2, algorithm="moead", pop_size=20, max_evaluations=200,
                    seed=1, survival="stochastic", inner_archive="nd", mating="each")
    assert {"survival", "inner_archive", "mating"} <= set(r.ignored) and not r.inner_archive
    try:
        Mz.minimize(p.evaluate, p.bounds, 2, algorithm="nsga2", pop_size=20, max_evaluations=200,
                    mating="each")
        raise AssertionError("mating = each with SBX accepted")
    except ValueError as e:
        assert "crossover" in str(e), e
    r = Mz.minimize(p.evaluate, p.bounds, 2, algorithm="nsga2", pop_size=20, max_evaluations=200,
                    seed=1, mating="each", crossover="none")
    assert r.evaluations >= 200 and "mating" not in r.ignored


@test
def integer_and_discrete_variables_are_rounded_after_the_operators():
    """Task 7, §10: with var_types an integer variable is always an integer and
    a discrete one always from its list — in the genome (int_repair =
    lamarck) or only at the evaluation (baldwin, the genome continuous); all
    real is the run without var_types, bit for bit; the declared RE problems;
    a campaign records dup_x_share."""
    if not _have_numpy() or _core_with("coords_state") is None:
        print("  skip  integer_and_discrete...: no NumPy or stale _core"); return
    import numpy as np
    import mootation
    from mootation.benchmarks import get
    from mootation.run import campaign as C
    from mootation.run.config import load
    assert callable(mootation.minimize)
    Mz = sys.modules["mootation.minimize"]
    p = get("RE25")
    assert p.var_types[0] == "int" and p.var_types[2][0] == "discrete"
    assert get("RE22").var_types[0][1][:3] == [0.2, 0.31, 0.4] and get("RE33c").var_types == ["real"] * 4
    values = set(p.var_types[2][1])
    seen = []

    def f(x):
        seen.append(list(x))
        return p.evaluate(x)
    for repair in ("lamarck", "baldwin"):
        seen.clear()
        r = Mz.minimize(f, p.bounds, 2, algorithm="nsga2", pop_size=20, max_evaluations=600,
                        seed=1, var_types=p.var_types, int_repair=repair)
        X = np.asarray(r.variables)
        assert all(isinstance(x[0], int) and x[2] in values for x in seen), repair
        assert all(p.bounds[0][0] <= x[0] <= p.bounds[0][1] for x in seen)
        genome_int = bool(np.all(X[:, 0] == np.round(X[:, 0])) and set(X[:, 2]) <= values)
        assert genome_int == (repair == "lamarck"), (repair, X[:3])
    q = get("ZDT1")
    a = Mz.minimize(q.evaluate, q.bounds, 2, algorithm="nsga2", pop_size=20, max_evaluations=400,
                    seed=3)
    b = Mz.minimize(q.evaluate, q.bounds, 2, algorithm="nsga2", pop_size=20, max_evaluations=400,
                    seed=3, var_types=["real"] * q.n_vars)
    assert a.objectives == b.objectives
    text = """algorithms = [ { name = "nsga2", pop = 20, gens = 0, params = { eta_int = 3.0 } } ]
[run]
name = "ty"
[problem]
kind = "builtin"
[benchmarks]
runs = 1
problems = ["RE36"]
[campaign]
out = "res"
budget_fe = 600
metrics = ["igdp_norm"]
"""
    with tempfile.TemporaryDirectory() as td:
        cfg_path = Path(td) / "ty.toml"
        cfg_path.write_text(text, encoding="utf-8")
        cfg = load(cfg_path)
        assert validate(cfg) == [], validate(cfg)
        spec = C.campaign_spec(cfg)
        root = C.out_root(cfg, spec)
        for job in C.expand_jobs(cfg, spec):
            assert C.run_job(job, root, spec, quiet=True) == "done", job
        meta = json.loads((root / "RE36" / "nsga2" / "run_1" / "meta.json").read_text(
            encoding="utf-8"))
        assert 0.0 <= meta["final"]["dup_x_share"] < 1.0, meta["final"]


@test
def bit_string_problems_their_fronts_and_the_binary_genome():
    """Supplement 2, §3: OneMinMax, LOTZ and OJZJ against every one of the 2^n
    strings (n <= 12): the front, its size; the binary genome reaches the
    cores and comes back (result.binary_variables, a population file round
    trip); a core without a binary genome refuses by name; a campaign records
    cover_gen."""
    if not _have_numpy() or _core_with("coords_state") is None:
        print("  skip  bit_string...: no NumPy or stale _core"); return
    import itertools
    import numpy as np
    import mootation
    from mootation.benchmarks import binary as B
    from mootation.benchmarks import get
    from mootation.run import campaign as C
    from mootation.run.config import load
    assert callable(mootation.minimize)
    Mz = sys.modules["mootation.minimize"]
    for kind, n, k in (("OMM", 8, 0), ("LOTZ", 10, 0), ("OJZJ", 10, 3), ("OJZJ", 12, 2)):
        F = np.array([B.objectives(kind, x, k) for x in itertools.product((0, 1), repeat=n)])
        nd = {tuple(f) for f in F if not np.any(np.all(F >= f, axis=1) & np.any(F > f, axis=1))}
        front = {tuple(r) for r in B.front(kind, n, k)}
        assert nd == front, (kind, n, k, sorted(nd - front), sorted(front - nd))
        assert len(front) == (n - 2 * k + 3 if kind == "OJZJ" else n + 1)
    p = get("OJZJ_n10_k3")
    assert p.var_types == ["bin"] * 10 and len(p.pareto_front(1000)) == 7
    r = Mz.minimize(p.evaluate, p.bounds, 2, algorithm="nsga2", pop_size=16, max_evaluations=800,
                    seed=1, var_types=p.var_types, save_population=None)
    assert len(r.binary_variables) == 16 and all(set(b) <= {0, 1} for b in r.binary_variables)
    assert all(np.allclose(p.evaluate(b), f) for b, f in zip(r.binary_variables, r.objectives))
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "pop.csv"
        r = Mz.minimize(p.evaluate, p.bounds, 2, algorithm="sms_emoa", pop_size=8,
                        max_evaluations=200, seed=2, var_types=p.var_types, save_population=path,
                        bin_crossover="none")
        back = mootation.load_population(path)
        assert sorted(map(tuple, back.binary)) == sorted(map(tuple, r.binary_variables))
        again = Mz.minimize(p.evaluate, p.bounds, 2, algorithm="sms_emoa", pop_size=8,
                            max_evaluations=100, seed=2, var_types=p.var_types,
                            seed_population=back)
        assert again.evaluations <= 100
    try:
        Mz.minimize(p.evaluate, p.bounds, 2, algorithm="mosa", pop_size=8, max_evaluations=100,
                    var_types=p.var_types)
        raise AssertionError("mosa took a binary genome")
    except ValueError as e:
        assert "mosa" in str(e), e
    text = """algorithms = [ { name = "nsga2", pop = 44, gens = 0, params = { bin_crossover = "none", inner_archive = "nd" } } ]
[run]
name = "bits"
[problem]
kind = "builtin"
[benchmarks]
runs = 1
problems = ["OMM_n10"]
[campaign]
out = "res"
budget_fe = 3000
record_grid = "generations"
record_every = 1
metrics = ["igdp_norm"]
"""
    with tempfile.TemporaryDirectory() as td:
        cfg_path = Path(td) / "bits.toml"
        cfg_path.write_text(text, encoding="utf-8")
        cfg = load(cfg_path)
        assert validate(cfg) == [], validate(cfg)
        spec = C.campaign_spec(cfg)
        root = C.out_root(cfg, spec)
        for job in C.expand_jobs(cfg, spec):
            assert C.run_job(job, root, spec, quiet=True) == "done", job
        fin = json.loads((root / "OMM_n10" / "nsga2" / "run_1" / "meta.json").read_text(
            encoding="utf-8"))["final"]
        assert fin["cover_reached"] and fin["cover_archive_gen"] <= fin["cover_gen"], fin
        assert fin["cover_evals"] <= 3000 + 44, fin


@test
def the_corrected_re22c_and_re33c_and_their_fronts():
    """Task 7, §9: RE22c is RE22 with x1's list as its source has it, 3.10
    for the original's "3,10" and the 7.00 it leaves out (77 values), RE33c is
    RE33 with x4 in [2, 11]; the originals are
    unchanged; their fronts, built here, are not beaten by 20 000 random points
    of the box, and their frames are the fronts' own."""
    if not _have_numpy():
        print("    (skipped: no NumPy)"); return
    import numpy as np
    from mootation.benchmarks import get
    from mootation.benchmarks import re as RE
    from mootation.benchmarks.properties import properties
    from mootation.families import family
    values = list(RE.RE22C_VALUES)
    assert len(values) == 77 and 3.1 in values and 7.0 in values and 10.0 not in values
    assert values.count(3.0) == 1 and values == sorted(values)
    a, b = get("RE22c"), get("RE22")
    assert a.evaluate([3.11, 10.0, 20.0]) != b.evaluate([3.11, 10.0, 20.0])   # 3.10 against 3.0
    assert a.evaluate([7.01, 10.0, 20.0]) != b.evaluate([7.01, 10.0, 20.0])   # 7.00 against 7.11
    assert a.evaluate([5.0, 10.0, 20.0]) == b.evaluate([5.0, 10.0, 20.0])
    c, d = get("RE33c"), get("RE33")
    assert c.bounds[3] == (2.0, 11.0) and d.bounds[3] == (11.0, 20.0)
    x = [60.0, 90.0, 2000.0, 11.0]
    assert c.evaluate(x) == d.evaluate(x)
    rng = np.random.default_rng(5)
    for p, rows in ((a, 1000), (c, 1500)):
        Z = np.asarray(p.pareto_front(1000), float)
        assert Z.shape == (rows, p.n_obj) and not p.front_exact
        assert np.allclose(Z.min(axis=0), p.ideal) and np.allclose(Z.max(axis=0), p.nadir)
        lo = np.array([u for u, _ in p.bounds])
        hi = np.array([v for _, v in p.bounds])
        F = np.array([p.evaluate(list(lo + (hi - lo) * rng.random(len(lo))))
                      for _ in range(20_000)])
        F = F[np.all(np.isfinite(F), axis=1)]
        span = Z.max(axis=0) - Z.min(axis=0)
        beaten = [np.any(np.all(F <= z - 1e-6 * span, axis=1)) for z in Z]
        assert not any(beaten), (p.name, sum(beaten))
        assert family(p.name) == "RE" and properties(p.name) is not None


@test
def das_cmop_follows_its_table_and_its_fronts_hold():
    """Task 7, §8: DAS-CMOP1-9 (Fan et al. 2020, Table 2) at hand-checked
    points; the triplet's parameters (Eq. 7), the equality at zeta = 1; the
    sixteen triplets of Table 3 and Figure 6's registered; the front's points
    satisfy every constraint and constructed feasible points do not beat it;
    the properties name the triplet's kinds of difficulty."""
    if not _have_numpy():
        print("    (skipped: no NumPy)"); return
    import math
    import numpy as np
    from mootation.benchmarks import families, get
    from mootation.benchmarks import dascmop as D
    from mootation.benchmarks.properties import properties
    assert len(families()["DAS-CMOP"]) == 9 * 16 + 3
    assert D.params(0.5, 0.5, 0.5) == (0.0, 0.5, 0.5 - math.log(0.5), 0.25)
    b, d, e, r = D.params(0.0, 0.0, 0.0)
    assert (b, d, math.isinf(e), r) == (-1.0, 0.0, True, 0.0)
    assert D.params(0.0, 1.0, 0.0)[2] == 0.5                       # e = d: g = d exactly
    x = np.full(30, math.sin(0.5 * math.pi * 0.3))
    x[0] = 0.3
    assert D.g_value(1, x) == 0.0
    f, c = D.evaluate(1, x, (0.0, 0.0, 0.0))
    assert np.allclose(f, [0.3, 1.0 - 0.09]) and max(c) <= 0.0       # all hold at (0, 0, 0)
    f, c = D.evaluate(2, x, (0.5, 0.5, 0.5))
    assert np.allclose(f, [0.3, 1.0 - math.sqrt(0.3)]) and c[1] > 0  # g = 0 < d: infeasible
    z = np.full(30, 0.5)
    z[0] = 0.2
    assert D.g_value(4, z) == 0.0                                   # (n - 1) - (n - 1)
    y = np.full(30, 0.5)
    y[:2] = [0.2, 0.7]
    assert D.g_value(7, y) == 0.0
    f, _ = D.evaluate(7, y, (0.0, 0.0, 0.0))
    assert np.allclose(f, [0.14, 0.56, 0.3]) and abs(sum(f) - 1.0) < 1e-12
    rng = np.random.default_rng(3)
    for name in ("DAS-CMOP1_f6", "DAS-CMOP5_t08", "DAS-CMOP7_t04"):
        p = get(name)
        i, t = int(name[8]), D.TRIPLETS[7] if name.endswith("t08") else (
            D.FIGURE6 if name.endswith("f6") else D.TRIPLETS[3])
        Z = np.asarray(p.pareto_front(1000), float)
        assert len(Z) > 100 and np.allclose(Z.min(axis=0), p.ideal)
        # feasible points made on purpose: x1 (x2) at random, g = d + a little
        bb, dd, ee, rr = D.params(*t)
        pts = []
        for _ in range(4000):
            x = np.full(30, 0.5)
            x[0], x[1] = rng.random(), rng.random()
            if i <= 3:
                x[1:] = math.sin(0.5 * math.pi * x[0])
            gx = D.g_value(i, x)
            f, c = D.evaluate(i, x, t)
            G = dd + (min(ee, dd + 1.0) - dd) * rng.random() * 0.2 - gx
            f = list(np.asarray(f) + G)
            F = np.asarray([f])
            cc = D.constraints_c(i, F, x[0], x[1], gx + G, t)[0]
            if np.all(cc >= 0):
                pts.append(f)
        P = np.asarray(pts)
        assert len(P) > 50, name
        span = Z.max(axis=0) - Z.min(axis=0)
        beaten = [np.any(np.all(P <= z - 1e-3 * span, axis=1)) for z in Z]
        assert not any(beaten), (name, sum(beaten))
    assert properties("DAS-CMOP3_t02")["difficulty"] == "feasibility"
    assert properties("DAS-CMOP8_t12")["difficulty"] == "diversity+feasibility+convergence"
    assert properties("DAS-CMOP4_t01")["multimodal"] and properties("ZDT1")["difficulty"] is None


@test
def das_cmop_mmo_and_the_wfg8_wfg9_optimum_match_pymoo():
    """DAS-CMOP1-9's objectives and constraints, Omni-test's and SYM-PART
    simple's objectives and WFG8's and WFG9's optimal distance variables give
    pymoo 0.6.2's values at fixed points (tests/data/pymoo-0.6.2_das-cmop.json,
    _omni-sympart.json, _wfg-optimum.json). At zeta = 1 the equality constraint
    is not compared (g_not_compared): its tolerance here is the paper's 10^-6
    (Eq. 13)."""
    if not _have_numpy():
        print("    (skipped: no NumPy)"); return
    from mootation.benchmarks import dascmop as D, mmo, rotated
    ref = reference_data("pymoo-0.6.2_das-cmop")
    for case in ref["cases"]:
        fo, co = D.evaluate(case["problem"], case["x"], tuple(case["triplet"]))
        skip = set(case.get("g_not_compared", []))
        what = (case["problem"], case["triplet"])
        assert_close(list(fo), case["f"], ref["tolerance"], what)
        assert_close([v for k, v in enumerate(co) if k not in skip],
                     [v for k, v in enumerate(case["g"]) if k not in skip], ref["tolerance"], what)
    ref = reference_data("pymoo-0.6.2_omni-sympart")
    for case in ref["cases"]:
        assert_close(mmo.evaluate(case["problem"], case["x"]), case["f"], ref["tolerance"],
                     case["problem"])
    ref = reference_data("pymoo-0.6.2_wfg-optimum")
    for case in ref["cases"]:
        mine = (rotated.wfg8_optimum(case["positions"], case["l"]) if case["function"] == "wfg8_optimum"
                else rotated.wfg9_optimum(case["l"]))
        assert_close(list(mine), case["y"], ref["tolerance"], case["function"])


def _lir_attain(L, i, z):
    """An x at which LIR-CMOP<i> takes the front point z: x1 and g1, g2 (g1
    alone for 13-14) read back from the objectives, each g spread evenly
    over its distance variables."""
    import math
    import numpy as np
    x = np.empty(30)
    if i >= 13:
        r = math.sqrt(float(np.sum(np.square(z))))
        x[0] = math.asin(min(z[2] / r, 1.0)) / (0.5 * math.pi)
        x[1] = math.atan2(z[1], z[0]) / (0.5 * math.pi)
        x[2:] = 0.5 + math.sqrt(max(r - 1.7057, 0.0) / 280.0)
        return x
    a, b = z
    if i <= 4:
        u, g1, g2 = a - 0.5, 0.5, 0.5
    elif i <= 8:
        u = min(a - L.SHIFT, 1.0)
        g1, g2 = (a - L.SHIFT - u) / 10.0, (b - L.SHIFT - float(L.shape(i, u))) / 10.0
    else:
        # u in [h^-1(f2 / 1.7057), 1): f2 reachable; and u <= f1 / 1.7057
        t = min(max(b / 1.7057, 0.0), 1.0)
        least = (1.0 - t) ** 2 if L._sqrt_shape(i) else math.sqrt(1.0 - t)
        u = min(a / 1.7057, 0.5 * (least + 1.0))
        h = float(L.shape(i, u))
        g1 = (a / (1.7057 * u) - 1.0) / 10.0 if u > 0 else 0.0
        g2 = (b / (1.7057 * h) - 1.0) / 10.0 if h > 0 else 0.0
    x[0] = u
    if i <= 4:
        x[L.J1], x[L.J2] = math.sin(0.5 * math.pi * u), math.cos(0.5 * math.pi * u)
    else:
        x[L.J1] = np.sin(0.5 * (L.J1 + 1) / 30.0 * math.pi * u)
        x[L.J2] = np.cos(0.5 * (L.J2 + 1) / 30.0 * math.pi * u)
    for J, g in ((L.J1, g1), (L.J2, g2)):
        d = math.sqrt(max(g, 0.0) / len(J))
        x[J] = np.where(x[J] + d <= 1.0, x[J] + d, x[J] - d)
    return x


@test
def lir_cmop_follows_its_table_and_its_fronts_hold():
    """Task 7, §8: LIR-CMOP1-14 (Fan et al. 2019, arXiv:1707.08767, Table 6)
    at hand-checked points; every front point satisfies the constraints and
    is reached by a feasible x (_lir_attain), and feasible points made on
    purpose do not beat the front; as the paper's text
    says, 5's and 6's fronts are the unconstrained ones and 7's and 8's lie
    on a constraint's boundary, all but their end at f2's floor."""
    if not _have_numpy():
        print("    (skipped: no NumPy)"); return
    import math
    import numpy as np
    from mootation.benchmarks import families, get
    from mootation.benchmarks import lir_cmop as L
    from mootation.benchmarks.properties import properties
    assert families()["LIR-CMOP"] == sorted(f"LIR-CMOP{i}" for i in range(1, 15))
    x = np.empty(30)
    x[0], x[L.J1], x[L.J2] = 0.3, math.sin(0.15 * math.pi), math.cos(0.15 * math.pi)
    f, c = L.evaluate(1, x)                                  # g1 = g2 = 0: outside the band
    assert np.allclose(f, [0.3, 0.91]) and np.allclose(c, [0.255, 0.255])
    y = np.full(30, 0.5)
    f, c = L.evaluate(13, y)                                 # the sphere of radius 1.7057
    assert np.allclose(f, 1.7057 * np.array([0.5, 0.5, math.sqrt(0.5)])) and max(c) < 0.0
    assert L.evaluate(14, y)[1][2] > 0.0                     # 2.56 < g < 3.0625: c3 fails
    rng = np.random.default_rng(5)
    for i in range(1, 15):
        p = get(f"LIR-CMOP{i}")
        Z = np.asarray(p.pareto_front(1000), float)
        assert len(Z) > 50 and np.allclose(Z.min(axis=0), p.ideal), i
        if i >= 5:
            assert L.constraints_c(i, Z).min() >= -1e-9, i
        for z in Z:                                          # every point reached
            x = _lir_attain(L, i, z)
            f, c = L.evaluate(i, x)
            assert 0.0 <= x.min() and x.max() <= 1.0, (i, z)
            assert np.allclose(f, z, rtol=0.0, atol=1e-9) and max(c) <= 1e-9, (i, z, f, c)
        # feasible points made on purpose: g1, g2 in the band (1-4), the
        # distance variables near their optimum (5-14)
        pts = []
        for _ in range(3000):
            x = np.empty(30)
            x[0], x[1] = rng.random(), rng.random()
            if i <= 4:
                s, t = math.sin(0.5 * math.pi * x[0]), math.cos(0.5 * math.pi * x[0])
                x[L.J1], x[L.J2] = s, t
                for J, v in ((L.J1, s), (L.J2, t)):
                    d = math.sqrt((0.5 + 0.01 * rng.random()) / 4.0)
                    x[J[:4]] = v + d if v + d <= 1.0 else v - d
            elif i <= 12:
                x[L.J1] = np.sin(0.5 * (L.J1 + 1) / 30.0 * math.pi * x[0])
                x[L.J2] = np.cos(0.5 * (L.J2 + 1) / 30.0 * math.pi * x[0])
                x[2:] += rng.normal(0.0, 0.15 * rng.random(), 28)
            else:
                x[2:] = 0.5 + rng.normal(0.0, 0.03 * rng.random(), 28)
            f, c = L.evaluate(i, np.clip(x, 0.0, 1.0))
            if max(c) <= 0.0:
                pts.append(f)
        P = np.asarray(pts)
        assert len(P) > 50, (i, len(P))
        span = Z.max(axis=0) - Z.min(axis=0)
        beaten = [np.any(np.all(P <= z - 1e-3 * span, axis=1)) for z in Z]
        assert not any(beaten), (i, sum(beaten))
    for i in (5, 6):
        Z = L.front(i, 1000)
        assert np.allclose(Z[:, 1], L._least_f2(i, Z[:, 0]), atol=1e-9), i
    for i in (7, 8):                                         # but the end at f2's floor
        Z = L.front(i, 1000)
        edge = np.abs(L.constraints_c(i, Z)).min(axis=1) < 1e-6
        floor = np.abs(Z[:, 1] - L._least_f2(i, Z[:, 0])) < 1e-9
        assert np.all(edge | floor) and floor.sum() == 1, i
    row = properties("LIR-CMOP3")
    assert row["front"] == "concave+disconnected" and not row["multimodal"]
    assert properties("LIR-CMOP13")["front_dim"] == 2



# ── task 8: every core writes back the point it read ────────────────────────

def _points_off(p, rows, eq_tol) -> list:
    """The rows of current_points() whose stored values are not the ones their
    x gives: objectives, constraint values in the vault's convention (an
    equality as |h| - eq_tol), the violation summed in order as the vault adds
    it, and failed exactly when the evaluation fails."""
    import math
    off = []
    for x, f, g, cv, failed in rows:
        try:
            rf = list(p.evaluate(x))
            rg = list(p.constraints(x)) if p.has_cons else []
            if p.n_eq:
                rg = rg[:-p.n_eq] + [abs(h) - eq_tol for h in p.equalities(x)]
            bad = not all(math.isfinite(v) for v in rf + rg)
        except Exception:                                # noqa: BLE001
            bad = True
        if bad or failed:
            if bad != failed:
                off.append(("failed", failed, bad))
            continue
        rcv = 0.0
        for v in rg:
            if v > 0.0:
                rcv += v
        if f != rf or g != rg or cv != rcv:
            off.append(("values", f[:2], rf[:2], cv, rcv))
    return off


@test
def every_core_keeps_the_values_its_points_have():
    """Task 8: after every step, every point of the population and of the
    vault's archive carries the objectives, constraint values and violation
    its x gives, and is failed exactly when its evaluation fails — on every
    core, on C1-DTLZ1_3D (inequalities), DOC3 (equalities) and
    DTLZ2_3D_fail30r (failures), 2 000 evaluations, seeds 1-3; and on DOC3
    under the linear tolerance schedule, at the tolerance of the moment. 18
    cores used to write their population back with every constraint at 0."""
    if not _have_numpy() or _core_with("Config") is None \
            or not hasattr(__import__("mootation")._core, "current_points"):
        print("    (skipped: no NumPy or no compiled core)"); return
    from mootation import _core, algorithms
    from mootation.benchmarks import get, solve
    cases = [(prob, {}) for prob in ("C1-DTLZ1_3D", "DOC3", "DTLZ2_3D_fail30r")]
    cases.append(("DOC3", {"eq_tol_schedule": "linear", "eq_tol0": 1.0}))
    wrong = []
    for alg in sorted(algorithms()):
        pop = {"pop_size": 100} if alg in ("moead_m2m", "sms_m2m") else {}   # K S = N
        for prob, extra in cases:
            p = get(prob)
            for seed in (1, 2, 3):
                seen = []

                def observe(gen, F, _p=p, _seen=seen):
                    pop_rows, arc_rows = _core.current_points()
                    _seen += _points_off(_p, pop_rows + arc_rows, _core.current_eq_tol())

                solve(prob, alg, max_evaluations=2000, seed=seed, on_generation=observe,
                      record_every=1, **pop, **extra)
                if seen:
                    wrong.append((alg, prob, extra, seed, len(seen), seen[0]))
    assert not wrong, wrong[:5]

# ── task 7, supplement 3, §2: failed evaluations ──────────────────────────────

def _failing_zdt1(x):
    """ZDT1 failing on about a third of its box: NaN on half of it, an exception
    on the other half (a hash of x0..x2)."""
    import math
    h = (math.sin(12.9898 * x[0] + 78.233 * x[1] + 37.719 * x[2]) * 43758.5453) % 1.0
    if h < 0.15:
        return [math.nan, 0.0]
    if h < 0.30:
        raise RuntimeError("the solver did not converge")
    g = 1 + 9 * sum(x[1:]) / (len(x) - 1)
    return [x[0], g * (1 - math.sqrt(x[0] / g))]


@test
def every_algorithm_survives_a_third_of_its_evaluations_failing():
    """Supplement 3, §2.5: every algorithm on a function that fails on a third
    of its box (NaN or an exception) to 2 000 evaluations: no error, an answer
    of finite values, the failures counted and inside the budget."""
    import math
    if _core_with("fail_raise") is None:
        print("  skip  every_algorithm_survives...: stale _core"); return
    import mootation
    bounds = [(0.0, 1.0)] * 10
    for alg in mootation.algorithms():
        r = mootation.minimize(_failing_zdt1, bounds, 2, algorithm=alg, pop_size=20,
                               max_evaluations=2000, seed=1)
        assert r.objectives and all(math.isfinite(v) for row in r.objectives for v in row), alg
        assert 0 < r.n_failed < r.evaluations and r.evaluations >= 2000, (alg, r.n_failed)


@test
def a_failure_raises_logs_and_stops_as_asked():
    """on_failure = raise lets the user's exception through and raises on NaN;
    max_failure_share stops a run; the evaluation log marks a failed row and
    keeps its error; a None row of batch and a constraint that raises fail
    too; a function that never fails runs as without the guard."""
    import math
    if _core_with("fail_raise") is None:
        print("  skip  a_failure_raises...: stale _core"); return
    import mootation
    from mootation.failures import FailureShareExceeded
    bounds = [(0.0, 1.0)] * 10
    for f, kind in ((_failing_zdt1, (RuntimeError,)),
                    (lambda x: [math.nan, 1.0], (RuntimeError,))):
        try:
            mootation.minimize(f, bounds, 2, pop_size=20, max_evaluations=400, seed=1,
                               on_failure="raise")
            raise AssertionError("on_failure = raise did not raise")
        except kind as e:
            assert "did not converge" in str(e) or "non-finite" in str(e), e
    try:
        mootation.minimize(_failing_zdt1, bounds, 2, pop_size=20, max_evaluations=2000,
                           seed=1, max_failure_share=0.1, failure_window=50)
        raise AssertionError("max_failure_share did not stop the run")
    except FailureShareExceeded as e:
        assert "of the last 50 evaluations failed" in str(e), e
    with tempfile.TemporaryDirectory() as td:
        log = Path(td) / "evals.csv"
        r = mootation.minimize(_failing_zdt1, bounds, 2, pop_size=20, max_evaluations=200,
                               seed=1, log_evaluations=str(log))
        rows = [ln.split(",") for ln in log.read_text("utf-8").splitlines()
                if ln and not ln.startswith("#")]
        assert rows[0][-3:] == ["cv", "failed", "error"], rows[0]
        failed = [row for row in rows[1:] if row[-2] == "1"]
        assert len(failed) == r.n_failed and any("did not converge" in row[-1] for row in failed)
    def batch(X):
        return [None if i % 4 == 0 else [x[0], 1.0 - x[0]] for i, x in enumerate(X)]
    r = mootation.minimize(lambda x: [x[0], 1.0 - x[0]], bounds, 2, pop_size=20,
                           max_evaluations=400, seed=1, batch=batch)
    assert r.n_failed > 0, r.n_failed

    def cons(x):
        if x[1] > 0.9:
            raise ValueError("the mesh did not build")
        return [x[0] - 0.8]
    r = mootation.minimize(lambda x: [x[0], 1.0 - x[0]], bounds, 2, pop_size=20,
                           max_evaluations=400, seed=1, constraints=cons, n_cons=1)
    assert r.n_failed > 0 and all(c <= 0.0 for c in r.cv), (r.n_failed, r.cv)
    # no failure: the guard changes nothing
    plain = mootation.minimize(lambda x: [x[0], 1.0 - x[0]], bounds, 2, pop_size=20,
                               max_evaluations=400, seed=3)
    bare = mootation.minimize(lambda x: [x[0], 1.0 - x[0]], bounds, 2, pop_size=20,
                              max_evaluations=400, seed=3, on_failure="raise")
    assert plain.objectives == bare.objectives and plain.n_failed == 0


@test
def the_failing_problems_fail_where_they_say():
    """benchmarks/failing.py: the 18 failing copies, their own family; NaN inside
    the region; random and boundary take about their share of the box,
    near_front about its share of the Pareto set; the front stays the base's
    where the region misses the Pareto set and is marked otherwise; a campaign
    records n_failed in final and fail_share in its trajectory."""
    if not _have_numpy():
        print("    (skipped: no NumPy)"); return
    import math
    import numpy as np
    from mootation.benchmarks import get, names
    from mootation.benchmarks import failing as FL
    from mootation.benchmarks.registry import FAILING
    from mootation.families import family
    fails = [n for n in names() if family(n) == "failing"]
    assert sorted(fails) == sorted(FAILING) and len(fails) == 18, fails
    rng = np.random.default_rng(4)
    for name in ("ZDT1_fail10r", "ZDT1_fail30b", "DTLZ2_3D_fail30r", "WFG4_3D_fail10b"):
        p = get(name)
        lo = np.array([u for u, _ in p.bounds]); hi = np.array([v for _, v in p.bounds])
        X = lo + (hi - lo) * rng.random((3000, p.n_vars))
        bad = [not all(math.isfinite(v) for v in p.evaluate(list(x))) for x in X]
        share = FAILING[name][1] / 100.0
        assert abs(np.mean(bad) - share) < 0.03, (name, np.mean(bad))
    assert get("ZDT1_fail10r").front_exact            # the balls miss ZDT1's Pareto set
    assert not get("ZDT1_fail30b").front_exact         # the band takes all of it
    assert not get("DTLZ2_3D_fail10n").front_exact
    n = get("DTLZ2_3D_fail10n")
    S = np.asarray(get("DTLZ2_3D").pareto_set(FL.SAMPLE))
    kept = n.pareto_set(1000)
    assert abs(1 - len(kept) / len(S) - 0.10) < 0.05, len(kept)
    if _core_with("fail_raise") is None:
        return
    from mootation.run import campaign as C
    from mootation.run.config import load
    text = "\n".join([
        '[run]', 'name = "fail"', '[problem]', 'kind = "builtin"', '[benchmarks]',
        'problems = ["ZDT1_fail30r"]', 'runs = 1', '[campaign]', 'budget_fe = 400',
        'metrics = ["igdp_norm"]', '[[algorithms]]', 'name = "nsga2"', 'pop = 20', 'gens = 0', ''])
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "c.toml"
        path.write_text(text, encoding="utf-8")
        cfg = load(path)
        assert validate(cfg) == [], validate(cfg)
        spec = C.campaign_spec(cfg)
        root = C.out_root(cfg, spec)
        job = C.expand_jobs(cfg, spec)[0]
        assert C.run_job(job, root, spec, quiet=True) == "done"
        d = root / "ZDT1_fail30r" / "nsga2" / "run_1"
        meta = json.loads((d / "meta.json").read_text("utf-8"))
        T = [json.loads(t) for t in (d / "trajectory.jsonl").read_text().splitlines()]
        assert meta["final"]["n_failed"] > 0 and math.isfinite(meta["final"]["igdp_norm"])
        assert all("fail_share" in t for t in T) and any(t["fail_share"] > 0 for t in T)


@test
def equalities_hold_within_their_tolerance_and_its_schedule():
    """Supplement 3, §3: minimize(eq_constraints=..., n_eq=...): every row's
    cv is max(|h| - eq_tol, 0) beside the inequalities' max(g, 0), so a point
    with |h| <= eq_tol is feasible; under eq_tol_schedule = "linear" the
    tolerance falls from eq_tol0 to eq_tol in proportion to the evaluations
    spent and the answer is taken at eq_tol; the misuses are refused."""
    import math
    if _core_with("eq_tol") is None:
        print("  skip  equalities_hold...: stale _core"); return
    import mootation
    from mootation import _core

    def f(x):
        return [x[0], 1.0 - math.sqrt(x[0]) + x[1] + x[2]]

    def h(x):
        return [x[1] - 0.25]

    def g(x):
        return [x[0] - 0.8]

    bounds = [(0.0, 1.0)] * 3
    r = mootation.minimize(f, bounds, 2, pop_size=20, n_gen=40, seed=1,
                           eq_constraints=h, n_eq=1, eq_tol=0.01)
    assert r.eq_tol == 0.01 and len(r.cv) == len(r.variables) > 0
    for x, cv in zip(r.variables, r.cv):
        assert cv == max(abs(x[1] - 0.25) - 0.01, 0.0), (x, cv)
    assert any(cv == 0.0 and abs(x[1] - 0.25) > 0.0 for x, cv in zip(r.variables, r.cv))
    r = mootation.minimize(f, bounds, 2, pop_size=20, n_gen=40, seed=1, constraints=g,
                           eq_constraints=h, n_eq=1, eq_tol=0.01)
    for x, cv in zip(r.variables, r.cv):
        assert cv == max(x[0] - 0.8, 0.0) + max(abs(x[1] - 0.25) - 0.01, 0.0), (x, cv)
    # the linear schedule, read by an observer after every step: the tolerance
    # of a step is the one set before it, from the evaluations spent so far
    seen = []
    r = mootation.minimize(f, bounds, 2, pop_size=20, max_evaluations=1000, seed=2,
                           eq_constraints=h, n_eq=1, eq_tol=1e-4,
                           eq_tol_schedule="linear", eq_tol0=0.5,
                           on_generation=lambda gen, F: seen.append((gen, _core.current_eq_tol())),
                           record_every=1)
    assert seen[0] == (0, 0.5), seen[0]
    for gen, tol in seen[1:]:
        spent = 20 * gen                         # setup and the steps before this one
        assert tol == 0.5 + (1e-4 - 0.5) * min(1.0, spent / 1000.0), (gen, tol)
    assert seen[-1][1] < 0.02 and r.eq_tol == 1e-4
    for x, cv in zip(r.variables, r.cv):
        assert cv == max(abs(x[1] - 0.25) - 1e-4, 0.0), (x, cv)
    for kw, what in ((dict(eq_constraints=h), "n_eq"),
                     (dict(eq_constraints=h, n_eq=1, eq_tol_schedule="linear", eq_tol0=0.5),
                      "max_evaluations"),
                     (dict(eq_constraints=h, n_eq=1, eq_tol_schedule="linear",
                           max_evaluations=500), "eq_tol0"),
                     (dict(n_eq=1), "eq_constraints"),
                     (dict(eq_constraints=h, n_eq=1, eq_tol_schedule="cosine"), "fixed or linear")):
        try:
            mootation.minimize(f, bounds, 2, pop_size=10, n_gen=2, **kw)
        except ValueError as e:
            assert what in str(e), (kw, e)
        else:
            raise AssertionError(f"{kw} was accepted")


@test
def doc_follows_its_supplement_and_its_fronts_hold():
    """Task 7, §8: DOC1-9 (Liu and Wang 2019, supplementary file S-I): the
    sizes of Table I (D, inequalities, equalities), hand-checked points at
    g = 1, the share of feasible random points of DOC1 and DOC4 (Table I:
    26.97 %, 0.53 %), the fronts inside the objective constraints with the
    pieces and points the paper states, and the equalities: |h| - 1e-4 in
    constraints, h in equalities, the same run either way."""
    if not _have_numpy():
        print("    (skipped: no NumPy)"); return
    import math
    import numpy as np
    from mootation.benchmarks import families, get
    from mootation.benchmarks import doc as D
    from mootation.benchmarks.properties import properties
    assert families()["DOC"] == [f"DOC{i}" for i in range(1, 10)]
    table = {1: (6, 7, 0), 2: (16, 7, 0), 3: (10, 6, 4), 4: (8, 6, 0), 5: (8, 4, 5),
             6: (11, 10, 0), 7: (11, 3, 3), 8: (10, 7, 0), 9: (11, 14, 0)}
    for i, (n, nic, nec) in table.items():
        lo = [b[0] for b in D.BOUNDS[i]]
        _, ineq, eq = D.evaluate(i, lo)
        p = get(f"DOC{i}")
        assert (p.n_vars, len(ineq), len(eq), p.n_eq) == (n, nic, nec, nec), i
        c = p.constraints(lo)
        assert c[:nic] == ineq and c[nic:] == [abs(h) - 1e-4 for h in eq], i
    # at g = 1 the objectives are the unconstrained front's
    x = [0.25, 193.724510070035, 20.0, 20.0, 200.0, 6.5, 6.0, 5.0]
    assert np.allclose(D.evaluate(5, x)[0], [0.25, 0.5], atol=1e-12)
    x = [0.3, 0.6, 600.0, 1400.0, 7048.2480205286 + 1.0 - 2000.0] + [300.0] * 5
    assert np.allclose(D.evaluate(8, x)[0], [0.18, 0.12, 0.7], atol=1e-9)
    x = [0.0, 0.5] + [0.0] * 9                                 # g = 1.8660254038
    assert np.allclose(D.evaluate(9, x)[0], [1.8660254038 * math.sqrt(0.5)] * 2 + [0.0])
    rng = np.random.default_rng(11)
    for i, lo_hi in ((1, (25.5, 28.5)), (4, (0.33, 0.75))):
        b = np.array(D.BOUNDS[i])
        X = b[:, 0] + (b[:, 1] - b[:, 0]) * rng.random((20000, len(b)))
        share = 100.0 * np.mean([max(D.constraints(i, x)) <= 0.0 for x in X])
        assert lo_hi[0] < share < lo_hi[1], (i, share)
    for i in range(1, 10):
        p = get(f"DOC{i}")
        Z = np.asarray(p.pareto_front(1000), float)
        assert np.allclose(Z.min(axis=0), p.ideal) and np.allclose(Z.max(axis=0), p.nadir), i
        assert max(max(D.objective_constraints(i, z)) for z in Z) <= 1e-12, i
    assert len(D.front(4)) == 21 and len(D.front(5)) == 14
    assert sorted(np.round(D.front(5)[:, 0] * 20).astype(int)) == list(range(9)) + list(range(16, 21))
    ends = [D._circle_ends(p) for p, _ in D.CIRCLES]          # the paper: 0.050, 0.2202, 0.3830 (sic),
    assert np.allclose(ends, [(0.0500, 0.2203), (0.3803, 0.6247), (0.7439, 1.0)], atol=1e-4)
    Z = D.front(3, 10 ** 6)
    Z = Z[np.argsort(Z[:, 0])]
    cut = np.flatnonzero(np.diff(Z[:, 0]) > 0.01)
    pieces = [Z[0, 0]] + [v for k in cut for v in (Z[k, 0], Z[k + 1, 0])] + [Z[-1, 0]]
    assert np.allclose(pieces, [0, 0.3403, 0.4782, 0.6553, 0.7553, 0.8782, 0.9403, 1], atol=1e-4)
    assert np.allclose(D.front(9)[:, 2], 0.0) and properties("DOC9")["front"].startswith("degenerate")
    assert properties("DOC9")["front_dim"] == 1 and properties("DOC5")["multimodal"]
    # the equalities as such or as |h| - 1e-4: the same run
    if _core_with("Config") is None:
        print("    (the runs skipped: no compiled core)"); return
    from mootation.benchmarks import solve
    from mootation import minimize
    p = get("DOC3")
    a = solve("DOC3", "nsga2", pop_size=20, max_evaluations=600, seed=3)
    b = minimize(p.evaluate, p.bounds, 2, pop_size=20, max_evaluations=600, seed=3,
                 constraints=p.constraints)
    assert a.objectives == b.objectives and a.cv == b.cv and a.eq_tol == 1e-4


@test
def categorical_variables_keep_to_their_labels_and_repeats_cost_nothing():
    """Supplement 3, §1: ("cat", labels) reaches fn as a label, never anything
    else, on every core that takes the integer genome, and the 17 others
    refuse it by name; the result's genes are label indices; with
    budget_counts = "unique" a decoded point is evaluated once however its
    real underlay differs (int_repair = "baldwin"), and the budget counts
    distinct points; unique_evaluations <= evaluations; the labels survive the
    evaluation log and a seed population."""
    import math
    if _core_with("gene_levels") is None:
        print("  skip  categorical_variables...: stale _core"); return
    import mootation
    from mootation.var_types import Layout
    labels = ["steel", "titanium", "cfrp", "glass"]
    pen = {"steel": 0.3, "titanium": 0.0, "cfrp": 0.6, "glass": 0.9}
    seen = set()

    def f(x):
        a, m, k = x
        seen.add(m)
        return [a + pen[m] + 0.05 * k, 1 - math.sqrt(a) + pen[m]]

    vt = ["real", ("cat", labels), "int"]
    bounds = [(0.0, 1.0), (0.0, 0.0), (0.0, 4.0)]          # a label's bounds are ignored
    refused = []
    for alg in mootation.algorithms():
        seen.clear()
        try:
            r = mootation.minimize(f, bounds, 2, algorithm=alg, pop_size=20,
                                   max_evaluations=400, seed=1, var_types=vt)
        except ValueError as e:
            assert "takes no binary or categorical variables" in str(e), (alg, e)
            refused.append(alg)
            continue
        assert seen <= set(labels) and len(seen) >= 2, (alg, seen)
        assert all(0 <= b[0] < 4 for b in r.binary_variables), alg
        assert 0 < r.unique_evaluations <= r.evaluations, alg
    assert len(refused) == 24, refused
    # the best label wins, as the decoded rows show
    r = mootation.minimize(f, bounds, 2, pop_size=20, max_evaluations=2000, seed=1, var_types=vt)
    lay = Layout(vt, bounds)
    rows = [lay.to_user(list(x) + list(b)) for x, b in zip(r.variables, r.binary_variables)]
    assert {row[1] for row in rows} == {"titanium"}, rows[:3]
    # a repeat is evaluated once: the real underlay of an integer differs
    # (baldwin), the decoded point does not
    calls = []

    def g(x):
        calls.append(tuple(x))
        return [x[0] + 0.1 * x[1], 1.0 - x[0] + 0.1 * (6 - x[1])]

    r = mootation.minimize(g, [(0.0, 1.0), (0.0, 6.0)], 2, pop_size=20, max_evaluations=600,
                           seed=2, var_types=[("discrete", [0.0, 0.5, 1.0]), "int"],
                           int_repair="baldwin", budget_counts="unique")
    assert len(calls) == len(set(calls)) == r.unique_evaluations, (len(calls), r.unique_evaluations)
    assert r.evaluations > r.unique_evaluations
    calls.clear()
    r = mootation.minimize(g, [(0.0, 1.0), (0.0, 6.0)], 2, pop_size=20, max_evaluations=200,
                           seed=2, var_types=[("discrete", [0.0, 0.5, 1.0]), "int"],
                           int_repair="baldwin")
    assert len(calls) == r.evaluations == 200 and r.unique_evaluations == len(set(calls)) < 200
    # real variables only: a fresh float each call, unique as many as calls (or fewer)
    r = mootation.minimize(lambda x: [x[0], 1 - x[0] + x[1]], [(0, 1), (0, 1)], 2,
                           pop_size=20, max_evaluations=400, seed=3)
    assert 0 < r.unique_evaluations <= r.evaluations
    # the labels in the evaluation log and back through a seed population
    with tempfile.TemporaryDirectory() as d:
        log = Path(d) / "log.csv"
        mootation.minimize(f, bounds, 2, pop_size=10, n_gen=3, seed=4, var_types=vt,
                           log_evaluations=log)
        text = log.read_text(encoding="utf-8")
        assert any(m in text for m in labels) and "'steel'" not in text
        seed_rows = [[0.5, "glass", 2], [0.1, "steel", 0]] * 5
        pop = mootation.minimize(f, bounds, 2, pop_size=10, n_gen=1, seed=4, var_types=vt,
                                 save_population=Path(d) / "pop.csv")
        r = mootation.minimize(f, bounds, 2, pop_size=10, n_gen=2, seed=5, var_types=vt,
                               seed_population=Path(d) / "pop.csv")
        assert all(0 <= b[0] < 4 for b in r.binary_variables) and pop.binary_variables
        assert Layout(vt, bounds).split(seed_rows[0]) == ([0.5, 2.0], [3])


@test
def mixed_integer_and_categorical_problems_keep_their_optima():
    """Supplement 3, §1.4: bbob-biobj-mixint (Tusar et al. 2019, §3.1.1): n/5
    variables of each arity 2, 4, 8, 16, the rest continuous; the levels
    equidistant, 8/(l+1) apart, one of them the optimum; each objective takes
    its optimal value at its mixed-integer optimum, the ideal; the nadir is
    the other's value there. catZDT1: ZDT1's front from "rastrigin" alone,
    the easy labels' best curves above it."""
    if not _have_numpy():
        print("    (skipped: no NumPy)"); return
    import math
    import numpy as np
    from mootation.benchmarks import families, get
    from mootation.benchmarks import bbob_biobj_mixint as M
    from mootation.benchmarks import categorical as C
    from mootation.benchmarks.properties import properties
    assert len(families()["bbobbiobjmixint"]) == 110 and families()["catZDT"] == ["catZDT1"]
    assert M.arities(10) == [2, 2, 4, 4, 8, 8, 16, 16]
    z = M.levels(0.3, 4)
    assert np.allclose(np.diff(z), 1.6) and np.isclose(z, 0.3).sum() == 1
    for D in (5, 10):
        for fnum in (1, 7, 30, 55):
            p = get(M.name(fnum, D))
            mi = p.evaluate
            assert p.var_types == ["int"] * (4 * D // 5) + ["real"] * (D // 5)
            a = mi.optimum(mi.alpha, mi.z_alpha)
            b = mi.optimum(mi.beta, mi.z_beta)
            assert math.isclose(mi(a)[0], p.ideal[0], abs_tol=1e-12)
            assert math.isclose(mi(b)[1], p.ideal[1], abs_tol=1e-12)
            assert p.nadir == (mi(b)[0], mi(a)[1])
            rng = np.random.default_rng(fnum)
            for _ in range(200):                      # nothing beats the ideal
                x = [int(rng.integers(0, l)) for l in mi.arities] + list(rng.uniform(-5, 5, D // 5))
                f = mi(x)
                assert f[0] >= p.ideal[0] - 1e-9 and f[1] >= p.ideal[1] - 1e-9
    assert properties("bbobbiobjmixint07_n10_2D")["bbob_groups"] == \
        properties("bbobbiobj07_n10_2D")["bbob_groups"]
    # catZDT1
    p = get("catZDT1")
    y_best = {"rastrigin": 0.5, "linear": 0.0, "quadratic": 0.75}
    for label, g0 in C.LEAST_G.items():
        assert math.isclose(C.g(label, [y_best[label]] * 9), g0, abs_tol=1e-12)
    f1 = np.linspace(0.0, 1.0, 101)
    curves = {c: np.array([p.evaluate([c, u] + [y_best[c]] * 9)[1] for u in f1])
              for c in C.LABELS}
    assert np.allclose(curves["rastrigin"], 1.0 - np.sqrt(f1), atol=1e-12)
    assert np.all(curves["linear"] > curves["rastrigin"])
    assert np.all(curves["quadratic"] > curves["linear"])
    assert np.allclose(p.pareto_front(101), np.column_stack([f1, 1.0 - np.sqrt(f1)]))
    assert p.evaluate([0, 0.3] + [0.5] * 9) == p.evaluate(["rastrigin", 0.3] + [0.5] * 9)
    row = properties("catZDT1")
    assert row["front"] == "convex" and row["multimodal"] and row["deceptive"]
    from mootation.var_types import Layout                    # the run archive's numbers
    assert Layout(C.var_types(), C.bounds()).numeric(["linear"] + [0.5] * 10)[0] == 1.0


@test
def eps_level_follows_fans_rule_and_is_feasibility_at_zero():
    """Supplement 3, §4.1: constraint_mode = "eps_level" with eps held at 0
    (eps_schedule tc = 0) is the feasibility-first run bit for bit, on every
    core; otherwise eps follows Fan et al.'s Eq. 8 once per generation — 0.9
    of itself while the population is mostly infeasible, 1.1 times the largest
    violation seen once 95 % of it is feasible, 0 from 80 % of the budget on;
    the answer's cv is the violation itself."""
    if _core_with("eps_level") is None:
        print("  skip  eps_level...: stale _core"); return
    import math
    import mootation
    from mootation import _core
    from mootation.benchmarks import get
    p = get("LIR-CMOP2")
    for alg in mootation.algorithms():
        pop = max(20, len(p.bounds) + 1) if alg.startswith("bc_") else 20   # bc: n + 1 parents
        a = mootation.minimize(p.evaluate, p.bounds, 2, algorithm=alg, pop_size=pop,
                               max_evaluations=600, seed=1, constraints=p.constraints)
        b = mootation.minimize(p.evaluate, p.bounds, 2, algorithm=alg, pop_size=pop,
                               max_evaluations=600, seed=1, constraints=p.constraints,
                               constraint_mode="eps_level", eps_schedule={"tc": 0.0})
        assert a.objectives == b.objectives and a.cv == b.cv, alg
    # the schedule on a mostly infeasible start: rule 2, then rule 4
    levels = []
    mootation.minimize(p.evaluate, p.bounds, 2, pop_size=50, max_evaluations=5000, seed=2,
                       constraints=p.constraints, constraint_mode="eps_level",
                       on_generation=lambda g, F: levels.append(_core.current_cv_level()),
                       record_every=1)
    assert levels[0] > 0.0 and len(levels) == 100
    for g in range(1, 10):
        assert levels[g] == (1.0 - 0.1) * levels[g - 1], g
    assert all(v == 0.0 for v in levels[80:]) and levels[79] > 0.0
    # rule 3 where 95 % are feasible: 1.1 times the largest violation so far
    seen = {"max": 0.0}
    hist = []

    def g(x):
        v = x[0] - 0.96
        seen["max"] = max(seen["max"], v)
        return [v]

    def f(x):
        return [x[0], 1.0 - math.sqrt(x[0]) + sum(x[1:])]

    r = mootation.minimize(f, [(0.0, 1.0)] * 4, 2, pop_size=100, max_evaluations=4000, seed=3,
                           constraints=g, constraint_mode="eps_level",
                           on_generation=lambda gen, F: hist.append((_core.current_cv_level(),
                                                                     seen["max"])),
                           record_every=1)
    rises = [k for k in range(1, len(hist)) if hist[k][0] > hist[k - 1][0]]
    assert rises, hist[:5]
    for k in rises:
        assert hist[k][0] == (1.0 + 0.1) * hist[k - 1][1], (k, hist[k - 1], hist[k])
    for x, cv in zip(r.variables, r.cv):
        assert cv == max(x[0] - 0.96, 0.0)
    for bad in (dict(constraint_mode="eps"), dict(constraint_mode="eps_level"),
                dict(eps_schedule={"tc": 0.5})):
        try:
            mootation.minimize(f, [(0.0, 1.0)] * 4, 2, pop_size=10, n_gen=2, **bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{bad} was accepted")


@test
def ctaea_runs_feasible_spends_n_per_generation_and_wants_a_lattice():
    """Supplement 3, §4.3: C-TAEA on C2-DTLZ2 (3 objectives) answers with
    feasible points near the front; a generation costs N evaluations (one
    child of each of N pairs), the start N; pop_size must be a Das-Dennis
    lattice size (91 at three objectives), and the run layer knows it."""
    if _core_with("eps_level") is None or "ctaea" not in __import__("mootation").algorithms():
        print("  skip  ctaea...: stale _core"); return
    import numpy as np
    import mootation
    from mootation.benchmarks import get
    from mootation.run import metrics as M
    from mootation.run.algorithms import EXACT_LATTICE
    p = get("C2-DTLZ2_3D")
    seen = []
    r = mootation.minimize(p.evaluate, p.bounds, 3, algorithm="ctaea", pop_size=91, n_gen=60,
                           seed=1, constraints=p.constraints,
                           on_generation=lambda g, F: seen.append(len(F)), record_every=1)
    assert r.evaluations == 91 * 61 and len(r.objectives) == 91, r.evaluations
    F = np.asarray([f for f, c in zip(r.objectives, r.cv) if c <= 0])
    assert len(F) == 91
    igdp = M.compute(F, ref_front=np.asarray(p.pareto_front(1000)), ideal=p.ideal, nadir=p.nadir,
                     which=["igdp"])["igdp"]
    assert igdp < 0.1, igdp
    assert "ctaea" in EXACT_LATTICE
    try:
        mootation.minimize(p.evaluate, p.bounds, 3, algorithm="ctaea", pop_size=90, n_gen=2,
                           seed=1, constraints=p.constraints)
    except (ValueError, RuntimeError) as e:
        assert "91" in str(e) or "lattice" in str(e).lower() or "das" in str(e).lower(), e
    else:
        raise AssertionError("ctaea took pop_size = 90 at three objectives")


@test
def ccmo_spends_n_per_generation_and_finds_mw1s_feasible_front():
    """Supplement 3, §4.2: CCMO costs 2N at the start and N a generation (N/2
    children from each population); on a problem without constraints both
    populations solve the same problem and it runs as any core; on MW1 the
    answer is feasible and near the front."""
    if _core_with("eps_level") is None or "ccmo" not in __import__("mootation").algorithms():
        print("  skip  ccmo...: stale _core"); return
    import math
    import numpy as np
    import mootation
    from mootation.benchmarks import get
    from mootation.run import metrics as M
    r = mootation.minimize(lambda x: [x[0], 1.0 - math.sqrt(x[0]) + sum(x[1:])],
                           [(0.0, 1.0)] * 5, 2, algorithm="ccmo", pop_size=40, n_gen=25, seed=1)
    assert r.evaluations == 2 * 40 + 25 * 40 and len(r.objectives) == 40
    assert min(f[1] + math.sqrt(f[0]) for f in r.objectives) < 1.2
    p = get("MW1")
    r = mootation.minimize(p.evaluate, p.bounds, 2, algorithm="ccmo", pop_size=100,
                           max_evaluations=20000, seed=2, constraints=p.constraints)
    assert r.evaluations == 20000
    F = np.asarray([f for f, c in zip(r.objectives, r.cv) if c <= 0])
    assert len(F) == 100
    igdp = M.compute(F, ref_front=np.asarray(p.pareto_front(1000)), ideal=p.ideal,
                     nadir=p.nadir, which=["igdp"])["igdp"]
    assert igdp < 0.01, igdp


@test
def i_cmop_puts_every_feasible_set_before_every_infeasible_one():
    """Supplement 3, §4.4: Vodopija et al.'s I_CMOP (Eq. 12): minus the
    normalised hypervolume with the reference point 1 when a feasible point is
    inside [ideal, nadir], the distance to that box while none is, capped at
    1; 1 plus the least violation without a feasible point; so any feasible
    set scores below any infeasible one (their Eq. 11); in compute on the
    whole answer, infeasible points included."""
    if not _have_numpy():
        print("    (skipped: no NumPy)"); return
    import numpy as np
    from mootation.run import metrics as M
    ideal, nadir = (0.0, 0.0), (1.0, 1.0)
    F = np.array([[0.5, 0.5], [0.2, 2.0]])
    assert M.i_cmop(F, [0.0, 0.0], ideal, nadir) == -0.25
    assert M.i_cmop(np.array([[1.5, 0.5]]), [0.0], ideal, nadir) == 0.5
    assert M.i_cmop(np.array([[3.0, 3.0]]), [0.0], ideal, nadir) == 1.0
    assert M.i_cmop(F, [0.3, 0.7], ideal, nadir) == 1.3
    assert M.i_cmop(F, [0.0, 0.7], ideal, nadir) == M.i_cmop(F[:1], [0.0], ideal, nadir)
    rng = np.random.default_rng(1)
    feas = [M.i_cmop(rng.uniform(0, 5, (8, 2)), np.zeros(8), ideal, nadir) for _ in range(50)]
    infeas = [M.i_cmop(rng.uniform(0, 5, (8, 2)), rng.uniform(1e-9, 1, 8), ideal, nadir)
              for _ in range(50)]
    assert max(feas) <= 1.0 < min(infeas)
    from mootation.benchmarks import get
    p = get("MW1")
    X = [[0.5] * 15, [0.1] * 15]
    got = M.compute([p.evaluate(x) for x in X], ideal=p.ideal, nadir=p.nadir,
                    which=["i_cmop", "hv"], X=X, constraints=p.constraints)
    v = M.violations(X, p.constraints)
    assert got["hv"] is None and got["i_cmop"] == min(v) + 1.0


@test
def multimodal_problems_have_every_part_of_their_pareto_sets():
    """Supplement 3, §5.4-5.5: the CEC 2019 problems (benchmarks/mmo.py) at the
    report's own points, their Pareto-set samples on the front's equation to
    1e-9 and inside the box, every part present (as many separate pieces as
    mmo.parts says for SYM-PART, Omni-test, MMF1, MMF4, MMF8, MMF9 and MMF14),
    and the property table's multimodal_ps and ps_parts."""
    if not _have_numpy():
        print("    (skipped: no NumPy)"); return
    import math
    import numpy as np
    from mootation.benchmarks import families, get, mmo
    from mootation.benchmarks.properties import properties
    assert len(mmo.NAMES) == 22
    assert sorted(families()["MMF"]) == sorted(n for n in mmo.NAMES if n.startswith("MMF"))
    # by hand: SYM-PART's middle and corner parts, Omni-test's corner, MMF1's middle
    assert get("SYM-PART_simple").evaluate([0.0, 0.0]) == [1.0, 1.0]
    assert get("SYM-PART_simple").evaluate([10.0, 10.0]) == [1.0, 1.0]
    f = get("SYM-PART_rotated").evaluate([10 / math.sqrt(2), -10 / math.sqrt(2)])
    assert abs(f[0] - 1.0) < 1e-12 and abs(f[1] - 1.0) < 1e-12, f   # r = (10, 0)
    f = get("Omni-test").evaluate([1.25, 3.25, 5.25])
    assert abs(f[0] + 3 / math.sqrt(2)) < 1e-12 and abs(f[1] + 3 / math.sqrt(2)) < 1e-12, f
    assert get("MMF1").evaluate([2.0, 0.0]) == [0.0, 1.0 + 2 * math.sin(math.pi) ** 2]
    assert get("MMF14").n_obj == 3 and get("MMF14").pop_size == 300 and get("MMF1").n_gen == 50
    for name in mmo.NAMES:
        p = get(name)
        X = np.asarray(p.pareto_set(400), float)
        lo, hi = np.array(p.bounds, float).T
        assert len(X) >= 100 and np.all((X >= lo) & (X <= hi)), name
        F = np.array([p.evaluate(list(x)) for x in X])
        assert mmo.front_gap(name, F).max() < 1e-9, (name, mmo.front_gap(name, F).max())
        Z = np.asarray(p.pareto_front(500), float)
        assert mmo.front_gap(name, Z).max() < 1e-9, name
        assert np.allclose(Z.min(0), p.ideal, atol=1e-9) and np.allclose(Z.max(0), p.nadir,
                                                                         atol=1e-9), name
        row = properties(name)
        assert row["multimodal_ps"] is True and row["ps_parts"] == mmo.parts(name), name
    # the parts apart: single linkage at a tenth of the box's diagonal
    for name in ("SYM-PART_simple", "SYM-PART_rotated", "Omni-test", "MMF1", "MMF4", "MMF8",
                 "MMF9", "MMF14"):
        p = get(name)
        X = np.asarray(p.pareto_set(400), float)
        lo, hi = np.array(p.bounds, float).T
        U = (X - lo) / (hi - lo)
        label = list(range(len(U)))
        def root(i):
            while label[i] != i:
                label[i] = label[label[i]]
                i = label[i]
            return i
        D = np.sqrt(((U[:, None, :] - U[None, :, :]) ** 2).sum(-1))
        for i, j in zip(*np.nonzero(D < 0.1 * math.sqrt(U.shape[1]) / 2)):
            label[root(i)] = root(j)
        pieces = len({root(i) for i in range(len(U))})
        # MMF1's two parts meet at x1 = 2, MMF4's and MMF8's pairs at x1 = 0
        want = {"MMF1": 1, "MMF4": 2, "MMF8": 2}.get(name, mmo.parts(name))
        assert pieces == want, (name, pieces, want)
    assert properties("IPolygon_3D")["ps_parts"] == 4 and properties("ZDT1")["multimodal_ps"] is None


@test
def psp_sees_the_lost_parts_of_a_pareto_set():
    """Supplement 3, §5.3 and 5.5: on a Pareto-set sample IGDX = 0, CR = 1 and
    PSP = CR / IGDX is infinite; on ONE part of it (SYM-PART's middle one,
    MMF1's left one, MMF14's lower one) CR < 1 and IGDX > 0, while IGD+ is 0
    to the front's sampling — the decision-space indicators see what the
    objectives cannot."""
    if not _have_numpy():
        print("    (skipped: no NumPy)"); return
    import numpy as np
    from mootation.benchmarks import get
    from mootation.run import metrics as M
    from mootation.run.metric_names import HIGHER_IS_BETTER, NEEDS_SET, SET_METRICS
    assert "psp" in NEEDS_SET and "psp" in HIGHER_IS_BETTER and SET_METRICS == ("igdx", "cr", "psp")
    for name, part in (("SYM-PART_simple", lambda X: (np.abs(X[:, 0]) <= 1) & (X[:, 1] == 0)),
                       ("MMF1", lambda X: X[:, 0] <= 2.0),
                       ("MMF14", lambda X: X[:, 2] < 0.5)):
        p = get(name)
        S = np.asarray(p.pareto_set(600), float)
        kw = dict(ref_front=np.asarray(p.pareto_front(2000), float), ideal=p.ideal,
                  nadir=p.nadir, pareto_set=S, bounds=p.bounds,
                  which=("igdx", "cr", "psp", "igdp_norm"))
        whole = M.compute(np.array([p.evaluate(list(x)) for x in S]), X=S, **kw)
        assert whole["igdx"] == 0.0 and whole["cr"] == 1.0 and whole["psp"] == float("inf"), whole
        X = S[part(S)]
        assert 0 < len(X) < len(S), name
        one = M.compute(np.array([p.evaluate(list(x)) for x in X]), X=X, **kw)
        assert one["cr"] < 1.0 and one["igdx"] > 0.05, (name, one)
        assert abs(one["psp"] - one["cr"] / one["igdx"]) < 1e-12, one
        assert one["igdp_norm"] < 0.02, (name, one)


@test
def a_campaign_records_igdx_cr_and_psp_where_there_is_a_pareto_set():
    """[campaign] set_metrics (on by default): igdx, cr and psp along the
    trajectory and in `final` for a problem with a Pareto-set sample, beside
    `metrics`; none for one without (RE21); off with set_metrics = false. The
    CEC 2019 ranks read them."""
    if not _have_numpy() or _core_with("max_evaluations") is None:
        print("  skip  a_campaign_records_igdx...: no NumPy or stale _core"); return
    from mootation.run import campaign as C
    from mootation.run import report as R
    from mootation.run.config import load
    text = """algorithms = [ { name = "nsga2", pop = 0, gens = 0 } ]
[run]
name = "mm"
[problem]
kind = "builtin"
[benchmarks]
runs = 1
problems = ["MMF1", "RE21"]
[campaign]
out = "res"
budget_fe = 1000
metrics = ["igd"]
"""
    for on in (True, False):
        with tempfile.TemporaryDirectory() as td:
            cfg_path = Path(td) / "c.toml"
            cfg_path.write_text(text + ("" if on else "set_metrics = false\n"), encoding="utf-8")
            cfg = load(cfg_path)
            assert validate(cfg) == [], validate(cfg)
            spec = C.campaign_spec(cfg)
            assert spec.set_metrics is on
            root = C.out_root(cfg, spec)
            for job in C.expand_jobs(cfg, spec):
                assert C.run_job(job, root, spec, quiet=True) == "done", job
                d = root / job.rel_dir
                meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
                T = [json.loads(l) for l in (d / "trajectory.jsonl").read_text().splitlines()]
                has = on and job.problem == "MMF1"
                assert all(("psp" in t) is has and ("cr" in t) is has for t in T), (job.problem, T[-1])
                assert ("psp" in meta["final"]) is has and (("igdx" in meta["metrics"]) is has)
                if has:
                    f = meta["final"]
                    assert 0 < f["cr"] <= 1 and f["igdx"] > 0 and abs(f["psp"] - f["cr"] / f["igdx"]) < 1e-9
            if on:
                rows = C.scan_results(root)
                table = R.cec2019_ranks(rows)
                assert table["nsga2"]["igdx"] == (1.0, 1) and table["nsga2"]["igd"] == (1.0, 2)
                assert table["nsga2"]["mean"] == 1.0, table
                assert "1/psp" in R.format_cec2019_ranks(table)
    raises(ConfigError, C.campaign_spec, loads(text + "set_metrics = 1\n"))


@test
def cec2019_ranks_average_igd_igdx_and_one_over_psp():
    """report.cec2019_ranks: per problem the medians ranked, 1/PSP as PSP
    reversed (higher PSP better), and the mean of the three mean ranks."""
    from mootation.run import report as R
    def row(prob, alg, igd, igdx, psp):
        return {"status": "done", "problem": prob, "algorithm": alg,
                "final": {"igd": igd, "igdx": igdx, "psp": psp}}
    rows = [row("P", "a", 1.0, 0.1, 5.0), row("P", "b", 2.0, 0.05, 9.0),
            row("Q", "a", 1.0, 0.1, 5.0), row("Q", "b", 0.5, 0.2, 1.0)]
    t = R.cec2019_ranks(rows)
    assert t["a"]["igd"] == (1.5, 2) and t["b"]["igd"] == (1.5, 2)
    assert t["a"]["igdx"] == (1.5, 2) and t["a"]["psp"] == (1.5, 2)
    assert t["b"]["psp"] == (1.5, 2) and t["a"]["mean"] == 1.5
    rows.append(row("P", "c", 0.1, 0.01, 99.0))
    t = R.cec2019_ranks(rows)
    assert t["c"]["igd"] == (1.0, 1) and t["c"]["psp"] == (1.0, 1) and t["c"]["mean"] == 1.0


@test
def dn_nsga2_keeps_more_of_sym_parts_parts_than_nsga2():
    """Supplement 3, §5.1 and 5.5: DN-NSGA-II (algorithms/dn_nsga2.hpp) on
    SYM-PART simple at the competition's budget keeps more of the nine parts
    than NSGA-II, by the cover rate over 10 seeds, and on every seed at least
    as many parts within 0.1 of a part; dn_cf reaches it and only it, and the
    integer genome is refused."""
    if (not _have_numpy() or _core_with("Config") is None
            or "dn_nsga2" not in __import__("mootation").algorithms()):
        print("  skip  dn_nsga2...: no NumPy or stale _core"); return
    import numpy as np
    import mootation
    from mootation.benchmarks import get
    from mootation.run import metrics as M
    p = get("SYM-PART_simple")
    S = np.asarray(p.pareto_set(900), float)
    centres = [(t1 * 10.0, t2 * 10.0) for t1 in (-1, 0, 1) for t2 in (-1, 0, 1)]
    def parts(X):
        return sum(any(abs(x[0] - a) <= 1.1 and abs(x[1] - b) <= 0.1 for x in X)
                   for a, b in centres)
    cr = {"nsga2": [], "dn_nsga2": []}
    kept = {"nsga2": [], "dn_nsga2": []}
    for seed in range(1, 11):
        for alg in cr:
            r = mootation.minimize(p.evaluate, p.bounds, 2, algorithm=alg, pop_size=p.pop_size,
                                   n_gen=p.n_gen, seed=seed)
            X = np.asarray(r.variables, float)
            cr[alg].append(M.cover_rate(X, S))
            kept[alg].append(parts(X))
    print(f"    SYM-PART simple, 10 seeds: CR nsga2 {np.mean(cr['nsga2']):.3f}, dn_nsga2 "
          f"{np.mean(cr['dn_nsga2']):.3f}; parts kept {kept['nsga2']} vs {kept['dn_nsga2']}")
    assert np.mean(cr["dn_nsga2"]) > np.mean(cr["nsga2"]), cr
    assert sum(kept["dn_nsga2"]) > sum(kept["nsga2"]), kept
    f = lambda x: [x[0], 1.0 - x[0] + x[1]]
    r = mootation.minimize(f, [(0.0, 1.0)] * 2, 2, algorithm="dn_nsga2", pop_size=20, n_gen=3,
                           seed=1, dn_cf=5)
    assert "dn_cf" not in r.ignored and r.evaluations == 80
    r = mootation.minimize(f, [(0.0, 1.0)] * 2, 2, algorithm="nsga2", pop_size=20, n_gen=3,
                           seed=1, dn_cf=5)
    assert "dn_cf" in r.ignored
    raises(ValueError, mootation.minimize, f, [(0.0, 1.0)] * 2, 2, algorithm="dn_nsga2",
           pop_size=20, n_gen=3, seed=1, dn_cf=0.5)


@test
def rwmop_problems_as_the_paper_prints_them():
    """Task 7, §8: RWMOP's RCM01-18, 21-23, 25, 26, 28, 29 (benchmarks/rwmop.py). Where the
    problem is one of Tanabe and Ishibuchi's (re.py: RE31-33, 35, 36, 41), its
    objectives agree with theirs to 1e-12 and its constraints too (RCM08's in
    the paper's order); values by hand; the readings the header lists (RCM06's
    g11, RCM08's g8, RCM15's list); RCM20 as printed has no feasible point;
    the reference fronts are non-dominated and frame the problems."""
    if not _have_numpy():
        print("    (skipped: no NumPy)"); return
    import math
    import numpy as np
    from mootation.benchmarks import families, get
    from mootation.benchmarks import re as RE
    from mootation.benchmarks import rwmop as RW
    from mootation.run.metrics import nondominated_mask
    names = sorted(families()["RCM"])
    assert names == [f"RCM{i:02d}" for i in list(range(1, 19)) + [21, 22, 23, 25, 26, 28, 29]], names
    rng = np.random.default_rng(5)
    for i, model, order in ((3, RE._truss, None), (4, RE._weld, None), (5, RE._brake, None),
                            (6, RE._reducer, None), (7, RE._gear, None),
                            (8, RE._car, [0, 1, 3, 2, 5, 4, 6, 7, 8, 9])):
        lo, hi = np.array(RW.PROBLEMS[i][1], float).T
        for x in lo + (hi - lo) * rng.random((200, len(lo))):
            f, g = RW.evaluate(i, x)
            rf, rv = model(x)
            assert np.allclose(f, rf, rtol=1e-12, atol=0), (i, f, rf)
            v = np.maximum(np.array(g), 0.0)[order or slice(None)]
            assert np.allclose(v, rv, rtol=1e-12, atol=1e-9), (i, v, rv)
    # by hand: RCM09 at its lower bounds, RCM16 at (0.05, 0.2)
    f, g = RW.evaluate(9, [1.0, math.sqrt(2), math.sqrt(2), 1.0])
    assert abs(f[0] - 1400.0) < 1e-9 and abs(f[1] - 0.04) < 1e-15 and g == []
    f, g = RW.evaluate(16, [0.05, 0.2])
    assert abs(f[0] - 0.25 * 7800 * math.pi * 0.2 * 0.05 ** 2) < 1e-12
    assert abs(f[1] - 64 * 0.2 ** 3 / (3 * 307e8 * math.pi * 0.05 ** 4)) < 1e-18
    # the readings: RCM06's g11 against 1100 (as printed, 110, nothing is feasible:
    # its left side is at least 754 in the box); RCM08's g8 bounds f2 (f1 >= 15.58)
    lo, hi = np.array(RW.PROBLEMS[6][1], float).T
    X = lo + (hi - lo) * rng.random((2000, 7))
    lhs = [RW.evaluate(6, x)[1][10] + 1100.0 for x in X]
    assert min(lhs) > 754 and min(RW.evaluate(8, x)[0][0] for x in
                                  np.array(RW.PROBLEMS[8][1]).T[0][None]) > 15.5
    assert RW.WIRE == tuple(sorted(RW.WIRE)) and len(RW.WIRE) == 42
    raises(ValueError, RW.rcm02, [0.05, 0.5, 0.4, 0.4, 4.0])         # the layers out of order
    # RCM20 as printed: no feasible point (the paper's Table 5 has its algorithms
    # feasible, so what ran was not what is printed; not registered)
    best = math.inf
    for x in np.column_stack([rng.uniform(1, 16, 20000), rng.uniform(1, 16, 20000),
                              rng.uniform(1e-6, 16e-6, 20000), rng.uniform(1, 16, 20000)]):
        try:
            best = min(best, sum(max(c, 0.0) for c in RW.rcm20(x)[1]))
        except (ValueError, ZeroDivisionError):
            pass
    assert best > 100, best
    for key in names:
        p = get(key)
        Z = RW.front(key)                     # the file; pareto_front(n) thins it to n
        want = {"RCM02": 1658, "RCM07": 17, "RCM15": 64, "RCM22": 472, "RCM23": 524,
                "RCM26": 687, "RCM28": 1548, "RCM29": 2}.get(key, 2000)  # the header's sizes
        assert len(Z) == want and nondominated_mask(Z).all(), key
        assert p.front_exact is (key == "RCM28"), key                 # its curves are exact
        assert np.allclose(Z.min(0), p.ideal) and np.allclose(Z.max(0), p.nadir), key
        assert p.n_gen * p.pop_size >= 0.99 * RW.max_evaluations(int(key[3:])), key
    assert get("RCM15").var_types[2][1] == list(RW.WIRE) and get("RCM07").var_types == ["int"] * 4
    assert get("RCM09").has_cons is False and get("RCM08").has_cons is True
    # the chemical and process problems: the equalities as DOC's, |h| - 1e-4 last
    assert [get(f"RCM{i}").n_eq for i in (22, 23, 25, 26, 28, 29)] == [4, 4, 0, 1, 5, 0]
    x1 = 10 / (0.9 * (1 - math.exp(-2.5)))                 # RCM28: reactor 1, x5 = 5
    f, g = RW.evaluate(28, [x1, 0, 10, 0, 5.0, 0, 1, 0])
    assert max(g) <= 0 and RW.equalities(28, [x1, 0, 10, 0, 5.0, 0, 1, 0]) == [0.0] * 5
    Z = RW.front("RCM28")
    assert np.min(np.abs(Z - np.array(f)).sum(1)) < 0.05      # on the exact front's curve
    # as printed, exp(+0.5 x5): a reactor's output would be <= 0, h4 never holds
    assert 0.9 * (1 - math.exp(0.5 * 5.0)) * x1 < 0
    # RCM24 and RCM27: one-point fronts, not registered
    f, g = RW.evaluate(24, [0.0, 1e4 / 600, 0.0, 50.0, 2e6, 600.0, 100.0, 600.0, 700.0])
    assert max(g) <= 0 and f[1] == 0.0 and abs(f[0] - 35 * (50 / 3) ** 0.6) < 1e-9
    x1 = 0.2 + math.log(2.1)
    f, g = RW.evaluate(27, [x1, -2.1, 1])
    assert max(g) <= 1e-12 and abs(f[0] - 1.0765430833) < 1e-9
    assert "RCM24" not in names and "RCM27" not in names


@test
def budget_check_tells_a_schedule_from_none():
    """budget.check: RVEA's APD reads t/t_max, NSGA-II reads nothing of the budget."""
    if not _have_numpy() or _core_with("on_generation") is None:
        print("  skip  budget_check...: no NumPy or stale _core"); return
    from mootation.run.budget import check
    res = {r["algorithm"]: r for r in check(["nsga2", "rvea", "random_search"],
                                            problem="ZDT1", short=600, long=1200)}
    assert not res["nsga2"]["dependent"] and res["nsga2"]["compared"] >= 6, res["nsga2"]
    assert res["rvea"]["dependent"] and res["rvea"]["by_headers"], res["rvea"]
    assert not res["random_search"]["dependent"], res["random_search"]


@test
def small_budgets_read_hard_and_start_from_a_design():
    """Task 5, run B1: the last record at or below a hard budget and the archive
    after exactly that many evaluations, reduced to answer_k; a Sobol start and
    a plan of half the budget before NSGA-II, the plan a budget-dependent run."""
    if not _have_numpy() or not _have_scipy() or _core_with("on_generation") is None:
        print("  skip  small_budgets...: no NumPy, no SciPy (the Sobol start) or stale _core")
        return
    from mootation import designs
    from mootation.run import campaign as C
    from mootation.run.config import ConfigError, load, loads
    F = [[0, 3], [1, 1], [3, 0], [2, 2], [4, 4], [1, 3.5]]
    assert list(designs.nondominated(F)) == [0, 1, 2]
    assert sorted(designs.start_points(F, 4)[:3]) == [0, 1, 2] and len(designs.start_points(F, 4)) == 4
    assert set(designs.start_points(F, 2)) <= {0, 1, 2}
    text = "\n".join([
        '[run]', 'name = "small"', '[problem]', 'kind = "builtin"', '[benchmarks]',
        'problems = ["ZDT1"]', 'runs = 1', '[campaign]', 'budget_fe = 300',
        'record_grid = "generations"', 'metrics = ["igdp_norm", "gdp_norm", "hv_h"]',
        'hard_budgets = [100, 300]', 'answer_k = 10', 'ladder = [100]',
        '[[algorithms]]', 'name = "nsga2"', 'pop = 20', 'gens = 0',
        '[[algorithms]]', 'name = "nsga2"', 'label = "nsga2_sobol"', 'pop = 20', 'gens = 0',
        'init = "sobol"',
        '[[algorithms]]', 'name = "nsga2"', 'label = "nsga2_plan"', 'pop = 20', 'gens = 0',
        'init = "sobol"', 'init_share = 0.5',
        '[[algorithms]]', 'name = "nsga2"', 'label = "nsga2_lhs"', 'pop = 20', 'gens = 0',
        'init = "lhs"', ''])
    with tempfile.TemporaryDirectory() as td:
        cfg_path = Path(td) / "c.toml"
        cfg_path.write_text(text, encoding="utf-8")
        cfg = load(cfg_path)
        assert validate(cfg) == [], validate(cfg)
        spec = C.campaign_spec(cfg)
        root = C.out_root(cfg, spec)
        jobs = C.expand_jobs(cfg, spec)
        assert [j.key for j in jobs] == ["nsga2", "nsga2_sobol", "nsga2_plan", "nsga2_lhs",
                                         "nsga2_plan@100"]
        for job in jobs:
            assert C.run_job(job, root, spec, quiet=True) == "done", job

        def run(label):
            d = root / "ZDT1" / label / "run_1"
            return (json.loads((d / "meta.json").read_text("utf-8")),
                    [json.loads(t) for t in (d / "trajectory.jsonl").read_text().splitlines()])
        m, T = run("nsga2")
        at = m["hard_at"]
        assert at["100"]["population"]["fe"] == 100 and at["100"]["population"]["gen"] == 4, at
        rec = next(t for t in T if t["fe"] == 100)
        assert at["100"]["population"]["igdp_norm"] == rec["igdp_norm"]
        assert at["100"]["archive"]["fe"] == 100 and 0 < at["100"]["archive"]["n"] <= 10, at
        assert at["300"]["population"]["fe"] == 300 and m["answer_k"] == 10
        m, T = run("nsga2_sobol")
        assert m["init"] == "sobol" and T[0]["fe"] == 20 and T[0]["n"] == 20 and m["fe"] == 300, m
        m, T = run("nsga2_lhs")
        assert m["init"] == "lhs" and T[0]["fe"] == 20 and m["fe"] == 300, m
        # the plan: 150 points, then generations of 20 past 300 (310); read at 290
        m, T = run("nsga2_plan")
        assert m["budget_dependent"] and T[0]["fe"] == 150 and m["fe"] == 310, (m, T[0])
        assert m["hard_at"]["300"]["population"]["fe"] == 290, m["hard_at"]
        assert m["hard_at"]["300"]["archive"]["fe"] == 300, m["hard_at"]
        assert m["hard_at"]["100"]["population"] is None       # still planning at 100
        m, T = run("nsga2_plan@100")
        assert m["ladder_of"] == "nsga2_plan" and T[0]["fe"] == 50, (m, T[0])
        assert m["hard_at"]["100"]["population"]["fe"] == 90, m["hard_at"]
    bad = text.replace('init_share = 0.5', 'init_share = 0.5\n[[algorithms]]\nname = '
                       '"random_search"\npop = 20\ngens = 0\ninit = "sobol"')
    assert any("baseline" in e for e in validate(loads(bad))), validate(loads(bad))
    try:
        C.campaign_spec(loads(text.replace('record_grid = "generations"', 'record_grid = "log"')))
        raise AssertionError("hard_budgets on the log grid")
    except ConfigError:
        pass


@test
def a_ladder_campaign_runs_marks_and_covers():
    """Rungs run as jobs, meta.json says budget_dependent and ladder_of, and
    --cover reads rungs for rvea and trajectory records for nsga2."""
    if not _have_numpy() or _core_with("on_generation") is None:
        print("  skip  a_ladder_campaign...: no NumPy or stale _core"); return
    from mootation.run import campaign as C
    from mootation.run import cover as V
    from mootation.run.config import load
    text = (_LADDER.replace('problems = ["DTLZ2_3D"]', 'problems = ["ZDT1"]')
            .replace("budget_fe = 25000", 'budget_fe = 1200' + chr(10)
                     + 'metrics = ["igdp_norm", "gdp_norm", "hv_h"]')
            .replace("record_at = [10000, 2500, 5000]", "record_at = [600]")
            .replace("ladder = [2500, 10000, 30000]", "ladder = [600]"))
    with tempfile.TemporaryDirectory() as td:
        cfg_path = Path(td) / "c.toml"
        cfg_path.write_text(text, encoding="utf-8")
        cfg = load(cfg_path)
        assert validate(cfg) == [], validate(cfg)
        spec = C.campaign_spec(cfg)
        root = C.out_root(cfg, spec)
        jobs = C.expand_jobs(cfg, spec)
        assert [j.key for j in jobs[4:]] == ["rvea@600", "rvea@600"]
        for job in jobs:
            assert C.run_job(job, root, spec, quiet=True) == "done", job
        m = json.loads((root / "ZDT1" / "rvea@600" / "run_1" / "meta.json").read_text("utf-8"))
        assert m["budget_dependent"] and m["ladder_of"] == "rvea" and m["budget_fe"] == 600, m
        m = json.loads((root / "ZDT1" / "nsga2" / "run_1" / "meta.json").read_text("utf-8"))
        assert m["budget_dependent"] is False and m["ladder_of"] is None and m["record_at"] == [600]
        T = [json.loads(l) for l in (root / "ZDT1" / "nsga2" / "run_1" / "trajectory.jsonl")
             .read_text().splitlines()]
        assert any(600 <= t["fe"] < 700 for t in T), [t["fe"] for t in T]   # the rung's record
        data = V.collect(root, (600, 1200))
        rung = json.loads((root / "ZDT1" / "rvea@600" / "run_2" / "meta.json").read_text("utf-8"))
        assert data["values"][("ZDT1", "rvea", 2)][600]["igdp_norm"] == rung["final"]["igdp_norm"]
        first = next(t for t in T if t["fe"] >= 600)
        assert data["values"][("ZDT1", "nsga2", 1)][600]["igdp_norm"] == first["igdp_norm"]
        assert not data["missing"] and data["dependent"] == {"rvea"}
        text_out = V.run(root, taus=(10.0, 1e-9), floor_taus=(1e6, 0.0), budgets=(600, 1200),
                         min_seeds=2)
        assert "igdp <= 1e+06 x floor (hv_h gap <= 10) at 600: 1/1 covered" in text_out, text_out
        assert "gdp <= 1e-09 (hv_h gap <= 1e-09) at 1200: 0/1 covered" in text_out, text_out
        with (root / "_cover" / "igdp_floor.csv").open(encoding="utf-8") as fh:
            assert fh.read().splitlines()[1].startswith("ZDT1,100,1000,")
        meta = json.loads((root / "_cover" / "cover_meta.json").read_text(encoding="utf-8"))
        assert meta["analysis"]["commit"] and "floor" in meta["levels"]["igdp"], meta
        assert meta["floor_taus"] == [1e6, 0.0] and meta["excluded"] == {}, meta
        with (root / "_cover" / "cover_summary.csv").open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert rows[0]["level"] == "igdp_norm <= 1e+06 x greedy floor | hv_h gap <= 10", rows[0]
        assert rows[0]["analysis_commit"] == meta["analysis"]["commit"][:12] + \
            ("+dirty" if meta["analysis"]["dirty"] else ""), rows[0]
        # every table over all problems, the unmarked ones and the interior ones
        # (task 6): ZDT1's optimum is on the boundary, so none of those here
        assert [r_["subset"] for r_ in rows[:3]] == ["all", "unmarked", "interior"], rows[:3]
        assert (rows[0]["problems"], rows[2]["problems"]) == ("1", "0"), rows[:3]
        assert meta["floor_method"] == "greedy" and meta["reference_version"] >= 2, meta
        with (root / "_cover" / "floor_review.csv").open(encoding="utf-8") as fh:
            review = list(csv.DictReader(fh))
        assert [r["problem"] for r in review] == ["ZDT1"] and float(review[0]["best_ratio"]) > 0
        assert float(review[0]["floor_greedy"]) < float(review[0]["floor_dss"]), review[0]
        for name in ("cover_summary.csv", "cover_curve.csv", "cover_nobody.csv",
                     "best_known_hv.csv", "cover_marks.csv"):
            assert (root / "_cover" / name).is_file(), name


@test
def one_pass_over_the_front_gives_what_each_indicator_computes_alone():
    """compute() reads igdp_norm, gdp_norm, eps_norm, tau90 and gap_max off one
    pass over the differences (front_distances); the values must be those of
    igd_plus, gd_plus, eps_plus and dplus_distances, bit for bit (the reference
    sizes cross the 512-row chunks)."""
    try:
        import numpy as np
    except ImportError:
        return
    from mootation.run import metrics as M
    rng = np.random.default_rng(3)
    for _ in range(30):
        m, n, k = int(rng.integers(2, 7)), int(rng.integers(1, 200)), int(rng.integers(1, 1300))
        F, ref = rng.random((n, m)) * 1.3, rng.random((k, m))
        ideal, nadir = np.zeros(m), np.ones(m) * (1 + rng.random())
        out = M.compute(F, ref_front=ref, ideal=ideal, nadir=nadir,
                        which=["igdp_norm", "gdp_norm", "eps_norm", "tau90", "gap_max"])
        G, R = M._normalised(F, ref, ideal, nadir)
        d = M.dplus_distances(G, R)
        assert out["igdp_norm"] == M.igd_plus(G, R) and out["gdp_norm"] == M.gd_plus(G, R)
        assert out["eps_norm"] == M.eps_plus(G, R)
        assert out["tau90"] == float(np.quantile(d, 0.9)) and out["gap_max"] == float(d.max())


@test
def costly_indicators_can_wait_for_the_record_at_counts():
    """record_at_metrics: those indicators only at the records that reach a
    record_at count and at the last one; --recompute-trajectory fills the rest
    from the snapshots."""
    if not _have_numpy() or _core_with("on_generation") is None:
        print("  skip  costly_indicators...: no NumPy or stale _core"); return
    from mootation import _core
    if not hasattr(_core, "current_variables"):
        print("  skip  costly_indicators...: stale _core"); return
    from mootation.run import campaign as C
    from mootation.run.config import ConfigError, load
    text = """algorithms = [
    { name = "nsga2", pop = 0, gens = 0 },
]
[run]
name = "sparse"
[problem]
kind = "builtin"
[benchmarks]
runs = 1
problems = ["DTLZ2_3D"]
[campaign]
out = "res"
budget_fe = 1500
record_grid = "log"
record_at = [500, 1000]
metrics = ["igdp_norm", "igdx"]
record_at_metrics = ["igdx"]
snapshots = true
snapshot_variables = "pareto_set"
"""
    with tempfile.TemporaryDirectory() as td:
        cfg_path = Path(td) / "c.toml"
        cfg_path.write_text(text, encoding="utf-8")
        cfg = load(cfg_path)
        assert validate(cfg) == [], validate(cfg)
        spec = C.campaign_spec(cfg)
        root = C.out_root(cfg, spec)
        for job in C.expand_jobs(cfg, spec):
            assert C.run_job(job, root, spec, quiet=True) == "done", job
        d = root / "DTLZ2_3D" / "nsga2" / "run_1"
        T = [json.loads(l) for l in (d / "trajectory.jsonl").read_text().splitlines()]
        assert all("igdp_norm" in t for t in T)
        marks = [t["fe"] for t in T if "igdx" in t]
        want = [next(t["fe"] for t in T if t["fe"] >= b) for b in (500, 1000)] + [T[-1]["fe"]]
        assert marks == want, (marks, want)
        assert C.recompute_trajectory(root, ["igdx"]) == {"done": 1}
        again = [json.loads(l) for l in (d / "trajectory.jsonl").read_text().splitlines()]
        assert all(u["igdx"] is not None for u in again)
        for t, u in zip(T, again):
            if "igdx" in t:
                assert abs(u["igdx"] - t["igdx"]) <= 1e-5 * max(1.0, t["igdx"]), (t, u)
        bad = Path(td) / "bad.toml"
        bad.write_text(text.replace('record_at_metrics = ["igdx"]', 'record_at_metrics = ["hv"]'),
                       encoding="utf-8")
        try:
            C.campaign_spec(load(bad))
            raise AssertionError("record_at_metrics outside metrics accepted")
        except ConfigError:
            pass


@test
def a_budget_that_no_population_divides_still_finds_its_rungs():
    """At a population of 91 a budget of 1200 runs to 1274 evaluations: --cover
    and --at count from the budget asked for (budget_nominal), so rvea's full
    run stands at 1200 and its rung at 600, not missing."""
    if not _have_numpy() or _core_with("on_generation") is None:
        print("  skip  a_budget_that_no_population...: no NumPy or stale _core"); return
    from mootation.run import campaign as C
    from mootation.run import cover as V
    from mootation.run.config import load
    text = (_LADDER.replace("budget_fe = 25000", 'budget_fe = 1200' + chr(10)
                            + 'metrics = ["igdp_norm", "gdp_norm", "hv_h"]')
            .replace("record_at = [10000, 2500, 5000]", "record_at = [600]")
            .replace("ladder = [2500, 10000, 30000]", "ladder = [600]"))
    with tempfile.TemporaryDirectory() as td:
        cfg_path = Path(td) / "c.toml"
        cfg_path.write_text(text, encoding="utf-8")
        cfg = load(cfg_path)
        assert validate(cfg) == [], validate(cfg)
        spec = C.campaign_spec(cfg)
        root = C.out_root(cfg, spec)
        for job in C.expand_jobs(cfg, spec):
            assert C.run_job(job, root, spec, quiet=True) == "done", job
        m = json.loads((root / "DTLZ2_3D" / "rvea" / "run_1" / "meta.json").read_text("utf-8"))
        assert m["budget_fe"] == 1274 and m["budget_nominal"] == 1200, m
        data = V.collect(root, (600, 1200))
        assert not data["missing"], data["missing"]
        full = json.loads((root / "DTLZ2_3D" / "rvea" / "run_2" / "meta.json").read_text("utf-8"))
        assert data["values"][("DTLZ2_3D", "rvea", 2)][1200]["igdp_norm"] == \
            full["final"]["igdp_norm"]
        rows = C.rungs_in_place(C.scan_results(root), 0.5)
        assert {r["algorithm"]: r.get("rung_of") for r in rows if r["seed"] == 1} == \
            {"nsga2": None, "rvea": 1274}, rows


@test
def the_trajectory_keeps_variables_checkpoints_and_recomputes_from_snapshots():
    """Decision-space indicators along the trajectory (the core hands its
    variables over through current_variables, a baseline passes them), the
    snapshots' variables where the problem has a Pareto set, the archive
    checkpoints and the tables and --cover reading them in the archive scenario,
    record_seconds, and --recompute-trajectory from the snapshots."""
    if not _have_numpy() or _core_with("on_generation") is None:
        print("  skip  the_trajectory_keeps_variables...: no NumPy or stale _core"); return
    import numpy as np
    from mootation import _core
    if not hasattr(_core, "current_variables"):
        print("  skip  the_trajectory_keeps_variables...: stale _core"); return
    from mootation.run import campaign as C
    from mootation.run import cover as V
    from mootation.run.config import load
    text = """algorithms = [
    { name = "nsga2", pop = 0, gens = 0 },
    { name = "random_search", pop = 0, gens = 0 },
]
[run]
name = "tr"
[problem]
kind = "builtin"
[benchmarks]
runs = 1
problems = ["WFG4_2D", "DTLZ2_3D"]
[campaign]
out = "res"
budget_fe = 1500
record_grid = "log"
record_at = [500, 1000, 1500]
metrics = ["igdp_norm", "gdp_norm", "hv_h", "nd_share", "r2", "pdist", "igdx", "cr"]
final_metrics = ["igdp_norm", "pdist", "igdx", "cr"]
operator_stats = true
archive_variables = "selected"
archive_checkpoints = true
snapshots = true
snapshot_variables = "pareto_set"
"""
    with tempfile.TemporaryDirectory() as td:
        cfg_path = Path(td) / "c.toml"
        cfg_path.write_text(text, encoding="utf-8")
        cfg = load(cfg_path)
        assert validate(cfg) == [], validate(cfg)
        spec = C.campaign_spec(cfg)
        root = C.out_root(cfg, spec)
        for job in C.expand_jobs(cfg, spec):
            assert C.run_job(job, root, spec, quiet=True) == "done", job
        for prob, pop, n in (("WFG4_2D", 100, 22), ("DTLZ2_3D", 91, 12)):
            for alg in ("nsga2", "random_search"):
                d = root / prob / alg / "run_1"
                meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
                T = [json.loads(l) for l in (d / "trajectory.jsonl").read_text().splitlines()]
                assert 0 < meta["record_seconds"] < meta["seconds"], meta["record_seconds"]
                # the last record is the answer: the same rows, the same variables
                for k in ("pdist", "igdx", "cr"):
                    assert T[-1][k] == meta["final"][k], (prob, alg, k, T[-1][k])
                assert T[-1]["pdist"] > 0 and (T[-1]["igdx"] is None) == (prob == "WFG4_2D")
                with np.load(d / "snapshots.npz") as z:
                    assert ("X" in z.files) == (prob == "DTLZ2_3D"), (prob, z.files)
                    assert z["F"].shape[0] == int(z["n"].sum()), z["F"].shape
                    if "X" in z.files:
                        assert z["X"].shape == (z["F"].shape[0], n), z["X"].shape
                # checkpoints before the end only: 1500 is the end (final_archive)
                assert sorted(meta["archive_at"], key=int) == ["500", "1000"], meta["archive_at"]
                for b, e in meta["archive_at"].items():
                    assert int(b) <= e["fe"] < int(b) + pop and 0 < e["n"] <= pop, (b, e)
                with np.load(d / "archive_at.npz") as z:
                    assert z["budget"].tolist() == [500, 1000] and ("X" in z.files) == \
                        (prob == "DTLZ2_3D"), (prob, z.files)
        # the tables and --cover read the checkpoints in the archive scenario
        rows = C.rungs_in_place(C.scenario_rows(C.scan_results(root), "archive"), 1000 / 1500)
        r = next(r for r in rows if r["problem"] == "DTLZ2_3D" and r["algorithm"] == "nsga2")
        assert C.value_at(r, "igdx", 1000 / 1500) == r["archive_at"]["1000"]["igdx"], r
        data = V.collect(root, (500, 1000, 1500), scenario="archive")
        v = data["values"][("DTLZ2_3D", "nsga2", 1)]
        assert v[500]["igdp_norm"] == r["archive_at"]["500"]["igdp_norm"], v
        assert v[1500]["igdp_norm"] == r["final_archive"]["igdp_norm"], v
        out = V.run(root, taus=(10.0,), floor_taus=(1e6,), budgets=(500, 1500), min_seeds=1,
                    scenario="archive")
        assert "igdp <= 1e+06 x floor (hv_h gap <= 10) at 500: 2/2 covered" in out, out
        assert (root / "_cover_archive" / "cover_summary.csv").is_file()
        # --recompute-trajectory: what the snapshots give back (float32 rows);
        # without the variables (WFG4_2D) pdist and igdx stay out
        before = {}
        for prob in ("DTLZ2_3D", "WFG4_2D"):
            d = root / prob / "nsga2" / "run_1"
            before[prob] = [json.loads(l) for l in (d / "trajectory.jsonl").read_text().splitlines()]
            blank = [{k: v for k, v in t.items() if k not in ("igdp_norm", "pdist", "igdx")}
                     for t in before[prob]]
            (d / "trajectory.jsonl").write_text("".join(json.dumps(t) + "\n" for t in blank),
                                                encoding="utf-8")
        counts = C.recompute_trajectory(root, ["igdp_norm", "pdist", "igdx"],
                                        hv_settings=(5, 100_000, 0, 0))
        assert counts == {"done": 4}, counts
        for prob, keys in (("DTLZ2_3D", ("igdp_norm", "pdist", "igdx")), ("WFG4_2D", ("igdp_norm",))):
            d = root / prob / "nsga2" / "run_1"
            again = [json.loads(l) for l in (d / "trajectory.jsonl").read_text().splitlines()]
            for t, u in zip(before[prob], again):
                for k in keys:
                    assert abs(u[k] - t[k]) <= 1e-5 * max(1.0, abs(t[k])), (prob, k, t[k], u[k])
                if prob == "WFG4_2D":
                    assert "pdist" not in u and "igdx" not in u, u


@test
def smpso_presets_knobs_and_refusals():
    """Task 9, B1: SMPSO (algorithms/smpso.hpp) — each preset resolves to its
    table row and Result.settings reports every knob as the run used it; a
    knob overrides its preset whatever the order; the budget is spent to the
    evaluation; a binary genome, a bad word or a bad value is refused; the
    knobs are smpso's alone; the omopso preset (turbulence thirds) is
    budget-dependent, the others are not."""
    if (not _have_numpy() or _core_with("Config") is None
            or "smpso" not in __import__("mootation").algorithms()):
        print("  skip  smpso...: no NumPy or stale _core"); return
    import mootation
    from mootation.run.budget import depends_on_budget
    f = lambda x: [x[0], 1.0 - x[0] ** 0.5 + sum(x[1:])]
    box = [(0.0, 1.0)] * 4
    def run(**kw):
        kw.setdefault("max_evaluations", 1000)
        return mootation.minimize(f, box, 2, algorithm="smpso", pop_size=20, seed=3, **kw)
    rows = {
        "jmetal":  {"chi_mode": "signed", "w_min": "0.1", "w_max": "0.1", "c_max": "2.5",
                    "vmax_frac": "0.5", "bound_repair": "clip", "vel_on_bound_low": "-1",
                    "turbulence": "every_k", "turbulence_k": "6", "archive_size": "20",
                    "archive_prune": "one_by_one", "answer": "leaders"},
        "platemo": {"chi_mode": "abs", "w_min": "0.1", "w_max": "0.5", "vel_on_bound_up": "0.001",
                    "turbulence": "prob", "turbulence_p": "0.15", "bound_repair": "clip"},
        "fair":    {"chi_mode": "abs", "w_max": "0.1", "bound_repair": "reflect",
                    "vel_on_bound_low": "-1", "turbulence": "every_k"},
        "omopso":  {"chi_mode": "none", "w_max": "0.5", "c_min": "1.5", "c_max": "2",
                    "vmax_frac": "0", "turbulence": "thirds", "answer": "eps_archive",
                    "eps_archive_eta": "0.0075"},
    }
    for preset, row in rows.items():
        r = run(smpso_preset=preset)
        s = dict(r.settings)
        assert s["smpso_preset"] == preset and not r.ignored, (preset, r.ignored)
        assert {k: s[k] for k in row} == row, (preset, s)
        assert r.evaluations == 1000 and 0 < len(r.objectives) <= 20, (preset, r.evaluations)
    # an override wins over the preset, given before or after it
    for kw in ({"chi_mode": "signed", "smpso_preset": "platemo"},
               {"smpso_preset": "platemo", "chi_mode": "signed"}):
        s = dict(run(**kw).settings)
        assert s["chi_mode"] == "signed" and s["w_max"] == "0.5", s
    s = dict(run(vmax_frac=0.25, w_min=0.2, w_max=0.4, archive_size=7, r_per_dim="yes",
                 leader_select="random", archive_prune="one_shot", turbulence="off").settings)
    assert (s["vmax_frac"], s["w_min"], s["w_max"], s["archive_size"], s["r_per_dim"],
            s["leader_select"], s["archive_prune"], s["turbulence"]) == \
        ("0.25", "0.2", "0.4", "7", "yes", "random", "one_shot", "off"), s
    assert len(run(archive_size=7).objectives) <= 7
    # the same seed, the same run
    a, b = run(smpso_preset="omopso"), run(smpso_preset="omopso")
    assert a.objectives == b.objectives and a.variables == b.variables
    for kw in ({"smpso_preset": "pso"}, {"chi_mode": "plus"}, {"turbulence": "half"},
               {"vmax_frac": -0.1}, {"turbulence_p": 1.5}, {"archive_size": 0},
               {"w_min": 0.6, "w_max": 0.5}):
        assert raises(ValueError, run, **kw) is not None, kw
    e = raises(ValueError, mootation.minimize, lambda x: [x[0], 1.0 - x[0] + x[1]],
               [(0.0, 1.0), (0.0, 1.0)], 2, algorithm="smpso", pop_size=10,
               max_evaluations=100, seed=1, var_types=["real", "bin"])
    assert e is not None and "smpso" in str(e), e
    r = mootation.minimize(f, box, 2, algorithm="nsga2", pop_size=20, n_gen=2, seed=1,
                           vmax_frac=0.3, smpso_preset="fair")
    assert {"vmax_frac", "smpso_preset"} <= set(r.ignored) and list(r.settings) == [], r.ignored
    assert not depends_on_budget("smpso") and not depends_on_budget("smpso", {"smpso_preset": "fair"})
    assert depends_on_budget("smpso", {"smpso_preset": "omopso"})
    assert depends_on_budget("smpso", {"turbulence": "thirds"})
    assert not depends_on_budget("smpso", {"smpso_preset": "omopso", "turbulence": "every_k"})
    assert depends_on_budget("rvea") and not depends_on_budget("nsga2", {"turbulence": "thirds"})


@test
def smpso_settings_reach_meta_json():
    """Task 9, B1: every knob of an smpso run goes into its meta.json, the
    preset resolved; the omopso preset's run is marked budget-dependent."""
    if (not _have_numpy() or _core_with("Config") is None
            or "smpso" not in __import__("mootation").algorithms()):
        print("  skip  smpso_settings...: no NumPy or stale _core"); return
    from mootation.run import campaign as C
    from mootation.run.config import load
    text = """algorithms = [
    { name = "smpso", label = "jm", pop = 20, gens = 0 },
    { name = "smpso", label = "om", pop = 20, gens = 0, params = { smpso_preset = "omopso", archive_size = 10 } },
]
[run]
name = "pso"
[problem]
kind = "builtin"
[benchmarks]
runs = 1
problems = ["ZDT1"]
[campaign]
out = "res"
budget_fe = 600
metrics = ["igdp_norm"]
final_metrics = ["igdp_norm"]
"""
    with tempfile.TemporaryDirectory() as td:
        cfg_path = Path(td) / "c.toml"
        cfg_path.write_text(text, encoding="utf-8")
        cfg = load(cfg_path)
        assert validate(cfg) == [], validate(cfg)
        spec = C.campaign_spec(cfg)
        root = C.out_root(cfg, spec)
        for job in C.expand_jobs(cfg, spec):
            assert C.run_job(job, root, spec, quiet=True) == "done", job
        jm = json.loads((root / "ZDT1" / "jm" / "run_1" / "meta.json").read_text(encoding="utf-8"))
        om = json.loads((root / "ZDT1" / "om" / "run_1" / "meta.json").read_text(encoding="utf-8"))
        assert jm["settings"]["smpso_preset"] == "jmetal" and jm["settings"]["chi_mode"] == "signed"
        assert om["settings"]["answer"] == "eps_archive" and om["settings"]["archive_size"] == "10"
        assert len(jm["settings"]) == len(om["settings"]) >= 20, jm["settings"]
        assert jm["budget_dependent"] is False and om["budget_dependent"] is True


@test
def smpso_reaches_the_zdt4_front():
    """Task 9, B1, acceptance 1: the jmetal preset, swarm and leaders 20,
    10 000 evaluations on ZDT4 — the non-dominated of every evaluation within
    IGD+ 0.005 (normalised) of the front on each of three seeds."""
    if (not _have_numpy() or _core_with("Config") is None
            or "smpso" not in __import__("mootation").algorithms()):
        print("  skip  smpso_reaches...: no NumPy or stale _core"); return
    import numpy as np
    import mootation
    from mootation.benchmarks import get
    from mootation.run import metrics as M
    p = get("ZDT4")
    lo, hi = np.asarray(p.ideal, float), np.asarray(p.nadir, float)
    ref = (np.asarray(p.pareto_front(1000), float) - lo) / (hi - lo)
    for seed in (1, 2, 3):
        seen = []
        def f(x):
            y = p.evaluate(x)
            seen.append(list(y))
            return y
        mootation.minimize(f, p.bounds, 2, algorithm="smpso", pop_size=20,
                           max_evaluations=10000, seed=seed)
        A = M.nondominated(np.asarray(seen, float))
        assert len(seen) == 10000 and M.igd_plus((A - lo) / (hi - lo), ref) < 0.005, seed


def _have_scipy():
    try:
        import scipy.optimize  # noqa: F401
        return True
    except ImportError:
        return False


@test
def scalar_presets_budget_and_refusals():
    """Task 9, A1: the scalarization methods (mootation.scalar) — every preset
    runs through minimize(), spends evaluations only through fn (calls =
    result.evaluations, the finite differences counted) and never past the
    budget; the answer is non-dominated and pop_size at most; a knob it lacks
    is reported, its own knobs are reported by the cores; a binary genome,
    eps_level, laumanns with nbi and a bad word are refused; a failing fn does
    not stop the run."""
    if not _have_numpy() or not _have_scipy() or _core_with("Config") is None:
        print("  skip  scalar_presets...: no NumPy, SciPy or _core"); return
    import math
    import mootation
    from mootation.run.metrics import nondominated
    from mootation.scalar import NAMES, PRESETS
    assert tuple(PRESETS) == ("scalar", "eps_sqp", "tch_sqp", "nbi_sqp", "nnc_sqp", "eps_nm",
                              "eps_cmaes") and NAMES == tuple(PRESETS) + ("boostdms", "mosqp")
    calls = [0]
    def f(x):
        calls[0] += 1
        return [x[0], 1.0 - math.sqrt(x[0]) + sum(v * v for v in x[1:])]
    box = [(0.0, 1.0)] * 4
    for name in PRESETS:
        calls[0] = 0
        r = mootation.minimize(f, box, 2, algorithm=name, pop_size=20, max_evaluations=1500,
                               seed=2)
        F = r.objectives
        assert r.evaluations == calls[0] <= 1500, (name, r.evaluations, calls[0])
        assert 0 < len(F) <= 20 and len(nondominated(F)) == len(F), (name, len(F))
        s = dict(r.settings)
        assert s["local"] in ("sqp", "nm", "cmaes") and not r.ignored, (name, r.ignored)
    assert dict(mootation.minimize(f, box, 2, algorithm="nbi_sqp", pop_size=10,
                                   max_evaluations=600, seed=1).settings)["scalarization"] == "nbi"
    r = mootation.minimize(f, box, 2, algorithm="scalar", pop_size=10, max_evaluations=600,
                           seed=1, eta_c=15, sub_plan="gap", scalarization="tch")
    assert r.ignored == ["eta_c"] and dict(r.settings)["sub_plan"] == "gap", r.ignored
    r = mootation.minimize(f, box, 2, algorithm="nsga2", pop_size=10, n_gen=2, seed=1,
                           scalarization="nbi", n_sub=5)
    assert {"scalarization", "n_sub"} <= set(r.ignored), r.ignored
    for kw in ({"var_types": ["real", "bin", "real", "real"]},
               {"constraint_mode": "eps_level", "constraints": lambda x: [x[0] - 0.5]},
               {"sub_plan": "laumanns", "scalarization": "nbi"}, {"local": "bfgs"},
               {"eps_objective": 3}, {"n_sub": 2.5}):
        assert raises((ValueError, TypeError), mootation.minimize, f, box, 2,
                      algorithm="scalar", pop_size=10, max_evaluations=300, seed=1,
                      **kw) is not None, kw
    k = [0]
    def g(x):
        k[0] += 1
        if k[0] % 9 == 0:
            raise RuntimeError("the model did not converge")
        return f(x)
    for name in ("eps_sqp", "eps_nm", "eps_cmaes"):
        k[0] = 0
        r = mootation.minimize(g, box, 2, algorithm=name, pop_size=10, max_evaluations=800,
                               seed=1)
        assert r.n_failed > 0 and len(r.objectives) > 0, (name, r.n_failed)
        assert all(math.isfinite(v) for row in r.objectives for v in row)


@test
def scalar_anchors_constraints_and_the_front():
    """Task 9, A4: eps_sqp's anchors on ZDT1 within 1e-12 of the true minima
    (0, 0); on DOC3 every point of the answer within eq_tol of the equalities
    and feasible; nbi_sqp's subproblem solutions on DTLZ2_3D on the front
    (| ||F|| - 1 | < 1e-8) and evenly spread (Spacing < 0.06); laumanns on ZDT1
    finds points on the front; the observer sees gen 0, the records and the end."""
    if not _have_numpy() or not _have_scipy() or _core_with("Config") is None:
        print("  skip  scalar_anchors...: no NumPy, SciPy or _core"); return
    import math
    import numpy as np
    import mootation
    from mootation.benchmarks import get, solve
    def zdt1(x):                                      # the formula, not a shared registry object
        g = 1.0 + 9.0 * sum(x[1:]) / (len(x) - 1)
        return [x[0], g * (1.0 - math.sqrt(x[0] / g))]
    class p:                                          # noqa: N801
        bounds, evaluate = [(0.0, 1.0)] * 30, staticmethod(zdt1)
    seen = []
    def f(x):
        y = p.evaluate(x)
        seen.append(y)
        return y
    gens = []
    # 4000, not 3000: at 3000 an anchor's stage has about six SQP iterations,
    # too few to reach its minimum with every BLAS thread count (a campaign's
    # workers run one)
    r = mootation.minimize(f, p.bounds, 2, algorithm="eps_sqp", pop_size=50,
                           max_evaluations=4000, n_sub=5, seed=1,
                           on_generation=lambda g, F, X: gens.append((g, len(F), len(X))),
                           record_every=1)
    assert np.max(np.abs(np.min(np.asarray(seen), axis=0))) < 1e-12, np.min(seen, axis=0)
    assert gens[0][0] == 0 and gens[-1][0] == -(-r.evaluations // 50), gens[-3:]
    assert all(n == x for _, n, x in gens) and len(gens) >= r.evaluations // 50, len(gens)
    S = np.asarray(r.solutions)       # n_sub 5 and the anchors
    err = np.abs(S[:, 1] - (1 - np.sqrt(S[:, 0])))
    assert len(S) == 7 and err.max() < 1e-6, (len(S), err, S)
    d = get("DOC3")
    r = solve(d, "eps_sqp", max_evaluations=6000, seed=1)
    H = np.abs(np.array([d.equalities(x) for x in r.variables]))
    G = np.array([list(d.constraints(x))[:-d.n_eq] for x in r.variables])
    assert len(H) and H.max() <= 1e-4 and G.max() <= 0.0 and max(r.cv) == 0.0, (H.max(), G.max())
    q = get("DTLZ2_3D")
    r = mootation.minimize(q.evaluate, q.bounds, 3, algorithm="nbi_sqp", pop_size=91,
                           max_evaluations=10000, seed=1)
    S = np.asarray(r.solutions)
    D = np.abs(S[:, None, :] - S[None, :, :]).sum(axis=2)
    np.fill_diagonal(D, np.inf)
    near = D.min(axis=1)
    spacing = np.sqrt(np.sum((near - near.mean()) ** 2) / (len(near) - 1))
    assert len(S) > 50 and np.max(np.abs(np.linalg.norm(S, axis=1) - 1)) < 1e-8, len(S)
    assert spacing < 0.06, spacing
    r = mootation.minimize(p.evaluate, p.bounds, 2, algorithm="scalar", sub_plan="laumanns",
                           pop_size=50, max_evaluations=10000, seed=1)
    S = np.asarray(r.solutions)
    assert len(S) > 10 and np.max(np.abs(S[:, 1] - (1 - np.sqrt(S[:, 0])))) < 1e-6


@test
def scalar_runs_in_a_campaign():
    """Task 9, A1: a campaign runs the presets by name; meta.json carries the
    settings and marks the run budget-dependent."""
    if not _have_numpy() or not _have_scipy() or _core_with("Config") is None:
        print("  skip  scalar_runs...: no NumPy, SciPy or _core"); return
    from mootation.run import campaign as C
    from mootation.run.config import load
    text = """algorithms = [
    { name = "eps_sqp", pop = 20, gens = 0 },
    { name = "scalar", label = "ps_gap", pop = 20, gens = 0, params = { scalarization = "ps", sub_plan = "gap" } },
]
[run]
name = "scalar"
[problem]
kind = "builtin"
[benchmarks]
runs = 1
problems = ["ZDT1"]
[campaign]
out = "res"
budget_fe = 1500
metrics = ["igdp_norm"]
final_metrics = ["igdp_norm"]
operator_stats = true
"""
    with tempfile.TemporaryDirectory() as td:
        cfg_path = Path(td) / "c.toml"
        cfg_path.write_text(text, encoding="utf-8")
        cfg = load(cfg_path)
        assert validate(cfg) == [], validate(cfg)
        spec = C.campaign_spec(cfg)
        root = C.out_root(cfg, spec)
        for job in C.expand_jobs(cfg, spec):
            assert C.run_job(job, root, spec, quiet=True) == "done", job
        for key, sc in (("eps_sqp", "eps"), ("ps_gap", "ps")):
            meta = json.loads((root / "ZDT1" / key / "run_1" / "meta.json").read_text(encoding="utf-8"))
            assert meta["settings"]["scalarization"] == sc and meta["budget_dependent"] is True
            assert meta["fe"] <= 1500 and meta["status"] == "done", meta["fe"]
            T = [json.loads(l) for l in (root / "ZDT1" / key / "run_1" / "trajectory.jsonl")
                 .read_text().splitlines()]
            assert T[0]["gen"] == 0 and T[-1]["igdp_norm"] < 0.1, T[-1]


@test
def polish_improves_and_loses_nothing():
    """Task 9, A2: polish = sqp / nm after a core — the budget split exactly
    (the core's share, then the polishing's), every point of the core's own
    answer weakly dominated by the polished answer, the settings reported;
    polish = off is the plain run, bit for bit; without max_evaluations, or
    with a share out of (0, 1), refused."""
    if not _have_numpy() or not _have_scipy() or _core_with("Config") is None:
        print("  skip  polish...: no NumPy, SciPy or _core"); return
    import numpy as np
    import mootation
    from mootation.benchmarks import get
    p = get("DTLZ2_3D")
    calls = [0]
    def f(x):
        calls[0] += 1
        return p.evaluate(x)
    plain = mootation.minimize(p.evaluate, p.bounds, 3, algorithm="nsga2", pop_size=91,
                               max_evaluations=4000, seed=3)
    off = mootation.minimize(p.evaluate, p.bounds, 3, algorithm="nsga2", pop_size=91,
                             max_evaluations=4000, seed=3, polish="off")
    assert off.objectives == plain.objectives and off.variables == plain.variables
    core = mootation.minimize(p.evaluate, p.bounds, 3, algorithm="nsga2", pop_size=91,
                              max_evaluations=4000, seed=3)
    # 91 points: the union outgrows pop_size and is cut (POLISH-2)
    for how, k in (("sqp", 10), ("nm", 10), ("sqp", 91)):
        calls[0] = 0
        r = mootation.minimize(f, p.bounds, 3, algorithm="nsga2", pop_size=91,
                               max_evaluations=5000, seed=3, polish=how, polish_share=0.2,
                               polish_points=k)
        s = dict(r.settings)
        assert s["polish"] == how and s["polish_points"] == str(k), s
        assert r.evaluations == calls[0] <= 5000 + 91, (r.evaluations, calls[0])
        A, B = np.asarray(r.objectives), np.asarray(core.objectives)
        assert len(A) <= 91
        assert all(np.any(np.all(A <= b, axis=1)) for b in B), (how, k)
    for kw in ({"polish": "sqp", "n_gen": 5}, {"polish": "sqp", "max_evaluations": 900,
                                               "polish_share": 1.0},
               {"polish": "bfgs", "max_evaluations": 900}):
        kw.setdefault("max_evaluations", 0)
        assert raises(ValueError, mootation.minimize, p.evaluate, p.bounds, 3,
                      algorithm="nsga2", pop_size=91, seed=1, **kw) is not None, kw
    from mootation.run.budget import depends_on_budget
    assert depends_on_budget("nsga2", {"polish": "nm"}) and not depends_on_budget("nsga2")


@test
def r2_emoa_bc_and_paes_take_their_knobs():
    """Task 9, B3-B5: r2_emoa takes r2_mode, r2_space, r2_weights, r2_offset
    and normalize, refuses a bad word and keeps SMS-EMOA's binary genome;
    bc_spx and bc_rex take bc_selection, bc_lambda and crossover spx or rex,
    spend λ evaluations a step and refuse sbx, a population below n + 1 and a
    binary genome; paes spends one evaluation a step, bounds its answer by
    archive_size, takes paes_depth and refuses a binary genome."""
    if not _have_numpy() or _core_with("Config") is None or \
            "paes" not in __import__("mootation").algorithms():
        print("  skip  r2_emoa_bc_and_paes...: no NumPy or stale _core"); return
    import math
    import mootation
    f = lambda x: [x[0], 1.0 - math.sqrt(x[0]) + sum(v * v for v in x[1:])]
    box = [(0.0, 1.0)] * 4
    def run(alg, **kw):
        kw.setdefault("max_evaluations", 1000)
        return mootation.minimize(f, box, 2, algorithm=alg, pop_size=20, seed=4, **kw)
    for kw in ({}, {"r2_mode": "discrete", "r2_weights": 50}, {"r2_space": "angle"},
               {"r2_offset": 0.1, "normalize": True}):
        r = run("r2_emoa", **kw)
        assert not r.ignored and r.evaluations == 1000 and len(r.objectives) == 20, (kw, r.ignored)
    assert run("r2_emoa", r2_mode="discrete").objectives != run("r2_emoa").objectives
    for kw in ({"r2_mode": "integral"}, {"r2_weights": 1}, {"r2_offset": -1}):
        assert raises(ValueError, run, "r2_emoa", **kw) is not None, kw
    r = mootation.minimize(lambda x: [x[0] + x[2], 1.0 - x[0] + x[1]], [(0.0, 1.0)] * 3, 2,
                           algorithm="r2_emoa", pop_size=10, max_evaluations=200, seed=1,
                           var_types=["real", "bin", "real"])
    assert r.evaluations == 200
    for alg in ("bc_spx", "bc_rex"):
        r = run(alg)
        assert not r.ignored and len(r.objectives) == 20, r.ignored
        assert 1000 <= r.evaluations < 1000 + 40, r.evaluations          # λ = 10n = 40
        for kw in ({"bc_selection": "bf"}, {"bc_selection": "ba"}, {"bc_lambda": 7},
                   {"crossover": "rex" if alg == "bc_spx" else "spx"}):
            r = run(alg, **kw)
            assert not r.ignored and len(r.objectives) == 20, (alg, kw, r.ignored)
        assert run(alg, bc_lambda=7).evaluations < 1000 + 7
        for kw in ({"crossover": "sbx"}, {"bc_selection": "bx"}):
            assert raises(ValueError, run, alg, **kw) is not None, (alg, kw)
        assert raises(ValueError, mootation.minimize, f, box, 2, algorithm=alg, pop_size=4,
                      max_evaluations=200, seed=1) is not None
        assert raises(ValueError, run, alg, var_types=["real", "bin", "real", "real"]) is not None
    r = run("paes")
    assert r.evaluations == 1000 and 0 < len(r.objectives) <= 20 and not r.ignored
    assert len(run("paes", archive_size=7).objectives) <= 7
    assert not run("paes", paes_depth=3).ignored
    assert raises(ValueError, run, "paes", paes_depth=0) is not None
    assert raises(ValueError, run, "paes", var_types=["real", "bin", "real", "real"]) is not None


@test
def direct_search_methods_are_deterministic_and_take_their_knobs():
    """Task 9, C1-C4: dmulti_mads, multiglods and boostdms — one seed, one
    trajectory; multiglods and boostdms (as dms) draw nothing at their
    defaults, dmulti_mads draws its poll directions from the seed; the knobs
    reach them, a bad word is refused, a binary genome refused; the budget is
    not overrun by more than one poll; a run that converges ends early."""
    if not _have_numpy() or _core_with("Config") is None or \
            "multiglods" not in __import__("mootation").algorithms():
        print("  skip  direct_search...: no NumPy or stale _core"); return
    import math
    import mootation
    f = lambda x: [x[0], 1.0 - math.sqrt(x[0]) + sum((v - 0.3) ** 2 for v in x[1:])]
    box = [(0.0, 1.0)] * 3
    def run(alg, seed=1, **kw):
        kw.setdefault("max_evaluations", 600)
        return mootation.minimize(f, box, 2, algorithm=alg, pop_size=20, seed=seed, **kw)
    for alg in ("dmulti_mads", "multiglods"):
        a, b = run(alg), run(alg)
        assert a.objectives == b.objectives, alg
        assert a.evaluations <= 600 + 2 * 3 + 1 and len(a.objectives) <= 20, (alg, a.evaluations)
        assert raises(ValueError, run, alg, var_types=["real", "bin", "real"]) is not None, alg
    assert run("dmulti_mads").objectives != run("dmulti_mads", seed=2).objectives
    assert run("multiglods").objectives == run("multiglods", seed=2).objectives
    for kw in ({"mads_tau": 0.25}, {"mads_delta0": 0.3}, {"mads_select": "first"},
               {"mads_poll": "opportunistic"}, {"mads_search": "none"}):
        assert not run("dmulti_mads", **kw).ignored, kw
    for kw in ({"mads_tau": 1.0}, {"mads_select": "crowding"}):
        assert raises(ValueError, run, "dmulti_mads", **kw) is not None, kw
    for kw in ({"glods_search_every": 1}, {"glods_local_tol": 0.01}):
        assert not run("multiglods", **kw).ignored, kw
    r = run("multiglods", max_evaluations=200000, glods_local_tol=0.05)
    assert r.evaluations < 200000                     # every active point local: the end
    if _have_scipy():
        a, b = run("boostdms", max_evaluations=400), run("boostdms", max_evaluations=400, seed=5)
        assert a.objectives == b.objectives and a.evaluations <= 400 + 6
        assert dict(a.settings)["boost_scalarization"] == "eps"
        r = run("boostdms", max_evaluations=400, boost_scalarization="minmax")
        assert dict(r.settings)["boost_scalarization"] == "minmax" and not r.ignored
        assert raises(ValueError, run, "boostdms", boost_scalarization="nbi") is not None
        assert run("boostdms", max_evaluations=200, eta_c=10).ignored == ["eta_c"]
        from mootation.scalar.boostdms import models
        import numpy as np
        rng = np.random.default_rng(3)
        S = rng.uniform(-1, 1, (40, 3))                 # regression: 40 > 10 coefficients
        q = lambda s: 1.0 + s @ [1.0, -2.0, 0.5] + 0.5 * s @ np.diag([2.0, 1.0, 4.0]) @ s
        (c, g, H), = models(S, np.array([[q(s)] for s in S]))
        assert abs(c - 1) < 1e-10 and np.allclose(g, [1, -2, 0.5]) and \
            np.allclose(H, np.diag([2.0, 1.0, 4.0]))   # a quadratic is its own model
        S = rng.uniform(-1, 1, (7, 3))                  # minimum Frobenius norm: 4 < 7 < 10
        (c, g, H), = models(S, np.array([[1.0 + s @ [1.0, -2.0, 0.5]] for s in S]))
        assert np.allclose(H, 0, atol=1e-9) and np.allclose(g, [1, -2, 0.5])   # linear: H = 0


@test
def mosqp_spreads_refines_and_keeps_its_budget():
    """Task 9, A3: mosqp — every evaluation through fn and counted, never past
    the budget; the answer non-dominated and on ZDT2's front (f2 = 1 - f1²)
    within 1e-3 for most of its points; DOC3's equalities held within eq_tol;
    its knobs reported, a bad value refused."""
    if not _have_numpy() or not _have_scipy() or _core_with("Config") is None:
        print("  skip  mosqp...: no NumPy, SciPy or _core"); return
    import numpy as np
    import mootation
    from mootation.benchmarks import get, solve
    from mootation.run.metrics import nondominated
    p = get("ZDT2")
    calls = [0]
    def f(x):
        calls[0] += 1
        return p.evaluate(x)
    r = mootation.minimize(f, p.bounds, 2, algorithm="mosqp", pop_size=30, max_evaluations=6000,
                           seed=1)
    F = np.asarray(r.objectives)
    assert r.evaluations == calls[0] <= 6000 and len(nondominated(F)) == len(F)
    assert np.mean(np.abs(F[:, 1] - (1 - F[:, 0] ** 2)) <= 1e-3) > 0.5, F
    s = dict(r.settings)
    assert s["mosqp_tau"] == "1e-08" and s["mosqp_spread"] == "20" and not r.ignored, s
    d = get("DOC3")
    r = solve(d, "mosqp", max_evaluations=8000, seed=1)
    H = np.abs(np.array([d.equalities(x) for x in r.variables]))
    assert len(H) and H.max() <= 1e-4 and max(r.cv) == 0.0, H.max()
    assert raises(ValueError, mootation.minimize, f, p.bounds, 2, algorithm="mosqp",
                  pop_size=10, max_evaluations=100, seed=1, mosqp_beta=1.5) is not None


def main() -> int:
    import os
    failed = []
    for fn in TESTS:
        try:
            fn()
            print(f"  ok    {fn.__name__}")
        except Exception as e:
            failed.append(fn.__name__)
            print(f"  FAIL  {fn.__name__}")
            traceback.print_exc()
            if os.environ.get("GITHUB_ACTIONS"):
                # an annotation of the run, readable where the log is not
                where = traceback.extract_tb(e.__traceback__)[-1]
                what = f"{type(e).__name__}: {e}".replace("\n", " ")[:300]
                print(f"::error title=run layer: {fn.__name__}::line {where.lineno} of "
                      f"{Path(where.filename).name}: {what}", flush=True)
    print(f"\n{len(TESTS) - len(failed)}/{len(TESTS)} passed")
    if failed:
        print("failed: " + ", ".join(failed), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

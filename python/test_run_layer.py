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
    assert len(names) == 63, f"expected 63 algorithms, got {len(names)}"
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


# WFG1-9 computed by pymoo 0.6.2 (Apache-2.0) at z_i = 2i ((0.2 + 0.61803398875 i)
# mod 1), i = 1..n, n = 2(M - 1) + 20, k = 2(M - 1) (pymoo's own check k >= 4
# switched off for two objectives; its formulas take any k).
WFG_PYMOO = {
    "WFG1_2D": [2.94457111371373, 0.984463667946229],
    "WFG2_2D": [1.56018000168202, 2.58401057188495],
    "WFG3_2D": [1.92005922421703, 2.15775332546703],
    "WFG4_2D": [1.06082206911698, 4.11744877311357],
    "WFG5_2D": [2.11800671369633, 2.87937749198748],
    "WFG6_2D": [2.40542892580166, 2.63121418749276],
    "WFG7_2D": [1.41950839693793, 4.09133831260672],
    "WFG8_2D": [2.39648814554069, 2.94153233426548],
    "WFG9_2D": [2.5186944014566, 2.84934810074639],
    "WFG1_3D": [2.8546793490372, 0.986095172540787, 0.983907195792332],
    "WFG2_3D": [0.82679872235157, 1.50799865683367, 3.5623166626168],
    "WFG3_3D": [1.19465803897627, 2.17459792976725, 2.92293079298993],
    "WFG4_3D": [0.808919232494887, 1.42943803100311, 5.99026621995833],
    "WFG5_3D": [1.39407860311448, 3.233207890784, 4.08123403040685],
    "WFG6_3D": [2.19350949593799, 2.50347929233896, 3.65367831709079],
    "WFG7_3D": [1.05548835074729, 3.65224891375816, 3.8084321803318],
    "WFG8_3D": [1.64942102045718, 3.55495622372161, 4.06690900271006],
    "WFG9_3D": [2.59493606696002, 1.53531219866495, 2.89798125913124],
    "WFG1_5D": [2.74311768171211, 0.984998204601443, 0.982909799340561, 0.989916761065143,
                0.984931525725869],
    "WFG2_5D": [0.676417783080577, 0.726435055095487, 0.749958260369459, 2.3140552754439,
                5.46366463005119],
    "WFG3_5D": [0.781636058762556, 1.0225502901116, 1.33505588785009, 3.63578307639656,
                4.39802151400641],
    "WFG4_5D": [0.440625792531076, 0.82211778702428, 1.46048629468114, 2.485691259948,
                9.73179472185055],
    "WFG5_5D": [1.03301639684897, 1.27789756743023, 2.34095640418625, 5.94907285773911,
                6.4592071066976],
    "WFG6_5D": [1.665899248303, 2.77726275996572, 1.99790793998006, 4.29413930705103,
                5.61093671295352],
    "WFG7_5D": [0.522167218252031, 0.818507633120548, 1.18220863368219, 5.53851129340892,
                8.13147381144415],
    "WFG8_5D": [1.09426481964525, 1.98625054181705, 2.31824154804536, 6.33946683803821,
                6.2575951397301],
    "WFG9_5D": [1.30252124113123, 1.53941226180596, 3.72620814103324, 2.35122447554018,
                8.90208480581622],
    "WFG1_10D": [2.53327822406374, 0.983047504448265, 0.988390421827318, 0.984198075227453,
                 0.982459104750291, 0.984272189825867, 0.993084421871517, 0.984921101103002,
                 1.0002598138056, 0.988281431989272],
    "WFG2_10D": [0.63320591634379, 0.633222255451482, 0.634416819846423, 0.635237980725482,
                 0.63385898587121, 0.643513215788716, 0.835847867605265, 0.850323323469961,
                 4.33561372609034, 10.2234514527581],
    "WFG3_10D": [0.637422307808985, 0.641331479065863, 0.677431925217375, 0.715800702563992,
                 0.729268549416129, 0.948210281362007, 1.86727367169773, 2.4461740778462,
                 7.25489223052921, 8.0921652206685],
    "WFG4_10D": [0.374507445498651, 0.390264000428367, 0.421110723075865, 0.486526104888682,
                 0.478962461381912, 0.829506990802944, 2.00964095377182, 3.32188393651244,
                 5.16757171133372, 19.1268633556046],
    "WFG5_10D": [0.610088244804657, 0.609391420355498, 0.793939057036668, 1.14936744444178,
                 2.41570064523498, 1.97278329165451, 3.12626840546702, 5.33752479334254,
                 12.7010620434512, 12.3664665657488],
    "WFG6_10D": [1.04152277620799, 1.21426205036138, 2.34446693647205, 1.74266715166473,
                 1.97360291095962, 3.82934936845458, 7.96189227280386, 4.14667169866042,
                 8.76945292247326, 10.5027462812524],
    "WFG7_10D": [0.521677113848181, 0.526911209047744, 0.551987208865328, 0.572235730382281,
                 0.546918865218975, 0.754664561146427, 2.0067047062035, 2.80484794988333,
                 12.1406876947442, 15.365748780637],
    "WFG8_10D": [0.768102925810941, 0.825344643455878, 1.26308977415434, 1.42193466882926,
                 1.13053233008945, 2.07261331212912, 5.10620208145035, 4.94356002846582,
                 13.3290116435579, 11.7625787520084],
    "WFG9_10D": [0.961296168094168, 1.01748770974016, 0.98341455578532, 2.42516566633991,
                 2.38309048842606, 2.36032333348164, 2.69514238740955, 10.7573701356826,
                 1.42566114316174, 15.321084303775],
}

def wfg_at_fixed_points() -> dict:
    """{key of WFG_PYMOO: MOOtation's objectives at that key's fixed point}."""
    import numpy as np
    from mootation.benchmarks import wfg
    out = {}
    for key in WFG_PYMOO:
        name, M = key.split("_")[0], int(key.split("_")[1][:-1])
        i = np.arange(1, 2 * (M - 1) + 21)
        out[key] = wfg.WFG_FUNCS[name](list(2.0 * i * ((0.2 + 0.61803398875 * i) % 1.0)), M)
    return out


@test
def wfg_matches_pymoo_with_and_without_numba():
    """WFG1-9 at 2, 3, 5 and 10 objectives give pymoo's values at fixed points:
    on the path this interpreter takes and, in a child process with
    NUMBA_DISABLE_JIT=1, on the interpreted one. Until 2026-10-09 the path
    without numba got f_2 .. f_{M-1} wrong from three objectives on."""
    if not _have_numpy():
        print("    (skipped: no NumPy)"); return
    import os
    import subprocess
    from mootation.benchmarks import wfg
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
        for key, ref in WFG_PYMOO.items():
            worst = max(abs(a - b) for a, b in zip(got[key], ref))
            assert len(got[key]) == len(ref) and worst < 1e-9, (path, key, worst)


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
        if len(f) != p.n_obj or any(v != v for v in f):
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
    6, item 12.5): rows 0, 5 and 11 of 12 points at seed 5, to 1e-12."""
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
        "lhs_search": [[0.0671787427199933, 0.87864713808322, 5.03133728213074],
                       [0.328950833854345, -1.08414312656047, 8.04914019274116],
                       [0.871689838192273, 0.412373696896755, 2.23928782044247]],
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
    assert len(fams) == 8 and fams[0][1][0] == "nsga2", fams
    assert fams[-1] == ("Direct search", ("dms",)), fams[-1]
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
    assert len(PP.BUDGET_SCHEDULED) == 17 and PP.budget_note("rvea") == "*"
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

# pymoo 0.6.2's C-DTLZ (problems/many/cdtlz.py, Apache-2.0) at the points of
# c_dtlz_fixed_points: (objectives, constraints), <= 0 satisfied. The convex
# C2-DTLZ2, for which pymoo has no class, is its DTLZ2 turned convex (f_i^4,
# f_M^2) under its constraint_c4_cylindrical, the paper's Eq. 6.
C_DTLZ_PYMOO = {
    "C1-DTLZ1_3D": [
        ([59.7984043146437, 118.137038268974, 69.8737772743156],
         [471.327180624427]),
        ([88.9682583664649, 115.05557038731, 45.3836916666593],
         [482.687143618648]),
        ([123.325279701835, 106.730020890775, 20.5403237653231],
         [493.344474127426]),
        ([11.6481296007396, 23.0119105714592, 13.6107112340644],
         [91.0045990678383]),
        ([61.8356237688568, 79.9670926868747, 31.5430349505318],
         [335.177157829016]),
    ],
    "C1-DTLZ3_3D": [
        ([378.041371309325, 220.432887766739, 922.570821542082],
         [-1087002992591.2]),
        ([224.323791039201, 183.254881448657, 985.648141542881],
         [-1113778974154.13]),
        ([87.4754767992258, 97.9946256132122, 1014.5968659256],
         [-1095399201993.48]),
        ([71.1148807980232, 41.4665159085698, 173.548502838375],
         [-1357728433.99052]),
        ([151.186148241474, 123.507183720111, 664.291314581048],
         [-229772324436.839]),
    ],
    "C2-DTLZ2_3D": [
        ([0.684588560766381, 0.399178092226137, 1.67066749529848],
         [0.917799536095413]),
        ([0.41736525869089, 0.340954566824441, 1.83384602082331],
         [0.825742962243431]),
        ([0.159784418525468, 0.178998901694088, 1.85328249918289],
         [0.625662690623007]),
        ([0.370600175935276, 0.216093986500306, 0.904411354746914],
         [0.0331782905061725]),
        ([0.219229662775171, 0.179093379599993, 0.963265236756179],
         [-0.0785144735123583]),
    ],
    "C2-ConvexDTLZ2_3D": [
        ([0.21964359926911, 0.02539023923308, 2.79112987984688],
         [-4.71590560624918]),
        ([0.0303434657310284, 0.0135140663684382, 3.3629912280895],
         [-7.39131553117562]),
        ([0.000651835045328287, 0.00102660048460497, 3.43465602177757],
         [-7.81010696590413]),
        ([0.0188635090441306, 0.00218057347903423, 0.817959898595148],
         [-0.384151422473147]),
        ([0.00230992172400047, 0.00102876961256665, 0.927879916342938],
         [-0.521286501495012]),
    ],
    "C3-DTLZ1_3D": [
        ([59.7984043146437, 118.137038268974, 69.8737772743156],
         [-306.607624172577, -364.946258126907, -316.682997132248]),
        ([88.9682583664649, 115.05557038731, 45.3836916666593],
         [-337.375778786899, -363.463090807744, -293.791212087093]),
        ([123.325279701835, 106.730020890775, 20.5403237653231],
         [-372.920904059768, -356.325645248708, -270.135948123256]),
        ([11.6481296007396, 23.0119105714592, 13.6107112340644],
         [-58.9188810070028, -70.2826619777224, -60.8814626403276]),
        ([61.8356237688568, 79.9670926868747, 31.5430349505318],
         [-234.18137517512, -252.312844093138, -203.888786356795]),
    ],
    "C3-DTLZ4_3D": [
        ([1.40783736378174, 9.7137204167572e-48, 9.10274982407908e-15],
         [0.504498489285019, -0.982006042859924, -0.982006042859924]),
        ([1.43980337503174, 2.04099820292739e-36, 4.28112523765757e-09],
         [0.481741560311801, -1.0730337587528, -1.0730337587528]),
        ([1.46356538642609, 1.92096083262571e-27, 0.000444025668062882],
         [0.464493892755065, -1.14202383750336, -1.14202368963426]),
        ([1.0005, 6.90319601325218e-48, 6.46900091820764e-15],
         [0.7497499375, -0.00100024999999992, -0.00100024999999992]),
        ([1.002, 1.4203885299881e-36, 2.97935646111283e-09],
         [0.748999, -0.0040039999999999, -0.0040039999999999]),
    ],
    "C1-DTLZ1_5D": [
        ([36.1218374535393, 27.0132218586119, 3.03717546512012, 130.729104259922,
          77.3215280167352],
         [521.671891435612]),
        ([3.54171545726692, 1.72762838741242, 92.1271873706594, 125.955185124205,
          49.6830467768852],
         [528.508510640564]),
        ([16.1633356690998, 4.76994112552867, 114.907149541453, 117.561067574421,
          22.6247720184261],
         [543.510941185049]),
        ([6.35843485552335, 4.75506849967716, 0.53462624553912, 23.0119105714592,
          13.6107112340644],
         [91.0045990678383]),
        ([2.24858300166456, 1.09684582852584, 58.4901949386664, 79.9670926868747,
          31.5430349505318],
         [335.177157829016]),
    ],
    "C1-DTLZ3_5D": [
        ([17.5977005666561, 22.1171892212406, 391.349728325773, 228.787232930179,
          957.535999193056],
         [-1261322431809.4]),
        ([114.026429374905, 201.47732425991, 19.7216343864504, 189.807371500353,
          1020.89112983807],
         [-1281748885880.08]),
        ([31.0302444335286, 82.95988644939, 21.8691001705988, 102.204099285361, 1058.18006008785],
         [-1296018463027.65]),
        ([3.18949322982626, 4.00862743497276, 70.9301368244946, 41.4665159085698,
          173.548502838375],
         [-1354953222.94151]),
        ([74.196713491333, 131.100792904447, 12.8328183577615, 123.507183720111,
          664.291314581048],
         [-229736251225.296]),
    ],
    "C2-DTLZ2_5D": [
        ([0.0304706544058254, 0.0382962095892014, 0.677627300138466, 0.396148160418738,
          1.65798641714771],
         [0.801453308495799]),
        ([0.201216973509685, 0.355537375331783, 0.0348018227498706, 0.334943969152602,
          1.80151763547293],
         [0.672724244998741]),
        ([0.0554638619600905, 0.148283578626342, 0.0390891137081371, 0.182680934617247,
          1.89140478436634],
         [0.604567031948086]),
        ([0.0166213700825164, 0.0208901149237555, 0.369637421750437, 0.216093986500306,
          0.904411354746914],
         [-0.0568217094938274]),
        ([0.107590018443692, 0.19010460252016, 0.0186084140230469, 0.179093379599993,
          0.963265236756179],
         [-0.168514473512358]),
    ],
    "C2-ConvexDTLZ2_5D": [
        ([8.62039419848259e-07, 2.1509145888177e-06, 0.21084513170011, 0.0246280810512119,
          2.74891895944631],
         [-5.76966973331538]),
        ([0.00163930004228304, 0.0159786854717592, 1.46692517902439e-06, 0.0125860267358389,
          3.24546579091998],
         [-8.33683515985183]),
        ([9.4632524362562e-06, 0.000483473009898146, 2.33465812654689e-06, 0.00111371200065977,
          3.57741205832389],
         [-10.1853751865018]),
        ([7.63250816372747e-08, 1.90442255662988e-07, 0.0186682552088692, 0.00218057347903423,
          0.817959898595148],
         [-0.478066548663102]),
        ([0.000133994794765356, 0.00130608224558252, 1.19905040706128e-07, 0.00102876961256665,
          0.927879916342938],
         [-0.637229112541956]),
    ],
    "C3-DTLZ1_5D": [
        ([36.1218374535393, 27.0132218586119, 3.03717546512012, 130.729104259922,
          77.3215280167352],
         [-309.344704507468, -300.23608891254, -276.260042519049, -403.95197131385,
          -350.544395070664]),
        ([3.54171545726692, 1.72762838741242, 92.1271873706594, 125.955185124205,
          49.6830467768852],
         [-275.576478573696, -273.762391503842, -364.161950487089, -397.989948240635,
          -321.717809893315]),
        ([16.1633356690998, 4.76994112552867, 114.907149541453, 117.561067574421,
          22.6247720184261],
         [-291.189601598029, -279.796207054458, -389.933415470382, -392.58733350335,
          -297.651037947355]),
        ([6.35843485552335, 4.75506849967716, 0.53462624553912, 23.0119105714592,
          13.6107112340644],
         [-53.6291862617866, -52.0258199059404, -47.8053776518023, -70.2826619777224,
          -60.8814626403276]),
        ([2.24858300166456, 1.09684582852584, 58.4901949386664, 79.9670926868747,
          31.5430349505318],
         [-174.594334407928, -173.442597234789, -230.83594634493, -252.312844093138,
          -203.888786356795]),
    ],
    "C3-DTLZ4_5D": [
        ([1.43031003708889, 1.26320600690361e-24, 0.02046721678758, 9.86978649937e-48,
          9.24899971033251e-15],
         [0.488134392487665, -1.04620570916025, -1.04589152893798, -1.04620570916025,
          -1.04620570916025]),
        ([1.40669439005996, 1.23036065344279e-17, 4.63046436309942e-127, 1.99406444794389e-36,
          4.18267866251128e-09],
         [0.505302723243461, -0.978789107026155, -0.978789107026155, -0.978789107026155,
          -0.978789107026155]),
        ([1.46652437881813, 1.35452422387409e-11, 1.39067988622334e-81, 1.92484457334673e-27,
          0.000444923385777343],
         [0.462326363626204, -1.15069395162473, -1.15069395162473, -1.15069395162473,
          -1.15069380315711]),
        ([1.000397581688, 8.83520496753379e-25, 0.0143153258015671, 6.90319601325218e-48,
          6.46900091820764e-15],
         [0.749596241085396, -0.00100024999999992, -0.000846553585396204, -0.00100024999999992,
          -0.00100024999999992]),
        ([1.002, 8.76396027069628e-18, 3.29831790374017e-127, 1.4203885299881e-36,
          2.97935646111283e-09],
         [0.748999, -0.0040039999999999, -0.0040039999999999, -0.0040039999999999,
          -0.0040039999999999]),
    ],
}


def c_dtlz_fixed_points(name: str, M: int) -> list:
    """Five points of a C-DTLZ problem: x_i = (0.1 j + 0.61803398875 i) mod 1,
    i = 1..n, for j = 1, 2, 3, and the first two again with the distance
    variables at 0.5 + 0.01 j, near the front."""
    import numpy as np
    from mootation.benchmarks import c_dtlz
    i = np.arange(1, c_dtlz.n_vars(name, M) + 1)
    X = [(0.1 * j + 0.61803398875 * i) % 1.0 for j in (1, 2, 3)]
    for j in (1, 2):
        x = X[j - 1].copy()
        x[M - 1:] = 0.5 + 0.01 * j
        X.append(x)
    return [list(x) for x in X]


@test
def c_dtlz_matches_pymoo():
    """C1-DTLZ1, C1-DTLZ3, C2-DTLZ2, C2-ConvexDTLZ2, C3-DTLZ1 and C3-DTLZ4 at
    three and five objectives give pymoo's objectives and constraints at five
    points each, to 1e-12 relative (task 6, item 5.3)."""
    if not _have_numpy():
        print("    (skipped: no NumPy)"); return
    from mootation.benchmarks import get, names
    assert sorted(C_DTLZ_PYMOO) == sorted(n for n in names() if n[:3] in ("C1-", "C2-", "C3-"))
    for key, rows in C_DTLZ_PYMOO.items():
        name, M = key.split("_")[0], int(key.split("_")[1][:-1])
        p = get(key)
        assert p.has_cons and p.n_obj == M, key
        for x, (f, g) in zip(c_dtlz_fixed_points(name, M), rows):
            for mine, theirs in ((p.evaluate(x), f), (p.constraints(x), g)):
                assert len(mine) == len(theirs), key
                for a, b in zip(mine, theirs):
                    assert abs(a - b) <= 1e-12 * max(1.0, abs(b)), (key, a, b)


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
    from mootation.benchmarks import c_dtlz, get
    from mootation.benchmarks.registry import PROBLEMS, _strictly_dominated_by
    for key in C_DTLZ_PYMOO:
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
    approximate = {n for n in C_DTLZ_PYMOO if not PROBLEMS[n].front_exact}
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


MW_PYMOO = {
    'MW1': [
        ([0.71803398875, 13.214831471811495],
         [12.743616327026453]),
        ([0.8180339887500001, 13.09481618133752],
         [12.855731607265628]),
        ([0.9180339887500001, 13.814877542974049],
         [13.242568908058608]),
        ([0.71803398875, 0.38967110956249995],
         [0.10770215142474622]),
        ([0.8180339887500001, 0.3046711095624999],
         [-0.33368103776665164]),
    ],
    'MW2': [
        ([0.71803398875, 14.920726860505866],
         [14.601710630714738]),
        ([0.8180339887500001, 13.764880051111463],
         [13.582911114184038]),
        ([0.9180339887500001, 13.82280391221085],
         [13.709824366193962]),
        ([0.71803398875, 0.28196601125],
         [-0.000898742608043089]),
        ([0.8180339887500001, 0.1819660112499999],
         [-0.09410664910208602]),
    ],
    'MW3': [
        ([0.71803398875, 8.094089269929531],
         [7.7525333686682245, -7.878949123827623]),
        ([0.8180339887500001, 7.138204286564312],
         [6.7861287410245765, -6.913081663930229]),
        ([0.9180339887500001, 8.156576442859906],
         [7.856933955345026, -8.008732724659035]),
        ([0.71803398875, 0.28196601125],
         [-0.4816253562271867, 0.14585987315874788]),
        ([0.8180339887500001, 0.1819660112499999],
         [-0.22360051086835128, 0.06839067121440906]),
    ],
    'MW4_3D': [
        ([2.4074764930215173, 1.2186123405254712, 9.233932193356084],
         [11.860021026820545]),
        ([1.3129237797123703, 1.0152358695516868, 10.466315721553626],
         [11.644848895619123]),
        ([0.517012746240339, 0.5974021274178792, 12.481645940612763],
         [12.290075651220402]),
        ([0.1872062641254702, 0.0947597471245298, 0.71803398875],
         [-1.489914148655162e-05]),
        ([0.10261646075047019, 0.0793495504995297, 0.8180339887500001],
         [-0.2889037178347636]),
    ],
    'MW4_5D': [
        ([0.04739098936795808, 0.06337080499585229, 2.302452570484893, 1.2215167266896687, 9.255939934465909],
         [11.89067079753233]),
        ([0.41069716640932624, 0.8419475583555581, 0.07164677396970069, 1.0240261102988426, 10.556936470571117],
         [11.512524415114656]),
        ([0.09930797108030845, 0.33651318306583444, 0.0793959722184599, 0.5953273098401709, 12.438296348758698],
         [12.446548287452234]),
        ([0.003676378775965696, 0.004916020653076532, 0.17861386469642798, 0.0947597471245298, 0.71803398875],
         [-1.489914148655162e-05]),
        ([0.03182402794055731, 0.06524068051369647, 0.005551752296216407, 0.0793495504995297, 0.8180339887500001],
         [-0.2889037178347636]),
    ],
    'MW5': [
        ([9.926935040014042, 9.62242275901668],
         [188.8847679470211, -190.0403631504468, -190.21662325034885]),
        ([11.280807378505276, 7.93167611477308],
         [187.8825721563648, -189.91745381954496, -188.0672890731219]),
        ([13.398895578694468, 5.787024114788165],
         [210.60389400344525, -212.1280608171511, -211.9175655505448]),
        ([0.71803398875, 0.6960080394648935],
         [-1.2502910948796733, 0.0946958914540002, -0.08156420844805529]),
        ([0.8180339887500001, 0.5751698820781254],
         [-1.2855289442360522, -0.7493527189440884, 1.1008120274789635]),
    ],
    'MW6': [
        ([11.302920598355175, 10.956199468473345],
         [246.52908323706487]),
        ([11.357296508785744, 7.985456574569011],
         [189.58652919295434]),
        ([16.19014304284512, 6.992572459464124],
         [221.92832259142097]),
        ([0.7898373876250001, 0.7656088434113829],
         [0.2087048229945303]),
        ([0.8998373876250002, 0.632686870285938],
         [0.19638329499265295]),
    ],
    'MW7': [
        ([6.327404012786312, 6.133308632796529],
         [76.21351632616076, -76.33101632626463]),
        ([6.508473331800788, 4.576188630598],
         [61.86102537267868, -61.991637707042955]),
        ([8.330800810883202, 3.5980984331826047],
         [79.82480394647712, -81.44060740484436]),
        ([0.71803398875, 0.6960080394648935],
         [-0.43999999999999995, 0.3224999998961307]),
        ([0.8180339887500001, 0.5751698820781254],
         [-0.44070212089776883, 0.3100897865334946]),
    ],
    'MW8_3D': [
        ([4.681411400606608, 2.7296933938362153, 11.4244997759788],
         [158.58258852304203]),
        ([2.84463371222378, 2.3238418505834115, 12.498932542262645],
         [169.13740408895842]),
        ([1.230286073531258, 1.3782311064112025, 14.269649507160745],
         [206.02309507462743]),
        ([0.37022994598928705, 0.2158781083919141, 0.9035078469000136],
         [-0.3034453343802277]),
        ([0.21835623782387528, 0.17837986015935536, 0.9594275266495808],
         [0.4219074896832845]),
    ],
    'MW8_5D': [
        ([0.23481972460116143, 0.295126756033352, 5.222082003277305, 3.052884939451789, 12.777143171321576],
         [198.68442929374217]),
        ([1.6737812117114785, 2.9574631230744863, 0.2894916668113129, 2.7861611909032566, 14.98554678676917],
         [243.38315126400357]),
        ([0.5287627328823229, 1.4136561628624746, 0.3726546594453679, 1.741582118879825, 18.031639475229493],
         [329.57708892315725]),
        ([0.016604765317199233, 0.02086924567807744, 0.3692681535968397, 0.2158781083919141, 0.9035078469000136],
         [-0.3034453343802277]),
        ([0.10716137295188428, 0.189347213665498, 0.018534276915385315, 0.17837986015935536, 0.9594275266495808],
         [0.4219074896832845]),
    ],
    'MW9': [
        ([9.926935040014042, 2.4918396422855285],
         [2386.6352271564665]),
        ([11.280807378505276, 1.565629891754707],
         [3803.4543816240634]),
        ([13.398895578694468, 0.7300268708210166],
         [7377.665032993134]),
        ([0.71803398875, 0.18023947476875202],
         [0.19499677833779422]),
        ([0.8180339887500001, 0.11353251786735441],
         [0.09370647090707551]),
    ],
    'MW10': [
        ([0.10873700699655219, 15.638004795959537],
         [-187.9346745708994, 188.9021876370522, 211.7221534725455]),
        ([0.7168273152158293, 14.54767818792988],
         [-243.26348169976345, 281.9485808486947, 196.35910738143915]),
        ([4.086990035798998, 13.607694204181474],
         [-11389.61415816595, 12552.948012097924, 951.2029315611368]),
        ([0.006953044940368545, 0.9999516551660572],
         [-0.9995166007422546, 0.99922651771526, 0.0]),
        ([0.0491552863341598, 0.9975837578254068],
         [-0.9759601810052395, 0.9614276986002024, 0.0]),
    ],
    'MW11': [
        ([6.634571538607728, 6.431053684958233],
         [-4339.939320138415, 9640.103559079847, -137.63448445036977, 128.66062640287947]),
        ([7.694093677897531, 5.409813057005904],
         [-7442.844888359077, 16425.90202386341, -168.79301909553226, 151.30751771089382]),
        ([9.661293026068458, 4.172742103520482],
         [-17754.69746426834, 38953.07346433702, -284.80089802791935, 237.51590336924264]),
        ([1.0154534051351005, 0.9843040089319607],
         [0.0458753657818632, -7.133581721568655, -0.005312314620440676, -0.3618857058392573]),
        ([1.1568747613724102, 0.8134130479034185],
         [0.4157431125729019, -9.696409744459077, -0.0816234580332177, -0.26290088211100365]),
    ],
    'MW12': [
        ([9.926935040014042, 2.9197577317355434],
         [119.85320432386361, -84.1558877206173]),
        ([11.280807378505276, 1.6683404972947211],
         [121.47205850955179, -78.32687616264249]),
        ([13.398895578694468, 1.4597694412200057],
         [164.557729705835, -104.22979685950831]),
        ([0.71803398875, 0.21119159960764278],
         [0.09592457249788622, -0.12600903533691502]),
        ([0.8180339887500001, 0.12098063425811212],
         [0.1374585422943775, -0.1669432390486059]),
    ],
    'MW13': [
        ([13.119270668108797, 21.09677798522716],
         [10913313.034142453, -2991.6235522750226]),
        ([15.854995248451887, 15.087514793077155],
         [130211173.61212951, -3300.9618909869605]),
        ([16.773277923328017, 10.18988750590217],
         [256499344.65840703, -3010.964868724286]),
        ([1.077050983125, 1.7319793184078192],
         [1.4404102626308757, -1.7408005379080078]),
        ([1.2270509831250003, 1.1676540780777576],
         [2.327261319639106, -3.0499043475966925]),
    ],
    'MW14_3D': [
        ([1.077050983125, 0.5041019662500003, 26.383313418832614],
         [22.432711737395834]),
        ([1.2270509831250003, 0.65410196625, 25.282511569306422],
         [21.6917997697786]),
        ([1.3770509831250002, 0.8041019662500002, 16.168127124799234],
         [13.585181064266797]),
        ([1.077050983125, 0.5041019662500003, 3.6989642628867916],
         [-0.25163741854998767]),
        ([1.2270509831250003, 0.65410196625, 3.24737935596801],
         [-0.3433324435598104]),
    ],
    'MW14_5D': [
        ([1.077050983125, 0.5041019662500003, 1.4311529493750001, 0.8582039325000004, 21.10532915902515],
         [17.978744451635123]),
        ([1.2270509831250003, 0.65410196625, 0.08115294937500028, 1.0082039325000005, 28.659431827705614],
         [24.586843647869337]),
        ([1.3770509831250002, 0.8041019662500002, 0.23115294937499975, 1.1582039325000006, 24.27148321570986],
         [20.646660622521367]),
        ([1.077050983125, 0.5041019662500003, 1.4311529493750001, 0.8582039325000004, 2.735584336395335],
         [-0.39100037099469054]),
        ([1.2270509831250003, 0.65410196625, 0.08115294937500028, 1.0082039325000005, 3.7948410700682453],
         [-0.27774710976802996]),
    ],
}


def mw_fixed_points(name: str, m: int) -> list:
    """Five points of an MW problem: x_i = top ((0.1 j + 0.61803398875 i) mod
    1), i = 1..n, for j = 1, 2, 3, top the box's bound, and the first two
    again with the distance variables at g's optimum (mw.optimal_distance)."""
    import numpy as np
    from mootation.benchmarks import mw
    n, top = mw.n_vars(name, m), mw.upper(name)
    i = np.arange(1, n + 1)
    X = [top * ((0.1 * j + 0.61803398875 * i) % 1.0) for j in (1, 2, 3)]
    for j in (1, 2):
        x = X[j - 1].copy()
        x[m - 1:] = mw.optimal_distance(name, x[:m - 1])
        X.append(x)
    return [list(x) for x in X]


@test
def mw_matches_pymoo():
    """MW1-14 (MW4, MW8 and MW14 at three and five objectives) give pymoo's
    objectives and constraints at five points each, to 1e-12 relative (the
    literal has pymoo's MW12 and MW13 constraints in the paper's order:
    TASK6 mw_pymoo_values.py) (task 6, item 5)."""
    if not _have_numpy():
        print("    (skipped: no NumPy)"); return
    from mootation.benchmarks import get, names
    assert sorted(MW_PYMOO) == sorted(n for n in names() if n.startswith("MW"))
    for key, rows in MW_PYMOO.items():
        name = key.split("_")[0]
        p = get(key)
        assert p.has_cons, key
        for x, (f, g) in zip(mw_fixed_points(name, p.n_obj), rows):
            for mine, theirs in ((p.evaluate(x), f), (p.constraints(x), g)):
                assert len(mine) == len(theirs), key
                for a, b in zip(mine, theirs):
                    assert abs(a - b) <= 1e-12 * max(1.0, abs(b)), (key, a, b)


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
    from mootation.benchmarks import get, mw
    from mootation.benchmarks.registry import _strictly_dominated_by
    for key in MW_PYMOO:
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
    want = sorted(f"{b}_{lv}" for b in noisy.BASES for lv in ("n1", "n5"))
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
        assert p.evaluate(x[:p.n_vars]) == p.evaluate(x[:p.n_vars]) != q.evaluate(x[:p.n_vars])


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


# ── task 6, item 1: rotated DTLZ and WFG ────────────────────────────────────

# sha256 (16 hex digits) of every rotated problem's matrix (Q, Lambda Q, I or
# Lambda), rounded to 1e-12, and of every rotation of the registered
# bbob-biobj problems: numpy does not promise its normal stream across
# versions, and a changed draw must fail here, not move results silently.
ROTATION_HASHES = {
    "c0DTLZ2_3D": "1e24edff38434edf", "c0DTLZ2_5D": "1e24edff38434edf",
    "r0DTLZ1_3D": "1a8475b42f5782e9", "r0DTLZ1_5D": "1a8475b42f5782e9",
    "r0DTLZ3_3D": "a3c81aeea2e1ab58", "r0DTLZ3_5D": "a3c81aeea2e1ab58",
    "rDTLZ1_3D": "0b3008ed21263fbc", "rDTLZ1_5D": "ef5a865bfb3c0415",
    "rDTLZ2_3D": "9d2a751088421ed6", "rDTLZ2_5D": "219ddeb235a454a0",
    "rDTLZ3_3D": "ef85240a1c1c76ee", "rDTLZ3_5D": "1f7ffcdb60f9914f",
    "rDTLZ4_3D": "4767919b46de0568", "rDTLZ4_5D": "6134b078d8446b73",
    "rWFG4_3D": "f52d69d1a9090e8f", "rWFG4_5D": "836e2efbfecb0621",
    "rWFG5_3D": "786c215ca3c4a44c", "rWFG5_5D": "db857c63767a80d8",
    "rWFG6_3D": "8bd9224b78e079c3", "rWFG6_5D": "c19dac79a30b4db7",
    "rWFG7_3D": "bf02effac24cdcde", "rWFG7_5D": "45d4935de1c51b2e",
    "rcDTLZ2_3D": "f5494f9446406f34", "rcDTLZ2_5D": "26423ce721190dfa",
}
BBOB_ROTATION_HASH = "b20703a7d9b960f5"


def _matrix_hash(A, digest=None) -> str:
    import hashlib
    import numpy as np
    d = digest or hashlib.sha256()
    d.update((np.round(np.asarray(A, float), 12) + 0.0).tobytes())
    return d.hexdigest()[:16]


@test
def the_rotations_are_rotations_and_numpy_still_draws_them():
    """Every Q of rotated.py is orthogonal (1e-12), the conditioned ones are
    Lambda Q with the stated Lambda, the controls I or Lambda; the hash of every
    matrix, and of bbob-biobj's rotations, is the one recorded (task 6, item
    1.3)."""
    if not _have_numpy():
        print("    (skipped: no NumPy)"); return
    import hashlib
    import numpy as np
    from mootation.benchmarks import bbob_biobj as BB
    from mootation.benchmarks import names
    from mootation.benchmarks import rotated as R
    from mootation.benchmarks.registry import ROTATED
    assert sorted(ROTATED) == sorted(ROTATION_HASHES) == sorted(
        n for n in names() if n.split("_")[0] in R.SPECS)
    for name, r in ROTATED.items():
        family, rotate, condition = R.SPECS[name.split("_")[0]]
        k = len(r.optimum)
        Q = R.rotation(family, r.M, k) if rotate else np.eye(k)
        assert np.allclose(Q.T @ Q, np.eye(k), atol=1e-12), name
        L = R.conditioning(k) if condition else np.ones(k)
        assert np.array_equal(r.A, L[:, None] * Q), name
        assert _matrix_hash(r.A) == ROTATION_HASHES[name], (name, _matrix_hash(r.A))
    assert R.conditioning(10)[[0, -1]].tolist() == [1.0, 1e3 ** 0.5]   # cond(Lambda^2) = 1e3
    d = hashlib.sha256()
    for n in sorted(names()):
        if n.startswith("bbobbiobj"):
            p = BB.problem(int(n[9:11]), int(n[13:15]), 1)
            for A in (p.alpha.R, p.alpha.Q, p.beta.R, p.beta.Q):
                if A is not None:
                    d.update((np.round(A, 12) + 0.0).tobytes())
    assert d.hexdigest()[:16] == BBOB_ROTATION_HASH, d.hexdigest()[:16]


@test
def a_rotated_problem_is_its_base_where_its_map_is_the_identity():
    """At Q = I and c = 1/2 rDTLZ is DTLZ, at Q = I rWFG is WFG; rDTLZ2 and
    rDTLZ4 are their Q = I construction (the isotropic g), rDTLZ1, rDTLZ3 and
    rcDTLZ2 are not; the Pareto set lands on the front; 10^4 uniform points
    give finite values; no point dominates the front; the properties
    (task 6, item 1.3 and supplement 2)."""
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

    for name, r in ROTATED.items():
        p, base = get(name), get(f"{r.family}_{r.M}D")
        ident = R.Rotated(name.split("_")[0], r.M, p.n_vars)
        ident.A = np.eye(len(r.optimum))
        if not r.wfg:
            ident.optimum = np.full(len(r.optimum), 0.5)
        X = box(p, 2000 if r.wfg else 10_000)
        assert max(np.abs(np.subtract(ident.evaluate(x), base.evaluate(list(x)))).max()
                   for x in X[:2000]) < 1e-12, name
        F = np.array([p.evaluate(list(x)) for x in box(p, 10_000)])
        assert np.all(np.isfinite(F)), name
        S = np.asarray(p.pareto_set(100), float)
        FS = np.array([p.evaluate(list(x)) for x in S])
        if r.wfg:                      # the base's own values at the same point
            assert np.abs(FS - [base.evaluate(list(x)) for x in S]).max() < 1e-9, name
        else:
            G = FS if r.family == "DTLZ1" else FS ** 2
            assert np.abs(G.sum(axis=1) - (0.5 if r.family == "DTLZ1" else 1.0)).max() < 1e-9
            assert max(p.distance_fn(list(x)) for x in S) < 1e-9, name
        row = properties(name)
        assert (row["centre"], row["boundary"], p.cyclic_vars) == (False, False, ()), name
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
    for name in ("rDTLZ1_3D", "rcDTLZ2_3D", "rWFG4_3D", "rWFG7_5D"):
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
    fam = families()
    assert sorted(fam["r0DTLZ"]) == ["r0DTLZ1_3D", "r0DTLZ1_5D", "r0DTLZ3_3D", "r0DTLZ3_5D"]
    assert len(fam["rDTLZ"]) == 8 and len(fam["rWFG"]) == 8 and len(fam["c0DTLZ"]) == 2


# ── task 6, item 6: the RE and CRE real-world problems ─────────────────────

# ryojitanabe/reproblems' Python version (MIT), commit 7876b4e, at the points of
# re_fixed_points: the objectives, and for CRE (objectives, violations); RE91's
# four normal draws fixed to (0.3, -1.2, 0.7, 0.1).
RE_VALUES = {
    "RE21": [
        [2296.19966267654, 0.0224006268779211],
        [2363.81993589829, 0.010694494579839],
        [2541.29537671572, 0.0103509537974754],
        [1918.18048446298, 0.0227004216064676],
        [2094.55106976404, 0.0196141259265501],
    ],
    "RE22": [
        [477.308696700677, 1.67802962571761],
        [364.124224800677, 281.745089042506],
        [451.252382100677, 235.10847278545],
        [90.5165394006772, 175.64552048523],
        [183.908696700677, 153.773550758615],
    ],
    "RE23": [
        [300027.379122309, 0],
        [36913.0018192696, 1048415.51346437],
        [81031.7631339557, 132879.65741633],
        [25653.5270098798, 0.93729191023875],
        [59742.035659015, 0.67899191023875],
    ],
    "RE24": [
        [2059.25690531063, 0],
        [2653.60690531063, 0],
        [3247.95690531063, 0],
        [3838.80690531063, 0],
        [4433.15690531063, 0],
    ],
    "RE25": [
        [45.9848597660687, 15.5915205646927],
        [2.64270541433649, 3762993.35880804],
        [6.72943289314108, 1460272.86003018],
        [0.773714054613419, 675532.331472527],
        [3.82410835115629, 287947.696876453],
    ],
    "RE31": [
        [458.453388226319, 0.473660842035115, 458.353388226319],
        [404.63078837806, 0.915709760793062, 404.53078837806],
        [474.624478403442, 0.700845350202228, 474.524478403442],
        [122.812713947747, 31.434072279866, 122.712713947747],
        [197.034781116135, 4.3143962461802, 196.934781116135],
    ],
    "RE32": [
        [73.0834059056722, 0.000866062630698507, 0.711252914531249],
        [84.4591675474363, 2.51311155360689, 336740.951825653],
        [132.319695471005, 0.131392118383155, 19039.7531525377],
        [11.5539710451005, 0.0280293035610294, 6602.68689531989],
        [22.0491216577812, 0.0096278638027483, 0],
    ],
    "RE33": [
        [1.63745404625136, 1.74121173130994, 6.20833828176535],
        [1.93022170549791, 4.17074637362964, 5.18847050625],
        [2.25488047641816, 3.23816992070285, 4.18847050625],
        [5.58454923875761, 2.94239916699888, 0],
        [6.24242365117445, 2.38748090528687, 0],
    ],
    "RE34": [
        [1685.90227199309, 9.17707521215899, 0.150723807700163],
        [1681.04962289309, 10.1512522609983, 0.101379808517313],
        [1685.33472739309, 10.4337505434367, 0.103990603046463],
        [1684.90517489309, 10.9881277119835, 0.100638582518613],
        [1689.19027939309, 11.4407059944219, 0.100297377047763],
    ],
    "RE35": [
        [5393.74023602842, 808.306475716548, 0.621355728680025],
        [3479.61803219421, 748.683057127226, 0.502699935963833],
        [3600.72287105523, 1680.05522243232, 380.441221329506],
        [3346.75389973403, 1517.23889799609, 218.819648324174],
        [3718.7468016905, 1374.85217637395, 76.3387196713279],
    ],
    "RE36": [
        [5.17478881987578, 58, 0.246615036773305],
        [6.5388431372549, 51, 0.443419872638133],
        [6.4935, 56, 0.436877795411918],
        [4.61257423971377, 54, 0.165499096770131],
        [4.90854137115839, 59, 0.208201034649891],
    ],
    "RE37": [
        [0.624043399578703, 0.516744080634531, 0.591438973230051],
        [0.610428282464368, 0.231391524255244, 0.769115649696007],
        [0.540797977512683, 0.310692287475706, 0.772495311959053],
        [0.234307449628998, 0.575082868781669, 0.977299636747565],
        [0.205642744677313, 0.649445632002131, 0.975392573011686],
    ],
    "RE41": [
        [31.5889729782052, 3.97604251910976, 12.1567884813263, 1.51017522147498],
        [27.3281737782052, 4.04523823722413, 12.0895724654494, 0.222664362503699],
        [30.0473665782052, 3.96804633118601, 12.3446935959829, 0],
        [27.8665673782052, 3.88743442514788, 12.4563248602853, 1.93212405354312],
        [30.5857681782052, 3.80340251910976, 12.2319367070119, 0.59607405354312],
    ],
    "RE42": [
        [-844.039935810605, 12516.8326628499, 6286.87465719817, 8.01425205603963],
        [-710.446251977143, 11410.6263748209, 7696.70403186347, 7.9104790745108],
        [-874.697002123148, 12122.8298814322, 6031.1741758049, 7.34566802351604],
        [-522.869879527343, 5725.29421084565, 7928.31049405119, 1.45957136461411],
        [-481.327875089567, 6898.29057643774, 9913.60278142385, 0.93830782671353],
    ],
    "RE61": [
        [76239.1115560854, 977.80486515, 1148410.48601048, 4623455.37529136, 12488.9526343154, 0],
        [68550.9249160854, 1109.80486515, 1405222.69285524, 1449896.74314552, 1743.81213340406,
         0],
        [70472.9715760854, 1241.80486515, 1662034.8997, 1108316.66061568, 2389.1477836894, 0],
        [72395.0182360854, 53.8048651500001, 1918847.10654476, 847209.172656927, 30872.2649998094,
         0.234890599992375],
        [74317.0648960854, 185.80486515, 2175659.31338952, 647615.801277689, 10529.5458542863, 0],
    ],
    "RE91": [
        [31.5527390561583, 0.779424460463896, 23.7548090689652, 0.627077664697238,
         0.991051539019921, 0.996951984194581, 0.988246879476977, 0.981717476528617,
         0.949194557976996],
        [27.2866898561583, 0.794398394309973, 27.0282939855199, 0.528812260112356,
         1.04837678840925, 0.88590636907852, 1.00471630900557, 0.94948381741016,
         0.952811328264829],
        [30.0006326561583, 0.638030528156051, 30.3187875101449, 0.473624181980561,
         0.967492662798577, 0.841486741462459, 0.971533832496039, 0.914709552231096,
         0.998262894273729],
        [27.8145834561583, 0.550180062002129, 33.9008230996388, 0.643734877395679,
         0.898100412187905, 0.950735825731249, 0.950551355986508, 0.964158199693599,
         0.982314041963281],
        [30.5285342561583, 0.455638995848206, 37.2326076492638, 0.630573122810798,
         0.840200036577234, 0.921733460615188, 0.928713879476977, 0.948898681989283,
         0.964363406213343],
    ],
    "CRE21": [
        ([458.453388226319, 0.473660842035115],
         [458.353388226319, 0, 0]),
        ([404.63078837806, 0.915709760793062],
         [404.53078837806, 0, 0]),
        ([474.624478403442, 0.700845350202228],
         [474.524478403442, 0, 0]),
        ([122.812713947747, 31.434072279866],
         [122.712713947747, 0, 0]),
        ([197.034781116135, 4.3143962461802],
         [196.934781116135, 0, 0]),
    ],
    "CRE22": [
        ([73.0834059056722, 0.000866062630698507],
         [0, 0, 0.711252914531249, 0]),
        ([84.4591675474363, 2.51311155360689],
         [0, 336740.240572739, 0.711252914531249, 0]),
        ([132.319695471005, 0.131392118383155],
         [0, 19039.0418996232, 0.711252914531249, 0]),
        ([11.5539710451005, 0.0280293035610294],
         [6602.68689531989, 0, 0, 0]),
        ([22.0491216577812, 0.0096278638027483],
         [0, 0, 0, 0]),
    ],
    "CRE23": [
        ([1.63745404625136, 1.74121173130994],
         [6.18847050624998, 0.0198677755153691, 0, 0]),
        ([1.93022170549791, 4.17074637362964],
         [5.18847050625, 0, 0, 0]),
        ([2.25488047641816, 3.23816992070285],
         [4.18847050625, 0, 0, 0]),
        ([5.58454923875761, 2.94239916699888],
         [0, 0, 0, 0]),
        ([6.24242365117445, 2.38748090528687],
         [0, 0, 0, 0]),
    ],
    "CRE24": [
        ([5393.74023602842, 808.306475716548],
         [0, 0, 0, 0, 0, 0, 0.477094815742525, 0, 0.1442609129375, 0, 0]),
        ([3479.61803219421, 748.683057127226],
         [0, 0, 0, 0, 0, 0, 0.403439023026332, 0, 0.0992609129375008, 0, 0]),
        ([3600.72287105523, 1680.05522243232],
         [0, 0, 0, 0, 0, 0, 0.331737984246441, 0, 0.0542609129375009, 380.055222432322, 0]),
        ([3346.75389973403, 1517.23889799609],
         [0, 0, 0, 0, 0, 0, 1.57148941514907, 0, 0.0092609129374992, 217.238897996088, 0]),
        ([3718.7468016905, 1374.85217637395],
         [0, 0, 0, 0, 0, 0, 1.48654329737629, 0, 0, 74.8521763739516, 0]),
    ],
    "CRE25": [
        ([5.17478881987578, 58],
         [0.246615036773305]),
        ([6.5388431372549, 51],
         [0.443419872638133]),
        ([6.4935, 56],
         [0.436877795411918]),
        ([4.61257423971377, 54],
         [0.165499096770131]),
        ([4.90854137115839, 59],
         [0.208201034649891]),
    ],
    "CRE31": [
        ([31.5889729782052, 3.97604251910976, 12.1567884813263],
         [0, 0, 0, 0.0204011679318589, 0, 0, 1.48977405354312, 0, 0, 0]),
        ([27.3281737782052, 4.04523823722413, 12.0895724654494],
         [0, 0, 0, 0.0237020717364438, 0, 0, 0.153724053543122, 0.0452382372241331, 0, 0]),
        ([30.0473665782052, 3.96804633118601, 12.3446935959829],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        ([27.8665673782052, 3.88743442514788, 12.4563248602853],
         [0, 0, 0, 0, 0, 0, 1.93212405354312, 0, 0, 0]),
        ([30.5857681782052, 3.80340251910976, 12.2319367070119],
         [0, 0, 0, 0, 0, 0, 0.59607405354312, 0, 0, 0]),
    ],
    "CRE32": [
        ([-844.039935810605, 12516.8326628499, 6286.87465719817],
         [0, 0, 2.79434353659295, 0, 0, 0, 0, 0, 5.21990851944668]),
        ([-710.446251977143, 11410.6263748209, 7696.70403186347],
         [0, 3.44046173978733, 3.57512136817347, 0, 0.894895966550003, 0, 0, 0, 0]),
        ([-874.697002123148, 12122.8298814322, 6031.1741758049],
         [0, 2.78746099360625, 4.33231106335979, 0, 0.225895966550004, 0, 0, 0, 0]),
        ([-522.869879527343, 5725.29421084565, 7928.31049405119],
         [0.52957229715341, 0, 0, 0.929999067460704, 0, 0, 0, 0, 0]),
        ([-481.327875089567, 6898.29057643774, 9913.60278142385],
         [0.333505261448607, 0, 0, 0.604802565264922, 0, 0, 0, 0, 0]),
    ],
    "CRE51": [
        ([76239.1115560854, 977.80486515, 1148410.48601048, 4623455.37529136, 12488.9526343154],
         [0, 0, 0, 0, 0, 0, 0]),
        ([68550.9249160854, 1109.80486515, 1405222.69285524, 1449896.74314552, 1743.81213340406],
         [0, 0, 0, 0, 0, 0, 0]),
        ([70472.9715760854, 1241.80486515, 1662034.8997, 1108316.66061568, 2389.1477836894],
         [0, 0, 0, 0, 0, 0, 0]),
        ([72395.0182360854, 53.8048651500001, 1918847.10654476, 847209.172656927,
          30872.2649998094],
         [0.234890599992375, 0, 0, 0, 0, 0, 0]),
        ([74317.0648960854, 185.80486515, 2175659.31338952, 647615.801277689, 10529.5458542863],
         [0, 0, 0, 0, 0, 0, 0]),
    ],
}


def re_fixed_points(lb, ub) -> list:
    """Five points of a box: lb + (ub - lb) u, u_i = (0.1 j + 0.61803398875 i)
    mod 1, i = 1..n, j = 1..5."""
    import numpy as np
    lb, ub = np.asarray(lb, float), np.asarray(ub, float)
    i = np.arange(1, len(lb) + 1)
    return [lb + (ub - lb) * ((0.1 * j + 0.61803398875 * i) % 1.0) for j in range(1, 6)]


@test
def re_problems_match_reproblems():
    """RE21-91 and CRE21-51 give the objectives (and the CRE their constraint
    violations) of ryojitanabe/reproblems' Python version at five points each,
    to 1e-12 relative; RE91 with its four normal draws fixed (task 6, item 6)."""
    if not _have_numpy():
        print("    (skipped: no NumPy)"); return
    import numpy as np
    from mootation.benchmarks import get, names
    from mootation.benchmarks import re as RE
    assert sorted(RE_VALUES) == sorted(n for n in names() if n[:2] == "RE" or n[:3] == "CRE")
    for name, rows in RE_VALUES.items():
        p = get(name)
        lb = [b[0] for b in p.bounds]
        ub = [b[1] for b in p.bounds]
        for x, want in zip(re_fixed_points(lb, ub), rows):
            if name == "RE91":
                pairs = [(RE.re91(x, [0.3, -1.2, 0.7, 0.1]).tolist(), want)]
            elif p.has_cons:
                pairs = [(p.evaluate(list(x)), want[0]), (p.constraints(list(x)), want[1])]
            else:
                pairs = [(p.evaluate(list(x)), want)]
            for mine, theirs in pairs:
                assert len(mine) == len(theirs), name
                for a, b in zip(mine, theirs):
                    assert abs(a - b) <= 1e-12 * max(1.0, abs(b)), (name, a, b)


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
    for name in RE_VALUES:
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
                                       "pcx", "de", "de_pbest", "portfolio"}, words
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
    if not _have_numpy() or _core_with("on_generation") is None:
        print("  skip  small_budgets...: no NumPy or stale _core"); return
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

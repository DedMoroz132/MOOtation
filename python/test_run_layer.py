#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Tests for the config / parser / ledger layer.

Plain asserts and a hand-rolled runner: this package has no dependencies, and
requiring pytest to test a zero-dependency package would defeat the point.

    python python/test_mootation_run.py
"""

from __future__ import annotations

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
    assert len(names) == 59, f"expected 59 algorithms, got {len(names)}"
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
    for name in ("random_search", "sobol_search"):
        if name == "sobol_search":
            try:
                import scipy  # noqa: F401
            except ImportError:
                continue
        runs = []
        for _ in range(2):
            arc = GridArchive(2, p.n_vars, ideal=p.ideal, nadir=p.nadir)
            spent = [0]

            def ev(x, arc=arc, spent=spent):
                spent[0] += 1
                f = p.evaluate(x)
                arc.add(f, x)
                return f
            res = run_baseline(name, ev, p.bounds, pop=100, max_evaluations=1234, seed=7,
                               archive=arc)
            assert spent[0] == 1234, (name, spent[0])
            assert len(res.objectives) == min(100, len(arc)), (name, len(res.objectives))
            runs.append(np.asarray(res.objectives))
        assert np.array_equal(runs[0], runs[1]), name                   # a seed is a seed


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
    """properties.py covers campaign_all's problem list; the groups read key=value."""
    if not _have_numpy():
        print("    (skipped: no NumPy)"); return
    from mootation.benchmarks.properties import KEYS, label, properties
    from mootation.run.config import load
    cfg = load(HERE / "examples" / "campaign_all.toml")
    assert validate(cfg) == [], validate(cfg)[:3]
    names = cfg.benchmark_problems
    assert len(names) > 100, len(names)
    missing = [n for n in names if properties(n) is None]
    assert not missing, missing[:10]
    assert properties("DTLZ5_3D")["front"] == "degenerate"
    assert properties("DTLZ5_5D")["front"] == "degenerate+mixed"
    assert properties("WFG3_3D")["front"] == "degenerate+mixed"
    assert properties("shiftDTLZ2_3D")["centre"] is False and properties("DTLZ2_3D")["centre"]
    assert properties("SDTLZ1_3D")["scaled"] is True
    assert label("WFG5_3D", "deceptive") == "deceptive=yes"
    assert label("ZCAT1_3D", "front") == "front=?"
    assert set(KEYS) == set(properties("ZDT1")), properties("ZDT1")


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
    assert len(PP.BUDGET_SCHEDULED) == 16 and PP.budget_note("rvea") == "*"
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
problems = ["ZDT1", "DTLZ2_3D"]
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
        assert json.loads((root / "ZDT1" / "gsemo" / "run_1" / "meta.json").read_text(
            encoding="utf-8"))["final"]["igdx"] is None             # ZDT1 has no Pareto set
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
        code, txt = cli("--ranks", "igdp", "--by", "front")
        assert code == 0 and "grouped by front" in txt, txt
        code, txt = cli("--compare", "igdx")
        assert code == 0 and "DTLZ2_3D" in txt and "ZDT1" not in txt, txt
        for argv in (("--gap", "igdp"), ("--zero-share",),
                     ("--ecdf", "igdp_norm", "--interpolation", "linear")):
            code, txt = cli(*argv)
            assert code == 0 and txt.strip(), (argv, txt)
        assert cli("--ecdf", "hv", "--scenario", "archive")[0] == 1
        assert cli("--ci")[0] == 1 and cli("--gap", "nonsense")[0] == 1
        assert cli("--ranks", "igdp", "--reference", "nobody")[0] == 1


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
        for alg, maker in (("flat", lambda: rng.random((100, 2))),
                           ("centre", lambda: 0.45 + 0.1 * rng.random((100, 2))),
                           ("edges", lambda: np.where(rng.random((100, 2)) < 0.5, 0.0, 1.0))):
            for seed in (1, 2, 3):
                d = Path(td) / alg / str(seed)
                d.mkdir(parents=True)
                X = maker()
                F = rng.random((100, 2))
                with (d / "final.csv").open("w", encoding="utf-8") as fh:
                    fh.write("# final\nf1,f2,x1,x2\n")
                    for f, x in zip(F, X):
                        fh.write(",".join(f"{v:.10g}" for v in (*f, *x)) + "\n")
                rows.append({"problem": "uninformative_n02_2D", "algorithm": alg,
                             "status": "done", "dir": d})
        rep = R.structural_bias(rows)
        flat, centre, edges = (rep[("uninformative_n02_2D", a)] for a in ("flat", "centre", "edges"))
        assert abs(flat["edge"] - 0.2) < 0.05 and abs(flat["centre"] - 0.2) < 0.05, flat
        assert flat["p_min"] > 1e-3 and flat["runs"] == 3 and flat["points"] == 300, flat
        assert centre["centre"] == 1.0 and centre["edge"] == 0.0 and centre["p_min"] < 1e-12, centre
        assert edges["edge"] == 1.0 and edges["centre"] == 0.0, edges
        text = R.format_bias(rep)
        assert text.index("edges") < text.index("flat") and "centre" in text


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


# ── task 2, step 3: operators and switchable variation ──────────────────────

@test
def operator_words_are_knobs_of_the_run_layer():
    from mootation.run.knobs import text_knobs
    words = text_knobs()
    assert set(words["crossover"]) == {"sbx", "uniform", "blx_alpha"}, words
    assert {"polynomial", "gaussian", "cauchy", "uniform_reset", "mixture",
            "mixture_cauchy"} == set(words["mutation"]), words
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
    """nsga2, ibea_eplus, spea2_sde, agemoea take crossover/mutation; the defaults are SBX + PM."""
    if not _have_numpy() or _core_with("operator_stats") is None:
        print("  skip  switchable_operators...: no NumPy or stale _core"); return
    import numpy as np
    from mootation import minimize
    from mootation.benchmarks import get
    p = get("ZDT4")                         # x_1 in [0, 1], the rest in [-5, 5]
    lo = np.array([b[0] for b in p.bounds]); hi = np.array([b[1] for b in p.bounds])
    kw = dict(bounds=p.bounds, n_objs=2, pop_size=100, max_evaluations=1500, seed=4)
    for alg in ("nsga2", "ibea_eplus", "spea2_sde", "agemoea"):
        plain = minimize(p.evaluate, algorithm=alg, **kw)
        same = minimize(p.evaluate, algorithm=alg, crossover="sbx", mutation="polynomial", **kw)
        assert np.array_equal(np.array(plain.objectives), np.array(same.objectives)), alg
        assert dict(plain.operators) == {"sbx": "none", "polynomial": "none"}, plain.operators
        for extra in ({"crossover": "blx_alpha", "blx_alpha": 1.0, "bound_repair": "reflect"},
                      {"crossover": "uniform", "mutation": "gaussian", "mutation_scale": 0.3},
                      {"mutation": "mixture_cauchy", "mixture_q": 0.5, "bound_repair": "random"},
                      {"mutation": "uniform_reset"}):
            r = minimize(p.evaluate, algorithm=alg, **kw, **extra)
            assert not r.ignored, (alg, extra, r.ignored)
            X = np.array(r.variables)
            assert np.all((X >= lo) & (X <= hi)), (alg, extra)
            ops = dict(r.operators)
            name = extra.get("crossover", "sbx")
            assert name in ops, (alg, extra, r.operators)
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
    assert list(words["dms_init"]) == ["line", "single"], words
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


def main() -> int:
    failed = []
    for fn in TESTS:
        try:
            fn()
            print(f"  ok    {fn.__name__}")
        except Exception:
            failed.append(fn.__name__)
            print(f"  FAIL  {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(TESTS) - len(failed)}/{len(TESTS)} passed")
    if failed:
        print("failed: " + ", ".join(failed), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

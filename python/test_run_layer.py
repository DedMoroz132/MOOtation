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
    assert len(names) == 58, f"expected 58 algorithms, got {len(names)}"
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
    assert len(fams) == 7 and fams[0][1][0] == "nsga2", fams
    assert all(name and "─" not in name for name, _ in fams), fams



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

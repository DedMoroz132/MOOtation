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
    assert len(names) == 60, f"expected 60 algorithms, got {len(names)}"
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
    e = raises(ConfigError, loads, "[run]\nname='x'\nscratch='s'\n")
    assert "ledger" in str(e), e


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
    # TUI_SPEC.md's own example writes DTLZ2_M3. The registry calls it
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

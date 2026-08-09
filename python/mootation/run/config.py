# SPDX-License-Identifier: Apache-2.0
"""Reading and checking a run description.

The file format is TOML (TUI_SPEC.md §1): `tomllib` is in the standard library
from Python 3.11, comments survive, and there are none of YAML's surprises
(`no` becoming False, the Norway problem). Arrays of tables map onto a sequence
of steps without inventing anything.

`validate()` implements TUI_SPEC.md §7. It is cheap and it runs before anything
expensive starts, because the loss it prevents is the expensive one: a run
budgeted for a day that dies ten minutes in on a typo in a path.

Everything reports through ConfigError with a `path` naming the offending key,
so the message points at a line the user can edit rather than at a stack frame.
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .algorithms import algorithm_names, check_pop
from .knobs import knob_names

# ── Errors ──────────────────────────────────────────────────────────────────


class ConfigError(Exception):
    """A problem with the run description, named by the key that carries it."""

    def __init__(self, path: str, message: str):
        self.path = path
        self.message = message
        super().__init__(f"{path}: {message}" if path else message)


# ── Model ───────────────────────────────────────────────────────────────────

ON_FAIL = ("penalty", "skip", "abort")
ON_SIZE_MISMATCH = ("truncate", "pad", "error")
PARSERS = ("csv", "json", "regex", "columns")   # plus "python:module.func"
PROBLEM_KINDS = ("external", "builtin")


@dataclass
class Step:
    name: str
    argv: list[str]                 # already resolved for this platform
    timeout: float | None = None
    shell: bool = False


@dataclass
class Output:
    parser: str
    source: str | None = None       # `from`, a path template
    objectives: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    take: str = "last"              # regex: first | last | nth
    nth: int | None = None
    patterns: dict[str, str] = field(default_factory=dict)
    line: int = -1                  # columns
    fields: list[int] = field(default_factory=list)


@dataclass
class Algorithm:
    name: str
    pop: int
    gens: int
    params: dict[str, float] = field(default_factory=dict)


@dataclass
class WarmStart:
    source: str
    on_size_mismatch: str = "error"


@dataclass
class Config:
    # [run]
    name: str
    scratch: str
    ledger: str
    workers: int = 1
    on_fail: str = "penalty"
    penalty: float = 1e8
    resume: bool = True
    warm_start: WarmStart | None = None

    # [problem]
    kind: str = "external"
    n_vars: int = 0
    n_objs: int = 0
    bounds: list[tuple[float, float]] = field(default_factory=list)
    input_template: str | None = None
    input_write_to: str | None = None
    steps: list[Step] = field(default_factory=list)
    output: Output | None = None

    # [[algorithms]] / [benchmarks]
    algorithms: list[Algorithm] = field(default_factory=list)
    benchmarks: dict[str, Any] = field(default_factory=dict)

    # Resolved by validate() from [benchmarks]: the concrete problem names
    # the family/objective cross product expands to.
    benchmark_problems: list[str] = field(default_factory=list)

    # provenance
    source_path: Path | None = None

    @property
    def n_cons(self) -> int:
        return len(self.output.constraints) if self.output else 0


# ── Loading ─────────────────────────────────────────────────────────────────


def _req(table: dict, key: str, where: str, typ=None):
    if key not in table:
        raise ConfigError(where, f"missing required key '{key}'")
    v = table[key]
    if typ is not None and not isinstance(v, typ):
        want = typ.__name__ if isinstance(typ, type) else "/".join(t.__name__ for t in typ)
        raise ConfigError(f"{where}.{key}", f"expected {want}, got {type(v).__name__}")
    return v


def _opt(table: dict, key: str, default, where: str, typ=None):
    if key not in table:
        return default
    return _req(table, key, where, typ)


def _platform_key() -> str:
    s = platform.system().lower()
    if s.startswith("win"):
        return "windows"
    if s == "darwin":
        return "darwin"
    return "linux"


def _resolve_argv(step: dict, where: str) -> list[str]:
    """Pick run_<platform> over run (TUI_SPEC.md §2).

    A command is an argv ARRAY, never a string: no quoting rules, no escaping
    of spaces in paths, no difference between cmd and sh. If neither the
    platform-specific key nor the generic one is present, that is a config
    error here rather than a mysterious failure at run time.
    """
    key = f"run_{_platform_key()}"
    argv = step.get(key, step.get("run"))
    if argv is None:
        raise ConfigError(
            where,
            f"no command for this platform: neither '{key}' nor 'run' is set",
        )
    if not isinstance(argv, list) or not argv:
        raise ConfigError(
            f"{where}.{key if key in step else 'run'}",
            "a command must be a non-empty array of arguments, e.g. "
            '["solver", "--mesh", "{scratch}/mesh.msh"]',
        )
    for i, a in enumerate(argv):
        if not isinstance(a, str):
            raise ConfigError(
                f"{where}.run[{i}]",
                f"every argument must be a string, got {type(a).__name__}",
            )
    return list(argv)


def loads(text: str, *, source_path: Path | None = None) -> Config:
    """Parse a run description from TOML text."""
    try:
        raw = tomllib.loads(text)
    except tomllib.TOMLDecodeError as e:
        raise ConfigError("", f"TOML does not parse: {e}") from e

    run = _req(raw, "run", "", dict)
    cfg = Config(
        name=_req(run, "name", "run", str),
        scratch=_req(run, "scratch", "run", str),
        ledger=_req(run, "ledger", "run", str),
        workers=_opt(run, "workers", 1, "run", int),
        on_fail=_opt(run, "on_fail", "penalty", "run", str),
        penalty=float(_opt(run, "penalty", 1e8, "run", (int, float))),
        resume=_opt(run, "resume", True, "run", bool),
        source_path=source_path,
    )

    if cfg.on_fail not in ON_FAIL:
        raise ConfigError("run.on_fail",
                          f"must be one of {', '.join(ON_FAIL)}; got '{cfg.on_fail}'")
    if cfg.workers < 1:
        raise ConfigError("run.workers", f"must be >= 1, got {cfg.workers}")

    if "warm_start" in run:
        ws = _req(run, "warm_start", "run", dict)
        mode = _opt(ws, "on_size_mismatch", "error", "run.warm_start", str)
        if mode not in ON_SIZE_MISMATCH:
            raise ConfigError(
                "run.warm_start.on_size_mismatch",
                f"must be one of {', '.join(ON_SIZE_MISMATCH)}; got '{mode}'")
        cfg.warm_start = WarmStart(
            source=_req(ws, "from", "run.warm_start", str),
            on_size_mismatch=mode,
        )

    # ── [problem] ───────────────────────────────────────────────────────────
    prob = _req(raw, "problem", "", dict)
    cfg.kind = _opt(prob, "kind", "external", "problem", str)
    if cfg.kind not in PROBLEM_KINDS:
        raise ConfigError("problem.kind",
                          f"must be one of {', '.join(PROBLEM_KINDS)}; got '{cfg.kind}'")

    if cfg.kind == "external":
        cfg.n_vars = _req(prob, "n_vars", "problem", int)
        cfg.n_objs = _req(prob, "n_objs", "problem", int)
        raw_bounds = _req(prob, "bounds", "problem", list)
        for i, b in enumerate(raw_bounds):
            if not isinstance(b, list) or len(b) != 2:
                raise ConfigError(f"problem.bounds[{i}]",
                                  "expected a [lower, upper] pair")
            lo, hi = b
            if not isinstance(lo, (int, float)) or not isinstance(hi, (int, float)):
                raise ConfigError(f"problem.bounds[{i}]", "bounds must be numbers")
            cfg.bounds.append((float(lo), float(hi)))

        if "input" in prob:
            pin = _req(prob, "input", "problem", dict)
            cfg.input_template = _req(pin, "template", "problem.input", str)
            cfg.input_write_to = _req(pin, "write_to", "problem.input", str)

        for i, st in enumerate(_opt(prob, "steps", [], "problem", list)):
            where = f"problem.steps[{i}]"
            if not isinstance(st, dict):
                raise ConfigError(where, "each step must be a table")
            cfg.steps.append(Step(
                name=_opt(st, "name", f"step{i}", where, str),
                argv=_resolve_argv(st, where),
                timeout=(float(st["timeout"]) if "timeout" in st else None),
                shell=bool(_opt(st, "shell", False, where, bool)),
            ))

        if "output" in prob:
            cfg.output = _load_output(_req(prob, "output", "problem", dict))

    # ── [[algorithms]] ──────────────────────────────────────────────────────
    for i, a in enumerate(_opt(raw, "algorithms", [], "", list)):
        where = f"algorithms[{i}]"
        if not isinstance(a, dict):
            raise ConfigError(where, "each algorithm must be a table")
        params = _opt(a, "params", {}, where, dict)
        for k, v in params.items():
            if not isinstance(v, (int, float, bool)):
                raise ConfigError(f"{where}.params.{k}",
                                  f"expected a number, got {type(v).__name__}")
        cfg.algorithms.append(Algorithm(
            name=_req(a, "name", where, str),
            pop=_req(a, "pop", where, int),
            gens=_req(a, "gens", where, int),
            params={k: float(v) for k, v in params.items()},
        ))

    cfg.benchmarks = _opt(raw, "benchmarks", {}, "", dict)
    return cfg


def _load_output(out: dict) -> Output:
    where = "problem.output"
    parser = _req(out, "parser", where, str)
    if parser not in PARSERS and not parser.startswith("python:"):
        raise ConfigError(
            f"{where}.parser",
            f"must be one of {', '.join(PARSERS)} or 'python:module.function'; "
            f"got '{parser}'")

    o = Output(
        parser=parser,
        source=_opt(out, "from", None, where, str),
        objectives=list(_opt(out, "objectives", [], where, list)),
        constraints=list(_opt(out, "constraints", [], where, list)),
        take=_opt(out, "take", "last", where, str),
        nth=(int(out["nth"]) if "nth" in out else None),
        patterns=dict(_opt(out, "patterns", {}, where, dict)),
        line=int(_opt(out, "line", -1, where, int)),
        fields=[int(f) for f in _opt(out, "fields", [], where, list)],
    )
    if o.take not in ("first", "last", "nth"):
        raise ConfigError(f"{where}.take",
                          f"must be first, last or nth; got '{o.take}'")
    if o.take == "nth" and o.nth is None:
        raise ConfigError(f"{where}.nth", "take = \"nth\" requires an 'nth' index")
    return o


def load(path: str | os.PathLike) -> Config:
    """Parse a run description from a TOML file."""
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as e:
        raise ConfigError("", f"cannot read {p}: {e}") from e
    return loads(text, source_path=p)


# ── Validation (TUI_SPEC.md §7) ──────────────────────────────────────────────

_PLACEHOLDER = re.compile(r"\{x\[(\d+)\]\}")


def _executable_findable(argv0: str) -> bool:
    """Is the program either on PATH or an existing file at the given path?"""
    if shutil.which(argv0) is not None:
        return True
    p = Path(argv0)
    # A bare name that is not on PATH is not findable; a path is checked as one.
    return p.is_file() if (p.is_absolute() or len(p.parts) > 1) else False


def validate(cfg: Config, *, base: Path | None = None) -> list[str]:
    """Return a list of problems. Empty means the run can start.

    Problems are collected rather than raised one at a time: a user fixing a
    config wants every complaint at once, not one per edit-run cycle.

    `base` is the directory relative paths resolve against — the config file's
    own directory by default, so a config is portable between machines as long
    as its neighbours travel with it.
    """
    if base is None:
        base = cfg.source_path.parent if cfg.source_path else Path.cwd()
    problems: list[str] = []

    def bad(where: str, msg: str) -> None:
        problems.append(f"{where}: {msg}")

    # ── the problem ─────────────────────────────────────────────────────────
    if cfg.kind == "external":
        if cfg.n_vars < 1:
            bad("problem.n_vars", f"must be >= 1, got {cfg.n_vars}")
        if cfg.n_objs < 1:
            bad("problem.n_objs", f"must be >= 1, got {cfg.n_objs}")

        if len(cfg.bounds) != cfg.n_vars:
            bad("problem.bounds",
                f"has {len(cfg.bounds)} entries but n_vars = {cfg.n_vars}; "
                "there must be exactly one [lower, upper] pair per variable")
        for i, (lo, hi) in enumerate(cfg.bounds):
            if not lo < hi:
                bad(f"problem.bounds[{i}]",
                    f"lower ({lo}) must be strictly less than upper ({hi})")

        # input template
        if cfg.input_template:
            tpl = (base / cfg.input_template)
            if not tpl.is_file():
                bad("problem.input.template", f"file not found: {tpl}")
            else:
                text = tpl.read_text(encoding="utf-8", errors="replace")
                for m in _PLACEHOLDER.finditer(text):
                    idx = int(m.group(1))
                    if idx >= cfg.n_vars:
                        bad("problem.input.template",
                            f"uses {{x[{idx}]}} but n_vars = {cfg.n_vars} "
                            f"(valid indices are 0..{cfg.n_vars - 1})")
            if not cfg.input_write_to:
                bad("problem.input.write_to", "required when a template is given")

        # steps
        if not cfg.steps:
            bad("problem.steps", "an external problem needs at least one step")
        for st in cfg.steps:
            if not _executable_findable(st.argv[0]):
                bad(f"problem.steps.{st.name}",
                    f"executable not found on PATH and not an existing file: "
                    f"{st.argv[0]!r} (platform: {_platform_key()})")
            if st.timeout is not None and st.timeout <= 0:
                bad(f"problem.steps.{st.name}",
                    f"timeout must be positive, got {st.timeout}")

        # output
        if cfg.output is None:
            bad("problem.output", "an external problem needs an output section")
        else:
            o = cfg.output
            if len(o.objectives) != cfg.n_objs:
                bad("problem.output.objectives",
                    f"lists {len(o.objectives)} names but n_objs = {cfg.n_objs}")
            if o.parser in ("csv", "json", "regex", "columns") and not o.source \
                    and o.parser != "json":
                bad("problem.output.from",
                    f"parser '{o.parser}' needs a 'from' path")
            if o.parser == "regex":
                wanted = set(o.objectives) | set(o.constraints)
                missing = wanted - set(o.patterns)
                if missing:
                    bad("problem.output.patterns",
                        "no pattern for: " + ", ".join(sorted(missing)))
                for key, pat in o.patterns.items():
                    try:
                        rx = re.compile(pat)
                    except re.error as e:
                        bad(f"problem.output.patterns.{key}",
                            f"not a valid regular expression: {e}")
                        continue
                    if rx.groups < 1:
                        bad(f"problem.output.patterns.{key}",
                            "must contain a capturing group for the value, "
                            r"e.g. 'MASS\s*=\s*([0-9.eE+-]+)'")
            if o.parser == "columns":
                need = len(o.objectives) + len(o.constraints)
                if len(o.fields) != need:
                    bad("problem.output.fields",
                        f"lists {len(o.fields)} indices but "
                        f"{need} values are expected "
                        f"({len(o.objectives)} objectives + "
                        f"{len(o.constraints)} constraints)")
                for f in o.fields:
                    if f < 0:
                        bad("problem.output.fields",
                            f"column indices are zero-based and non-negative, got {f}")

    # ── algorithms ──────────────────────────────────────────────────────────
    try:
        known = set(algorithm_names())
    except (FileNotFoundError, ValueError) as e:
        known = set()
        bad("algorithms", f"cannot read the algorithm registry: {e}")

    if not cfg.algorithms and not cfg.benchmarks:
        bad("algorithms", "no algorithms and no benchmarks — nothing to run")

    valid_knobs = set(knob_names())
    for i, a in enumerate(cfg.algorithms):
        where = f"algorithms[{i}] ({a.name})"
        if known and a.name not in known:
            near = sorted(n for n in known if n.startswith(a.name[:3]))
            hint = f"; did you mean: {', '.join(near[:5])}" if near else ""
            bad(where, f"unknown algorithm '{a.name}'{hint}")
        if a.pop < 2:
            bad(where, f"pop must be >= 2, got {a.pop}")
        if a.gens < 1:
            bad(where, f"gens must be >= 1, got {a.gens}")

        unknown = sorted(set(a.params) - valid_knobs)
        if unknown:
            bad(where,
                f"unknown parameter(s): {', '.join(unknown)}. "
                f"Known: {', '.join(sorted(valid_knobs))}")

        if cfg.n_objs >= 1 and a.pop >= 2:
            reason = check_pop(a.name, a.pop, cfg.n_objs, a.params)
            if reason:
                bad(where, reason)

    # ── benchmarks ──────────────────────────────────────────────────────────
    if cfg.benchmarks:
        # NOT named `problems`: that is the complaint list this function
        # returns, and shadowing it here silently replaced every complaint
        # with the selection.
        cfg.benchmark_problems = _benchmark_selection(cfg.benchmarks, bad)

    # ── warm start ──────────────────────────────────────────────────────────
    if cfg.warm_start:
        src = base / cfg.warm_start.source
        if not src.is_file():
            bad("run.warm_start.from", f"file not found: {src}")
        else:
            n_objs = _peek_warm_start_objs(src)
            if n_objs is not None and cfg.n_objs and n_objs != cfg.n_objs:
                bad("run.warm_start.from",
                    f"seed population has {n_objs} objectives but the run "
                    f"declares n_objs = {cfg.n_objs}; a warm start cannot change "
                    "the number of objectives")

    return problems


def _benchmark_selection(spec: dict, bad) -> list:
    """Resolve and check a [benchmarks] section against the real registry.

    Two spellings, as in TUI_SPEC.md: an explicit `problems` list, or the
    cross product of `families` and `objectives`. Names are checked against
    the registry rather than a pattern, because the registry is what the run
    will actually look them up in — `DTLZ2_M3` looks plausible and does not
    exist; the name is `DTLZ2_3D`.
    """
    known_keys = {"families", "objectives", "runs", "problems"}
    unknown = sorted(set(spec) - known_keys)
    if unknown:
        bad("benchmarks", f"unknown key(s): {', '.join(unknown)}. "
                          f"Known: {', '.join(sorted(known_keys))}")

    runs = spec.get("runs", 31)
    if not isinstance(runs, int) or runs < 1:
        bad("benchmarks.runs", f"must be an integer >= 1, got {runs!r}")

    try:
        from ..benchmarks import names as bench_names, get as bench_get
    except ImportError as e:
        bad("benchmarks", str(e))
        return []

    registry = set(bench_names())

    if "problems" in spec:
        wanted = list(spec["problems"])
        missing = [p for p in wanted if p not in registry]
        for p in missing:
            # Suggest by the FULL stem first — "DTLZ2" from "DTLZ2_M3" — before
            # falling back to the family. Sorting the family alphabetically
            # would offer DTLZ1_10D..DTLZ1_4D for a DTLZ2 typo, which is a
            # worse answer than none.
            head = p.split("_")[0]
            near = sorted(n for n in registry if n.upper().startswith(head.upper()))
            if not near:
                fam = "".join(c for c in head if c.isalpha())
                near = sorted(n for n in registry
                              if n.upper().startswith(fam.upper()))
            hint = f"; did you mean: {', '.join(near[:5])}" if near else ""
            bad("benchmarks.problems", f"unknown problem '{p}'{hint}")
        return [p for p in wanted if p in registry]

    fams = spec.get("families")
    objs = spec.get("objectives")
    if not fams or not objs:
        bad("benchmarks",
            "give either 'problems = [...]' or both 'families = [...]' and "
            "'objectives = [...]'")
        return []

    selected = []
    for fam in fams:
        for m in objs:
            hits = [n for n in registry
                    if n.upper().startswith(str(fam).upper())
                    and n.endswith(f"_{m}D")]
            if not hits:
                avail = sorted({n.split("_")[-1] for n in registry
                                if n.upper().startswith(str(fam).upper())})
                bad("benchmarks",
                    f"no problems for family '{fam}' at {m} objectives"
                    + (f"; available sizes: {', '.join(avail)}" if avail
                       else f"; no family '{fam}' in the registry"))
            selected.extend(hits)
    return sorted(set(selected))


def _peek_warm_start_objs(path: Path) -> int | None:
    """Objective count of the first usable record of a JSONL population."""
    import json
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    return None
                f = rec.get("f")
                if isinstance(f, list):
                    return len(f)
                return None
    except OSError:
        return None
    return None

# SPDX-License-Identifier: Apache-2.0
"""Reading objective values out of whatever a solver produced (TUI_SPEC.md §3).

Four declarative strategies, from clean formats to dirty ones, plus an escape
hatch into Python for the formats nobody can describe declaratively (HDF5, VTU,
anything binary). Without that hatch the schema is dead on arrival: solver
output formats cannot be anticipated.

A parse failure is an EVALUATION failure. There is no fallback value, no zero,
no NaN quietly substituted: a corrupted point poisons the archive and every
metric computed from it afterwards. ParseError propagates to the run's `on_fail`
policy, which is where that decision belongs.
"""

from __future__ import annotations

import csv
import importlib
import json
import re
from pathlib import Path
from typing import Any, Mapping


class ParseError(Exception):
    """The solver's output could not be read as numbers."""


def _to_float(value: Any, what: str) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError) as e:
        raise ParseError(f"{what}: {value!r} is not a number") from e
    if f != f:  # NaN
        raise ParseError(f"{what}: value is NaN")
    return f


# ── csv: by column name ─────────────────────────────────────────────────────


def _parse_csv(path: Path, names: list[str]) -> dict[str, float]:
    try:
        with path.open("r", encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
    except OSError as e:
        raise ParseError(f"cannot read {path}: {e}") from e
    if not rows:
        raise ParseError(f"{path}: no data rows")
    # The last row: solvers that append per iteration leave the final state last,
    # and a single-row file is unaffected.
    row = rows[-1]
    missing = [n for n in names if n not in row]
    if missing:
        raise ParseError(
            f"{path}: no column named {', '.join(missing)}; "
            f"columns present: {', '.join(k for k in row if k)}")
    return {n: _to_float(row[n], f"{path}:{n}") for n in names}


# ── json: by dotted path ────────────────────────────────────────────────────


def _dig(doc: Any, dotted: str, where: str) -> Any:
    cur = doc
    for part in dotted.split("."):
        if isinstance(cur, list):
            try:
                cur = cur[int(part)]
                continue
            except (ValueError, IndexError) as e:
                raise ParseError(f"{where}: '{dotted}' — no element {part}") from e
        if not isinstance(cur, Mapping) or part not in cur:
            raise ParseError(f"{where}: '{dotted}' — no key '{part}'")
        cur = cur[part]
    return cur


def _parse_json(path: Path, names: list[str]) -> dict[str, float]:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except OSError as e:
        raise ParseError(f"cannot read {path}: {e}") from e
    except json.JSONDecodeError as e:
        raise ParseError(f"{path}: not valid JSON: {e}") from e
    out = {}
    for n in names:
        out[n] = _to_float(_dig(doc, n, str(path)), f"{path}:{n}")
    return out


# ── regex: for solver logs, the commonest real case ─────────────────────────


def _parse_regex(path: Path, names: list[str], patterns: Mapping[str, str],
                 take: str = "last", nth: int | None = None) -> dict[str, float]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        raise ParseError(f"cannot read {path}: {e}") from e

    out = {}
    for n in names:
        pat = patterns.get(n)
        if pat is None:
            raise ParseError(f"{path}: no pattern configured for '{n}'")
        matches = re.findall(pat, text)
        if not matches:
            raise ParseError(f"{path}: pattern for '{n}' did not match: {pat}")
        # `take = last` matters: solvers print the value every iteration and it
        # is almost always the final one that is wanted.
        if take == "first":
            hit = matches[0]
        elif take == "nth":
            if nth is None:
                raise ParseError(f"take = 'nth' needs an index, for '{n}'")
            try:
                hit = matches[nth]
            except IndexError as e:
                raise ParseError(
                    f"{path}: pattern for '{n}' matched {len(matches)} times, "
                    f"no index {nth}") from e
        else:
            hit = matches[-1]
        if isinstance(hit, tuple):      # more than one capturing group
            hit = hit[0]
        out[n] = _to_float(hit, f"{path}:{n}")
    return out


# ── columns: positional, for Fortran-shaped output ──────────────────────────


def _parse_columns(path: Path, names: list[str], line: int,
                   fields: list[int]) -> dict[str, float]:
    try:
        lines = [ln for ln in path.read_text(encoding="utf-8",
                                             errors="replace").splitlines()
                 if ln.strip()]
    except OSError as e:
        raise ParseError(f"cannot read {path}: {e}") from e
    if not lines:
        raise ParseError(f"{path}: file is empty")
    try:
        row = lines[line]
    except IndexError as e:
        raise ParseError(
            f"{path}: no line {line} (file has {len(lines)} non-blank lines)") from e

    cells = row.split()
    if len(fields) != len(names):
        raise ParseError(
            f"{path}: {len(fields)} column indices for {len(names)} values")
    out = {}
    for n, idx in zip(names, fields):
        if idx >= len(cells):
            raise ParseError(
                f"{path}: line {line} has {len(cells)} columns, no index {idx}: "
                f"{row.strip()!r}")
        out[n] = _to_float(cells[idx], f"{path}:{n}")
    return out


# ── python:module.function — the escape hatch ───────────────────────────────


def _parse_python(spec: str, scratch: Path, names: list[str]) -> dict[str, float]:
    target = spec[len("python:"):]
    if "." not in target:
        raise ParseError(
            f"parser '{spec}': expected 'python:module.function'")
    mod_name, func_name = target.rsplit(".", 1)
    try:
        mod = importlib.import_module(mod_name)
    except ImportError as e:
        raise ParseError(f"parser '{spec}': cannot import '{mod_name}': {e}") from e
    func = getattr(mod, func_name, None)
    if func is None or not callable(func):
        raise ParseError(f"parser '{spec}': '{mod_name}' has no callable "
                         f"'{func_name}'")
    try:
        result = func(scratch)
    except Exception as e:      # the user's code — report, never let it escape raw
        raise ParseError(f"parser '{spec}' raised {type(e).__name__}: {e}") from e
    if not isinstance(result, Mapping):
        raise ParseError(
            f"parser '{spec}' returned {type(result).__name__}, expected a dict "
            'like {"mass": 1.2, "drag": 3.4}')
    missing = [n for n in names if n not in result]
    if missing:
        raise ParseError(f"parser '{spec}' did not return: {', '.join(missing)}")
    return {n: _to_float(result[n], f"{spec}:{n}") for n in names}


# ── entry point ─────────────────────────────────────────────────────────────


def parse_output(output, scratch: Path, subst) -> tuple[list[float], list[float]]:
    """Read objectives and constraints for one evaluation.

    `output` is a config.Output; `scratch` is this worker's directory; `subst`
    expands {scratch} and friends in the `from` path.

    Returns (objectives, constraints) in the order the config lists them.
    Raises ParseError on anything that is not a clean read.
    """
    names = list(output.objectives) + list(output.constraints)
    if not names:
        raise ParseError("no objectives configured")

    if output.parser.startswith("python:"):
        values = _parse_python(output.parser, scratch, names)
    else:
        if not output.source:
            raise ParseError(f"parser '{output.parser}' needs a 'from' path")
        path = Path(subst(output.source))
        if not path.is_file():
            raise ParseError(f"output file not produced: {path}")
        if output.parser == "csv":
            values = _parse_csv(path, names)
        elif output.parser == "json":
            values = _parse_json(path, names)
        elif output.parser == "regex":
            values = _parse_regex(path, names, output.patterns,
                                  output.take, output.nth)
        elif output.parser == "columns":
            values = _parse_columns(path, names, output.line, output.fields)
        else:
            raise ParseError(f"unknown parser '{output.parser}'")

    n_obj = len(output.objectives)
    ordered = [values[n] for n in names]
    return ordered[:n_obj], ordered[n_obj:]

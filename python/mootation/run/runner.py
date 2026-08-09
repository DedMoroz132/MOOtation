# SPDX-License-Identifier: Apache-2.0
"""Turning a decision vector into objective values by running other programs.

One evaluation is: substitute the vector into an input template, run the
configured steps in order, parse whatever they produced, journal the result.

Three rules this module exists to enforce:

* **A failure is a failure.** A step that returns non-zero, times out, or
  produces output that will not parse marks the evaluation `failed` and hands
  it to the run's `on_fail` policy. Nothing is silently substituted, because
  one fabricated point poisons the archive and every metric taken from it.
* **Scratch is per worker and overwritten; the journal is per run and appended.**
  Mixing them is what makes a run both enormous on disk and impossible to
  resume (TUI_SPEC.md §4).
* **The journal is a cache.** Evolutionary algorithms re-propose identical
  individuals far more often than one expects, and a cache hit costs a
  dictionary lookup instead of an hour of solver time.

Commands are argv arrays and are executed directly — no shell between the
config and the process, so there is nothing to quote and nothing to escape.
`shell = true` exists for the rare step that genuinely needs a pipeline, and it
is opt-in per step.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

from .config import Config
from .ledger import Ledger, Record
from .parsers import ParseError, parse_output


class EvaluationFailed(Exception):
    """One evaluation could not be completed. Carries a human-readable reason."""


class AbortRun(Exception):
    """`on_fail = "abort"` and an evaluation failed."""


def _subst(text: str, *, scratch: Path, name: str, worker: int) -> str:
    """Expand {scratch}, {name} and {worker} (with format specs) in a path."""
    return text.format(scratch=scratch.as_posix(), name=name, worker=worker)


class Evaluator:
    """Runs one worker's evaluations. Not thread-safe; make one per worker."""

    def __init__(self, cfg: Config, worker: int = 0, *,
                 base: Path | None = None, ledger: Ledger | None = None):
        self.cfg = cfg
        self.worker = worker
        self.base = base or (cfg.source_path.parent if cfg.source_path
                             else Path.cwd())
        self.scratch = Path(_subst(cfg.scratch, scratch=Path("."),
                                   name=cfg.name, worker=worker))
        if not self.scratch.is_absolute():
            self.scratch = self.base / self.scratch
        self.ledger = ledger
        self._own_ledger = ledger is None
        if self.ledger is None:
            led_path = self.base / _subst(cfg.ledger, scratch=Path("."),
                                          name=cfg.name, worker=worker)
            self.ledger = Ledger(led_path, resume=cfg.resume)

    # ── one evaluation ──────────────────────────────────────────────────────

    def evaluate(self, x: list[float], *, gen: int = -1, idx: int = -1
                 ) -> tuple[list[float], list[float]]:
        """Objective and constraint values for one decision vector.

        Returns (objectives, constraints). Applies the `on_fail` policy, so the
        caller never sees a partially evaluated point.
        """
        cached = self.ledger.lookup(x)
        if cached is not None:
            return list(cached.f), ([cached.cv] if cached.cv else [])

        t0 = time.monotonic()
        try:
            f, g = self._run_once(x)
        except (EvaluationFailed, ParseError) as e:
            elapsed = time.monotonic() - t0
            self.ledger.append(Record(gen=gen, idx=idx, x=list(x), f=[],
                                      status="failed", t=elapsed))
            return self._apply_on_fail(str(e))

        elapsed = time.monotonic() - t0
        cv = sum(max(0.0, v) for v in g)
        self.ledger.append(Record(gen=gen, idx=idx, x=list(x), f=list(f),
                                  cv=cv, status="ok", t=elapsed))
        return f, g

    def _apply_on_fail(self, reason: str) -> tuple[list[float], list[float]]:
        policy = self.cfg.on_fail
        n_obj = self.cfg.n_objs
        n_con = self.cfg.n_cons
        if policy == "abort":
            raise AbortRun(reason)
        if policy == "skip":
            # "Skip" still has to return numbers — the algorithm asked for a
            # point and the contract is one row per candidate. The penalty is
            # what makes the point unattractive; the journal records that it
            # was never really evaluated.
            return [float("inf")] * n_obj, [float("inf")] * n_con
        p = self.cfg.penalty
        return [p] * n_obj, [p] * n_con

    # ── the mechanics ───────────────────────────────────────────────────────

    def _run_once(self, x: list[float]) -> tuple[list[float], list[float]]:
        self._prepare_scratch()
        self._write_input(x)
        for step in self.cfg.steps:
            self._run_step(step)
        if self.cfg.output is None:
            raise EvaluationFailed("no [problem.output] section")
        return parse_output(self.cfg.output, self.scratch, self._path)

    def _path(self, template: str) -> str:
        p = Path(_subst(template, scratch=self.scratch, name=self.cfg.name,
                        worker=self.worker))
        return str(p if p.is_absolute() else self.base / p)

    def _prepare_scratch(self) -> None:
        """Empty the worker's scratch directory.

        Overwritten rather than accumulated: this is where meshes and VTU files
        land, and keeping them per evaluation is how a run fills a disk. Stale
        files are actively harmful too — a solver that fails silently would
        otherwise be "parsed" from the previous evaluation's output.
        """
        if self.scratch.exists():
            shutil.rmtree(self.scratch, ignore_errors=True)
        self.scratch.mkdir(parents=True, exist_ok=True)

    def _write_input(self, x: list[float]) -> None:
        if not self.cfg.input_template:
            return
        tpl_path = self.base / self.cfg.input_template
        try:
            tpl = tpl_path.read_text(encoding="utf-8")
        except OSError as e:
            raise EvaluationFailed(f"cannot read template {tpl_path}: {e}") from e
        try:
            rendered = tpl.format(x=list(x), scratch=self.scratch.as_posix(),
                                  name=self.cfg.name, worker=self.worker)
        except (IndexError, KeyError) as e:
            raise EvaluationFailed(
                f"{tpl_path}: placeholder out of range for n_vars="
                f"{self.cfg.n_vars}: {e}") from e
        out = Path(self._path(self.cfg.input_write_to or "{scratch}/input"))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(rendered, encoding="utf-8")

    def _run_step(self, step) -> None:
        argv = [self._path(a) if ("{scratch}" in a or "{name}" in a
                                  or "{worker}" in a) else a
                for a in step.argv]
        log = self.scratch / f"{step.name}.log"
        try:
            with log.open("wb") as fh:
                proc = subprocess.run(
                    " ".join(argv) if step.shell else argv,
                    shell=step.shell,
                    cwd=str(self.base),
                    stdout=fh,
                    stderr=subprocess.STDOUT,
                    timeout=step.timeout,
                )
        except subprocess.TimeoutExpired as e:
            raise EvaluationFailed(
                f"step '{step.name}' exceeded its {step.timeout:g}s timeout") from e
        except FileNotFoundError as e:
            raise EvaluationFailed(
                f"step '{step.name}': executable not found: {argv[0]!r}") from e
        except OSError as e:
            raise EvaluationFailed(f"step '{step.name}': {e}") from e

        if proc.returncode != 0:
            tail = ""
            try:
                lines = log.read_text(encoding="utf-8",
                                      errors="replace").splitlines()
                tail = " | ".join(lines[-3:])
            except OSError:
                pass
            raise EvaluationFailed(
                f"step '{step.name}' exited {proc.returncode}"
                + (f": {tail}" if tail else ""))

    def close(self) -> None:
        if self._own_ledger and self.ledger is not None:
            self.ledger.close()

    def __enter__(self) -> "Evaluator":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

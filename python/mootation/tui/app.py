# SPDX-License-Identifier: Apache-2.0
"""The Textual application: screens over a run description.

Config / Problems / Algorithms / Monitor always; Campaign / Compare / Explore
when the config describes a builtin benchmark campaign (mootation.run.campaign),
and then the app opens on Campaign.

Nothing here edits the config. The screens render what `config.load` and
`config.validate` already produced, plus what the journal on disk says, so the
UI cannot disagree with `--check`: they call the same functions. The one thing
the app does besides reading is run the campaign its config describes: the
Campaign tab starts it, stops it and changes its number of workers while it
runs.
"""

from __future__ import annotations

import atexit
import json
import os
import signal
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

from rich import box
from rich.align import Align
from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import (
    Button, DataTable, Footer, Header, Input, Static, TabbedContent, TabPane, Tree,
)

from ..run.algorithms import algorithm_families
from ..run.config import Config, load, validate, _platform_key
from ..run.ledger import Ledger
from ..run import campaign as _camp


# ── Config ──────────────────────────────────────────────────────────────────


class ConfigScreen(VerticalScroll):
    """What the file said, after resolution, and whether it will run."""

    def __init__(self, cfg: Config, problems: list[str]) -> None:
        super().__init__()
        self.cfg = cfg
        self.problems = problems

    def compose(self) -> ComposeResult:
        cfg = self.cfg
        base = cfg.source_path.parent if cfg.source_path else Path.cwd()

        verdict = Text()
        if self.problems:
            verdict.append(f"{len(self.problems)} problem(s) — this run cannot "
                           f"start\n\n", style="bold red")
            for p in self.problems:
                verdict.append("  ! ", style="red")
                verdict.append(p + "\n")
        else:
            total = sum(a.pop * a.gens for a in cfg.algorithms)
            verdict.append("OK", style="bold green")
            verdict.append(f" — {len(cfg.algorithms)} algorithm(s), about "
                           f"{total:,} evaluations at full budget\n")
        yield Static(verdict, classes="panel")

        run = Text()
        run.append("run\n", style="bold")
        for k, v in (("name", cfg.name), ("config", str(cfg.source_path or "-")),
                     ("base dir", str(base)), ("scratch", cfg.scratch),
                     ("ledger", cfg.ledger), ("workers", str(cfg.workers)),
                     ("on_fail", cfg.on_fail), ("resume", str(cfg.resume))):
            run.append(f"  {k:<10}", style="dim")
            run.append(f"{v}\n")
        if cfg.warm_start:
            run.append(f"  {'warm start':<10}", style="dim")
            run.append(f"{cfg.warm_start.source} "
                       f"[{cfg.warm_start.on_size_mismatch}]\n")
        yield Static(run, classes="panel")

        prob = Text()
        prob.append(f"problem — {cfg.kind}\n", style="bold")
        if cfg.kind == "external":
            for k, v in (("n_vars", cfg.n_vars), ("n_objs", cfg.n_objs),
                         ("n_cons", cfg.n_cons)):
                prob.append(f"  {k:<10}", style="dim")
                prob.append(f"{v}\n")
            if cfg.input_template:
                prob.append(f"  {'input':<10}", style="dim")
                prob.append(f"{cfg.input_template} -> {cfg.input_write_to}\n")
            prob.append(f"\n  steps, resolved for {_platform_key()}\n",
                        style="dim")
            for st in cfg.steps:
                prob.append(f"    {st.name:<10}", style="cyan")
                prob.append(" ".join(st.argv))
                if st.timeout:
                    prob.append(f"   timeout {st.timeout:g}s", style="dim")
                prob.append("\n")
            if cfg.output:
                o = cfg.output
                prob.append(f"\n  {'output':<10}", style="dim")
                prob.append(f"parser={o.parser}"
                            + (f"  from={o.source}" if o.source else "") + "\n")
                prob.append(f"    {'objectives':<12}", style="dim")
                prob.append(", ".join(o.objectives) + "\n")
                if o.constraints:
                    prob.append(f"    {'constraints':<12}", style="dim")
                    prob.append(", ".join(o.constraints) + "\n")
        elif cfg.benchmark_problems:
            prob.append(f"  {len(cfg.benchmark_problems)} benchmark problems "
                        f"selected — see the Problems tab\n")
        yield Static(prob, classes="panel")


# ── Algorithms ──────────────────────────────────────────────────────────────


class AlgorithmsScreen(VerticalScroll):
    """Which of the 58 are selected, with what parameters, and any objection."""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg

    def compose(self) -> ComposeResult:
        from ..run.algorithms import check_pop

        table = DataTable(zebra_stripes=True)
        table.add_columns("algorithm", "pop", "gens", "evaluations",
                          "parameters", "note")
        for a in self.cfg.algorithms:
            note = ""
            if self.cfg.n_objs:
                note = check_pop(a.name, a.pop, self.cfg.n_objs, a.params) or ""
            params = ", ".join(f"{k}={v:g}" for k, v in sorted(a.params.items()))
            table.add_row(a.name, str(a.pop), str(a.gens), f"{a.pop * a.gens:,}",
                          params or "-",
                          Text(note, style="red") if note else Text("ok",
                                                                    style="green"))
        yield table

        from ..run.algorithms import algorithm_names
        known = algorithm_names()
        yield Static(
            Text(f"\n{len(known)} algorithms are available; "
                 f"{len(self.cfg.algorithms)} selected. "
                 f"Names come from include/mootation/algorithms.def.\n",
                 style="dim"),
            classes="panel")


# ── Problems ────────────────────────────────────────────────────────────────


class ProblemsScreen(VerticalScroll):
    """The benchmark registry, filterable, or the external step chain."""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self._table: DataTable | None = None
        self._rows: list = []

    def compose(self) -> ComposeResult:
        if self.cfg.kind == "external":
            tree: Tree = Tree("external problem")
            tree.root.expand()
            if self.cfg.input_template:
                tree.root.add_leaf(
                    f"input: {self.cfg.input_template} -> {self.cfg.input_write_to}")
            steps = tree.root.add("steps", expand=True)
            for st in self.cfg.steps:
                node = steps.add(st.name, expand=True)
                node.add_leaf(" ".join(st.argv))
                if st.timeout:
                    node.add_leaf(f"timeout {st.timeout:g}s")
            if self.cfg.output:
                out = tree.root.add("output", expand=True)
                out.add_leaf(f"parser {self.cfg.output.parser}")
                out.add_leaf(f"objectives {', '.join(self.cfg.output.objectives)}")
                if self.cfg.output.constraints:
                    out.add_leaf(
                        f"constraints {', '.join(self.cfg.output.constraints)}")
            yield tree
            return

        try:
            from ..benchmarks import describe
        except ImportError as e:
            yield Static(Text(str(e), style="red"), classes="panel")
            return

        # describe(), not get(): get() samples a Pareto front per problem to
        # fix its reference frame, which costs minutes across the registry and
        # buys nothing for a listing.
        selected = set(self.cfg.benchmark_problems)
        self._rows = []
        for n, p in describe():
            self._rows.append((n, p.n_obj, p.n_vars, p.pop_size, p.n_gen,
                               "yes" if n in selected else "",
                               "yes" if p.pareto_front else ""))

        yield Input(placeholder="filter, e.g. DTLZ  or  _5D  or  MaF3",
                    id="bench-filter")
        table = DataTable(zebra_stripes=True, id="bench-table")
        table.add_columns("problem", "M", "n_vars", "pop", "gens",
                          "selected", "true PF")
        self._table = table
        self._fill("")
        yield table

    def _fill(self, needle: str) -> None:
        if self._table is None:
            return
        self._table.clear()
        needle = needle.strip().upper()
        shown = 0
        for row in self._rows:
            if needle and needle not in row[0].upper():
                continue
            self._table.add_row(*[str(c) for c in row])
            shown += 1
        self._table.border_title = f"{shown} of {len(self._rows)} problems"

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "bench-filter":
            self._fill(event.value)


# ── Monitor ─────────────────────────────────────────────────────────────────


class MonitorScreen(VerticalScroll):
    """Live progress, read from the journal.

    The journal is the source of truth: it is append-only and
    it is what a resume reads, so a monitor built on it shows the same thing a
    restart would see. Reading the scratch directories instead would show
    whatever the last worker happened to leave behind.
    """

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self._body: Static | None = None

    def compose(self) -> ComposeResult:
        self._body = Static(self._snapshot(), classes="panel")
        yield self._body

    def on_mount(self) -> None:
        self.set_interval(2.0, self._refresh)

    def _ledger_path(self) -> Path:
        base = (self.cfg.source_path.parent if self.cfg.source_path
                else Path.cwd())
        return base / self.cfg.ledger.format(name=self.cfg.name)

    def _snapshot(self) -> Text:
        t = Text()
        path = self._ledger_path()
        t.append("journal\n", style="bold")
        t.append(f"  {path}\n", style="dim")
        if not path.is_file():
            t.append("\n  nothing yet — this run has not written an "
                     "evaluation.\n", style="dim")
            return t

        ok = failed = 0
        gens: set = set()
        total_t = 0.0
        import json
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue          # a torn final line; see Ledger._load
                if rec.get("status") == "ok":
                    ok += 1
                else:
                    failed += 1
                gens.add(rec.get("gen", -1))
                total_t += float(rec.get("t", 0.0) or 0.0)

        done = ok + failed
        led = Ledger(path, resume=True)
        cached = len(led)
        led.close()

        t.append(f"\n  {'evaluations':<14}", style="dim")
        t.append(f"{done:,}")
        t.append(f"   ok {ok:,}", style="green")
        if failed:
            t.append(f"   failed {failed:,}", style="red")
        t.append(f"\n  {'distinct x':<14}", style="dim")
        t.append(f"{cached:,}")
        if done:
            t.append(f"   ({100.0 * (done - cached) / done:.1f}% were repeats "
                     f"served from the journal)", style="dim")
        t.append(f"\n  {'generations':<14}", style="dim")
        t.append(f"{len(gens)}")
        if ok:
            t.append(f"\n  {'mean eval':<14}", style="dim")
            t.append(f"{total_t / max(1, done):.2f}s")
        t.append(f"\n  {'journal size':<14}", style="dim")
        t.append(f"{path.stat().st_size / 1e6:.2f} MB\n")

        budget = sum(a.pop * a.gens for a in self.cfg.algorithms)
        if budget:
            frac = min(1.0, done / budget)
            width = 46
            filled = int(width * frac)
            t.append("\n  ")
            t.append("#" * filled, style="green")
            t.append("." * (width - filled), style="dim")
            t.append(f"  {100 * frac:5.1f}%  of {budget:,}\n")
        return t

    def _refresh(self) -> None:
        if self._body is not None:
            self._body.update(self._snapshot())



# ── Campaign ────────────────────────────────────────────────────────────────


def _campaign_ready(cfg: Config) -> bool:
    return cfg.kind == "builtin" and bool(cfg.benchmark_problems) and bool(cfg.algorithms)


# Campaigns this app started, by results root. Module-level so that a reload,
# which rebuilds every screen, does not lose track of them, and so that exit
# can stop them however the app ends.
_STARTED: dict = {}


def _kill_tree(proc: subprocess.Popen) -> None:
    """Stop a campaign runner and every worker process under it."""
    if proc.poll() is not None:
        return
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                           capture_output=True)
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except Exception:                                    # noqa: BLE001
        pass
    try:
        proc.wait(timeout=10)
    except Exception:                                    # noqa: BLE001
        pass


def _stop_started() -> None:
    for proc in list(_STARTED.values()):
        _kill_tree(proc)


atexit.register(_stop_started)

_GREEN = "#9ece6a"
_YELLOW = "#e0af68"
_DIM = "grey42"
_MAGENTA = "#bb9af7"
_CYAN = "#7dcfff"


def _hms(seconds: float) -> str:
    s = int(max(0.0, seconds))
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def _bar(frac: float, width: int, color: str) -> Text:
    frac = max(0.0, min(1.0, frac))
    n = int(round(frac * width))
    t = Text()
    t.append("━" * n, style=color)
    t.append("━" * (width - n), style=_DIM)
    return t


class CampaignScreen(Vertical):
    """A campaign dashboard, and the controls that run the campaign.

    The results tree is the contract between the runner and the UI: a run is
    whatever its meta.json says it is, whether it was produced on this machine
    or on a cluster shard and copied back. The scan runs in a worker thread
    every four seconds and does not re-read a job it has already seen finish,
    so a campaign of thousands of jobs does not stall the interface.

    Start launches `python -m mootation.run.campaign <config> --workers N` as
    a process group of its own, and Stop kills that group; jobs that were
    running then run again next time. Apply writes the worker count into
    `_workers.txt`, which a running campaign picks up within two seconds: a
    higher number starts workers at once, a lower one lets the surplus finish
    its current job, 0 drains the campaign. A campaign started elsewhere is
    shown from its heartbeat and resized the same way, and Stop drains it
    rather than killing it. Closing the app stops a campaign the app started.
    """

    DEFAULT_CSS = """
    CampaignScreen #camp-ctl { height: 1; padding: 0 1; }
    CampaignScreen #camp-ctl Static.label { width: auto; padding: 0 1 0 0; color: $text-muted; }
    CampaignScreen #camp-ctl Input { width: 8; border: none; height: 1; padding: 0; }
    CampaignScreen #camp-ctl Button { border: none; height: 1; min-width: 9; margin: 0 1; padding: 0 1; }
    CampaignScreen #camp-state { width: 1fr; height: 1; padding: 0 1; }
    CampaignScreen #camp-main { height: 1fr; }
    CampaignScreen #camp-dash-scroll { width: 1fr; }
    CampaignScreen #camp-probs-scroll { width: 34; }
    CampaignScreen #camp-dash { padding: 0 1; }
    CampaignScreen #camp-probs { padding: 0 1; }
    """

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.last_snapshot: dict | None = None
        self._scanning = False
        self._seen_done: set = set()
        self._history: deque = deque(maxlen=90)          # (time, done): ~6 minutes of scans
        try:
            self._spec = _camp.campaign_spec(cfg)
            self._jobs = _camp.expand_jobs(cfg, self._spec)
            self._root = _camp.out_root(cfg, self._spec)
            self._error = ""
        except Exception as e:                           # a config that cannot expand
            self._spec, self._jobs, self._root, self._error = None, [], None, str(e)
        self._by_index = {j.index: j for j in self._jobs}
        self._problems = list(dict.fromkeys(j.problem for j in self._jobs))
        self._groups = self._families([a.name for a in cfg.algorithms])

    @staticmethod
    def _families(names: list) -> list:
        try:
            fams = algorithm_families()
        except Exception:                                # noqa: BLE001
            fams = ()
        groups, placed = [], set()
        for family, members in fams:
            inside = [a for a in names if a in members]
            if inside:
                groups.append((family, inside))
                placed.update(inside)
        rest = [a for a in names if a not in placed]
        if rest:
            groups.append(("other", rest))
        return groups

    def compose(self) -> ComposeResult:
        default = os.cpu_count() or 1
        if self._root is not None:
            default = _camp.read_workers(self._root, default) or default
        with Horizontal(id="camp-ctl"):
            yield Static("workers", classes="label")
            yield Input(value=str(default), id="camp-workers", type="integer", max_length=3)
            yield Button("Apply", id="camp-apply", variant="primary")
            yield Button("Start", id="camp-start", variant="success")
            yield Button("Stop", id="camp-stop", variant="error")
            yield Static("", id="camp-state")
        with Horizontal(id="camp-main"):
            with VerticalScroll(id="camp-dash-scroll"):
                yield Static(Text(self._error or "reading the results ...",
                                  style="red" if self._error else _DIM), id="camp-dash")
            with VerticalScroll(id="camp-probs-scroll"):
                yield Static("", id="camp-probs")

    def on_mount(self) -> None:
        self.scan_now()
        self.set_interval(4.0, self.scan_now)

    # ── reading the results, off the UI thread ───────────────────────────────

    def scan_now(self) -> None:
        self._update_state()
        if self._scanning or self._root is None:
            return
        self._scanning = True
        self._scan()

    @work(thread=True)
    def _scan(self) -> None:
        try:
            snap = self._collect()
        except Exception as e:                           # noqa: BLE001
            snap = {"error": f"{type(e).__name__}: {e}"}
        self.app.call_from_thread(self._scan_done, snap)

    def _collect(self) -> dict:
        counts = {"done": 0, "failed": 0, "running": 0, "pending": 0}
        per_alg: dict = {}                               # algorithm -> [done, failed, total]
        per_prob: dict = {}                              # problem -> [done, total]
        failed = []
        for j in self._jobs:
            key = (j.problem, j.algorithm, j.seed)
            if key in self._seen_done:
                st = "done"
            else:
                st = _camp.job_status(self._root, j)
                if st == "done":
                    self._seen_done.add(key)
                elif st == "failed":
                    failed.append(j)
            counts[st] = counts.get(st, 0) + 1
            a = per_alg.setdefault(j.algorithm, [0, 0, 0])
            p = per_prob.setdefault(j.problem, [0, 0])
            a[2] += 1
            p[1] += 1
            if st == "done":
                a[0] += 1
                p[0] += 1
            elif st == "failed":
                a[1] += 1
        errors = []
        for j in failed[-4:]:
            try:
                meta = json.loads((self._root / j.rel_dir / "meta.json").read_text(encoding="utf-8"))
                errors.append(f"{j.algorithm} {j.problem} seed {j.seed}: {meta.get('error', '?')}")
            except (OSError, ValueError):
                pass
        return {"counts": counts, "per_alg": per_alg, "per_prob": per_prob,
                "errors": errors, "beat": _camp.runner_state(self._root), "time": time.time()}

    def _scan_done(self, snap: dict) -> None:
        self._scanning = False
        if "error" in snap:
            self.query_one("#camp-dash", Static).update(Text(snap["error"], style="red"))
            return
        self.last_snapshot = snap
        self._history.append((snap["time"], snap["counts"]["done"]))
        try:
            self.query_one("#camp-dash", Static).update(self._dashboard(snap))
            self.query_one("#camp-probs", Static).update(self._problems_panel(snap))
        except Exception as e:                           # noqa: BLE001
            self.query_one("#camp-dash", Static).update(Text(f"render error: {e!r}", style="red"))
        self._update_state()

    # ── drawing ──────────────────────────────────────────────────────────────

    def _rate(self) -> float | None:
        """Jobs finished per minute over the last few minutes of scans."""
        if len(self._history) < 2:
            return None
        (t0, d0), (t1, d1) = self._history[0], self._history[-1]
        if t1 - t0 < 1.0 or d1 <= d0:
            return None
        return (d1 - d0) / (t1 - t0) * 60.0

    def _flying(self, snap: dict) -> list:
        beat = snap.get("beat")
        if not (beat and beat.get("alive")):
            return []
        return [self._by_index[i] for i in beat.get("running", []) if i in self._by_index]

    def _dashboard(self, snap: dict) -> Panel:
        counts, per_alg = snap["counts"], snap["per_alg"]
        total, done = len(self._jobs), counts["done"]
        beat = snap.get("beat") or {}
        alive = bool(beat.get("alive")) or self._own() is not None
        per_prob = snap["per_prob"]
        probs_done = sum(1 for d, n in per_prob.values() if n and d >= n)
        flying = self._flying(snap)

        title = Align.center(Text(self.cfg.name, style=f"bold {_MAGENTA}"))
        summary = Text(justify="center")
        for label, value, style in (
                ("done ", f"{done}", f"bold {_GREEN}"),
                ("left ", f"{total - done}", "bold"),
                ("problems ", f"{probs_done}/{len(per_prob)}", "bold"),
                ("failed ", f"{counts['failed']}", f"bold {'red' if counts['failed'] else _GREEN}")):
            summary.append(label, style=_DIM)
            summary.append(value, style=style)
            summary.append("   ·   ", style=_DIM)
        summary.append("● running" if alive else "○ stopped",
                       style=f"bold {_GREEN if alive else _DIM}")

        frac = done / total if total else 0.0
        progress = Text("  ")
        progress.append_text(_bar(frac, 48, _GREEN))
        progress.append(f" {100 * frac:5.1f}%", style="bold")

        rate = self._rate()
        info = Text("  ")
        info.append("elapsed ", style=_DIM)
        info.append(_hms(time.time() - beat["started"]) if beat.get("alive") else "-")
        info.append("    speed ", style=_DIM)
        info.append(f"{rate:.1f} jobs/min" if rate else "-")
        info.append("    ETA ", style=_DIM)
        info.append(f"~{_hms((total - done) / rate * 60.0)}" if rate else "-")
        budget = Text("  ")
        budget.append("budget ", style=_DIM)
        budget.append("problem defaults" if self._spec.budget_fe == 0
                      else f"{self._spec.budget_fe:,} evaluations per run")
        budget.append(f"    metrics {', '.join(self._spec.metrics)}", style=_DIM)

        now1 = Text("  ")
        now1.append("NOW RUNNING  ", style=f"bold {_MAGENTA}")
        if beat.get("alive"):
            now1.append(f"{len(flying)} job(s)", style=f"bold {_CYAN}")
            now1.append(f"   workers {beat.get('active', 0)}/{beat.get('target', 0)}", style=_DIM)
        else:
            now1.append("- (the campaign is not running)", style=_DIM)
        now2 = Text("    ")
        if flying:
            now2.append(" · ".join(f"▶ {j.algorithm} {j.problem} s{j.seed}" for j in flying[:6]),
                        style=_CYAN)
            if len(flying) > 6:
                now2.append(f"  +{len(flying) - 6} more", style=_DIM)
        else:
            now2.append("-", style=_DIM)

        active = {j.algorithm for j in flying}
        table = Table(box=None, show_header=True, header_style=f"bold {_MAGENTA}",
                      padding=(0, 2, 0, 0), pad_edge=False, expand=False)
        table.add_column("ALGORITHM", no_wrap=True)
        table.add_column("PROGRESS", no_wrap=True)
        table.add_column("STATUS", no_wrap=True)
        table.add_column("DONE", no_wrap=True, justify="right")
        for family, members in self._groups:
            gd = sum(per_alg.get(a, (0, 0, 0))[0] for a in members)
            gt = sum(per_alg.get(a, (0, 0, 0))[2] for a in members)
            color = _GREEN if gt and gd >= gt else (_YELLOW if gd else _DIM)
            table.add_row(Text(f"● {family}", style=f"bold {color}"), Text(""), Text(""),
                          Text(f"{gd}/{gt}", style=_DIM))
            for a in members:
                d, f, n = per_alg.get(a, (0, 0, 0))
                if a in active:
                    dot, color, status = "▶", _CYAN, "running"
                elif n and d >= n:
                    dot, color, status = "●", _GREEN, "done"
                elif d:
                    dot, color, status = "◐", _YELLOW, "partial"
                else:
                    dot, color, status = "○", _DIM, "waiting"
                name = Text("   ")
                name.append(dot + " ", style=color)
                name.append(a, style=f"bold {_CYAN}" if a in active else "default")
                cell = Text("[", style=_DIM)
                cell.append_text(_bar(d / n if n else 0.0, 18, color if d or a in active else _DIM))
                cell.append("] ", style=_DIM)
                cell.append(f"{100 * d / max(1, n):3.0f}%", style=_DIM)
                stat = Text(status, style=f"bold {color}" if a in active else color)
                if f:
                    stat.append(f" !{f}", style="bold red")
                table.add_row(name, cell, stat,
                              Text(f"{d}/{n}", style=f"bold {_CYAN}" if a in active else "default"))

        blocks = [title, Text(""), summary, Text(""), progress, info, budget, Text(""),
                  now1, now2, Text(""), table]
        if snap["errors"]:
            fails = Text()
            for line in snap["errors"]:
                fails.append("  ! ", style="red")
                fails.append(line[:160] + "\n", style=_DIM)
            blocks += [Text(""), fails]
        return Panel(Group(*blocks), box=box.ROUNDED, border_style=_DIM, padding=(1, 2),
                     title=f"[{_DIM}]campaign[/]", subtitle=f"[{_DIM}]{self._root}[/]")

    def _problems_panel(self, snap: dict) -> Panel:
        per_prob = snap["per_prob"]
        busy = {j.problem for j in self._flying(snap)}
        groups: dict = {}
        for p in self._problems:
            groups.setdefault(_camp.problem_family(p), []).append(p)
        table = Table(box=None, show_header=False, padding=(0, 1, 0, 0), pad_edge=False)
        table.add_column(no_wrap=True)
        finished = 0
        for suite, probs in groups.items():
            complete = sum(1 for p in probs if per_prob.get(p, (0, 0))[1]
                           and per_prob[p][0] >= per_prob[p][1])
            finished += complete
            color = _GREEN if complete == len(probs) else (_YELLOW if complete else _DIM)
            head = Text()
            head.append(f"● {suite} ", style=f"bold {color}")
            head.append(f"{complete}/{len(probs)}", style=_DIM)
            table.add_row(head)
            for p in probs:
                d, n = per_prob.get(p, (0, 0))
                short = p[len(suite):] if p.startswith(suite) and len(p) > len(suite) else p
                if p in busy:
                    dot, color, tail = "▶", _CYAN, f"{d}/{n}"
                elif n and d >= n:
                    dot, color, tail = "●", _GREEN, "✓"
                elif d:
                    dot, color, tail = "◐", _YELLOW, f"{d}/{n}"
                else:
                    dot, color, tail = "○", _DIM, ""
                row = Text("  ")
                row.append(dot + " ", style=color)
                row.append(short.ljust(8), style=f"bold {_CYAN}" if p in busy else "default")
                if tail:
                    row.append(" " + tail, style=_DIM)
                table.add_row(row)
        return Panel(table, box=box.ROUNDED, border_style=_DIM, padding=(1, 1),
                     title=f"[{_DIM}]problems {finished}/{len(self._problems)}[/]")

    # ── running it ───────────────────────────────────────────────────────────

    def _own(self) -> subprocess.Popen | None:
        proc = _STARTED.get(str(self._root))
        return proc if proc is not None and proc.poll() is None else None

    def _heartbeat(self) -> dict | None:
        return _camp.runner_state(self._root) if self._root is not None else None

    def _update_state(self) -> None:
        if self._root is None:
            return
        st = self._heartbeat()
        alive = bool(st and st["alive"])
        t = Text()
        if self._own() is not None or alive:
            t.append("● running", style=f"bold {_GREEN}")
            if alive:
                t.append(f"   workers {st['active']} of {st['target']}", style=_CYAN)
                if self._own() is None:
                    t.append("   started elsewhere", style=_DIM)
            else:
                t.append("   starting ...", style=_DIM)
        else:
            t.append("○ stopped", style="bold")
            if st and st.get("state") in ("finished", "stopped", "broken"):
                t.append(f"   last run {st['state']}", style=_DIM)
        t.append("    s start · x stop · Enter applies workers", style=_DIM)
        self.query_one("#camp-state", Static).update(t)

    def _workers_value(self, at_least: int) -> int | None:
        try:
            n = int(self.query_one("#camp-workers", Input).value)
        except (ValueError, TypeError):
            n = -1
        if n < at_least:
            self.app.notify(f"workers must be a whole number >= {at_least}", severity="error")
            return None
        return n

    def apply(self) -> None:
        n = self._workers_value(0)
        if n is None or self._root is None:
            return
        _camp.write_workers(self._root, n)
        running = self._own() is not None or bool((self._heartbeat() or {}).get("alive"))
        self.app.notify(f"workers: {n}" + ("" if running else " (used when the campaign starts)"))
        self._update_state()

    def start(self) -> None:
        if self._root is None or self.cfg.source_path is None:
            self.app.notify("this config has no campaign to run", severity="warning")
            return
        if self._own() is not None or bool((self._heartbeat() or {}).get("alive")):
            self.app.notify("the campaign is already running; Apply changes its workers",
                            severity="warning")
            return
        n = self._workers_value(1)
        if n is None:
            return
        self._root.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        # The package this app was imported from, so a checkout runs without installing.
        here = str(Path(__file__).resolve().parents[2])
        env["PYTHONPATH"] = here + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        kw = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt"
              else {"start_new_session": True})
        log_path = self._root / "run_stdout.log"
        with log_path.open("a", encoding="utf-8") as log:
            _STARTED[str(self._root)] = subprocess.Popen(
                [sys.executable, "-m", "mootation.run.campaign", str(self.cfg.source_path),
                 "--workers", str(n)],
                cwd=str(self.cfg.source_path.parent), env=env,
                stdout=log, stderr=subprocess.STDOUT, **kw)
        self.app.notify(f"started with {n} worker(s); output goes to {log_path}")
        self._update_state()

    def stop(self) -> None:
        proc = self._own()
        if proc is not None:
            _kill_tree(proc)
            self.app.notify("stopped; the jobs that were running will run again next time")
        elif bool((self._heartbeat() or {}).get("alive")):
            _camp.write_workers(self._root, 0)
            self.app.notify("started elsewhere, so draining it: running jobs finish, then it stops")
        else:
            self.app.notify("nothing is running", severity="warning")
        self._update_state()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        action = {"camp-apply": self.apply, "camp-start": self.start,
                  "camp-stop": self.stop}.get(event.button.id)
        if action is not None:
            action()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "camp-workers":
            self.apply()
            self.query_one("#camp-dash-scroll").focus()


# ── Compare ─────────────────────────────────────────────────────────────────


class CompareScreen(VerticalScroll):
    """Two readings of the finished runs, switched with `t`.

    medians  median [q1, q3] of a final indicator per problem x algorithm, over
             seeds, the best per row in green.
    ranks    one row per algorithm: its rank on every problem (by that median,
             1 = best, ties averaged) averaged overall, per family and per
             objective count, with the number of problems it won. With dozens
             of algorithms this is the view that says which one is good where.

    Type a metric name in the box (igd, igdp, hv) and press Enter; press `e`
    to export the current view as CSV next to the results.
    """

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.metric = "igd"
        self.view = "medians"
        self._table: DataTable | None = None
        self._note: Static | None = None

    def compose(self) -> ComposeResult:
        yield Input(value=self.metric, placeholder="metric: igd | igdp | hv", id="cmp-metric")
        self._note = Static(Text(""), classes="panel")
        yield self._note
        self._table = DataTable(zebra_stripes=True, id="cmp-table")
        self._fill()
        yield self._table

    def _root(self) -> Path:
        return _camp.out_root(self.cfg, _camp.campaign_spec(self.cfg))

    def _fill(self) -> None:
        if self._table is None:
            return
        rows = _camp.scan_results(self._root())
        done = sum(1 for r in rows if r["status"] == "done")
        if self.view == "ranks":
            self._fill_ranks(rows, done)
        else:
            self._fill_medians(rows, done)

    def _fill_medians(self, rows: list, done: int) -> None:
        table = _camp.compare_table(rows, self.metric)
        algs = sorted({a for d in table.values() for a in d})
        self._table.clear(columns=True)
        self._table.add_columns("problem", *algs)
        lower_better = self.metric != "hv"
        for prob in sorted(table):
            cells = [prob]
            vals = {a: table[prob][a][0] for a in algs if a in table[prob]}
            best = (min if lower_better else max)(vals, key=vals.get) if vals else None
            for a in algs:
                if a in table[prob]:
                    med, q1, q3, n = table[prob][a]
                    txt = f"{med:.4g} [{q1:.3g}, {q3:.3g}] n={n}"
                    cells.append(Text(txt, style="bold green" if a == best else ""))
                else:
                    cells.append(Text("-", style="dim"))
            self._table.add_row(*cells)
        if self._note is not None:
            self._note.update(Text(
                f"{done} finished runs under {self._root()} — metric '{self.metric}' "
                f"({'lower' if lower_better else 'higher'} is better); best per row in green; "
                f"t: mean ranks, e: export CSV", style="dim"))

    def _fill_ranks(self, rows: list, done: int) -> None:
        ranks = _camp.rank_table(rows, self.metric)
        groups = ranks["groups"]
        self._table.clear(columns=True)
        self._table.add_columns("algorithm", "mean rank", "wins", "problems", *groups)
        best = {}
        for g in ("all", *groups):
            vals = [e[g][0] for _, e in ranks["algorithms"] if g in e]
            best[g] = min(vals) if vals else None

        def cell(e: dict, g: str) -> Text:
            if g not in e:
                return Text("-", style="dim")
            return Text(f"{e[g][0]:.2f}", style="bold green" if e[g][0] == best[g] else "")

        for alg, e in ranks["algorithms"]:
            self._table.add_row(alg, cell(e, "all"), str(e["wins"]), str(e["all"][1]),
                                *(cell(e, g) for g in groups))
        if self._note is not None:
            self._note.update(Text(
                f"{done} finished runs under {self._root()} — mean rank by median "
                f"'{self.metric}' over {ranks['n_problems']} problem(s), 1 = best, ties share "
                f"the average; best per column in green; t: medians, e: export CSV",
                style="dim"))

    def toggle_view(self) -> None:
        self.view = "ranks" if self.view == "medians" else "medians"
        self._fill()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "cmp-metric" and event.value.strip() in ("igd", "igdp", "hv"):
            self.metric = event.value.strip()
            self._fill()
            # Hand the keys back: with the box focused, `t` and `e` would be typed
            # into it instead of reaching the app.
            if self._table is not None:
                self._table.focus()

    def export_csv(self) -> Path:
        rows = _camp.scan_results(self._root())
        if self.view == "ranks":
            path = self._root() / f"ranks_{self.metric}.csv"
            _camp.write_rank_csv(_camp.rank_table(rows, self.metric), path)
            return path
        table = _camp.compare_table(rows, self.metric)
        path = self._root() / f"compare_{self.metric}.csv"
        _camp.write_compare_csv(table, path, self.metric)
        return path


# ── Explore ─────────────────────────────────────────────────────────────────


def _ascii_plot(xs: list, ys: list, width: int = 60, height: int = 12,
                log_y: bool = True) -> str:
    """A terminal plot of ys against xs (ascending in x)."""
    import math as _m
    pts = [(x, y) for x, y in zip(xs, ys)
           if isinstance(y, (int, float)) and _m.isfinite(y) and (y > 0 or not log_y)]
    if len(pts) < 2:
        return "(not enough finite points to plot)"
    xs = [p[0] for p in pts]
    ys = [(_m.log10(p[1]) if log_y else p[1]) for p in pts]
    x0, x1 = xs[0], xs[-1]
    y0, y1 = min(ys), max(ys)
    if y1 - y0 < 1e-12:
        y1 = y0 + 1e-12
    grid = [[" "] * width for _ in range(height)]
    for x, y in zip(xs, ys):
        c = int((x - x0) / max(1e-12, x1 - x0) * (width - 1))
        r = int((y1 - y) / (y1 - y0) * (height - 1))
        grid[r][c] = "*"
    lines = []
    for r in range(height):
        yv = y1 - (y1 - y0) * r / (height - 1)
        label = f"{10 ** yv:9.3g}" if log_y else f"{yv:9.3g}"
        lines.append(f"{label} |" + "".join(grid[r]))
    lines.append(" " * 10 + "+" + "-" * width)
    lines.append(" " * 11 + f"FE {x0:.0f}" + " " * max(1, width - 20) + f"{x1:.0f}")
    return "\n".join(lines)


class ExploreScreen(VerticalScroll):
    """One run at a time: pick it in the tree, see its trajectory."""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self._plot: Static | None = None
        self._tree: Tree | None = None
        self.metric = "igd"

    def compose(self) -> ComposeResult:
        yield Input(value=self.metric, placeholder="metric: igd | igdp | hv", id="exp-metric")
        self._tree = Tree("results")
        self._tree.root.expand()
        self._build_tree()
        yield self._tree
        self._plot = Static(Text("select a run in the tree", style="dim"), classes="panel")
        yield self._plot

    def _build_tree(self) -> None:
        if self._tree is None:
            return
        root = _camp.out_root(self.cfg, _camp.campaign_spec(self.cfg))
        rows = _camp.scan_results(root)
        by_prob: dict = {}
        for r in rows:
            by_prob.setdefault(r["problem"], {}).setdefault(r["algorithm"], []).append(r)
        self._tree.root.remove_children()
        for prob in sorted(by_prob):
            pn = self._tree.root.add(prob)
            for alg in sorted(by_prob[prob]):
                an = pn.add(alg)
                for r in sorted(by_prob[prob][alg], key=lambda r: (r["seed"] or 0)):
                    label = f"seed {r['seed']}  {r['status']}"
                    v = r["final"].get(self.metric)
                    if isinstance(v, float):
                        label += f"  {self.metric}={v:.4g}"
                    an.add_leaf(label, data=r["dir"])

    def show_run(self, run_dir) -> None:
        if self._plot is None:
            return
        traj = _camp.read_trajectory(Path(run_dir))
        xs = [t.get("fe", 0) for t in traj]
        ys = [t.get(self.metric) for t in traj]
        body = Text()
        body.append(f"{run_dir}\n", style="dim")
        body.append(f"{self.metric} vs evaluations ({len(traj)} records)\n", style="bold")
        body.append(_ascii_plot(xs, ys, log_y=(self.metric != "hv")) + "\n")
        if traj:
            body.append("\n  fe        gen     "
                        + "   ".join(f"{m:>9}" for m in ("igd", "igdp", "hv")) + "\n",
                        style="dim")
            step = max(1, len(traj) // 12)
            shown = traj[::step]
            if shown[-1] is not traj[-1]:
                shown.append(traj[-1])
            for t in shown:
                cells = []
                for m in ("igd", "igdp", "hv"):
                    v = t.get(m)
                    cells.append(f"{v:9.4g}" if isinstance(v, float) else f"{'-':>9}")
                body.append(f"  {t.get('fe', 0):<9} {t.get('gen', 0):<7} "
                            + "   ".join(cells) + "\n")
        self._plot.update(body)

    def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
        if event.node.data:
            self.show_run(event.node.data)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "exp-metric" and event.value.strip() in ("igd", "igdp", "hv"):
            self.metric = event.value.strip()
            self._build_tree()


# ── The app ─────────────────────────────────────────────────────────────────


class MootationApp(App):
    CSS = """
    .panel { padding: 1 2; }
    DataTable { height: auto; max-height: 100%; }
    Tree { padding: 1 2; }
    """
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("r", "reload", "Reload config"),
        ("e", "export", "Export compare CSV"),
        ("t", "toggle_view", "Medians / ranks"),
        ("s", "start_campaign", "Start campaign"),
        ("x", "stop_campaign", "Stop campaign"),
    ]

    def __init__(self, config_path: str | Path) -> None:
        super().__init__()
        self.config_path = Path(config_path)
        self.cfg = load(self.config_path)
        self.problems = validate(self.cfg)

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent(initial="tab-campaign" if _campaign_ready(self.cfg) else ""):
            with TabPane("Config", id="tab-config"):
                yield ConfigScreen(self.cfg, self.problems)
            with TabPane("Problems", id="tab-problems"):
                yield ProblemsScreen(self.cfg)
            with TabPane("Algorithms", id="tab-algorithms"):
                yield AlgorithmsScreen(self.cfg)
            with TabPane("Monitor", id="tab-monitor"):
                yield MonitorScreen(self.cfg)
            if _campaign_ready(self.cfg):
                with TabPane("Campaign", id="tab-campaign"):
                    yield CampaignScreen(self.cfg)
                with TabPane("Compare", id="tab-compare"):
                    yield CompareScreen(self.cfg)
                with TabPane("Explore", id="tab-explore"):
                    yield ExploreScreen(self.cfg)
        yield Footer()

    def _campaign(self) -> CampaignScreen | None:
        try:
            return self.query_one(CampaignScreen)
        except Exception:
            self.notify("no Campaign tab in this config", severity="warning")
            return None

    def action_start_campaign(self) -> None:
        screen = self._campaign()
        if screen is not None:
            screen.start()

    def action_stop_campaign(self) -> None:
        screen = self._campaign()
        if screen is not None:
            screen.stop()

    def action_toggle_view(self) -> None:
        try:
            screen = self.query_one(CompareScreen)
        except Exception:
            self.notify("no Compare tab in this config", severity="warning")
            return
        screen.toggle_view()

    def action_export(self) -> None:
        try:
            screen = self.query_one(CompareScreen)
        except Exception:
            self.notify("no Compare tab in this config", severity="warning")
            return
        try:
            path = screen.export_csv()
        except Exception as e:
            self.notify(str(e), severity="error", timeout=8)
            return
        self.notify(f"wrote {path}")

    def on_mount(self) -> None:
        self.title = f"MOOtation — {self.cfg.name}"
        self.sub_title = str(self.config_path)

    def on_unmount(self) -> None:
        # Closing the app stops a campaign it started. A reload rebuilds the
        # screens but keeps the app, so this is the place, not the screen.
        _stop_started()

    def action_reload(self) -> None:
        """Re-read the file. The config is edited elsewhere; this picks it up."""
        try:
            self.cfg = load(self.config_path)
            self.problems = validate(self.cfg)
        except Exception as e:                      # a half-saved file
            self.notify(str(e), severity="error", timeout=8)
            return
        self.notify("reloaded"
                    + (f" — {len(self.problems)} problem(s)" if self.problems
                       else " — OK"),
                    severity="warning" if self.problems else "information")
        self.refresh(recompose=True)


def run(config_path: str | Path) -> None:
    """Open the TUI on a run description."""
    MootationApp(config_path).run()

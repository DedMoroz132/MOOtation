# SPDX-License-Identifier: Apache-2.0
"""The Textual application: read-only screens over a run description.

Config / Problems / Algorithms / Monitor always; Campaign / Compare / Explore
when the config describes a builtin benchmark campaign (mootation.run.campaign).

Nothing here edits the config. The screens render what `config.load` and
`config.validate` already produced, plus what the journal on disk says, so the
UI cannot disagree with `--check`: they call the same functions.
"""

from __future__ import annotations

from pathlib import Path

from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import (
    DataTable, Footer, Header, Input, Static, TabbedContent, TabPane, Tree,
)

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


class CampaignScreen(VerticalScroll):
    """Progress of a benchmark campaign, read from the meta.json files.

    The results tree is the contract between the runner and the UI: a run
    is whatever its meta.json says it is, whether it
    was produced on this machine or on a cluster shard and copied back.
    """

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self._body: Static | None = None
        self._table: DataTable | None = None

    def compose(self) -> ComposeResult:
        self._body = Static(self._summary(), classes="panel")
        yield self._body
        self._table = DataTable(zebra_stripes=True, id="camp-table")
        self._fill()
        yield self._table

    def on_mount(self) -> None:
        self.set_interval(3.0, self._refresh)

    def _jobs(self):
        spec = _camp.campaign_spec(self.cfg)
        return spec, _camp.expand_jobs(self.cfg, spec), _camp.out_root(self.cfg, spec)

    def _summary(self) -> Text:
        t = Text()
        try:
            spec, jobs, root = self._jobs()
        except Exception as e:                       # a config that cannot expand
            t.append(str(e), style="red")
            return t
        counts = {"done": 0, "failed": 0, "running": 0, "pending": 0}
        for j in jobs:
            st = _camp.job_status(root, j)
            counts[st] = counts.get(st, 0) + 1
        t.append("campaign\n", style="bold")
        t.append(f"  {'results':<12}", style="dim"); t.append(f"{root}\n")
        t.append(f"  {'jobs':<12}", style="dim"); t.append(f"{len(jobs)}")
        t.append(f"   done {counts['done']}", style="green")
        t.append(f"   running {counts['running']}", style="yellow")
        t.append(f"   failed {counts['failed']}", style="red" if counts["failed"] else "dim")
        t.append(f"   pending {counts['pending']}\n", style="dim")
        t.append(f"  {'budget':<12}", style="dim")
        t.append(("problem defaults" if spec.budget_fe == 0 else f"{spec.budget_fe:,} FE")
                 + f", record every {spec.record_every} gen, metrics {', '.join(spec.metrics)}\n")
        frac = counts["done"] / max(1, len(jobs))
        width = 46
        filled = int(width * frac)
        t.append("\n  ")
        t.append("#" * filled, style="green")
        t.append("." * (width - filled), style="dim")
        t.append(f"  {100 * frac:5.1f}%\n")
        return t

    def _fill(self) -> None:
        if self._table is None:
            return
        try:
            spec, jobs, root = self._jobs()
        except Exception:
            return
        algs = [a.name for a in self.cfg.algorithms]
        probs = list(dict.fromkeys(j.problem for j in jobs))
        self._table.clear(columns=True)
        self._table.add_columns("problem", *algs)
        cell: dict = {}
        for j in jobs:
            st = _camp.job_status(root, j)
            c = cell.setdefault((j.problem, j.algorithm), {"done": 0, "failed": 0, "total": 0})
            c["total"] += 1
            if st in ("done", "failed"):
                c[st] += 1
        for p in probs:
            row = [p]
            for a in algs:
                c = cell.get((p, a), {"done": 0, "failed": 0, "total": 0})
                s = f"{c['done']}/{c['total']}"
                if c["failed"]:
                    s += f" !{c['failed']}"
                style = ("green" if c["done"] == c["total"] and c["total"]
                         else "red" if c["failed"] else "")
                row.append(Text(s, style=style))
            self._table.add_row(*row)

    def _refresh(self) -> None:
        if self._body is not None:
            self._body.update(self._summary())
        self._fill()


# ── Compare ─────────────────────────────────────────────────────────────────


class CompareScreen(VerticalScroll):
    """Median [q1, q3] of a final indicator per problem x algorithm, over seeds.

    Type a metric name in the box (igd, igdp, hv) and press Enter; press `e`
    to export the table as CSV next to the results.
    """

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.metric = "igd"
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
        done = sum(1 for r in rows if r["status"] == "done")
        if self._note is not None:
            self._note.update(Text(
                f"{done} finished runs under {self._root()} — metric '{self.metric}' "
                f"({'lower' if lower_better else 'higher'} is better); best per row in green; "
                f"press e to export CSV", style="dim"))

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "cmp-metric" and event.value.strip() in ("igd", "igdp", "hv"):
            self.metric = event.value.strip()
            self._fill()

    def export_csv(self) -> Path:
        rows = _camp.scan_results(self._root())
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
    ]

    def __init__(self, config_path: str | Path) -> None:
        super().__init__()
        self.config_path = Path(config_path)
        self.cfg = load(self.config_path)
        self.problems = validate(self.cfg)

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent():
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

#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Headless smoke test for the terminal interface.

Textual can be driven without a terminal through `App.run_test`, so every
screen is actually composed and rendered here rather than merely imported. A
screen that raises on a config shape it did not expect — a builtin problem with
no steps, an external one with no benchmarks — fails here.

The shipped campaign has no results next to it, so its Campaign and Compare
tabs render empty. It is exercised once more over a handful of finished runs
written by hand, which is the only way the dashboard, the medians and the ranks
draw real rows. Start is not pressed: it would launch a real campaign.

Skipped, not failed, when Textual is absent: the rest of mootation_run does not
need it.

    python python/test_tui.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

try:
    import textual  # noqa: F401
except ImportError:
    print("skipped: Textual is not installed (pip install textual)")
    raise SystemExit(0)

from mootation.tui.app import CampaignScreen, MootationApp  # noqa: E402

CONFIGS = ["examples/demo.toml", "examples/bench.toml", "examples/airfoil.toml",
           "examples/campaign.toml"]
TABS = ("tab-config", "tab-problems", "tab-algorithms", "tab-monitor")
CAMPAIGN_TABS = ("tab-campaign", "tab-compare", "tab-explore")

FAKE_PROBLEMS = (("DTLZ2_3D", 3), ("WFG4_5D", 5))
FAKE_ALGORITHMS = ("nsga2", "nsga3", "rvea")


def fake_campaign(td: Path) -> Path:
    """The shipped campaign config, with finished runs written by hand beside it."""
    cfg = td / "campaign.toml"
    cfg.write_text((HERE / "examples/campaign.toml").read_text(encoding="utf-8"),
                   encoding="utf-8")
    root = td / "results" / "dtlz_wfg_sweep"
    for problem, m in FAKE_PROBLEMS:
        for k, alg in enumerate(FAKE_ALGORITHMS):
            for seed in (1, 2):
                d = root / problem / alg / f"run_{seed}"
                d.mkdir(parents=True)
                meta = {"status": "done", "problem": problem, "algorithm": alg,
                        "seed": seed, "n_objs": m, "fe": 1000, "seconds": 1.0,
                        "final": {"igd": 0.1 * (k + 1) + 0.01 * seed,
                                  "igdp": 0.05 * (k + 1), "hv": 0.9 - 0.1 * k}}
                (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
                traj = [{"gen": g, "fe": 100 * g, "t": 0.1 * g, "n": 10, "finite": True,
                         "igd": 1.0 / (g + 1), "igdp": 0.5 / (g + 1), "hv": 0.1 * g}
                        for g in range(5)]
                (d / "trajectory.jsonl").write_text(
                    "".join(json.dumps(t) + "\n" for t in traj), encoding="utf-8")
    return cfg


async def exercise(cfg: Path, finished_runs: bool = False) -> None:
    app = MootationApp(cfg)
    async with app.run_test(size=(160, 60)) as pilot:
        present = {t.id for t in app.query("TabPane")}
        tabs = TABS + tuple(t for t in CAMPAIGN_TABS if t in present)
        if cfg.name == "campaign.toml":
            assert set(CAMPAIGN_TABS) <= present, "campaign config must show the campaign tabs"
            assert app.query_one("TabbedContent").active == "tab-campaign", \
                "a campaign config opens on its dashboard"
        for tab in tabs:
            app.query_one("TabbedContent").active = tab
            await pilot.pause()
        if "tab-campaign" in present:
            app.query_one("TabbedContent").active = "tab-campaign"
            screen = app.query_one(CampaignScreen)
            screen.scan_now()
            await app.workers.wait_for_complete()
            await pilot.pause()
            snap = screen.last_snapshot
            assert snap is not None, "the dashboard never finished a scan"
            if finished_runs:
                done = len(FAKE_PROBLEMS) * len(FAKE_ALGORITHMS) * 2
                assert snap["counts"]["done"] == done, snap["counts"]
                # Apply writes the worker count a running campaign would pick up;
                # Stop with nothing running must only say so.
                screen.query_one("#camp-workers").value = "3"
                screen.apply()
                screen.stop()
                await pilot.pause()
                results = cfg.parent / "results" / "dtlz_wfg_sweep"
                assert (results / "_workers.txt").read_text(encoding="utf-8").strip() == "3"
        if "tab-compare" in present:
            # Both readings of the results, and the export of each.
            app.query_one("TabbedContent").active = "tab-compare"
            await pilot.pause()
            table = app.query_one("#cmp-table")
            if finished_runs:
                assert table.row_count == len(FAKE_PROBLEMS), \
                    f"medians: one row per problem, got {table.row_count}"
                app.action_export()
                await pilot.pause()
            app.action_toggle_view()
            await pilot.pause()
            if finished_runs:
                assert table.row_count == len(FAKE_ALGORITHMS), \
                    f"ranks: one row per algorithm, got {table.row_count}"
                app.action_export()
                await pilot.pause()
            app.action_toggle_view()
            await pilot.pause()
        # Reload re-reads the file: the config is edited outside the UI, and a
        # half-saved file must not take the app down.
        app.action_reload()
        await pilot.pause()
    suffix = " + finished runs, dashboard, both compare views" if finished_runs else ""
    print(f"  ok    {cfg.name}{suffix}: {len(tabs)} screens + reload")


def main() -> int:
    failed = []
    for rel in CONFIGS:
        cfg = HERE / rel
        try:
            asyncio.run(exercise(cfg))
        except Exception as e:
            failed.append(rel)
            print(f"  FAIL  {rel}: {type(e).__name__}: {e}")
    rel = "examples/campaign.toml + finished runs"
    with tempfile.TemporaryDirectory() as td:
        try:
            cfg = fake_campaign(Path(td))
            asyncio.run(exercise(cfg, finished_runs=True))
            root = Path(td) / "results" / "dtlz_wfg_sweep"
            for name in ("compare_igd.csv", "ranks_igd.csv"):
                assert (root / name).is_file(), f"export did not write {name}"
        except Exception as e:
            failed.append(rel)
            print(f"  FAIL  {rel}: {type(e).__name__}: {e}")
    total = len(CONFIGS) + 1
    print("")
    print(f"{total - len(failed)}/{total} configs rendered")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

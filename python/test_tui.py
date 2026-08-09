#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Headless smoke test for the terminal interface.

Textual can be driven without a terminal through `App.run_test`, so every
screen is actually composed and rendered here rather than merely imported. A
screen that raises on a config shape it did not expect — a builtin problem with
no steps, an external one with no benchmarks — fails here.

Skipped, not failed, when Textual is absent: the rest of mootation_run does not
need it.

    python python/test_tui.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

try:
    import textual  # noqa: F401
except ImportError:
    print("skipped: Textual is not installed (pip install textual)")
    raise SystemExit(0)

from mootation.run.tui.app import MootationApp  # noqa: E402

CONFIGS = ["examples/demo.toml", "examples/bench.toml", "examples/airfoil.toml"]
TABS = ("tab-config", "tab-problems", "tab-algorithms", "tab-monitor")


async def exercise(cfg: Path) -> None:
    app = MootationApp(cfg)
    async with app.run_test() as pilot:
        for tab in TABS:
            app.query_one("TabbedContent").active = tab
            await pilot.pause()
        # Reload re-reads the file: the config is edited outside the UI, and a
        # half-saved file must not take the app down.
        app.action_reload()
        await pilot.pause()
    print(f"  ok    {cfg.name}: {len(TABS)} screens + reload")


def main() -> int:
    failed = []
    for rel in CONFIGS:
        cfg = HERE / rel
        try:
            asyncio.run(exercise(cfg))
        except Exception as e:
            failed.append(rel)
            print(f"  FAIL  {rel}: {type(e).__name__}: {e}")
    print("")
    print(f"{len(CONFIGS) - len(failed)}/{len(CONFIGS)} configs rendered")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

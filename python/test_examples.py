#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run every Python example.

An example that no longer runs is worse than no example, because it is the
first code anyone copies. Each is executed as a subprocess exactly the way a
user would run it, and a non-zero exit fails the test.

Skipped, not failed, when the compiled extension is absent — the examples are
about the optimizer, and a machine without it has nothing to check here.

    python python/test_examples.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

sys.path.insert(0, str(HERE))
try:
    import mootation
    mootation.algorithms()          # forces the extension to load
except Exception as e:              # noqa: BLE001 - any failure means "no build"
    print(f"skipped: the compiled extension is not available ({e})")
    raise SystemExit(0)

EXAMPLES = [
    "01_basic.py",
    "02_constraints.py",
    "03_benchmarks.py",
    "04_expensive.py",
    "05_many_objective.py",
    "06_restart.py",
]


def main() -> int:
    env_path = str(HERE)
    failed = []
    for name in EXAMPLES:
        script = HERE / "examples" / name
        if not script.is_file():
            failed.append(name)
            print(f"  MISSING  {name}")
            continue
        proc = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(ROOT),
            env={**__import__("os").environ, "PYTHONPATH": env_path},
            capture_output=True, text=True, timeout=900,
        )
        if proc.returncode == 0:
            print(f"  ok       {name}")
        else:
            failed.append(name)
            print(f"  FAIL     {name}  (exit {proc.returncode})")
            tail = (proc.stderr or proc.stdout).strip().splitlines()[-6:]
            for line in tail:
                print(f"           {line}")

    print("")
    print(f"{len(EXAMPLES) - len(failed)}/{len(EXAMPLES)} examples ran")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

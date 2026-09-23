#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Are the defaults unchanged? Build tests/compat_dump.cpp twice and compare.

    python tools/compat_check.py [--baseline REF] [--cxx g++|clang++|cl]

The driver is compiled once against the headers of a BASELINE commit and once
against the working tree's, on the same machine with the same compiler, and
the two fingerprint lists must be identical: every algorithm, run at its
defaults, returns bit for bit the population it returned at the baseline.
A stored golden population could not do this — the standard library's
distributions and the compiler's floating-point code differ between MSVC,
libstdc++ and libc++ — but the same compiler on the same machine is
deterministic.

The baseline is tests/compat_baseline.txt: a commit, and after it any
algorithm keys whose default behaviour a later change altered ON PURPOSE (a
fix), one per line with the commit and the reason. Those are reported, not
failed. Exit status: 0 identical (or only listed changes), 1 a difference.

On Windows run it from a Developer prompt (cl on PATH), or pass --cxx g++.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def baseline() -> tuple[str, dict]:
    ref, allowed = None, {}
    for line in (ROOT / "tests" / "compat_baseline.txt").read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        if ref is None:
            ref = line
        else:
            key, _, why = line.partition(" ")
            allowed[key] = why.strip()
    if ref is None:
        raise SystemExit("tests/compat_baseline.txt names no baseline commit")
    return ref, allowed


def export_include(ref: str, dest: Path) -> Path:
    """The baseline's include/ tree, written out without touching the checkout."""
    archive = subprocess.run(["git", "archive", "--format=tar", ref, "include"],
                             cwd=ROOT, capture_output=True, check=True).stdout
    import io
    import tarfile
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(dest, filter="data")
    return dest / "include"


def build(cxx: str, include: Path, out: Path) -> None:
    src = ROOT / "tests" / "compat_dump.cpp"
    if cxx == "cl":
        cmd = ["cl", "/nologo", "/std:c++17", "/O2", "/EHsc", "/bigobj", "/DNDEBUG",
               f"/I{include}", f"/I{ROOT / 'tests'}", str(src), f"/Fe:{out}",
               f"/Fo:{out.parent}\\"]
    else:
        cmd = [cxx, "-std=c++17", "-O2", "-DNDEBUG", f"-I{include}", f"-I{ROOT / 'tests'}",
               str(src), "-o", str(out)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)


def run(exe: Path) -> dict:
    out = subprocess.run([str(exe)], capture_output=True, text=True, check=True).stdout
    rows = {}
    for line in out.splitlines():
        parts = line.split(" ", 2)
        rows[(parts[0], parts[1])] = parts[2]
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--baseline", help="commit to compare against (default: tests/compat_baseline.txt)")
    ap.add_argument("--cxx", default="cl" if os.name == "nt" else "g++")
    ap.add_argument("--keep", help="write both fingerprint lists into this directory")
    ap.add_argument("--cache", help="keep the baseline's driver here and reuse it while the "
                                    "baseline and the compiler stay the same")
    args = ap.parse_args(argv)

    ref, allowed = baseline()
    ref = args.baseline or ref
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "base").mkdir()
        (td / "head").mkdir()
        exe = ".exe" if os.name == "nt" else ""
        base_exe = td / "base" / f"dump{exe}"
        if args.cache:
            sha = subprocess.run(["git", "rev-parse", ref], cwd=ROOT, capture_output=True,
                                 text=True, check=True).stdout.strip()
            base_exe = Path(args.cache) / f"compat_{sha[:12]}_{Path(args.cxx).stem}{exe}"
            base_exe.parent.mkdir(parents=True, exist_ok=True)
        if not base_exe.exists():
            build(args.cxx, export_include(ref, td / "base"), base_exe)
        build(args.cxx, ROOT / "include", td / "head" / f"dump{exe}")
        before = run(base_exe)
        after = run(td / "head" / f"dump{exe}")
        if args.keep:
            keep = Path(args.keep)
            keep.mkdir(parents=True, exist_ok=True)
            for name, rows in (("baseline", before), ("head", after)):
                (keep / f"{name}.txt").write_text(
                    "".join(f"{k} {p} {v}\n" for (k, p), v in sorted(rows.items())),
                    encoding="utf-8")

    changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
    unexpected = [k for k in changed if k[0] not in allowed]
    for k in changed:
        tag = "declared" if k[0] in allowed else "CHANGED"
        print(f"{tag:<9} {k[0]:<12} {k[1]:<8} {before.get(k)} -> {after.get(k)}"
              + (f"   ({allowed[k[0]]})" if k[0] in allowed else ""))
    print(f"{len(before)} fingerprints against {ref}: {len(changed)} differ, "
          f"{len(unexpected)} not declared")
    return 1 if unexpected else 0


if __name__ == "__main__":
    sys.exit(main())

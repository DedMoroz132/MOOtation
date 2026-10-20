# SPDX-License-Identifier: Apache-2.0
"""Does this build give the library's runs bit for bit? (task 14 of 2026-10-20, §7)

    mootation selftest                 (python -m mootation selftest)
    mootation.selftest()               -> (True, []) or (False, [what differs ...])

Since task 14 a run at a given seed is the same on every supported platform:
the cores' random numbers (random.hpp), their mathematics and the suite's
(detmath.hpp), no fused multiply-adds (fp_contract.hpp), sorts with one result.
A few short runs — the algorithms of CASES on two problems of the C++ suite, 300
evaluations each — are fingerprinted (FNV-1a of the answer's variables and
objectives, little-endian doubles) and compared with the list the repository
ships (selftest_golden.txt). Seconds. A difference names the run and what may
cause it: a build with -ffast-math, -Ofast, /fp:fast or with contraction into
fused multiply-adds (the library's CMake target forbids it; a build without
CMake may not), denormals flushed to zero in this process (another library
built with -ffast-math can switch that on), a 32-bit x87 build.

    mootation selftest --write         rewrite the list (maintainers: after a
                                       change that changes runs, with the reason
                                       in CHANGELOG)
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

CASES = ("nsga2", "nsga3", "moead_de", "sms_emoa", "mo_cma_es", "spea2_sde", "ibea_eplus",
         "rvea", "mosa", "nsga2_sdr")
PROBLEMS = ("ZDT1", "DTLZ2_3D")
EVALUATIONS = 300
POP = {"ZDT1": 20, "DTLZ2_3D": 21}     # 21: a Das-Dennis size at three objectives
SEED = 1
GOLDEN = Path(__file__).with_name("selftest_golden.txt")


def _fnv1a(rows) -> str:
    h = 1469598103934665603
    for row in rows:
        for v in row:
            for b in struct.pack("<d", float(v)):
                h ^= b
                h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return f"{h:016x}"


def fingerprints() -> dict:
    """{(algorithm, problem): fingerprint} of this build's runs."""
    from . import minimize
    from .benchmarks import get
    out = {}
    for name in PROBLEMS:
        p = get(name)
        for alg in CASES:
            r = minimize(p.evaluate, p.bounds, p.n_obj, algorithm=alg, pop_size=POP[name],
                         max_evaluations=EVALUATIONS, seed=SEED)
            out[(alg, name)] = _fnv1a(list(r.variables) + list(r.objectives))
    return out


def _golden() -> dict:
    out = {}
    if not GOLDEN.is_file():
        return out
    for line in GOLDEN.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].split()
        if len(line) == 3:
            out[(line[0], line[1])] = line[2]
    return out


def _hints() -> list:
    hints = []
    try:
        from . import _core
        env = dict(_core.fp_env())
        if env.get("ftz") or env.get("daz"):
            hints.append(f"denormals are flushed to zero or read as zero in this process ({env}): "
                         "a library built with -ffast-math may have switched it on")
    except (ImportError, AttributeError):
        hints.append("no compiled extension (mootation._core): the cores do not run")
    hints.append("a build with -ffast-math, -Ofast, /fp:fast or with contraction into fused "
                 "multiply-adds (-ffp-contract other than off; a build without the library's "
                 "CMake target, e.g. with -march=native) gives other runs")
    return hints


def selftest(verbose: bool = False) -> tuple:
    """(True, []) when every run matches the shipped fingerprints, else (False,
    the lines saying what differs and what may cause it)."""
    want = _golden()
    if not want:
        return False, [f"no fingerprint list ({GOLDEN.name}) in this installation"]
    got = fingerprints()
    bad = [f"{a} on {p}: {got.get((a, p), 'not run')} (expected {h})"
           for (a, p), h in sorted(want.items()) if got.get((a, p)) != h]
    if verbose:
        for (a, p), h in sorted(got.items()):
            print(f"  {a:12s} {p:10s} {h} {'ok' if want.get((a, p)) == h else 'DIFFERS'}")
    return (not bad), (bad + _hints() if bad else [])


def write() -> None:
    got = fingerprints()
    lines = ["# mootation selftest: the fingerprints of short runs (selftest.py), the same on every",
             "# supported platform; rewritten only with a change that changes runs (CHANGELOG)."]
    lines += [f"{a} {p} {h}" for (a, p), h in sorted(got.items())]
    GOLDEN.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {len(got)} fingerprints to {GOLDEN}")


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] == "--write":
        write()
        return 0
    ok, lines = selftest(verbose="-v" in argv)
    if ok:
        print(f"selftest: the {len(CASES) * len(PROBLEMS)} runs match the shipped "
              "fingerprints bit for bit")
        return 0
    print("selftest: DIFFERS")
    for line in lines:
        print("  " + line)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

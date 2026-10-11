# SPDX-License-Identifier: Apache-2.0
"""The reference values the run layer's tests hold against pymoo 0.6.2
(Apache-2.0), written to tests/data/pymoo-0.6.2_<what>.json.

Run by hand, with pymoo installed; CI never runs it and the tests never import
pymoo. From the repository root:

    uv run --no-project --python 3.13 --with pymoo==0.6.2 python tools/refs/gen_pymoo_refs.py

The points are MOOtation's (fixed formulas below, some through
python/mootation, imported from this checkout); the values are pymoo's at them.
Every file carries its library, version, license, command, date, the number of
points and the tolerance the test compares with.
"""

import datetime
import json
import sys
from pathlib import Path

import numpy as np
import pymoo
from pymoo.problems.many import cdtlz as PC
from pymoo.problems.many import dtlz as PD
from pymoo.problems.many import wfg as PW
from pymoo.problems.multi import dascmop as PDAS
from pymoo.problems.multi import mw as PMW
from pymoo.problems.multi.omnitest import OmniTest
from pymoo.problems.multi.sympart import SYMPART

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
from mootation.benchmarks import c_dtlz as C  # noqa: E402
from mootation.benchmarks import mw as W  # noqa: E402

OUT = ROOT / "tests" / "data"
COMMAND = ("uv run --no-project --python 3.13 --with pymoo==0.6.2 python "
           "tools/refs/gen_pymoo_refs.py")
G = 0.61803398875


def floats(v):
    return [float(t) for t in np.asarray(v, float).ravel()]


def write(what, module, about, tolerance, cases):
    assert pymoo.__version__ == "0.6.2", pymoo.__version__
    data = {"library": "pymoo", "version": pymoo.__version__, "license": "Apache-2.0",
            "module": module, "generated_by": "tools/refs/gen_pymoo_refs.py",
            "command": COMMAND, "date": datetime.date.today().isoformat(),
            "points": len(cases), "tolerance": tolerance, "about": about, "cases": cases}
    path = OUT / f"pymoo-0.6.2_{what}.json"
    path.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(path.name, len(cases))


def wfg():
    # pymoo refuses k < 4 (WFG.validate); the registry's WFG has k = 2(M - 1),
    # 2 at two objectives, and the formulas take any k: the check only is off.
    PW.WFG.validate = staticmethod(lambda l, k, n_obj: None)
    cases = []
    for M in (2, 3, 5, 10):
        n = 2 * (M - 1) + 20
        i = np.arange(1, n + 1)
        z = 2.0 * i * ((0.2 + G * i) % 1.0)
        for j in range(1, 10):
            f = getattr(PW, f"WFG{j}")(n_var=n, n_obj=M, k=2 * (M - 1)).evaluate(z[None, :])[0]
            cases.append({"problem": f"WFG{j}_{M}D", "x": floats(z), "f": floats(f)})
    write("wfg", "pymoo/problems/many/wfg.py",
          "WFG1-9 at 2, 3, 5 and 10 objectives, n = 2(M - 1) + 20, k = 2(M - 1), one point "
          "each: z_i = 2i ((0.2 + 0.61803398875 i) mod 1). f: the objectives of "
          "mootation.benchmarks.wfg.WFG_FUNCS[name](x, M).",
          {"kind": "absolute", "value": 1e-9}, cases)


def c_dtlz():
    classes = {"C1-DTLZ1": PC.C1DTLZ1, "C1-DTLZ3": PC.C1DTLZ3, "C2-DTLZ2": PC.C2DTLZ2,
               "C3-DTLZ1": PC.C3DTLZ1, "C3-DTLZ4": PC.C3DTLZ4}
    cases = []
    for M in C.SIZES:
        for name in C.NAMES:
            n = C.n_vars(name, M)
            i = np.arange(1, n + 1)
            X = [(0.1 * j + G * i) % 1.0 for j in (1, 2, 3)]
            for j in (1, 2):
                x = X[j - 1].copy()
                x[M - 1:] = 0.5 + 0.01 * j
                X.append(x)
            X = np.array(X)
            if name == "C2-ConvexDTLZ2":
                F = C.convex(PD.DTLZ2(n_var=n, n_obj=M).evaluate(X, return_values_of=["F"]))
                Gv = PC.constraint_c4_cylindrical(F, C.C2_CONVEX_R[M]).reshape(-1, 1)
            else:
                F, Gv = classes[name](n_var=n, n_obj=M).evaluate(X, return_values_of=["F", "G"])
            Gv = np.asarray(Gv, float).reshape(len(X), -1)
            for r in range(len(X)):
                cases.append({"problem": f"{name}_{M}D", "x": floats(X[r]), "f": floats(F[r]),
                              "g": floats(Gv[r])})
    write("c-dtlz", "pymoo/problems/many/cdtlz.py",
          "C1-DTLZ1, C1-DTLZ3, C2-DTLZ2, C2-ConvexDTLZ2, C3-DTLZ1 and C3-DTLZ4 at three and "
          "five objectives, five points each: x_i = (0.1 j + 0.61803398875 i) mod 1, j = 1, "
          "2, 3, and the first two with the distance variables at 0.5 + 0.01 j. "
          "C2-ConvexDTLZ2: pymoo's DTLZ2 turned convex (f_i^4, f_M^2) under its "
          "constraint_c4_cylindrical (the paper's Eq. 6). g <= 0 satisfied.",
          {"kind": "relative", "value": 1e-12}, cases)


def mw():
    cases = []
    for name in W.NAMES:
        for m in (W.SIZES if name in W.SCALABLE else (2,)):
            cls = getattr(PMW, name)
            p = cls(n_obj=m) if name in W.SCALABLE else cls()
            n, top = W.n_vars(name, m), W.upper(name)
            i = np.arange(1, n + 1)
            X = [top * ((0.1 * j + G * i) % 1.0) for j in (1, 2, 3)]
            for j in (1, 2):
                x = X[j - 1].copy()
                x[m - 1:] = W.optimal_distance(name, x[:m - 1])
                X.append(x)
            X = np.array(X)
            F, Gv = p.evaluate(X, return_values_of=["F", "G"])
            Gv = np.asarray(Gv, float).reshape(len(X), -1)
            if name in ("MW12", "MW13"):
                Gv = Gv[:, ::-1]                     # the order of the paper's equations
            key = f"{name}_{m}D" if name in W.SCALABLE else name
            for r in range(len(X)):
                cases.append({"problem": key, "x": floats(X[r]), "f": floats(F[r]),
                              "g": floats(Gv[r])})
    write("mw", "pymoo/problems/multi/mw.py",
          "MW1-14 (MW4, MW8 and MW14 at three and five objectives), five points each: "
          "x_i = top ((0.1 j + 0.61803398875 i) mod 1), j = 1, 2, 3, top the box's bound, "
          "and the first two with the distance variables at g's optimum "
          "(mootation.benchmarks.mw.optimal_distance). g <= 0 satisfied, in the order of "
          "the paper's equations.",
          {"kind": "relative", "value": 1e-12}, cases)


def fixed_points(n, count, lo=0.0, hi=1.0):
    return np.array([[lo + (hi - lo) * ((0.1 + G * (j + 1 + 7 * p)) % 1.0) for j in range(n)]
                     for p in range(count)])


def das_cmop():
    cases = []
    for i in range(1, 10):
        for trip in ((0.25, 0.25, 0.25), (0.5, 1.0, 0.5)):
            P = getattr(PDAS, f"DASCMOP{i}")(trip)
            X = fixed_points(30, 2)
            F, Gv = P.evaluate(X, return_values_of=["F", "G"])
            for r in range(len(X)):
                case = {"problem": i, "triplet": list(trip), "x": floats(X[r]),
                        "f": floats(F[r]), "g": floats(Gv[r])}
                if trip[1] == 1.0:
                    case["g_not_compared"] = [1 if i <= 6 else 2]
                cases.append(case)
    write("das-cmop", "pymoo/problems/multi/dascmop.py",
          "DAS-CMOP1-9 (problem: i) at the triplets (eta, zeta, gamma) = (0.25, 0.25, 0.25) "
          "and (0.5, 1, 0.5), two points each: x_j = (0.1 + 0.61803398875 (j + 1 + 7 p)) mod "
          "1, p = 0, 1. g <= 0 satisfied. At zeta = 1 the equality g = d "
          "(g_not_compared) is not compared: MOOtation takes the paper's tolerance 1e-6 "
          "(Eq. 13).",
          {"kind": "relative", "value": 1e-12}, cases)


def omni_sympart():
    cases = []
    X = fixed_points(3, 5, 0.0, 6.0)
    for x, f in zip(X, OmniTest(n_var=3).evaluate(X)):
        cases.append({"problem": "Omni-test", "x": floats(x), "f": floats(f)})
    centres = np.array([(-10, -10), (0, 0), (10, 10), (-10, 10), (10, -10)], float)
    X = centres + fixed_points(2, 5, -4.0, 4.0)
    for x, f in zip(X, SYMPART().evaluate(X)):
        cases.append({"problem": "SYM-PART_simple", "x": floats(x), "f": floats(f)})
    write("omni-sympart", "pymoo/problems/multi/omnitest.py, pymoo/problems/multi/sympart.py",
          "Omni-test (n = 3) at five points of [0, 6]^3 and SYM-PART simple at five points "
          "within 4 of five of its centres: x_j = lo + (hi - lo) ((0.1 + 0.61803398875 "
          "(j + 1 + 7 p)) mod 1).",
          {"kind": "relative", "value": 1e-12}, cases)


def wfg_optimum():
    PW.WFG.validate = staticmethod(lambda l, k, n_obj: None)
    cases = []
    w8 = PW.WFG8(n_var=8, n_obj=3, k=4, l=4)
    K = fixed_points(4, 3)
    Y = w8._positional_to_optimal(K) / (2 * (np.arange(8) + 1))
    for pos, y in zip(K, Y):
        cases.append({"function": "wfg8_optimum", "positions": floats(pos), "l": 4,
                      "y": floats(y[4:])})
    w9 = PW.WFG9(n_var=10, n_obj=3, k=4, l=6)
    Y = w9._positional_to_optimal(fixed_points(4, 1)) / (2 * (np.arange(10) + 1))
    cases.append({"function": "wfg9_optimum", "positions": None, "l": 6, "y": floats(Y[0][4:])})
    write("wfg-optimum", "pymoo/problems/many/wfg.py (_positional_to_optimal)",
          "WFG8's optimal distance variables for three position vectors (k = 4, l = 4) and "
          "WFG9's (l = 6), normalised (divided by 2i): y = "
          "mootation.benchmarks.rotated.wfg8_optimum(positions, l), wfg9_optimum(l).",
          {"kind": "relative", "value": 1e-12}, cases)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for make in (wfg, c_dtlz, mw, das_cmop, omni_sympart, wfg_optimum):
        make()

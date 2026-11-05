#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The problems with symmetries (task 19, §19.1.6): C++ against Python.

Every registry instance of suite/sym.hpp's families (benchmarks/symmetric.py,
ROWS) is made by C++ (_core.Native) and by NumPy (symmetric.numpy_family), and
the two are compared at the instance's start and at two points about it: the
residuals, the Jacobian, the objective, the gradient, the gauge directions
and canon, to 1e-12 of each column's scale (NumPy's exp, log and pow are not
dm::'s; the normal draws take math.log). The C++ invariants themselves (canon
bit for bit, the gauge test, f(x_opt) = f_opt) are tests/test_suite.cpp's.
"""

from __future__ import annotations

import pickle
import sys

import numpy as np

from mootation import _core
from mootation.benchmarks import symmetric as sym

TOL = 1e-12
failed = 0


def check(ok: bool, what: str) -> None:
    global failed
    if not ok:
        failed += 1
    print(("ok    " if ok else "FAIL  ") + what)


def close(a, b, scale=None) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    s = float(np.max(np.abs(b))) if scale is None else scale
    return float(np.max(np.abs(a - b))) / max(s, 1e-300) if a.size else 0.0


def main() -> int:
    for row in sym.ROWS:
        if row.family == "lj_cluster" and row.inst.dim > 26:
            continue                                   # the largest clusters: the same code as N <= 26
        nat = _core.Native(sym.spec(row))
        py = sym.numpy_family(row.family, row.inst)
        x0 = np.array(nat.start())
        rng = np.random.default_rng(7)
        worst = {"r": 0.0, "J": 0.0, "f": 0.0, "g": 0.0, "N": 0.0, "canon": 0.0}
        for t in range(3):
            x = x0 if t == 0 else x0 + 0.05 * (1.0 + np.abs(x0)) * rng.standard_normal(x0.size)
            lo, hi = np.array(nat.bounds).T
            x = np.clip(x, lo, hi)
            if nat.n_res:
                r, J = nat.residuals(list(x), nat.has_jacobian) if nat.has_jacobian else (nat.residuals(list(x)), None)
                worst["r"] = max(worst["r"], close(r, py.residuals(x)))
                if J is not None:
                    Jp = py.jacobian(x)
                    worst["J"] = max(worst["J"], max(close(np.array(J)[:, j], Jp[:, j], float(np.max(np.abs(Jp))))
                                                     for j in range(x.size)))
            else:
                worst["f"] = max(worst["f"], close([nat.evaluate(list(x))[0]], [py.value(x)]))
                worst["g"] = max(worst["g"], close(nat.gradient(list(x)), py.gradient(x)))
            if nat.gauge_dim:
                worst["N"] = max(worst["N"], close(nat.gauge_basis(list(x)), py.gauge_basis(x)))
            if nat.has_canon and hasattr(py, "canon"):
                worst["canon"] = max(worst["canon"], close(nat.canon(list(x)), py.canon(x)))
        seen = ", ".join(f"{k} {v:.1e}" for k, v in worst.items() if v)
        check(all(v <= TOL for v in worst.values()), f"{sym.name(row)}: C++ = NumPy to {TOL:g} ({seen or 'exact'})")
        # the spec travels: pickled and made again, the same problem
        again = _core.Native(nat.spec)
        check(again.evaluate(list(x0)) == nat.evaluate(list(x0)) and again.n_var == nat.n_var,
              f"{sym.name(row)}: its spec makes the same problem")
        # task 19, §19.5: the residuals' blocks, and the blocks as objectives (blocks_objectives.hpp)
        check(list(nat.res_blocks) == list(getattr(py, "res_blocks", [])),
              f"{sym.name(row)}: res_blocks {list(nat.res_blocks)} as NumPy's")
        if nat.res_blocks:
            bo = _core.Native({**sym.spec(row), "objectives": "blocks", "block_weights": [2.0] * len(nat.res_blocks)})
            r, at, want = py.residuals(x), 0, []
            for size in py.res_blocks:
                want.append(2.0 * 0.5 * float(np.sum(r[at:at + size] ** 2)))
                at += size
            got = bo.evaluate(list(x))
            made = _core.Native(bo.spec), pickle.loads(pickle.dumps(bo))
            check(bo.n_obj == len(want) and close(got, want) <= TOL and all(m.evaluate(list(x)) == got for m in made),
                  f"{sym.name(row)}: its blocks as objectives = NumPy's, and its spec and pickle make them again")
    check(abs(sym.resolution(1e2) - 2.44) < 0.005 and abs(sym.resolution(1e4) - 1.63) < 0.005,
          "resolution limit: Istratov & Vyvenko's Table I")
    blocks_instances()
    print(f"test_symmetric: {'FAILED, ' + str(failed) + ' check(s)' if failed else 'all checks passed'}")
    return 1 if failed else 0


_BLOCKS_TOML = """algorithms = [ { name = "nsga2", pop = 12, gens = 4 }, { name = "moead", pop = 10, gens = 4 } ]
[run]
name = "NAME"
[problem]
kind = "builtin"
[benchmarks]
runs = 1
problems = ["sumexp_K3_r4_snr1e3_blocks_2D", "chaboche1d_N3_s2_blocks_3D"]
[campaign]
out = "res/{name}"
storage = "store"
"""


def blocks_instances() -> None:
    """Task 19, §19.5: the registry's blocks instances (registry.py,
    _register_blocks): B objectives of the base's residual blocks, the frame
    [0, f(x0)] (BOR-1), every point's f_b the base's 1/2 ||r_b||^2; a small
    campaign of them gives the same files from both engines."""
    import tempfile
    from pathlib import Path
    from mootation.benchmarks import registry as R
    for base in R.BLOCK_BASES:
        b = R.PROBLEMS[base]
        sizes = list(b.native.res_blocks)
        key = base[:-len("_1D")] + f"_blocks_{len(sizes)}D"
        q = R.PROBLEMS.get(key)
        ok = q is not None and q.n_obj == len(sizes) and q.ideal == (0.0,) * len(sizes)
        if ok:
            x0 = list(b.x0)
            r, at, want = b.native.residuals(x0), 0, []
            for size in sizes:
                t = 0.0
                for v in r[at:at + size]:
                    t += v * v
                want.append(0.5 * t)
                at += size
            ok = list(q.nadir) == want and list(q.evaluate(x0)) == want and q.bounds == b.bounds
        check(ok, f"{key}: {len(sizes)} objectives of {base}'s blocks, the frame [0, f(x0)] (BOR-1)")
    from mootation.run import campaign as C
    from mootation.run.config import load, validate
    if C.engine_path() is None:
        print("  skip  the blocks campaign in both engines: no C++ engine")
        return
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_run_layer import _same_run
    import os
    with tempfile.TemporaryDirectory() as td:
        os.environ["MOOTATION_REFS_DIR"] = str(Path(td) / "refs")     # never the user's references
        made = {}
        for name, engine in (("p", "python"), ("c", "cxx")):
            path = Path(td) / f"{name}.toml"
            path.write_text(_BLOCKS_TOML.replace("NAME", name), encoding="utf-8")
            cfg = load(path)
            problems = validate(cfg)
            if problems:
                check(False, f"the blocks campaign's config: {problems}")
                return
            spec = C.campaign_spec(cfg)
            root, jobs = C.out_root(cfg, spec), C.expand_jobs(cfg, spec)
            made[name] = (root, jobs, C.run_campaign(cfg, workers=2, engine=engine))
        (rp, jobs, cp), (rc, _, cc) = made["p"], made["c"]
        same = len(jobs) == 4 and cp["done"] == cc["done"] == 4
        for job in jobs:
            try:
                _same_run(C.run_dir_of(rp, job), C.run_dir_of(rc, job))
            except AssertionError:
                same = False
        check(same, "a campaign of the blocks instances (nsga2, moead): the same files from both engines")


if __name__ == "__main__":
    sys.exit(main())

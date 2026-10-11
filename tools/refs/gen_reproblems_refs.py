# SPDX-License-Identifier: Apache-2.0
"""The reference values the run layer's tests hold against the authors' Python
version of the RE problems, github.com/ryojitanabe/reproblems (MIT), commit
7876b4e, reproblem_python_ver/reproblem.py — written to
tests/data/reproblems-7876b4e_re.json.

Run by hand with a checkout of that repository at that commit; CI never runs
it. From this repository's root:

    python tools/refs/gen_reproblems_refs.py <reproblems>/reproblem_python_ver/reproblem.py

Five points per problem, x = lb + (ub - lb) u, u_i = (0.1 j + 0.61803398875 i)
mod 1, j = 1..5; RE91 with its four normal draws fixed to (0.3, -1.2, 0.7, 0.1).
"""

import datetime
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
NAMES = ["RE21", "RE22", "RE23", "RE24", "RE25", "RE31", "RE32", "RE33", "RE34", "RE35", "RE36",
         "RE37", "RE41", "RE42", "RE61", "RE91", "CRE21", "CRE22", "CRE23", "CRE24", "CRE25",
         "CRE31", "CRE32", "CRE51"]
Z91 = [0.3, -1.2, 0.7, 0.1]


def main(path):
    spec = importlib.util.spec_from_file_location("reproblem", path)
    R = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(R)
    cases = []
    for name in NAMES:
        p = getattr(R, name)()
        i = np.arange(1, p.n_variables + 1)
        for j in range(1, 6):
            x = p.lbound + (p.ubound - p.lbound) * ((0.1 * j + 0.61803398875 * i) % 1.0)
            case = {"problem": name, "x": [float(v) for v in x]}
            if name == "RE91":
                draws = iter(Z91)
                saved = R.np.random.normal
                R.np.random.normal = lambda *a, _d=draws: next(_d)
                try:
                    f = p.evaluate(x)
                finally:
                    R.np.random.normal = saved
                case["normal_draws"] = Z91
            elif name.startswith("CRE"):
                f, g = p.evaluate(x)
                case["g"] = [float(v) for v in g]
            else:
                f = p.evaluate(x)
            case["f"] = [float(v) for v in f]
            cases.append(case)
    data = {"library": "reproblems (R. Tanabe, H. Ishibuchi), Python version",
            "version": "commit 7876b4e", "license": "MIT",
            "module": "reproblem_python_ver/reproblem.py",
            "generated_by": "tools/refs/gen_reproblems_refs.py",
            "command": "python tools/refs/gen_reproblems_refs.py <reproblem.py>",
            "date": datetime.date.today().isoformat(), "points": len(cases),
            "tolerance": {"kind": "relative", "value": 1e-12},
            "about": "RE21-91 and CRE21-51, five points each: x = lb + (ub - lb) u, u_i = "
                     "(0.1 j + 0.61803398875 i) mod 1, j = 1..5. f: the objectives; g, for "
                     "CRE: the constraint violations as the module returns them. RE91: its "
                     "four normal draws fixed (normal_draws).",
            "cases": cases}
    out = ROOT / "tests" / "data" / "reproblems-7876b4e_re.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    print(out.name, len(cases))


if __name__ == "__main__":
    main(sys.argv[1])

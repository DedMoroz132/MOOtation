# SPDX-License-Identifier: Apache-2.0
"""tests/data/scipy-<version>_sobol.txt: SciPy's unscrambled Sobol points
1..40 in 100 dimensions, the reference of tests/test_sobol.cpp (sobol.hpp).

    python tools/refs/gen_sobol_ref.py
"""
import warnings
from pathlib import Path

import scipy
from scipy.stats import qmc

warnings.simplefilter("ignore")            # "balance properties ... power of 2"
P = qmc.Sobol(d=100, scramble=False).random(41)[1:]
out = Path(__file__).resolve().parents[2] / "tests" / "data" / f"scipy-{scipy.__version__}_sobol.txt"
with out.open("w", newline="\n") as fh:
    fh.write(f"# scipy {scipy.__version__} qmc.Sobol(d=100, scramble=False): points 1..40 "
             "(index 0, the origin, left out)\n")
    for row in P:
        fh.write(" ".join(f"{v:.17g}" for v in row) + "\n")
print(out)

# SPDX-License-Identifier: Apache-2.0
"""tests/data/scipy-<version>_sobol.txt: SciPy's unscrambled Sobol points
1..40 in 100 dimensions, the reference of tests/test_sobol.cpp (sobol.hpp);
and tests/data/scipy-<version>_sobol_21201.txt (task 24, §24.3.2): points
1..40 in their first 300 dimensions and the FNV-1a (64 bits) of points 1..40
in all 21 201, their 32-bit integers row by row (each byte of an integer
little-endian first).

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

W = qmc.Sobol(d=21201, scramble=False).random(41)[1:]
h = 0xCBF29CE484222325
for row in W:
    for v in row:
        x = int(v * 4294967296.0)
        for k in range(4):
            h ^= (x >> (8 * k)) & 0xFF
            h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
out = out.with_name(f"scipy-{scipy.__version__}_sobol_21201.txt")
with out.open("w", newline="\n") as fh:
    fh.write(f"# scipy {scipy.__version__} qmc.Sobol(d=21201, scramble=False): points 1..40 in dimensions 1..300\n")
    fh.write(f"# fnv1a64 {h:016x} (points 1..40 in 21201 dimensions, tools/refs/gen_sobol_ref.py)\n")
    for row in W[:, :300]:
        fh.write(" ".join(f"{v:.17g}" for v in row) + "\n")
print(out)

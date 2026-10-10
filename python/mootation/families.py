# SPDX-License-Identifier: Apache-2.0
"""The family of a benchmark problem's name: one rule for the registry
(benchmarks.families) and the tables (run.campaign.problem_family). Kept apart
from both because the tables must work without NumPy, and the registry needs
it."""

from __future__ import annotations

# the noisy copies' suffixes (benchmarks/noisy.py, LEVELS)
NOISE_LEVELS = ("n1", "n5", "d1", "d5", "s1", "s5", "h1", "h5")
CONTROLS = ("r0", "c0")          # the Q = I controls of benchmarks/rotated.py
ENGINEERING = ("RE", "CRE")      # the real-world problems of benchmarks/re.py


def family(name: str) -> str:
    """DTLZ2_3D -> DTLZ, MaF10_8D -> MaF, ZDT1 -> ZDT: the name's stem without
    its digits. A noisy copy (ZDT1_n1, DTLZ2_3D_s5) -> noisy, a failing one
    (ZDT1_fail10r, benchmarks/failing.py) -> failing; a control of the rotated
    problems keeps its prefix, r0DTLZ1 -> r0DTLZ and c0DTLZ2 -> c0DTLZ, apart
    from rDTLZ and rcDTLZ."""
    if "_" in name and name.rsplit("_", 1)[1] in NOISE_LEVELS:
        return "noisy"
    if "_" in name and name.rsplit("_", 1)[1].startswith("fail"):
        return "failing"
    stem = name.split("_")[0]
    if stem[:2] == "RE" and stem.endswith("c") and stem[2:-1].isdigit():
        return "RE"                          # the corrected RE22c, RE33c
    head = stem[:2] if stem[:2] in CONTROLS else ""
    rest = stem[len(head):]
    return head + ("".join(c for c in rest if not c.isdigit()) or rest)


def is_engineering(name: str) -> bool:
    """A real-world problem (RE, CRE), or a noisy copy of one (RE21_n5): ranked
    apart from the synthetic ones (task 6, item 6), its front an approximation."""
    if "_" in name and name.rsplit("_", 1)[1] in NOISE_LEVELS:
        name = name.rsplit("_", 1)[0]
    return family(name) in ENGINEERING

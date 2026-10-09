# SPDX-License-Identifier: Apache-2.0
"""The family of a benchmark problem's name: one rule for the registry
(benchmarks.families) and the tables (run.campaign.problem_family). Kept apart
from both because the tables must work without NumPy, and the registry needs
it."""

from __future__ import annotations

NOISE_LEVELS = ("n1", "n5")      # the noisy copies' suffixes (benchmarks/noisy.py, SIGMA)
CONTROLS = ("r0", "c0")          # the Q = I controls of benchmarks/rotated.py
ENGINEERING = ("RE", "CRE")      # the real-world problems of benchmarks/re.py


def family(name: str) -> str:
    """DTLZ2_3D -> DTLZ, MaF10_8D -> MaF, ZDT1 -> ZDT: the name's stem without
    its digits. A noisy copy (ZDT1_n1) -> noisy; a control of the rotated
    problems keeps its prefix, r0DTLZ1 -> r0DTLZ and c0DTLZ2 -> c0DTLZ, apart
    from rDTLZ and rcDTLZ."""
    if "_" in name and name.rsplit("_", 1)[1] in NOISE_LEVELS:
        return "noisy"
    stem = name.split("_")[0]
    head = stem[:2] if stem[:2] in CONTROLS else ""
    rest = stem[len(head):]
    return head + ("".join(c for c in rest if not c.isdigit()) or rest)


def is_engineering(name: str) -> bool:
    """A real-world problem (RE, CRE): ranked apart from the synthetic ones
    (task 6, item 6), its front an approximation."""
    return family(name) in ENGINEERING

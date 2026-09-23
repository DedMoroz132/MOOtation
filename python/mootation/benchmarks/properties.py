# SPDX-License-Identifier: Apache-2.0
"""What each benchmark problem is: the table ranks are grouped by.

The vocabulary is Huband, Hingston, Barone & While's (IEEE TEVC 10(5), 2006,
Table I), and for ZDT, DTLZ and WFG the values are theirs too (Tables V, VII
and XV):

  front       the Pareto front's geometry — "linear", "concave", "convex",
              "mixed", "disconnected", "degenerate", joined by "+" when
              several hold (WFG1 "convex+mixed"); None where unknown
  multimodal  F5: multimodal or deceptive in any objective
  deceptive   F5, the deceptive kind (WFG5, WFG9)
  bias        F3: "substantially more solutions exist in some regions of
              fitness space than they do in others"
  scaled      R6 read as a property: the objectives' Pareto-optimal tradeoff
              ranges are dissimilar
  separable   F2: every parameter's optimum is independent of the others
  centre      the optimum at the centre of the domain — R2's "medial
              parameters": "the projection [of the Pareto set onto a
              parameter's domain] should cluster around the middle"

True / False, or None where the source leaves it open ("?" in Huband's table
for DTLZ5 and DTLZ6) or where nothing classifies it. Beyond the three suites
Huband analyses:

  IDTLZ1/2    DTLZ1/2 with the front inverted (Jain & Deb 2014, Eq. 9 for
              IDTLZ1; IDTLZ2 is this library's analogue, dtlz_variants.py)
  SDTLZ1/2    DTLZ1/2 with objective i scaled by p^(i-1) (Deb & Jain 2014,
              Table VIII): scaled
  shiftDTLZ   DTLZ1-4 with every distance variable's optimum 0.15-0.35 from
              the centre (dtlz_variants.py): not medial, the point of them
  ZCAT        Zapotecas-Martinez et al. 2023, Section 4: ZCAT3-4 linear,
              ZCAT5-10 "a high level of convexity and/or concavity"
              (mixed), ZCAT1-2 simplex-like and left None, ZCAT11-13
              disconnected, ZCAT14 degenerate, ZCAT15-16 degenerate and
              disconnected, ZCAT17-20 part degenerate. At the registered
              defaults — Level 1 (Z_1: "U/S", Table 4), no bias — they are
              unimodal and unbiased; objective i spans [0, i^2] (scaled); and
              with the complicated Pareto set, which the defaults switch on,
              every distance variable's optimum is g(y_I), a function of the
              position variables: not separable. Nor medial: sampled, those
              optima spread over the whole [0, 1] (None).
  bbob-biobj  Brockhoff et al. 2022, Section 5.1: each objective is one of
              ten bbob functions from five groups — separable (f1, f2), low
              or moderate conditioning (f6, f8), high conditioning (f13,
              f14), multimodal with global structure (f15, f17), multimodal
              with weak global structure (f20, f21). A pair is separable when
              both functions are, multimodal when either is; the front has no
              closed form (None), x_opt is uniform in [-4, 4]^n (not medial),
              no bias is built in.

At M >= 4 (WFG3: M >= 3) DTLZ5, DTLZ6 and WFG3 have a non-degenerate part
besides the curve (fronts_full.py), so their front reads "degenerate+mixed"
there.
"""

from __future__ import annotations

KEYS = ("front", "multimodal", "deceptive", "bias", "scaled", "separable", "centre")


def _p(front, multimodal, bias, scaled, separable, centre, deceptive=False) -> dict:
    return {"front": front, "multimodal": multimodal, "deceptive": deceptive,
            "bias": bias, "scaled": scaled, "separable": separable, "centre": centre}


# Huband et al. 2006, Table V (R2 x = medial parameters present; R6 check =
# dissimilar ranges).
_ZDT = {
    "ZDT1": _p("convex", False, False, False, True, False),
    "ZDT2": _p("concave", False, False, False, True, False),
    "ZDT3": _p("disconnected", True, False, True, True, False),
    "ZDT4": _p("convex", True, False, False, True, True),
    "ZDT6": _p("concave", True, True, True, True, False),
}

# Table VII. "S*": separable. DTLZ5/6: "?" where the table has it; their
# centre is ours — DTLZ5's distance optimum is x = 1/2 (medial), DTLZ6's x = 0
# (extremal).
_DTLZ = {
    "DTLZ1": _p("linear", True, False, False, True, True),
    "DTLZ2": _p("concave", False, False, False, True, True),
    "DTLZ3": _p("concave", True, False, False, True, True),
    "DTLZ4": _p("concave", False, True, False, True, True),
    "DTLZ5": _p("degenerate", False, False, None, None, True),
    "DTLZ6": _p("degenerate", False, True, None, None, False),
    "DTLZ7": _p("disconnected", True, False, False, True, False),
}

# Table XV; its caption: no extremal nor medial parameters, dissimilar
# tradeoff magnitudes for every WFG problem.
_WFG = {
    "WFG1": _p("convex+mixed", False, True, True, True, False),
    "WFG2": _p("convex+disconnected", True, False, True, False, False),
    "WFG3": _p("linear+degenerate", False, False, True, False, False),
    "WFG4": _p("concave", True, False, True, True, False),
    "WFG5": _p("concave", True, False, True, True, False, deceptive=True),
    "WFG6": _p("concave", False, False, True, False, False),
    "WFG7": _p("concave", False, True, True, True, False),
    "WFG8": _p("concave", False, True, True, False, False),
    "WFG9": _p("concave", True, True, True, False, False, deceptive=True),
}

_ZCAT_FRONT = {1: None, 2: None, 3: "linear", 4: "linear"}
for _i in range(5, 11):
    _ZCAT_FRONT[_i] = "mixed"
for _i in (11, 12, 13):
    _ZCAT_FRONT[_i] = "disconnected"
_ZCAT_FRONT[14] = "degenerate"
_ZCAT_FRONT[15] = _ZCAT_FRONT[16] = "degenerate+disconnected"
for _i in (17, 18, 19, 20):
    _ZCAT_FRONT[_i] = "degenerate+mixed"

_BBOB_SEPARABLE = {1, 2}
_BBOB_MULTIMODAL = {15, 17, 20, 21}


def _stem_and_m(name: str) -> tuple:
    stem = name.split("_")[0]
    tail = name.rsplit("_", 1)[-1]
    m = int(tail[:-1]) if tail.endswith("D") and tail[:-1].isdigit() else 2
    return stem, m


def properties(name: str) -> dict | None:
    """The row for one registry name, or None for a problem the table lacks."""
    stem, m = _stem_and_m(name)
    if stem in _ZDT:
        return dict(_ZDT[stem])
    if stem in _WFG:
        row = dict(_WFG[stem])
        if stem == "WFG3" and m >= 3:
            row["front"] = "degenerate+mixed"
        return row
    if stem in _DTLZ:
        row = dict(_DTLZ[stem])
        if stem in ("DTLZ5", "DTLZ6") and m >= 4:
            row["front"] = "degenerate+mixed"
        return row
    if stem.startswith("shiftDTLZ") and stem[5:] in _DTLZ:
        row = dict(_DTLZ[stem[5:]])
        row["centre"] = False
        return row
    if stem in ("IDTLZ1", "IDTLZ2"):
        row = dict(_DTLZ["DTLZ" + stem[-1]])
        row["front"] = "linear" if stem == "IDTLZ1" else "convex"
        return row
    if stem in ("SDTLZ1", "SDTLZ2"):
        row = dict(_DTLZ["DTLZ" + stem[-1]])
        row["scaled"] = True
        return row
    if stem.startswith("ZCAT") and stem[4:].isdigit():
        return _p(_ZCAT_FRONT[int(stem[4:])], False, False, True, False, None)
    if stem.startswith("bbobbiobj"):
        from .bbob_biobj import PAIRS
        fa, fb = PAIRS[int(stem[len("bbobbiobj"):]) - 1]
        return _p(None, fa in _BBOB_MULTIMODAL or fb in _BBOB_MULTIMODAL, False, None,
                  fa in _BBOB_SEPARABLE and fb in _BBOB_SEPARABLE, False)
    return None


def label(name: str, key: str) -> str:
    """The group a problem falls in for `key`: "front=concave", "bias=yes", ...;
    "key=?" when unknown."""
    row = properties(name)
    v = None if row is None else row.get(key)
    if v is None:
        return f"{key}=?"
    if isinstance(v, bool):
        return f"{key}={'yes' if v else 'no'}"
    return f"{key}={v}"

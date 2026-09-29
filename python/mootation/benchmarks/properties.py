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

  MOP1-7      Liu, Gu & Zhang, IEEE TEVC 18(3), 2014, Section III-C: the fronts
              as the paper states them (MOP1, 5 convex, MOP2, 3, 7 concave,
              MOP4 discontinuous, MOP6 linear); every Pareto set is nonlinear,
              x_j = sin(0.5 pi x_1) or x_1 x_2 (linkage, not separable, not
              medial). The paper does not classify modality: read off g,
              MOP1, 5, 6, 7's -0.9 t^2 + |t|^0.6 has a second basin at |t| = 1,
              the edge of the domain (multimodal), MOP2-4's |t|/(1 + e^|t|)
              rises all the way (not). Its "imbalance" (g multiplied by
              sin(pi x_1), zero at the ends) is none of Huband's biases: None.
  BT1-9       Li, Zhang & Deng, IEEE TCYB 47(1), 2017, Section III-B: every
              problem has a distance-related bias (bias), BT3 and BT4 a
              position-related one too; BT5's front is disconnected, BT9's the
              octant of the sphere, the others f2 = 1 - sqrt(f1) (convex); BT6
              and BT8 have simple nonlinear Pareto sets and BT7 a complicated
              one (linkage), the others constant optima sin(j pi / 2n)
              (separable, not medial); BT8 is multimodal.
  Polygon     a MOOtation construction (polygon.py): MaF8 of Cheng et al. 2017
              — "Linear, degenerate" in their Table 1 — with extra distance
              variables whose optimum is 0, the middle of [-1, 1] (medial). The
              Pareto set is 2-dimensional, so the front is a 2-dimensional
              manifold whatever M: degenerate from four objectives.
  IPolygon    Ishibuchi, Akedo & Nojima, GECCO 2011: m = 2 identical polygons
              in [0, 100]^2, each objective the distance to the nearer copy of
              a vertex, so every objective has two minima (multimodal). The
              registry's polygons are too close for the paper's equivalent
              Pareto regions (polygon_ishibuchi.py): at three objectives the
              Pareto set is a region between them, front unknown (None); at
              four it is the squares' inner halves, where f2 = f4
              ("degenerate"). Where it sits in the domain is not what the
              construction intends: centre None.

At M >= 4 (WFG3: M >= 3) DTLZ5, DTLZ6 and WFG3 have a non-degenerate part
besides the curve (fronts_full.py), so their front reads "degenerate+mixed"
there.

Besides Huband's vocabulary (task 4, item 5):

  degenerate, disconnected
              read off `front`: True when it names that geometry, None when
              the front is unknown
  linkage     the optimum of a distance variable depends on the position
              variables — a curved Pareto set (Li & Zhang, IEEE TEVC 13(2),
              2009): ZCAT at its defaults, MOP, BT6-8; not ZDT, DTLZ and its
              variants, BT1-5 and 9, Polygon, whose optima are constants.
              None where no source puts it so: WFG (its parameter-dependent
              biases are stated as biases, Huband Table XV), IPolygon (no
              distance variables), bbob-biobj (no closed form)
  n_obj, n_vars  M and D of the registry entry
  bbob_groups the two function groups of a bbob-biobj pair (Brockhoff et al.
              2022, Section 5.1), sorted and joined by "+": separable (f1,
              f2), moderate (f6, f8: low or moderate conditioning),
              ill-conditioned (f13, f14), multimodal (f15, f17: adequate
              global structure), weakly-structured (f20, f21); None elsewhere
"""

from __future__ import annotations

KEYS = ("front", "multimodal", "deceptive", "bias", "scaled", "separable", "centre",
        "degenerate", "disconnected", "linkage", "n_obj", "n_vars", "bbob_groups")


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
_BBOB_GROUP = {1: "separable", 2: "separable", 6: "moderate", 8: "moderate",
               13: "ill-conditioned", 14: "ill-conditioned", 15: "multimodal",
               17: "multimodal", 20: "weakly-structured", 21: "weakly-structured"}

# Liu, Gu & Zhang 2014, Section III-C (see the module docstring)
_MOP_FRONT = {1: "convex", 2: "concave", 3: "concave", 4: "disconnected", 5: "convex",
              6: "linear", 7: "concave"}
_MOP_MULTIMODAL = {1: True, 2: False, 3: False, 4: False, 5: True, 6: True, 7: True}

# Li, Zhang & Deng 2017, Section III-B and Appendix A
_BT_FRONT = {i: "convex" for i in (1, 2, 3, 4, 6, 7, 8)}
_BT_FRONT.update({5: "disconnected", 9: "concave"})
_BT_LINKAGE = {6, 7, 8}

# the curved Pareto sets (linkage), by family; WFG, IPolygon, bbob: None
_LINKAGE = {"ZDT": False, "DTLZ": False, "ZCAT": True, "MOP": True, "Polygon": False}


def _stem_and_m(name: str) -> tuple:
    stem = name.split("_")[0]
    tail = name.rsplit("_", 1)[-1]
    m = int(tail[:-1]) if tail.endswith("D") and tail[:-1].isdigit() else 2
    return stem, m


def properties(name: str) -> dict | None:
    """The row for one registry name, or None for a problem the table lacks."""
    row = _huband(name)
    if row is None:
        return None
    stem, m = _stem_and_m(name)
    front = row["front"]
    row["degenerate"] = None if front is None else "degenerate" in front
    row["disconnected"] = None if front is None else "disconnected" in front
    if stem.startswith("BT"):
        row["linkage"] = int(stem[2:]) in _BT_LINKAGE
    elif stem.startswith(("ZDT", "DTLZ", "shiftDTLZ", "IDTLZ", "SDTLZ")):
        row["linkage"] = False
    elif stem.startswith("ZCAT"):
        row["linkage"] = _LINKAGE["ZCAT"]
    elif stem.startswith("MOP"):
        row["linkage"] = _LINKAGE["MOP"]
    elif stem == "Polygon":
        row["linkage"] = _LINKAGE["Polygon"]
    else:
        row["linkage"] = None
    from .registry import PROBLEMS
    p = PROBLEMS.get(name)
    row["n_obj"] = p.n_obj if p is not None else m
    row["n_vars"] = p.n_vars if p is not None else None
    row["bbob_groups"] = None
    if stem.startswith("bbobbiobj"):
        from .bbob_biobj import PAIRS
        fa, fb = PAIRS[int(stem[len("bbobbiobj"):]) - 1]
        row["bbob_groups"] = "+".join(sorted((_BBOB_GROUP[fa], _BBOB_GROUP[fb])))
    return row


def _huband(name: str) -> dict | None:
    """Huband's seven keys for one registry name, or None."""
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
    if stem.startswith("MOP") and stem[3:].isdigit():
        i = int(stem[3:])
        return _p(_MOP_FRONT[i], _MOP_MULTIMODAL[i], None, False, False, False)
    if stem.startswith("BT") and stem[2:].isdigit():
        i = int(stem[2:])
        return _p(_BT_FRONT[i], i == 8, True, False, i not in _BT_LINKAGE, False)
    if stem == "Polygon":
        return _p("linear+degenerate" if m >= 4 else "linear", False, False, False, False, True)
    if stem == "IPolygon":
        return _p("degenerate" if m == 4 else None, True, False, False, False, None)
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

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
  boundary    the optimum on the boundary of the domain — R1's "extremal
              parameters": the projection "is a single value at the edge of
              the domain" (task 6 of 2026-10-09; Huband's Tables V, VII and
              XV for ZDT, DTLZ and WFG)

True / False, or None where the source leaves it open ("?" in Huband's table
for DTLZ5 and DTLZ6) or where nothing classifies it. Beyond the three suites
Huband analyses:

  IDTLZ1/2    DTLZ1/2 with the front inverted (Jain & Deb 2014, Eq. 9 for
              IDTLZ1; IDTLZ2 is this library's analogue, dtlz_variants.py)
  SDTLZ1/2    DTLZ1/2 with objective i scaled by p^(i-1) (Deb & Jain 2014,
              Table VIII): scaled
  minusDTLZ2  DTLZ2 negated (Ishibuchi, Setoguchi, Masuda & Nojima, IEEE TEVC
              21(2), 2017): the front -(1 + 0.25 k) times the sphere's octant,
              convex; g at its largest, every distance variable at 0 or 1, so
              the Pareto set is 2^k equivalent pieces (multimodal, as
              IPolygon's four) on the boundary; separable as DTLZ2
  shiftDTLZ   DTLZ1-4 with every distance variable's optimum 0.15-0.35 from
              the centre (dtlz_variants.py): not medial, the point of them
  wideDTLZ3   DTLZ3 with the distance variables in [0, 1.2]: DTLZ3's row,
              but not medial (the optimum at 0.5, the middle of the box at 0.6)
  C-DTLZ      Jain & Deb 2014 (c_dtlz.py): C1 and C2-DTLZ2 their DTLZ's row,
              the rest ours (see _CDTLZ below)
  MW1-14      Ma & Wang, IEEE TEVC 23(6), 2019, Table II (mw.py): the front is
              the table's constrained PF ("Discrete", MW5's, read as
              disconnected), with the unconstrained PF's geometry where the
              one is a part of the other (MW1, MW6, MW8). Multimodal where the
              distance function is g2 (MW2, 6, 8, 10, 13) or a sine makes f2 or
              f_m rise and fall along the positions (MW12, MW13, MW14, as
              ZDT3's and DTLZ7's in Huband's tables); bias where it is g1, the
              paper's "biased" one (MW1, 4, 5, 9, 12), and MW10's f1 = g x1^n
              (its "polynomial bias"); linkage, and not separable, where it is
              g3 (MW3, 7, 11, 14). Where the front lies at g = 1 (mw.ON_G1) the
              distance variables' optimum is a constant (g1, g2: separable) or
              g3's chain, inside the box: neither medial nor extremal; where it
              needs g > 1 they lie on level sets of g: separable, centre and
              boundary None (as C3-DTLZ's). Scaled where one objective's range
              on the front is at least twice another's (MW13, MW14; mw.frame).
  shiftZDT1-4, shiftZDT6, shiftDTLZ6-7
              the base problem's row with every distance variable's optimum
              0.15-0.35 from the centre and from the bounds (shifted.py):
              neither medial nor extremal
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
              optima spread over the whole [0, 1] (False; None until
              2026-10-10).
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
  SYM-PART, Omni-test, MMF
              the CEC 2019 competition's multimodal problems (mmo.py; Liang,
              Qu, Gong & Yue's report, Table I): the front as the table has
              it, MMF12's pieces read as disconnected; multimodal, every one
              having several global optima of its objectives; scaled where
              one objective's range on the front is at least twice another's
              (MMF9, 10, 11, 13); bias, separable and centre left open (the
              table has no column for them); boundary as measured() reads it.
  IPolygon    Ishibuchi, Akedo & Nojima, GECCO 2011: m = 4 identical polygons
              in [0, 100]^2, each objective the distance to the nearest copy
              of a vertex, so every objective has four minima and the Pareto
              set four equivalent regions (multimodal); the front is one
              polygon's, as MaF8's (linear, degenerate from four objectives);
              the polygons sit in the four quarters, not in the middle. (The
              instance before reference version 2 had two polygons too close
              for that: polygon_ishibuchi.py.)

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
  family_kind "engineering" for the real-world problems (RE, CRE, RCM),
              "synthetic" for the rest
  difficulty  a constrained problem's difficulty where its source names one:
              DAS-CMOP's kinds (diversity, feasibility, convergence, from its
              triplet), RWMOP's level ("low" for the mechanical RCM, every
              algorithm of the paper feasible in every run, §3.3.1)
  multimodal_ps, ps_parts
              the Pareto set is several parts that each map onto the whole
              front (supplement 3 of task 7, §5.4), and how many: the
              multimodal problems (mmo.parts; MMF6's pieces of a third and
              fourth part not counted), IPolygon (its four polygons), and
              minusDTLZ2 (2^k, every distance variable at 0 or 1). None
              elsewhere — not classified, though most have one part
  front_dim   d, the dimension of the Pareto front (reviewer, 2026-09-29: to
              read a ratio to the IGD+ floor alike across M as ratio^(-d)):
              M - 1 for a regular front, 1 at two objectives; 1 for the curves
              of DTLZ5 and DTLZ6 at three objectives and of ZCAT14-16; 2 for
              Polygon and IPolygon, one polygon's image whatever M; None where
              a degenerate front is not one of those (DTLZ5/6 from four
              objectives and WFG3, which have a non-degenerate part besides,
              ZCAT17-20, part degenerate)
  front_ref   "exact" where the registry's reference front lies on the true
              front (BenchProblem.front_exact), "approximate" where it is an
              approximation: RE and CRE (the authors' fronts), RE22c and
              RE33c, RCM but RCM28 (this library's long runs), C2-ConvexDTLZ2
              (a checked sample) and the failing copies whose region takes
              some of the Pareto set; None without a reference front. A rank
              table sets the approximate ones apart (campaign.rank_table)
  front_built how an approximate reference front was built, in a sentence
              (front_built below); None for an exact one
  undefined   where the problem is not defined in part of its box (its
              evaluation fails there: NaN, which a run counts as a failed
              evaluation), the sentence saying where: RCM02 and RCM24
              (rwmop.UNDEFINED, task 10 of 2026-10-18, §6); None elsewhere
  undefined_share
              the share of the box where it is not defined, from 100 000
              uniform points (rwmop.undefined_share); 0 elsewhere among the
              RCM, None for the rest (not measured)

MEASURED (task 6 of 2026-10-09). Where a problem has a Pareto-set sample
(pareto_sets.py), measured() reads centre and boundary off it: the share of
the distance variables' values within 1e-9 of the middle of their range, and
of either bound, in units of the range; a flag holds from 99 %. The tests hold
every boundary value of the table to it, and the centre values where the table
has one. IPolygon has no distance variables, so both of its variables count.
Without a sample the value is the source's — WFG: Table XV's caption ("no
extremal nor medial parameters"); MOP and BT: their Pareto sets, curves or
constants inside the box (Liu, Gu & Zhang 2014, Section III-C; Li, Zhang & Deng
2017, Section III-B); bbob-biobj: x_opt uniform in [-4, 4]^n inside [-5, 5]^n
(Brockhoff et al. 2022) — and DTLZ5/6 from four objectives, whose sample covers
only the curve, keep the value read at two and three.
"""

from __future__ import annotations

from .rotated import SPECS as _ROTATED

KEYS = ("front", "multimodal", "deceptive", "bias", "scaled", "separable", "centre",
        "boundary", "degenerate", "disconnected", "linkage", "n_obj", "n_vars",
        "bbob_groups", "front_dim", "family_kind", "difficulty", "multimodal_ps",
        "ps_parts", "front_ref", "front_built", "undefined", "undefined_share")


def _p(front, multimodal, bias, scaled, separable, centre, boundary,
       deceptive=False) -> dict:
    return {"front": front, "multimodal": multimodal, "deceptive": deceptive,
            "bias": bias, "scaled": scaled, "separable": separable, "centre": centre,
            "boundary": boundary}


# Huband et al. 2006, Table V (R2 x = medial parameters present; R1 x =
# extremal parameters present; R6 check = dissimilar ranges).
_ZDT = {
    "ZDT1": _p("convex", False, False, False, True, False, True),
    "ZDT2": _p("concave", False, False, False, True, False, True),
    "ZDT3": _p("disconnected", True, False, True, True, False, True),
    "ZDT4": _p("convex", True, False, False, True, True, False),
    "ZDT6": _p("concave", True, True, True, True, False, True),
}

# Table VII. "S*": separable. DTLZ5/6: "?" where the table has it; their
# centre and boundary are ours — DTLZ5's distance optimum is x = 1/2
# (medial), DTLZ6's x = 0 (extremal).
_DTLZ = {
    "DTLZ1": _p("linear", True, False, False, True, True, False),
    "DTLZ2": _p("concave", False, False, False, True, True, False),
    "DTLZ3": _p("concave", True, False, False, True, True, False),
    "DTLZ4": _p("concave", False, True, False, True, True, False),
    "DTLZ5": _p("degenerate", False, False, None, None, True, False),
    "DTLZ6": _p("degenerate", False, True, None, None, False, True),
    "DTLZ7": _p("disconnected", True, False, False, True, False, True),
}

# C-DTLZ (Jain & Deb 2014; c_dtlz.py). C1 and C2-DTLZ2 keep their DTLZ's row,
# C2-DTLZ2's front cut into pieces; the rest is ours, read off c_dtlz.py's
# FRONTS: C2-ConvexDTLZ2's front is the convex one with a hole and the points
# behind it, C3-DTLZ1's piecewise linear and bent towards the ideal point
# (convex), C3-DTLZ4's pieces of ellipsoids, concave, meeting in kinks that
# point to the ideal point (mixed). Where the Pareto set needs g > 0 (C3, and
# C2-ConvexDTLZ2 behind the hole) its distance variables lie on level sets of
# g: separable and centre unknown, None; C3-DTLZ4's g reaches 1 at the
# corners, so not medial there.
_CDTLZ = {
    "C1-DTLZ1": dict(_DTLZ["DTLZ1"]),
    "C1-DTLZ3": dict(_DTLZ["DTLZ3"]),
    "C2-DTLZ2": dict(_DTLZ["DTLZ2"], front="concave+disconnected"),
    "C2-ConvexDTLZ2": _p("convex", False, False, False, None, None, False),
    "C3-DTLZ1": _p("convex", True, False, False, None, None, False),
    "C3-DTLZ4": _p("mixed", False, True, False, None, False, False),
}

# MW1-14 (Ma & Wang 2019, Table II; see the module docstring)
_MW = {
    "MW1": _p("linear+disconnected", False, True, False, True, False, False),
    "MW2": _p("linear", True, False, False, True, False, False),
    "MW3": _p("mixed", False, False, False, False, None, None),
    "MW4": _p("linear", False, True, False, True, False, False),
    "MW5": _p("disconnected", False, True, False, None, None, None),
    "MW6": _p("concave+disconnected", True, False, False, True, False, False),
    "MW7": _p("disconnected", False, False, False, False, None, None),
    "MW8": _p("concave+disconnected", True, False, False, True, False, False),
    "MW9": _p("concave", False, True, False, None, None, None),
    "MW10": _p("disconnected", True, True, False, None, None, None),
    "MW11": _p("disconnected", False, False, False, False, None, None),
    "MW12": _p("mixed", True, True, False, None, None, None),
    "MW13": _p("disconnected", True, False, True, None, None, None),
    "MW14": _p("disconnected", True, False, True, False, False, False),
}
_MW_LINKAGE = {"MW3", "MW7", "MW11", "MW14"}             # g3

# Table XV; its caption: no extremal nor medial parameters, dissimilar
# tradeoff magnitudes for every WFG problem.
_WFG = {
    "WFG1": _p("convex+mixed", False, True, True, True, False, False),
    "WFG2": _p("convex+disconnected", True, False, True, False, False, False),
    "WFG3": _p("linear+degenerate", False, False, True, False, False, False),
    "WFG4": _p("concave", True, False, True, True, False, False),
    "WFG5": _p("concave", True, False, True, True, False, False, deceptive=True),
    "WFG6": _p("concave", False, False, True, False, False, False),
    "WFG7": _p("concave", False, True, True, True, False, False),
    "WFG8": _p("concave", False, True, True, False, False, False),
    "WFG9": _p("concave", True, True, True, False, False, False, deceptive=True),
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
    from ..families import NOISE_LEVELS
    stem = name.split("_")[0]
    last = name.rsplit("_", 1)[-1]
    if (last[:1] == "i" and last[1:].isdigit()) or last in NOISE_LEVELS:
        name = name.rsplit("_", 1)[0]       # a rotation's instance or a noisy copy, ..._3D_i2
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
    elif stem.startswith(("ZDT", "DTLZ", "shiftDTLZ", "shiftZDT", "IDTLZ", "SDTLZ", "wideDTLZ",
                          "C1-DTLZ", "C2-DTLZ2", "rDTLZ", "rcDTLZ", "r0DTLZ", "c0DTLZ",
                          "rZDT")):
        row["linkage"] = False
    elif stem.startswith(("ZCAT", "rZCAT")):
        row["linkage"] = _LINKAGE["ZCAT"]
    elif stem.startswith("MOP"):
        row["linkage"] = _LINKAGE["MOP"]
    elif stem in ("Polygon", "rcPolygon", "c0Polygon"):
        row["linkage"] = _LINKAGE["Polygon"]
    elif stem in _MW:
        row["linkage"] = stem in _MW_LINKAGE
    else:
        row["linkage"] = None
    from .registry import PROBLEMS
    p = PROBLEMS.get(name)
    if p is not None and p.failing is not None and not len(p.pareto_set(10)):
        # a failing copy (failing.py) whose region takes all of the base's
        # Pareto set: the optimum moved to the region's edge, not known here
        row["boundary"] = row["centre"] = None
    row["n_obj"] = p.n_obj if p is not None else m
    row["n_vars"] = p.n_vars if p is not None else None
    row["bbob_groups"] = None
    if stem.startswith("bbobbiobj"):
        from .bbob_biobj import PAIRS
        fa, fb = PAIRS[int(stem[len("bbobbiobj"):].removeprefix("mixint")) - 1]
        row["bbob_groups"] = "+".join(sorted((_BBOB_GROUP[fa], _BBOB_GROUP[fb])))
    row["front_dim"] = _front_dim(stem, row["n_obj"], front)
    from ..families import is_engineering
    row["family_kind"] = "engineering" if is_engineering(name) else "synthetic"
    # a constrained problem's kinds of difficulty, where its source names them
    # (DAS-CMOP's triplet: diversity, feasibility, convergence); None elsewhere
    row["difficulty"] = None
    from .registry import DASCMOP
    if name in DASCMOP:
        eta, zeta, gamma = DASCMOP[name][1]
        kinds = [k for k, v in (("diversity", eta), ("feasibility", zeta),
                                ("convergence", gamma)) if v > 0]
        row["difficulty"] = "+".join(kinds) if kinds else "none"
    elif stem.startswith("RCM") and stem[3:].isdigit():
        # RWMOP's level (Kumar et al. 2021, §3.3.1): every algorithm of the
        # paper feasible in every run on the mechanical problems, so "low"
        row["difficulty"] = "low"
    row["multimodal_ps"] = row["ps_parts"] = None
    from . import mmo
    if name in mmo.NAMES:
        row["multimodal_ps"], row["ps_parts"] = True, mmo.parts(name)
    elif stem == "IPolygon":
        row["multimodal_ps"], row["ps_parts"] = True, 4
    elif stem == "minusDTLZ2" and p is not None:
        row["multimodal_ps"], row["ps_parts"] = True, 2 ** (p.n_vars - p.n_obj + 1)
    row["front_ref"] = (None if p is None or not callable(p.pareto_front)
                        else "approximate" if approximate_front(name) else "exact")
    row["front_built"] = front_built(name)
    row["undefined"] = row["undefined_share"] = None
    if stem.startswith("RCM") and stem[3:].isdigit():
        from . import rwmop
        i = int(stem[3:])
        row["undefined"] = rwmop.UNDEFINED.get(i)
        row["undefined_share"] = rwmop.undefined_share(i)
    return row


def approximate_front(name: str) -> bool:
    """Is the registry's reference front of `name` an approximation
    (BenchProblem.front_exact False)? False without one."""
    from .registry import PROBLEMS
    p = PROBLEMS.get(name)
    return p is not None and callable(p.pareto_front) and not p.front_exact


def front_built(name: str) -> str | None:
    """How the approximate reference front of `name` was built (front_built
    above); None for an exact one."""
    if not approximate_front(name):
        return None
    from .registry import PROBLEMS
    p = PROBLEMS[name]
    if p.failing is not None:
        if len(p.pareto_set(10)):
            return ("the image of the base's Pareto-set sample outside the failing region "
                    "(failing.py, THE FRONT)")
        return ("the base's front, which no point outside the failing region reaches "
                "(failing.py, THE FRONT)")
    stem = name.split("_")[0]
    if stem.startswith("RCM"):
        from .rwmop import front_built as rcm_front_built
        return rcm_front_built(int(stem[3:]))
    if stem in ("RE22c", "RE33c"):
        return ("an exhaustive grid, then local refinement, thinned by DSS to 500 M points "
                "(re_fronts.py, benchmarks/_re/corrected/)")
    if stem.startswith("CRE"):
        return ("the feasible points of the authors' front of the RE problem it constrains "
                "(reproblems, benchmarks/_re/)")
    if stem.startswith("RE"):
        return ("the authors' approximation, at most 500 M points of every non-dominated "
                "point their optimizers found, as shipped (reproblems, benchmarks/_re/)")
    if stem == "C2-ConvexDTLZ2":
        return ("a sample of the convex front outside the cylinder and of the points behind "
                "its hole that nothing dominates, each checked (c_dtlz.py, FRONTS)")
    return "an approximation: the problem's module says how it was built"


def _front_dim(stem: str, m: int, front: str | None) -> int | None:
    """The Pareto front's dimension d (see front_dim above)."""
    if m == 2:
        return 1
    if stem in ("Polygon", "IPolygon"):
        return 2
    if stem in ("DTLZ5", "DTLZ6") and m == 3:
        return 1
    if stem in ("ZCAT14", "ZCAT15", "ZCAT16"):
        return 1
    if stem == "DOC9":                               # the quarter circle at f3 = 0
        return 1
    if front is not None and "degenerate" in front:
        return None
    return m - 1


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
        row["centre"] = row["boundary"] = False
        return row
    if stem.startswith("shiftZDT") and stem[5:] in _ZDT:
        row = dict(_ZDT[stem[5:]])
        row["centre"] = row["boundary"] = False
        return row
    if stem in _CDTLZ:
        return dict(_CDTLZ[stem])
    if stem in _MW:
        return dict(_MW[stem])
    if stem.startswith(("RE", "CRE")) and stem.lstrip("CRE").rstrip("c").isdigit():
        # Tanabe & Ishibuchi 2020: the true fronts are unknown; Section 2.6 and
        # Fig. 2 call RE2-4-3 (RE23) and RE2-3-5 (RE25) disconnected. The rest
        # of Huband's vocabulary is not given for them: None.
        front = "disconnected" if stem in ("RE23", "RE25") else None
        return _p(front, None, None, None, None, None, None, deceptive=None)
    if stem in _ROTATED:
        # rotated.py: the base's row, the optimum off centre and bound; not
        # separable where the rotation mixes an anisotropic g (rDTLZ2 and
        # rDTLZ4 are their Q = I construction: separable as DTLZ2)
        family, rotate, condition = _ROTATED[stem]
        if family.startswith("MaF"):
            return None                          # MaF has no row of its own
        row = _huband(family + ("" if family.startswith("ZDT") else f"_{m}D"))
        row["centre"] = row["boundary"] = False
        if rotate and (condition or family not in ("DTLZ2", "DTLZ4")):
            row["separable"] = False
        return row
    if stem.startswith("DAS-CMOP"):
        # Fan et al. 2020, Table 2: the unconstrained shapes (ZDT2-, ZDT1- and
        # ZDT3-like; DTLZ1-like plane; DTLZ2-like sphere), g Rastrigin-like in
        # DAS-CMOP4-8; the constraints of Types I and III cut the front
        from .dascmop import TRIPLETS  # noqa: F401  (the triplets are the registry's)
        i = int(stem[len("DAS-CMOP"):])
        shape = {1: "concave", 2: "convex", 3: "mixed", 4: "concave", 5: "convex", 6: "mixed",
                 7: "linear", 8: "concave", 9: "concave"}[i]
        from .registry import DASCMOP
        eta, _, gamma = DASCMOP.get(name, (i, (0.0, 0.0, 0.0)))[1]
        if eta > 0 or gamma > 0 or i in (3, 6):
            shape += "+disconnected"
        return _p(shape, 4 <= i <= 8, False, False, False, False, False)
    if stem.startswith("LIR-CMOP"):
        # Fan et al. 2019, Table 6 and Fig. 1-2: 1 - x1^2 concave, 1 - sqrt(x1)
        # convex, the sphere of 13-14; 3-4 and 9-12 cut into pieces (the
        # window sin(20 pi x1) >= 0.5, the sine constraint), 7-8 on an
        # ellipse's boundary, which bulges away from the origin
        i = int(stem[len("LIR-CMOP"):])
        shape = {1: "concave", 2: "convex", 3: "concave", 4: "convex", 5: "convex",
                 6: "concave", 7: "concave", 8: "concave", 9: "concave", 10: "convex",
                 11: "convex", 12: "concave", 13: "concave", 14: "concave"}[i]
        if i in (3, 4) or 9 <= i <= 12:
            shape += "+disconnected"
        return _p(shape, False, False, False, False, False, False)
    if stem.startswith("RCM") and stem[3:].isdigit():
        # RWMOP's mechanical problems (rwmop.py): nothing of Huband's classified
        return _p(None, None, None, None, None, None, None)
    if stem.startswith("DOC") and stem[3:].isdigit():
        # Liu and Wang 2019, Table I: the front's shape, and "multimodal"
        # where the table names the problem so (DOC3, 5, 6, 7, 9)
        i = int(stem[3:])
        shape = {1: "concave", 2: "convex+disconnected", 3: "concave+disconnected",
                 4: "linear+disconnected", 5: "linear+disconnected", 6: "mixed",
                 7: "mixed", 8: "linear+disconnected", 9: "degenerate+concave"}[i]
        return _p(shape, i in (3, 5, 6, 7, 9), False, False, False, False, False)
    if stem == "wideDTLZ3":
        row = dict(_DTLZ["DTLZ3"])
        row["centre"] = False
        return row
    if stem in ("IDTLZ1", "IDTLZ2"):
        row = dict(_DTLZ["DTLZ" + stem[-1]])
        row["front"] = "linear" if stem == "IDTLZ1" else "convex"
        return row
    if stem == "minusDTLZ2":
        return _p("convex", True, False, False, True, False, True)
    if stem in ("SDTLZ1", "SDTLZ2"):
        row = dict(_DTLZ["DTLZ" + stem[-1]])
        row["scaled"] = True
        return row
    if stem.startswith("ZCAT") and stem[4:].isdigit():
        return _p(_ZCAT_FRONT[int(stem[4:])], False, False, True, False, False, False)
    if stem.startswith("bbobbiobj"):
        # bbob-biobj-mixint (bbobbiobjmixint..) as its continuous pair: a
        # discretized separable function stays separable
        from .bbob_biobj import PAIRS
        fa, fb = PAIRS[int(stem[len("bbobbiobj"):].removeprefix("mixint")) - 1]
        return _p(None, fa in _BBOB_MULTIMODAL or fb in _BBOB_MULTIMODAL, False, None,
                  fa in _BBOB_SEPARABLE and fb in _BBOB_SEPARABLE, False, False)
    if stem == "catZDT1":
        # categorical.py: ZDT1's front, held by the multimodal g only, the
        # easy ones just above it (deceptive); its optimum y = 0.5, medial
        return _p("convex", True, False, False, False, True, False, deceptive=True)
    if stem.startswith("MOP") and stem[3:].isdigit():
        i = int(stem[3:])
        return _p(_MOP_FRONT[i], _MOP_MULTIMODAL[i], None, False, False, False, False)
    if stem.startswith("BT") and stem[2:].isdigit():
        i = int(stem[2:])
        return _p(_BT_FRONT[i], i == 8, True, False, i not in _BT_LINKAGE, False, False)
    if stem == "Polygon":
        return _p("linear+degenerate" if m >= 4 else "linear", False, False, False, False, True,
                  False)
    if stem == "IPolygon":
        return _p("linear+degenerate" if m >= 4 else "linear", True, False, False, False, False,
                  False)
    from . import mmo
    if name in mmo.NAMES:
        ideal, nadir = mmo.frame(name)
        span = [b - a for a, b in zip(ideal, nadir)]
        return _p(_MMO_FRONT.get(name, "convex"), True, None, max(span) >= 2 * min(span), None,
                  None, False)
    return None


# the CEC 2019 report's Table I, "Pareto front geometry"; MMF12's pieces
# disconnected besides
_MMO_FRONT = {"MMF4": "concave", "MMF8": "concave", "MMF12": "convex+disconnected",
              "MMF14": "concave", "MMF14_a": "concave", "MMF15": "concave",
              "MMF15_a": "concave"}


def interior(name: str) -> bool:
    """Does the table put the optimum neither at the centre nor on the
    boundary (centre and boundary not True; None, unknown, counts as
    neither)? A problem without a row: no."""
    row = properties(name)
    return row is not None and row["centre"] is not True and row["boundary"] is not True


def _positions(name: str, n_obj: int) -> int:
    """How many of the leading variables are position variables, whose
    optimum spans the box; the rest are the distance variables measured()
    reads (IPolygon: none — both of its variables count)."""
    stem, _ = _stem_and_m(name)
    if stem.startswith(("ZDT", "shiftZDT", "rZDT")):
        return 1
    if stem in ("Polygon", "rcPolygon", "c0Polygon"):
        return 2
    if stem == "IPolygon":
        return 0
    if stem.startswith("rWFG"):
        return 2 * (n_obj - 1)  # WFG's k
    return n_obj - 1            # DTLZ and its variants, ZCAT


def measured(name: str, n: int = 300) -> dict | None:
    """centre and boundary read off the problem's Pareto-set sample (MEASURED
    above): the shares of the distance variables' values within 1e-9 of the
    middle of the range and of a bound, in units of the range, and the flags
    at 99 %; None where the problem has no sample, or an empty one (a failing
    copy whose region takes all of its base's Pareto set)."""
    import numpy as np
    from .registry import PROBLEMS
    p = PROBLEMS[name]
    if p.pareto_set is None:
        return None
    k = _positions(name, p.n_obj)
    X = np.asarray(p.pareto_set(n), float)[:, k:]
    if not len(X):
        return None
    lo = np.array([b[0] for b in p.bounds], float)[k:]
    hi = np.array([b[1] for b in p.bounds], float)[k:]
    U = (X - lo) / (hi - lo)
    at_middle = float(np.mean(np.abs(U - 0.5) <= 1e-9))
    at_bound = float(np.mean((U <= 1e-9) | (U >= 1.0 - 1e-9)))
    return {"centre_share": at_middle, "boundary_share": at_bound,
            "centre": at_middle >= 0.99, "boundary": at_bound >= 0.99}


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

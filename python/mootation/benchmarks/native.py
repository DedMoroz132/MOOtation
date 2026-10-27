# SPDX-License-Identifier: Apache-2.0
"""The problems the C++ suite evaluates (task 11 of 2026-10-18, §1).

The pilot of the owner's decision of 2026-10-18 moved the most used and
simplest families to C++ (include/mootation/suite): ZDT1-4 and ZDT6; DTLZ1-7,
wideDTLZ3, shiftDTLZ1-4, 6, 7, IDTLZ1/2, minusDTLZ2, SDTLZ1/2 and C-DTLZ;
MaF1-7, MaF9 and MaF13; ZCAT1-20; shiftZDT1-4 and 6; and the rotations over
them (rotated.py: rZDT, rDTLZ, rcDTLZ, r0/c0DTLZ, rMaF, rcMaF, r0/c0MaF,
rZCAT). Task 12 of 2026-10-19 added MaF8 (§6: version 2, the C library's
hypot), the new MaF14 and MaF15 (§2), and bbob-biobj and bbob-biobj-mixint
(§1: the 110 problems of each; include/mootation/suite/bbob.hpp). Task 17,
supplement 1, stage 1: WFG1-9, MaF10-12 (WFG1, WFG2 and WFG9 with M - 1
positions) and rWFG1-9 (include/mootation/suite/wfg.hpp). Task 17,
supplement 3, stage 3a: DAS-CMOP1-9, LIR-CMOP1-14, MW1-14, DOC1-9, Polygon
with rcPolygon and c0Polygon, IPolygon, SYM-PART, Omni-test, MMF1-15, RE and
CRE (RE91 without a seed; its seeded stream, re.RE91Evaluator, stays
Python's), and the multimodal problems' and the polygons' Pareto sets
(attach: p.pareto_set). Not moved: every other family.

With the extension built, such a problem's evaluate, evaluate_batch,
constraints, constraints_batch and evaluate_with_constraints are C++ (attach):
one implementation, called through the binding, its loop over a batch in C++,
and minimize and a campaign hand it to the core, which then evaluates it
without calling Python (minimize's native). The NumPy code of these modules
stays as the reference, in BenchProblem.reference, and without the extension
it is what runs. The C++ suite's exp, log, pow, sin, cos, tan and hypot are the
library's own (include/mootation/detmath.hpp; task 14 of 2026-10-20), the same
bits on every platform; the NumPy code stays on NumPy, that is on the C
library, whose last bits may differ from machine to machine. The tests hold
the two within 4 ulps (near zero, within 2e-15 of the problem's scale; bbob
4e-15, WFG, MW, DOC and LIR-CMOP 1e-13) wherever NumPy's sin, cos, exp and
pow are the C library's (_rows.agrees) — bbob-biobj's, where its BLAS also
adds in the order bbob.hpp writes out.

spec(name) is what C++ needs of a problem: its family and sizes and the
numbers Python computes for it (centres, MaF14's and MaF15's groups, the
constants of C-DTLZ and SDTLZ, ...), as include/mootation/suite/suite.hpp,
Spec, reads them. What a problem draws — the rotations (a key: family, M,
instance, sizes), the bbob instances (F and instance), MaF8's and MaF9's
polygons, MaF14's and MaF15's linkage — C++ draws (task 17, supplement 5, §3).
"""

from __future__ import annotations

import re

import numpy as np

from . import c_dtlz as _cd
from . import dtlz_variants as _dv
from . import shifted as _shift

# the stems of the problems with an objective count (<stem>_<M>D), and of ZDT
_STEMS = re.compile(r"(DTLZ[1-7]|wideDTLZ3|shiftDTLZ[1-467]|IDTLZ[12]|minusDTLZ2|SDTLZ[12]|"
                    r"C1-DTLZ[13]|C2-DTLZ2|C2-ConvexDTLZ2|C3-DTLZ[14]|MaF([1-9]|1[0-5])|"
                    r"ZCAT([1-9]|1[0-9]|20)|WFG[1-9])_(\d+)D")
# MaF10-12: WFG1, WFG2 and WFG9 with M - 1 positions (Cheng et al. 2017, 2.10-2.12)
_MAF_WFG = {"MaF10": "WFG1", "MaF11": "WFG2", "MaF12": "WFG9"}
_ZDT = re.compile(r"(shift)?ZDT[12346]")
_BBOB = re.compile(r"bbobbiobj(mixint)?\d\d_n\d\d_2D")
# task 17, supplement 3, stage 3a
_DAS = re.compile(r"DAS-CMOP([1-9])_(t\d\d|f6)")
_OTHER = re.compile(r"LIR-CMOP\d+|DOC\d|(I?Polygon)_(\d+)D|(MW\d+)(_\d+D)?")
_MMO = re.compile(r"SYM-PART_(simple|rotated)|Omni-test|MMF\d+(_[aez])?")
_RE = re.compile(r"RE\d\dc?|CRE\d\d")
# BT1-9 and MOP1-7, and RWMOP's RCM problems (task 17, stage 5; bt.hpp, mop.hpp,
# rcm.hpp)
_BT_MOP = re.compile(r"BT[1-9]|MOP[1-7]|RCM\d\d|catZDT1")
# the bit strings (bits.hpp): OMM_n<n>, LOTZ_n<n>, OJZJ_n<n>_k<k>
_BITS = re.compile(r"(OMM|LOTZ)_n\d+|OJZJ_n\d+_k(\d+)")
# the verification set (task 17, supplement 2, §3; classic.hpp): its names
# and the suite's families
CLASSIC = {"SCH": "SCH", "FON": "FON", "POL": "POL", "KUR": "KUR", "SPH_2D": "SPH", "SPH_3D": "SPH",
           "QV": "QV", "KURZ": "KURZ"}


def _base(family: str, n_var: int, n_obj: int, **extra) -> dict:
    return {"family": family, "n_var": int(n_var), "n_obj": int(n_obj), **extra}


def _array(v) -> np.ndarray:
    return np.ascontiguousarray(v, dtype=float)


def _rotated(r) -> dict | None:
    """A rotated problem (rotated.Rotated) of the suite's families."""
    if r.kind not in ("dtlz", "maf", "zdt", "zcat", "wfg", "polygon"):
        return None
    # the matrices are drawn by C++ (suite.hpp, Rotation; task 17, supplement 5, §3)
    d = _base(r.family, r.n_vars, r.M, first=r.first,
              rotation={"family": r.family, "M": r.M, "instance": r.instance, "rotate": r.rotate,
                        "condition": r.condition, "sizes": r.sizes()})
    if r.kind == "polygon":
        d.update(map="rotate", centre=_array(r.optimum), vertices=_polygon_vertices(r.M))
    elif r.wfg:
        if r.family in ("WFG4", "WFG5", "WFG6", "WFG7"):
            d.update(map="rotate_wfg_035")
        else:
            d.update(map="rotate_wfg", centre=_array([] if r.family == "WFG8" else r.centre_u))
    elif r.kind == "zcat":
        d.update(map="rotate_zcat")
    elif r.family == "ZDT4":
        d.update(map="rotate", centre=_array(r.optimum))
    elif r.zdt:
        d.update(map="rotate_abs", centre=_array(r.optimum), spread=_array(r.spread))
    else:
        d.update(map="rotate_half", centre=_array(r.optimum))
    return d


def _polygon_vertices(M: int) -> np.ndarray:
    from .polygon import polygon_vertices
    return _array(polygon_vertices(M)).ravel()


def _stage3a(name: str, p) -> dict | None:
    """DAS-CMOP, LIR-CMOP, DOC, Polygon, IPolygon, SYM-PART, Omni-test, MMF, RE
    and CRE (task 17, supplement 3, stage 3a); BT, MOP, RCM, the bit strings and
    catZDT1 (task 17, stage 5)."""
    if _MMO.fullmatch(name) or _RE.fullmatch(name) or _BT_MOP.fullmatch(name):
        return _base(name, p.n_vars, p.n_obj)
    m = _BITS.fullmatch(name)
    if m is not None:
        k = int(m.group(2)) if m.group(2) else 0
        return _base(name.split("_")[0], p.n_vars, p.n_obj, k=k)
    m = _DAS.fullmatch(name)
    if m is not None:
        from . import dascmop
        i, tag = int(m.group(1)), m.group(2)
        triplet = dascmop.FIGURE6 if tag == "f6" else dascmop.TRIPLETS[int(tag[1:]) - 1]
        return _base(f"DAS-CMOP{i}", p.n_vars, p.n_obj, constants=_array(dascmop.params(*triplet)))
    m = _OTHER.fullmatch(name)
    if m is None:
        return None
    if m.group(1) == "Polygon":
        return _base("Polygon", p.n_vars, p.n_obj, vertices=_polygon_vertices(p.n_obj))
    if m.group(1) == "IPolygon":
        from .polygon_ishibuchi import _vertices
        return _base("IPolygon", p.n_vars, p.n_obj,
                     polygons=[_array(V).ravel() for V in _vertices(p.n_obj)])
    if m.group(3):
        return _base(m.group(3), p.n_vars, p.n_obj)   # MW<i>, MW<i>_<M>D
    return _base(name, p.n_vars, p.n_obj)             # LIR-CMOP<i>, DOC<i>


def _bbob(name: str, p) -> dict:
    """A bbob-biobj problem or a bbob-biobj-mixint one: F1..F55 at instance 1,
    its instances drawn by C++ (suite/bbob.hpp, biobj and mixint; task 17,
    supplement 5, §3)."""
    fnum = int(name[-9:-7])                            # bbobbiobj07_n10_2D: 7
    family = "bbob-biobj-mixint" if name.startswith("bbobbiobjmixint") else "bbob-biobj"
    return _base(family, p.n_vars, 2, bbob_fnum=fnum, bbob_iid=1)


def spec(name: str, p, rotated=None) -> dict | None:
    """The spec of _core.Native for the registered problem `name` (its
    BenchProblem p; rotated: its rotated.Rotated, if it is one), None for a
    problem the suite does not have."""
    if rotated is not None:
        return _rotated(rotated)
    if _BBOB.fullmatch(name):
        return _bbob(name, p)
    d = _stage3a(name, p)
    if d is not None:
        return d
    n = p.n_vars
    if name in CLASSIC:
        return _base(CLASSIC[name], n, p.n_obj)
    if name == "ZDT6_D100":
        return _base("ZDT6", n, 2)
    if _ZDT.fullmatch(name):
        base = name.replace("shift", "")
        if not name.startswith("shift"):
            return _base(base, n, 2)
        c = _shift.zdt_centres(base, n)
        if base == "ZDT4":
            return _base(base, n, 2, map="shift", first=1, centre=_array(c))
        return _base(base, n, 2, map="fold", first=1, centre=_array(c),
                     spread=_array(np.maximum(c, 1.0 - c)))
    m = _STEMS.fullmatch(name)
    if m is None:
        return None
    stem, M = m.group(1), int(m.group(m.lastindex))
    k = n - M + 1                                      # the distance block
    if stem in _MAF_WFG:
        return _base(_MAF_WFG[stem], n, M, k=M - 1)
    if stem.startswith(("DTLZ", "MaF", "ZCAT", "IDTLZ", "minusDTLZ", "WFG")):
        d = _base(stem, n, M)
        # MaF8's and MaF9's polygons and MaF14's and MaF15's linkage: drawn by
        # C++ (suite/maf.hpp; task 17, supplement 5, §3)
        if stem in ("MaF14", "MaF15"):
            from . import maf
            d.update(parts=_array(maf.lsmop_parts(M)))
        return d
    if stem == "wideDTLZ3":
        return _base("DTLZ3", n, M)
    if stem in ("shiftDTLZ1", "shiftDTLZ2", "shiftDTLZ3", "shiftDTLZ4"):
        return _base(stem[5:], n, M, map="cyclic", first=M - 1,
                     centre=_array(_dv.shift_centres(k)))
    if stem in ("shiftDTLZ6", "shiftDTLZ7"):
        c = _dv.shift_centres(k)
        return _base(stem[5:], n, M, map="fold", first=M - 1, centre=_array(c),
                     spread=_array(np.maximum(c, 1.0 - c)))
    if stem.startswith("SDTLZ"):
        return _base(stem, n, M, scale=_array(_dv._scale(M, stem)))
    if stem == "C1-DTLZ3":
        return _base(stem, n, M, r2=_cd.C1_DTLZ3_R[M] ** 2)
    if stem == "C2-DTLZ2":
        return _base(stem, n, M, r2=_cd.C2_DTLZ2_R[M] ** 2, centre_m=float(1.0 / np.sqrt(M)))
    if stem == "C2-ConvexDTLZ2":
        return _base(stem, n, M, r2=_cd.C2_CONVEX_R[M] ** 2)
    return _base(stem, n, M)                           # C1-DTLZ1, C3-DTLZ1, C3-DTLZ4


class Evaluate:
    """The evaluate of a problem of the suite: the objectives of one point, as
    a list. `native` is the _core.Native, which minimize and a campaign hand to
    the core."""

    __slots__ = ("native",)

    def __init__(self, native):
        self.native = native

    def __call__(self, x):
        return self.native.evaluate(x)


class Constraints:
    """The constraints of a problem of the suite: of one point, as a list."""

    __slots__ = ("native",)

    def __init__(self, native):
        self.native = native

    def __call__(self, x):
        return self.native.constraints(x)


def attach(problems: dict, rotated: dict) -> int:
    """Give every problem of the suite's families among `problems` (name ->
    BenchProblem) its C++ evaluation, the NumPy one kept in p.reference; the
    problems that have it already are left alone. Nothing without the
    extension. Returns how many were attached."""
    try:
        from .. import _core
    except ImportError:
        return 0
    Native = getattr(_core, "Native", None)
    if Native is None:                                 # an extension built before the suite
        return 0
    count = 0
    for name, p in problems.items():
        if p.native is not None:
            continue
        s = spec(name, p, rotated.get(name))
        if s is None:
            continue
        obj = Native(s)
        if obj.n_con != (len(p.constraints([lo for lo, _ in p.bounds])) if p.has_cons else 0):
            raise RuntimeError(f"{name}: the suite has {obj.n_con} constraints")
        p.reference = {"evaluate": p.evaluate, "evaluate_batch": p.evaluate_batch,
                       "constraints": p.constraints, "constraints_batch": p.constraints_batch,
                       "evaluate_with_constraints": p.evaluate_with_constraints}
        p.native = obj
        if obj.has_pareto_set and callable(p.pareto_set):   # the multimodal ones, the polygons
            p.reference["pareto_set"] = p.pareto_set
            p.pareto_set = obj.pareto_set
        p.evaluate = Evaluate(obj)
        p.evaluate_batch = obj.evaluate_batch
        if p.has_cons:
            p.constraints = Constraints(obj)
            p.constraints_batch = obj.constraints_batch
            p.evaluate_with_constraints = obj.evaluate_with_constraints
        count += 1
    return count

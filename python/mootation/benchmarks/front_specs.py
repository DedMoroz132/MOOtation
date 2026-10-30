# SPDX-License-Identifier: Apache-2.0
"""Which sampler builds a problem's reference front — data for the reference
builders in C++ (task 17, §2.1; include/mootation/refs/fronts.hpp), written
into the registry's export (run/registry_export.py).

front_spec(name) -> {"kind": the sampler, its numbers, "seed": the CRC-32 of
the name of the problem whose front it is (a shifted, noisy or rotated copy
takes its base's: registry.FRAME_OF), "whole": true where the registry takes
the sample as it is (no _exactly_n)}, or None where the C++ builders do not
have the sampler yet (the reference stays Python's).

The kinds and the Python functions they stand for are listed in fronts.hpp;
the comparison of the two is TASK17's scripts/stage5/refs_check.py.
"""

from __future__ import annotations

import zlib


def _is_exactly_n(p) -> bool:
    """Whether the registry thins p's sample to n (_exactly_n): its Python
    sampler's, kept in p.reference once the C++ one stands in for it
    (registry._cxx_references)."""
    pf = p.pareto_front
    if getattr(pf, "__qualname__", "").startswith("_cxx_sampler"):
        pf = p.reference["pareto_front"]
    return getattr(pf, "__qualname__", "").startswith("_exactly_n")


def _stem_m(name: str, p) -> tuple:
    return name.split("_")[0], int(p.n_obj)


def _kind(name: str, p) -> dict | None:
    """The sampler of the problem `name`, whose pareto_front is its own."""
    from . import binary as _bin
    from . import dtlz as _d
    from . import fronts_full as _ff
    from .registry import FULL_FRONT
    stem, M = _stem_m(name, p)
    if name in FULL_FRONT:
        return {"kind": "full", "file": _ff._path(stem, M).name}
    if stem == "DTLZ1":
        return {"kind": "dtlz1", "M": M}
    if stem == "shiftDTLZ1":
        return {"kind": "dtlz1", "M": M}
    if stem in ("DTLZ2", "DTLZ3", "DTLZ4", "wideDTLZ3", "shiftDTLZ2", "shiftDTLZ3", "shiftDTLZ4"):
        return {"kind": "dtlz2", "M": M}
    if stem in ("DTLZ5", "DTLZ6"):
        return {"kind": "dtlz5", "M": M}
    if stem == "DTLZ7":
        return {"kind": "dtlz7", "M": M}
    if stem in ("IDTLZ1", "IDTLZ2"):
        return {"kind": stem.lower(), "M": M}
    if stem == "minusDTLZ2":
        return {"kind": "minus_dtlz2", "M": M, "k": int(_d.DTLZ_K["DTLZ2"])}
    if stem in ("SDTLZ1", "SDTLZ2"):
        return {"kind": stem.lower(), "M": M}             # the factors: C++'s
    wfg = {"WFG1": "wfg1", "WFG2": "wfg2", "WFG3": "wfg3", "MaF10": "wfg1", "MaF11": "wfg2", "MaF12": "wfg_concave",
           **{f"WFG{i}": "wfg_concave" for i in range(4, 10)}}
    if stem in wfg:
        return {"kind": wfg[stem], "M": M}
    maf = {"MaF1": "maf1", "MaF2": "maf2", "MaF3": "maf3", "MaF4": "maf4", "MaF5": "maf5", "MaF6": "dtlz5",
           "MaF7": "dtlz7", "MaF8": "maf8", "MaF9": "maf9", "MaF13": "maf13", "MaF14": "lsmop14",
           "MaF15": "lsmop15", "Polygon": "polygon", "IPolygon": "ipolygon"}
    if stem in maf:
        return {"kind": maf[stem], "M": M}
    from .registry import DASCMOP
    if name in DASCMOP:
        i, t = DASCMOP[name]
        return {"kind": "dascmop", "i": int(i), "triplet": [float(v) for v in t]}
    if stem.startswith("MW") and stem[2:].isdigit():
        return {"kind": "mw", "i": int(stem[2:]), "M": M}
    from . import mmo as _mmo
    if name in _mmo.NAMES:
        return {"kind": "mmo", "name": name}
    if stem.startswith("ZCAT") and stem[4:].isdigit():
        return {"kind": "zcat", "i": int(stem[4:]), "M": M}
    if stem.startswith("DOC") and stem[3:].isdigit():
        return {"kind": "doc", "i": int(stem[3:])}
    if stem.startswith("LIR-CMOP") and stem[len("LIR-CMOP"):].isdigit():
        return {"kind": "lircmop", "i": int(stem[len("LIR-CMOP"):])}
    cdtlz = {"C1-DTLZ1": "c1_dtlz1", "C1-DTLZ3": "c1_dtlz3", "C2-DTLZ2": "c2_dtlz2",
             "C2-ConvexDTLZ2": "c2_convex", "C3-DTLZ1": "c3_dtlz1", "C3-DTLZ4": "c3_dtlz4"}
    if stem in cdtlz:
        return {"kind": cdtlz[stem], "M": M}
    if name in ("ZDT1", "ZDT4", "catZDT1"):
        return {"kind": "zdt1"}
    if name in ("ZDT2", "ZDT3", "ZDT6"):
        return {"kind": name.lower()}
    if name == "ZDT6_D100":
        return {"kind": "zdt6"}
    if name in ("BT1", "BT2", "BT3", "BT4", "BT6", "BT7", "BT8"):
        return {"kind": "bt_concave"}
    if name in ("BT5", "BT9"):
        return {"kind": name.lower()}
    if name in ("MOP1", "MOP5"):
        return {"kind": "mop1"}
    if name in ("MOP2", "MOP3", "MOP4", "MOP6", "MOP7"):
        return {"kind": name.lower()}
    if name in ("SCH", "FON"):
        return {"kind": name.lower()}
    if name in ("SPH_2D", "SPH_3D"):
        return {"kind": "sph", "M": M}
    if name in ("POL", "KUR"):
        return {"kind": "classic_grid", "name": name}
    if name == "QV":
        return {"kind": "qv"}
    from . import rwmop as _rw
    if name.startswith("RCM") and (_rw.DATA / f"{name}.dat").is_file():
        return {"kind": "text", "file": f"_rwmop/{name}.dat"}
    for kind, n, k in _bin.SIZES:
        if _bin.name(kind, n, k) == name:
            return {"kind": "binary", "which": kind, "n": int(n), "k": int(k)}
    return None


def front_spec(name: str) -> dict | None:
    """The front's sampler of the registry problem `name`, or None."""
    from .registry import FAILING, FRAME_OF, PROBLEMS
    p = PROBLEMS[name]
    if not callable(p.pareto_front) or name in FAILING:
        return None                                   # a failing copy: its own sample (not yet)
    src = name
    while src in FRAME_OF:                            # a noisy copy of a rotated problem: two steps
        src = FRAME_OF[src]
    q = PROBLEMS[src]
    spec = _kind(src, q)
    if spec is None:
        return None
    if _is_exactly_n(q):
        spec["seed"] = zlib.crc32(src.encode())
    else:
        spec["whole"] = True
    spec["of"] = src
    return spec


def _set_kind(name: str, p) -> dict | None:
    """The Pareto-set sampler of the problem `name` (its own, not a copy's)."""
    stem, M = _stem_m(name, p)
    nv = int(p.n_vars)
    half = {"M": M, "n_vars": nv}
    if stem in ("DTLZ1", "DTLZ2", "DTLZ3", "DTLZ4", "IDTLZ1", "IDTLZ2", "SDTLZ1", "SDTLZ2",
                "wideDTLZ3") or (stem == "DTLZ5" and M <= 3):
        return {"kind": "dtlz", **half, "rest": 0.5}
    if stem == "DTLZ6" and M <= 3:
        return {"kind": "dtlz", **half, "rest": 0.0}
    if stem == "DTLZ7":
        return {"kind": "dtlz7", **half, "rest": 0.0}
    if stem == "minusDTLZ2":
        return {"kind": "minus_dtlz", **half}
    if stem in ("C1-DTLZ1", "C1-DTLZ3"):
        return {"kind": "dtlz", **half, "rest": 0.5}
    if stem == "C2-DTLZ2":
        return {"kind": "c2_dtlz2", **half}
    from . import mmo as _mmo
    from .registry import ROTATED
    if name in _mmo.NAMES:
        return {"kind": "mmo", "name": name, "n_vars": nv}
    if stem in ("MaF14", "MaF15"):
        return {"kind": "lsmop", "M": M, "which": int(stem[3:])}
    r = ROTATED.get(name)
    if r is not None and r.kind in ("dtlz", "maf", "zdt", "wfg", "polygon"):
        return {"kind": "rotated", "family": r.family, "M": M, "n_vars": nv}
    if stem == "Polygon":
        return {"kind": "polygon", "M": M, "n_vars": nv}
    if stem == "IPolygon":
        return {"kind": "ipolygon", "M": M}
    for pre in ("ZCAT", "rZCAT"):
        if stem.startswith(pre) and stem[len(pre):].isdigit():
            return {"kind": "zcat", "i": int(stem[len(pre):]), "M": M}
    if stem in ("MW1", "MW2", "MW4", "MW6", "MW8", "MW14"):
        return {"kind": "mw", "i": int(stem[2:]), "M": M}
    # the shifted problems' optimum: C++'s (suite/spec_numbers.hpp)
    if stem in ("shiftDTLZ1", "shiftDTLZ2", "shiftDTLZ3", "shiftDTLZ4") or (stem == "shiftDTLZ6" and M <= 3):
        return {"kind": "dtlz", **half, "rest": "shift"}
    if stem == "shiftDTLZ7":
        return {"kind": "dtlz7", **half, "rest": "shift"}
    if name in ("ZDT1", "ZDT2", "ZDT3", "ZDT4", "ZDT6"):
        return {"kind": "zdt", "base": name, "rest": 0.0}
    if name.startswith("shiftZDT"):
        base = name[len("shift"):]
        return {"kind": "zdt", "base": base, "rest": "shift"}
    if name in ("SCH", "FON"):
        return {"kind": name.lower()}
    if name in ("SPH_2D", "SPH_3D"):
        return {"kind": "sph", "M": M}
    return None


def set_spec(name: str) -> dict | None:
    """The Pareto-set sampler of the registry problem `name` (a noisy copy:
    its base's), or None where it has none or C++ does not build it yet."""
    from .noisy import base_of, kind_of
    from .registry import FAILING, PROBLEMS
    p = PROBLEMS[name]
    if not callable(p.pareto_set) or name in FAILING:
        return None                                   # a failing copy: its base's set less the region
    src = base_of(name) if kind_of(name) is not None else name
    spec = _set_kind(src, PROBLEMS[src])
    if spec is not None:
        spec["of"] = src
    return spec

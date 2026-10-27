# SPDX-License-Identifier: Apache-2.0
"""The problem registry as data, for the campaign engine in C++ (task 17,
stage 5, step 4).

    python -m mootation.run.registry_export [--out FILE]

What the engine needs of a problem and cannot compute without Python yet: its
sizes and box, its default population and generations, its version, what it
modifies (a rotation, a shift, noise), whether its Pareto set has several
parts, which references it has, and — for a problem of the C++ suite — the
spec its C++ evaluation is built from (benchmarks/native.spec: the numbers
Python computes for it, centres, constants, vertices; what the suite draws
itself, the rotations and the bbob instances, stays out). And the flags that
send a problem to the Python engine: an evaluator of the run's own (noise,
uninformative problems), typed variables, a distance to the Pareto set
(g_min, g_median), which the suite does not compute yet. A failing copy of a
problem of the suite carries its base's spec and its failure region
(failing_region), and a noisy copy its base's spec and its noise (noise),
which the engine evaluates itself.

WHERE. Beside the cache of references (run/refs.py), under the benchmark
code's stamp (campaign._code_stamp): <refs root>/registry/<stamp>.json. The
same code writes the same file on every machine; a change of the benchmark
code is a new stamp and a new file.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

FORMAT = 1


def registry_path() -> Path:
    from .campaign import _code_stamp
    from .refs import refs_root
    return refs_root() / "registry" / f"{_code_stamp()}.json"


def _floats(v):
    import numpy as np
    return [float(x) for x in np.asarray(v, float).ravel()]


def _spec_json(s: dict | None):
    """A native spec with its arrays as lists of floats (flat, or a list of
    flat ones for matrices and polygons), as suite_module.spec_of reads them."""
    if s is None:
        return None
    out = {}
    for k, v in s.items():
        if k in ("matrices", "polygons"):
            out[k] = [_floats(m) for m in v]
        elif k == "rotation":
            out[k] = None if v is None else {"family": v["family"], "M": int(v["M"]),
                                             "instance": int(v["instance"]), "rotate": bool(v["rotate"]),
                                             "condition": bool(v["condition"]),
                                             "sizes": [int(x) for x in v["sizes"]]}
        elif k == "bbob":
            if v:
                raise ValueError("a spec handing over bbob functions is not exported")
        elif isinstance(v, (str, bool)) or v is None:
            out[k] = v
        elif isinstance(v, int):
            out[k] = int(v)
        elif isinstance(v, float):
            out[k] = float(v)
        else:
            out[k] = _floats(v)
    return out


def entry(name: str) -> dict:
    """What the engine reads of the registry problem `name`."""
    from ..benchmarks import native as N
    from ..families import family
    from ..benchmarks import registry as R
    from ..benchmarks.properties import modification_of, properties
    from ..benchmarks.noisy import kind_of as noise_kind
    from . import refs
    p = R.PROBLEMS[name]
    spec = None
    if p.native is not None:          # the spec it was built from, not the one it resolved
        spec = _spec_json(N.spec(name, p, R.ROTATED.get(name)))
    props = properties(name) or {}
    return {
        "name": name, "family": family(name), "n_var": int(p.n_vars), "n_obj": int(p.n_obj),
        "n_con": int(p.native.n_con) if p.native is not None else None,
        "has_cons": bool(p.has_cons), "n_eq": int(p.n_eq),
        "bounds": [[float(lo), float(hi)] for lo, hi in p.bounds],
        "pop_size": int(p.pop_size), "n_gen": int(p.n_gen),
        "problem_version": R.problem_version(name),
        "modification": modification_of(name),
        "multimodal_ps": bool(props.get("multimodal_ps")),
        "has_pareto_front": callable(p.pareto_front), "has_pareto_set": callable(p.pareto_set),
        "front_exact": bool(p.front_exact), "shared_reference": refs.shared(name),
        "dense_reference": R.has_dense_reference(name),
        "cyclic_vars": [int(i) for i in p.cyclic_vars],
        "stochastic": bool(getattr(p, "stochastic", False)),
        # what sends a run to the Python engine
        "own_evaluator": p.make_evaluator is not None or p.true_objectives is not None
                         or noise_kind(name) is not None,
        "typed": p.var_types is not None and any(t != "real" for t in p.var_types),
        "failing": p.failing is not None,
        "failing_region": _failing_region(name),
        "noise": _noise(name),
        "uninformative": _uninformative(name),
        "distance": p.distance_fn is not None,
        "spec": spec,
    }


def _uninformative(name: str) -> str | None:
    """An uninformative problem's kind (benchmarks/uninformative.py: f1, f2b,
    f3b, f4b, f5), whose run evaluator the engine makes from the run's seed."""
    from ..benchmarks import uninformative as U
    for problem in U.PROBLEMS:
        if any(U.name(n, problem) == name for n in U.SIZES):
            return problem
    return None


def _noise(name: str) -> dict | None:
    """A noisy copy of a problem of the C++ suite (benchmarks/noisy.py): the
    base's spec and its evaluator's numbers — the kind, sigma, the scale
    sigma (nadir - ideal) per objective and the ideal point (kind h) as the
    run's evaluator holds them; the cell's width comes from the campaign
    (noise_cell) and the box."""
    from ..benchmarks import native as N
    from ..benchmarks import noisy
    from ..benchmarks import registry as R
    kind = noisy.kind_of(name)
    if kind is None or name not in R.PROBLEMS:
        return None
    base = noisy.base_of(name)
    b = R.PROBLEMS.get(base)
    if b is None or b.native is None:
        return None
    ev = R.PROBLEMS[name].make_evaluator(0)
    return {"of": base, "kind": kind, "sigma": float(ev.sigma), "scale": [float(v) for v in ev.scale],
            "ideal": None if ev.ideal is None else [float(v) for v in ev.ideal],
            "spec": _spec_json(N.spec(base, b, R.ROTATED.get(base)))}


def _failing_region(name: str) -> dict | None:
    """A failing copy of a problem of the C++ suite (benchmarks/failing.py):
    the base's spec and the failure region in the box's normalised units —
    "b": the band within t of the faces; "r" and "n": the balls of one radius
    about the centres (the random balls, the tube about part of the Pareto-set
    sample)."""
    from ..benchmarks import native as N
    from ..benchmarks import registry as R
    if name not in R.FAILING:
        return None
    base = R.FAILING[name][0]
    b = R.PROBLEMS[base]
    if b.native is None:
        return None
    region = R.failing_region(name)
    out = {"of": base, "kind": region.kind, "spec": _spec_json(N.spec(base, b, R.ROTATED.get(base)))}
    if region.kind == "b":
        out["t"] = float(region.t)
    else:
        out["radius"] = float(region.radius)
        out["centres"] = [_floats(c) for c in region.centres]
    return out


def names() -> dict:
    """The lists a config is checked against and a campaign's jobs are made
    from that live in Python modules (run/config.py's validate, campaign.py):
    the baselines, the scalarization methods, the designs of a first
    population, the knobs, the text and list knobs, the metric names, the
    algorithms whose run depends on the budget."""
    from .. import designs
    from ..scalar import NAMES, OTHERS
    from .baselines import BASELINES
    from .budget import BUDGET_DEPENDENT
    from .knobs import LIST_KNOBS, knob_names, text_knobs
    from .metric_names import METRIC_NAMES
    return {"baselines": list(BASELINES), "scalar_names": list(NAMES), "scalar_others": list(OTHERS),
            "design_kinds": list(designs.KINDS), "knob_names": list(knob_names()),
            "text_knobs": [[k, list(v)] for k, v in text_knobs().items()], "list_knobs": list(LIST_KNOBS),
            "metric_names": list(METRIC_NAMES), "budget_dependent": sorted(BUDGET_DEPENDENT)}


def export(path: Path | None = None) -> Path:
    """Write the registry's file (atomically); its path."""
    from .. import __version__
    from ..benchmarks import registry as R
    from .campaign import _code_stamp
    path = path or registry_path()
    doc = {"format": FORMAT, "code_stamp": _code_stamp(), "reference_version": R.REFERENCE_VERSION,
           "mootation": __version__, "names": names(),
           "problems": [entry(n) for n in sorted(R.PROBLEMS)]}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(doc, separators=(",", ":")), encoding="utf-8")
    os.replace(tmp, path)
    return path


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m mootation.run.registry_export",
                                 description="Write the problem registry as data for the C++ engine.")
    ap.add_argument("--out", type=Path, default=None, help="the file (default: beside the references)")
    args = ap.parse_args(argv)
    path = export(args.out)
    print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())

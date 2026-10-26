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
uninformative problems), typed variables, a failing copy, a distance to the
Pareto set (g_min, g_median), which the suite does not compute yet.

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
        "name": name, "n_var": int(p.n_vars), "n_obj": int(p.n_obj),
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
        "distance": p.distance_fn is not None,
        "spec": spec,
    }


def export(path: Path | None = None) -> Path:
    """Write the registry's file (atomically); its path."""
    from .. import __version__
    from ..benchmarks import registry as R
    from .campaign import _code_stamp
    path = path or registry_path()
    doc = {"format": FORMAT, "code_stamp": _code_stamp(), "reference_version": R.REFERENCE_VERSION,
           "mootation": __version__, "problems": [entry(n) for n in sorted(R.PROBLEMS)]}
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

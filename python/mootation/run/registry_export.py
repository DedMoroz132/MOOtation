# SPDX-License-Identifier: Apache-2.0
"""The problem registry as data, for the campaign engine in C++ (task 17,
stage 5, step 4).

    python -m mootation.run.registry_export [--out FILE | --repo] [--check]

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

WHERE. The repository keeps one: data/registry.json (--repo), which the build
puts beside the engine and the installation in share/mootation, so that the
engine and `mootation refs build` run without Python (task 17, section 2.2;
docs/running.md has the order they look for it in). A CI test (--check) says
when it is behind the code. Without --out or --repo: beside the cache of
references (run/refs.py), under the benchmark code's stamp
(campaign._code_stamp) and this module's hash:
<refs root>/registry/<stamp>-<hash>.json. The same code writes the same file
on every machine; a change of the benchmark code or of this module is a new
file. The text is one problem per line, so that a change reads in a diff.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

FORMAT = 1


def registry_path() -> Path:
    """Under the benchmark code's stamp and this file's own hash: a change of
    either (a field more, as the engine learns more) is a new file."""
    import hashlib
    from .campaign import _code_stamp
    from .refs import refs_root
    mine = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:8]
    return refs_root() / "registry" / f"{_code_stamp()}-{mine}.json"


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
    from ..benchmarks.front_specs import front_spec, set_spec
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
        # RE91's run evaluator (re.RE91Evaluator): its draws from the run's
        # seed, which the engine makes as Python does
        "seeded_stream": "RE91" if name == "RE91" else None,
        **_layout(p),
        "distance": p.distance_fn is not None,
        "spec": spec,
        # the reference builders in C++ (refs/): the frame and the samplers
        "ideal": _frame(name)[0],
        "nadir": _frame(name)[1],
        "front": front_spec(name),
        "set": set_spec(name),
        "frame_files": _frame_files(name),
    }


def _frame_files(name: str) -> dict | None:
    """Where an RE or CRE problem's frame is (re.frame): its authors' ideal
    and nadir files, beside benchmarks/ (a CRE problem's those of its RE
    counterpart, without the last objective); C++ reads them (task 17,
    addendum 7, section 2.1)."""
    from ..benchmarks import re as _re
    if name in _re.RE:
        return {"dir": _re._data(name).relative_to(_re.DATA.parent).as_posix(), "name": name, "drop_last": False}
    if name in _re.CRE:
        base = _re.CRE[name][0]
        return {"dir": _re._data(base).relative_to(_re.DATA.parent).as_posix(), "name": base, "drop_last": True}
    return None


def _frame(name: str) -> tuple:
    """(ideal, nadir) as registry.get gives them (the frame read off the
    reference front where it is not exact), lists of floats; Nones without."""
    import warnings
    from ..benchmarks import registry as R
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")              # the degenerate fronts' note
        p = R.get(name)
    if p.ideal is None or p.nadir is None:
        return None, None
    return [float(v) for v in p.ideal], [float(v) for v in p.nadir]


def _layout(p) -> dict:
    """Typed variables (var_types.Layout): every variable's kind (real, int,
    discrete, bin, cat) and a categorical one's number of labels; the core's
    real genome's kinds (0 real, 1 integer, 2 discrete), its box (an
    integer's widened by just under a half on each side, a discrete one's
    from its least value to its largest), the discrete ones' values, sorted,
    and the integer genome's levels (gene_levels); for bits alone with an
    exact front, the front cover_gen looks for (run/campaign.py)."""
    from ..var_types import Layout
    if p.var_types is None:
        return {}
    layout = Layout(p.var_types, list(p.bounds))
    if layout.trivial():
        return {}
    out = {"user_kinds": list(layout.kinds),
           "cat_levels": [len(v) if k == "cat" else 0 for k, v in zip(layout.kinds, layout.values)],
           "var_kinds": [int(k) for k in layout.var_kinds()],
           "core_bounds": [[float(lo), float(hi)] for lo, hi in layout.core_bounds()],
           "discrete_values": [[float(v) for v in vs] for vs in layout.discrete_values()],
           "gene_levels": [int(v) for v in layout.gene_levels()]}
    if all(k == "bin" for k in layout.kinds) and p.front_exact and callable(p.pareto_front):
        import numpy as np
        out["cover_front"] = [[float(v) for v in row] for row in np.asarray(p.pareto_front(1000))]
    return out


def _uninformative(name: str) -> str | None:
    """An uninformative problem's kind (benchmarks/uninformative.py: f1, f2b,
    f3b, f4b, f5), whose run evaluator the engine makes from the run's seed."""
    from ..benchmarks import uninformative as U
    for problem in U.PROBLEMS:
        if any(U.name(n, problem) == name for n in U.SIZES):
            return problem
    return None


def _noise(name: str) -> dict | None:
    """A noisy copy of a problem of the C++ suite (benchmarks/noisy.py): its
    base and the base's spec, the kind. Sigma (from the name), the scale
    sigma (nadir - ideal) and the ideal point come from the base's frame in
    C++ (campaign/registry.hpp, derive_numbers; task 17, addendum 7, section
    2.2); the cell's width from the campaign (noise_cell) and the box."""
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
    return {"of": base, "kind": kind, "spec": _spec_json(N.spec(base, b, R.ROTATED.get(base)))}


def _failing_region(name: str) -> dict | None:
    """A failing copy of a problem of the C++ suite (benchmarks/failing.py):
    its base, the base's spec and the region's kind."""
    from ..benchmarks import native as N
    from ..benchmarks import registry as R
    if name not in R.FAILING:
        return None
    base = R.FAILING[name][0]
    b = R.PROBLEMS[base]
    if b.native is None:
        return None
    region = R.failing_region(name)
    # the band's t, the balls and the window: C++'s, from the name and the
    # base (campaign/registry.hpp, derive_numbers; task 17, addendum 7, section 2.3)
    return {"of": base, "kind": region.kind, "spec": _spec_json(N.spec(base, b, R.ROTATED.get(base)))}


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


REPO_FILE = Path(__file__).resolve().parents[3] / "data" / "registry.json"
REGENERATE = "python -m mootation.run.registry_export --repo"


FRAME_TABLE = Path(__file__).resolve().parents[1] / "benchmarks" / "_refframe_cache.json"
FRAME_WORKERS = 4


def _cxx_frames(head: dict, entries: list) -> None:
    """Every entry's ideal and nadir computed by C++ (_core.refs_frames:
    campaign/refs_build.hpp, frame_of; task 17, addendum 7, section 2.1),
    where it gives one; the others keep get()'s."""
    import tempfile
    from .. import _core
    from ..benchmarks import registry as R
    with tempfile.TemporaryDirectory(prefix="mootation-frames-") as td:
        tmp = Path(td) / "registry.json"
        tmp.write_text(json.dumps({**head, "problems": entries}), encoding="utf-8")
        got = _core.refs_frames(str(tmp), FRAME_WORKERS, str(Path(R.__file__).resolve().parent))
    for e in entries:
        if e["name"] in got:
            ideal, nadir = got[e["name"]]
            e["ideal"], e["nadir"] = [float(v) for v in ideal], [float(v) for v in nadir]


def text(cxx_frames: bool = False) -> str:
    """The registry's file as text: the header on the first line, then a
    problem per line. cxx_frames: the frames computed by C++ (the
    repository's file); else get()'s, which reads them from
    _refframe_cache.json, made from that file."""
    from .. import __version__
    from ..benchmarks import registry as R
    from .campaign import _code_stamp

    def dump(v):
        return json.dumps(v, separators=(",", ":"))

    head = {"format": FORMAT, "code_stamp": _code_stamp(), "reference_version": R.REFERENCE_VERSION,
            "mootation": __version__, "names": names()}
    entries = [entry(n) for n in sorted(R.PROBLEMS)]
    if cxx_frames:
        _cxx_frames(head, entries)
    problems = ",\n".join(dump(e) for e in entries)
    return dump(head)[:-1] + ',\n"problems":[\n' + problems + "\n]}\n"


def frame_table(registry_text: str) -> str:
    """_refframe_cache.json made from the registry's file: {name: [ideal,
    nadir]} of every problem with a frame (registry.get reads it; task 17,
    addendum 7, section 2.1: one source of the numbers)."""
    doc = json.loads(registry_text)
    table = {e["name"]: [e["ideal"], e["nadir"]] for e in doc["problems"]
             if e.get("ideal") is not None and e.get("nadir") is not None}
    return json.dumps(table, sort_keys=True)            # its format: one line, sorted keys (fronts_full.py)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(content)
    os.replace(tmp, path)


def export(path: Path | None = None, cxx_frames: bool = False) -> Path:
    """Write the registry's file (atomically); its path. With cxx_frames
    (the repository's file), _refframe_cache.json from it as well."""
    path = path or registry_path()
    content = text(cxx_frames)
    _write(path, content)
    if cxx_frames:
        _write(FRAME_TABLE, frame_table(content))
    return path


def check(path: Path) -> str | None:
    """None when the file is what the code writes now (the frames computed
    by C++ anew) and _refframe_cache.json is made from it; else what differs."""
    if not path.is_file():
        return f"{path} is missing"
    content = path.read_text(encoding="utf-8")
    if not FRAME_TABLE.is_file() or FRAME_TABLE.read_text(encoding="utf-8") != frame_table(content):
        return f"{FRAME_TABLE} is not made from {path}"
    old = content.splitlines()
    new = text(cxx_frames=True).splitlines()
    for i, (a, b) in enumerate(zip(old, new)):
        if a != b:
            what = "the header" if i == 0 else (json.loads(b.rstrip(",")).get("name", f"line {i + 1}")
                                               if b.startswith("{") else f"line {i + 1}")
            return f"{path} is behind the code: {what} differs"
    if len(old) != len(new):
        return f"{path} is behind the code: {len(old)} lines, the code writes {len(new)}"
    return None


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m mootation.run.registry_export",
                                 description="Write the problem registry as data for the C++ engine.")
    where = ap.add_mutually_exclusive_group()
    where.add_argument("--out", type=Path, default=None, help="the file (default: beside the references)")
    where.add_argument("--repo", action="store_true", help=f"the repository's {REPO_FILE.parent.name}/{REPO_FILE.name}")
    ap.add_argument("--check", action="store_true", help="only check that the file is what the code writes now")
    args = ap.parse_args(argv)
    path = REPO_FILE if args.repo else args.out
    if args.check:
        why = check(path or REPO_FILE)
        if why:
            print(f"{why}; write it anew with: {REGENERATE}", file=sys.stderr)
            return 1
        print(f"{path or REPO_FILE}: up to date")
        return 0
    print(export(path, cxx_frames=args.repo))
    return 0


if __name__ == "__main__":
    sys.exit(main())

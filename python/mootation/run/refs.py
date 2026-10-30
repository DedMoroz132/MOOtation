# SPDX-License-Identifier: Apache-2.0
"""The reference data of the registry's problems, built once per machine.

    mootation refs build [--problems A,B] [--workers N] [--n-ref 1000] [--rebuild]
    mootation refs build --extend --n-ref 5000 [--problems A,B]
    mootation refs check [--problems A,B] [--n-ref 1000]
    python -m mootation.run.refs ...          (the same, without the console script)

WHAT IS KEPT (task 12 of 2026-10-19, §3). For every problem of the registry
whose reference front is exact (BenchProblem.front_exact) or an approximation
fixed by the problem alone — a failing copy's (failing.py: the base's front,
or the base's values at the sample points outside the failed region) and
C2-ConvexDTLZ2's verified sample (task 13 of 2026-10-19, §5) — one file per
problem and reference size n_ref (the approximate fronts of RE, CRE and RWMOP,
and the pooled references, depend on a campaign and stay in its folder,
campaign.REFERENCES_DIR): the frame (ideal, nadir), the reference front of n_ref
points, the dense reference (registry.has_dense_reference: from four to ten
objectives), the Pareto-set sample of n_ref points, and the IGD+ floors of
cover.py at the problem's population — the greedy floor, DSS's and their
ratio, DSS's tax — with the problem's version (registry.problem_version), the
reference builder's (registry.REFERENCE_VERSION), the hash of the benchmark
code that built it, and the hash of the arrays (campaign.reference_hash). They
are the arrays a campaign built for itself before (campaign.reference_data),
so the same values.

WHERE. The user's cache, not the repository nor the installation:
%LOCALAPPDATA%\\mootation\\references\\v<REFERENCE_VERSION>\\ on Windows,
$XDG_CACHE_HOME/mootation/references/v<...>/ (else ~/.cache/...) elsewhere;
MOOTATION_REFS_DIR (or --refs-dir) puts it anywhere. <problem>/n<n_ref>.mrf.

WHO BUILDS. From reference version 5 (task 17, section 2.1) the C++ builders
(include/mootation/refs, campaign/refs_build.hpp; `mootation refs build`
without Python): an entry is one record of the store's format, .mrf. This
module hands them the registry's export of the problems asked for and reads
the entries back (_core.refs_load_entry); Python's samplers stay as the
cross-check (TASK17's refs_check.py). The .npz entries of before are read.
Every machine builds the same bytes of the arrays, so a cache can be copied
from one to another, and the hashes agree (campaign --merge checks them).

WHEN AN ENTRY IS OLD. When the problem's version or the builder's version is
not the one it was built with: `check` lists it and a campaign builds it
again. An entry built by other benchmark code but of the same versions is
listed by `check` as built by other code, and used: a change of a problem's
values or of its reference front raises its version (registry.PROBLEM_VERSION,
REFERENCE_VERSION), which is what a cache shared between commits and machines
can rely on.

A CAMPAIGN (campaign.run_campaign) checks before its first run that the
entries of all its problems are there and current, and builds the missing
ones itself, in parallel, before any run starts: no worker waits on another
for a reference. --no-build-refs stops instead, with the list.

EXTEND. `--extend --n-ref N` builds the entries of N points for problems that
have a smaller one. Where the builder's sample of n points is the first n of
its sample of N (the fronts taken in DSS order: DAS-CMOP, DOC, LIR-CMOP,
Polygon), the smaller sample stays a part of the larger and the entry says so
("extends"); elsewhere the larger is a sample of its own, and the entry says
that it was built anew.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ENV_DIR = "MOOTATION_REFS_DIR"


class MissingReferences(RuntimeError):
    """A campaign's problems whose entries are missing or old (--no-build-refs)."""

    def __init__(self, names, n_ref, st):
        self.names = list(names)
        lines = [f"  {n}: {st[n]}" for n in self.names]
        super().__init__(f"the cache of references ({store_dir()}) lacks {len(self.names)} "
                         f"entr{'y' if len(self.names) == 1 else 'ies'} at n_ref = {n_ref} "
                         f"(build them with: mootation refs build --n-ref {n_ref} --problems "
                         f"{','.join(self.names)}):\n" + "\n".join(lines))


def refs_root() -> Path:
    """The cache's root, before the builder's version."""
    if os.environ.get(ENV_DIR):
        return Path(os.environ[ENV_DIR])
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return base / "mootation" / "references"


def store_dir() -> Path:
    from ..benchmarks.registry import REFERENCE_VERSION
    return refs_root() / f"v{REFERENCE_VERSION}"


# the first reference version whose entries are records (.mrf), C++'s
RECORD_ENTRIES = 5


def entry_path(name: str, n_ref: int) -> Path:
    from ..benchmarks.registry import REFERENCE_VERSION
    ext = "mrf" if REFERENCE_VERSION >= RECORD_ENTRIES else "npz"
    return store_dir() / name / f"n{int(n_ref)}.{ext}"


def shared(name: str) -> bool:
    """Does `name` keep its reference in this cache: a registry problem with an
    exact reference front, a failing copy or C2-ConvexDTLZ2 (none without
    NumPy, which the registry needs). A failing copy by its kind, not by
    front_exact, which the copy sets only once its sample is tested."""
    try:
        from ..benchmarks.registry import PROBLEMS
    except ImportError:
        return False
    p = PROBLEMS.get(name)
    if p is None or not callable(p.pareto_front):
        return False
    return bool(p.front_exact) or p.failing is not None or name.startswith("C2-ConvexDTLZ2_")


def shared_problems() -> list:
    from ..benchmarks.registry import PROBLEMS
    return sorted(n for n in PROBLEMS if shared(n))


def load_entry(name: str, n_ref: int):
    """(arrays, meta) of the entry, or None where there is none or it cannot be
    read; arrays: ideal, nadir and those of ref, ref3, pset the problem has."""
    import numpy as np
    path = entry_path(name, n_ref)
    if not path.is_file():
        return None
    try:
        if path.suffix == ".mrf":
            from .. import _core
            arrays, meta = _core.refs_load_entry(str(path))
            return dict(arrays), json.loads(meta)
        with np.load(path, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files if k != "meta"}
            meta = json.loads(str(z["meta"]))
    except (OSError, ValueError, KeyError, RuntimeError):
        return None
    return arrays, meta


def status(name: str, meta) -> str:
    """"ok", "missing", or why the entry is old."""
    from ..benchmarks.registry import REFERENCE_VERSION, problem_version
    if meta is None:
        return "missing"
    why = []
    if meta.get("problem_version") != problem_version(name):
        why.append(f"problem version {meta.get('problem_version')} -> {problem_version(name)}")
    if meta.get("reference_version") != REFERENCE_VERSION:
        why.append(f"builder version {meta.get('reference_version')} -> {REFERENCE_VERSION}")
    return "old: " + ", ".join(why) if why else "ok"


def _registry_for(names) -> Path:
    """An export of the registry (registry_export) of `names` and of the bases
    of the failing copies among them, for the C++ builders: a file of its own,
    beside the cache, named by its content."""
    import hashlib
    from ..benchmarks import registry as R
    from . import registry_export as X
    want = set(names) | {R.FAILING[n][0] for n in names if n in R.FAILING}
    doc = {"format": X.FORMAT, "code_stamp": "", "reference_version": R.REFERENCE_VERSION,
           "mootation": "", "names": X.names(), "problems": [X.entry(n) for n in sorted(want)]}
    text = json.dumps(doc, separators=(",", ":"))
    path = refs_root() / "registry" / f"part-{hashlib.sha256(text.encode()).hexdigest()[:16]}.json"
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    return path


def _build_cxx(names, n_ref: int, workers: int, rebuild: bool, extend: bool, log=None) -> dict:
    """The entries of `names` built by the C++ builders; {name: what}."""
    from .. import __version__, _core
    from .campaign import _code_stamp
    data = Path(__file__).resolve().parent.parent / "benchmarks"
    return dict(_core.refs_build_cache(str(_registry_for(names)), list(names), int(n_ref), int(workers),
                                       bool(rebuild), bool(extend), str(data), _code_stamp(), __version__,
                                       log))


def build_entry(name: str, n_ref: int, extend: bool = False) -> dict:
    """Build the entry of `name` at n_ref (the C++ builders) and write it; its meta."""
    got = _build_cxx([name], n_ref, 1, True, extend)
    if got.get(name) != "built":
        raise RuntimeError(f"{name}: {got.get(name)}")
    return load_entry(name, n_ref)[1]


def floor_of(name: str, n_ref: int, pop: int):
    """(greedy, dss, rows, distinct) of the entry's floors at `pop`, or None."""
    got = load_entry(name, n_ref)
    if got is None or status(name, got[1]) != "ok":
        return None
    f = (got[1].get("floors") or {}).get(str(int(pop)))
    if f is None:
        return None
    return f["greedy"], f["dss"], got[1]["rows"], got[1]["distinct"]


def check(names, n_ref: int) -> dict:
    """{name: status} of the entries of `names` at n_ref."""
    out = {}
    for name in names:
        got = load_entry(name, n_ref)
        out[name] = status(name, None if got is None else got[1])
    return out


def build(names, n_ref: int = 1000, workers: int = 1, rebuild: bool = False,
          extend: bool = False, log=print) -> dict:
    """Build the entries of `names` at n_ref that are missing or old (all of
    them with rebuild or extend), `workers` threads of the C++ builders (the
    slow families first, refs_build.hpp's SLOW_FIRST). {name: "built" | "ok" |
    "failed: ..." | "not built: ..."}."""
    names = list(dict.fromkeys(names))
    if not names:
        return {}
    t0 = time.perf_counter()
    out = _build_cxx(names, n_ref, max(1, workers), rebuild, extend, log)
    failed = sum(v not in ("built", "ok") for v in out.values())
    built = sum(v == "built" for v in out.values())
    if built or failed:
        log(f"references: {built} built, {failed} failed, {time.perf_counter() - t0:.1f} s")
    return out


def cache_size() -> tuple:
    """(files, bytes) of this builder version's cache."""
    files = [f for f in store_dir().rglob("n*.*") if f.suffix in (".mrf", ".npz")]         if store_dir().is_dir() else []
    return len(files), sum(f.stat().st_size for f in files)


CONSTRAINT_SAMPLE = 10000             # the Sobol points the constraint properties read
CONSTRAINT_PROPS_VERSION = 1          # bump when constraint_properties() computes otherwise
# C-DTLZ's unconstrained problems (Jain & Deb 2014): the front without the
# constraints, which tells whether they move it
_UNCONSTRAINED = {"C1-DTLZ1": "DTLZ1", "C1-DTLZ3": "DTLZ3", "C2-DTLZ2": "DTLZ2",
                  "C3-DTLZ1": "DTLZ1", "C3-DTLZ4": "DTLZ4"}


def constraint_properties(name: str) -> dict | None:
    """What a constrained problem's constraints do (task 15, §5; after Tanabe &
    Oyama, CEC 2017), computed once and kept beside its reference
    (store_dir()/name/constraints.json); None for a problem without
    constraints.

      feasible_share   the share of 10 000 points of the box (scrambled Sobol,
                       designs.sobol_owen, seed 1) that meet every constraint
      violated_mean, violated_max
                       among the infeasible ones, how many constraints each
                       violates, on average and at most (None if none is)
      front            what the constraints do to the Pareto front, where the
                       problem has an unconstrained one (C-DTLZ): "same" — the
                       two fronts coincide; "part" — the constrained front is a
                       part of the other; "moved" — it is not on the other;
                       None where there is no unconstrained problem to compare.
                       Read from the two fronts' samples (3000 points each, in
                       the unconstrained front's range): a point lies on the
                       other front when it is no farther from that sample than
                       the sample's widest gap between neighbours.
      handling         does the problem test constraint handling (task 16,
                       §2.4): "tested", "untested" (an algorithm does no
                       worse without its handling), "unread" (no reading),
                       as set_constraint_handling() wrote it from a reading
                       of the user's; absent until then. Kept when the rest
                       is computed again.
    """
    import numpy as np
    from ..benchmarks.registry import PROBLEMS
    p = PROBLEMS[name]
    if not p.has_cons:
        return None
    path = store_dir() / name / "constraints.json"
    if path.is_file():
        try:
            got = json.loads(path.read_text(encoding="utf-8"))
            if got.get("version") == CONSTRAINT_PROPS_VERSION:
                return got
        except (OSError, ValueError):
            got = {}
    else:
        got = {}
    from ..designs import sobol_owen
    from ..var_types import Layout
    bounds = list(p.bounds)
    layout = Layout(p.var_types, bounds) if p.var_types is not None else None
    if layout is None or layout.trivial():
        lo = np.array([b[0] for b in bounds], float)
        hi = np.array([b[1] for b in bounds], float)
        X = lo + sobol_owen(CONSTRAINT_SAMPLE, len(lo), 1) * (hi - lo)
        G = np.asarray(p.constraints_batch(X), float)
    else:
        core = layout.core_bounds()
        levels = layout.gene_levels()
        U = sobol_owen(CONSTRAINT_SAMPLE, len(core) + len(levels), 1)
        rows = []
        for u in U:
            x = [lo + u[j] * (hi - lo) for j, (lo, hi) in enumerate(core)]
            x += [min(int(u[len(core) + g] * n), n - 1) for g, n in enumerate(levels)]
            rows.append(list(p.constraints(layout.to_user(x))))
        G = np.asarray(rows, float)
    violated = np.sum(G > 0.0, axis=1)
    bad = violated[violated > 0]
    out = {"version": CONSTRAINT_PROPS_VERSION, "sample": CONSTRAINT_SAMPLE,
           "feasible_share": float(np.mean(violated == 0)),
           "violated_mean": float(bad.mean()) if len(bad) else None,
           "violated_max": int(bad.max()) if len(bad) else None,
           "front": None}
    if "handling" in got:
        out["handling"] = got["handling"]
    stem, _, rest = name.partition("_")
    base = _UNCONSTRAINED.get(stem)
    if base is not None and f"{base}_{rest}" in PROBLEMS and callable(p.pareto_front):
        A = np.asarray(p.pareto_front(3000), float)
        B = np.asarray(PROBLEMS[f"{base}_{rest}"].pareto_front(3000), float)
        span = np.maximum(B.max(axis=0) - B.min(axis=0), 1e-12)

        def nearest(P, Q, skip_self=False):   # each row of P: its distance to the set Q, in Q's frame
            D = np.sqrt((((P[:, None, :] - Q[None, :, :]) / span) ** 2).sum(axis=2))
            if skip_self:
                np.fill_diagonal(D, np.inf)
            return D.min(axis=1)
        tol = float(nearest(B, B, skip_self=True).max())   # the sample's widest gap
        on_base = nearest(A, B) <= tol
        covers = nearest(B, A) <= tol
        out["front"] = ("moved" if not on_base.all() else "same" if covers.all() else "part")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


HANDLING = ("tested", "untested", "unread")


def set_constraint_handling(table: dict) -> int:
    """Write a reading of whether constrained problems test constraint
    handling into their constraints.json: table {"tested": [names],
    "untested": [...], "unread": [...]} and a "source" saying where it comes
    from (a run campaign's reading, kept by the user: mootation refs handling
    FILE). Returns the problems written."""
    from ..benchmarks.registry import PROBLEMS
    bad = sorted(set(table) - set(HANDLING) - {"source"})
    if bad:
        raise ValueError(f"handling: {', '.join(bad)}? the keys are {', '.join(HANDLING)}")
    names = [(n, k) for k in HANDLING for n in table.get(k, ())]
    unknown = [n for n, _ in names if n not in PROBLEMS or not PROBLEMS[n].has_cons]
    if unknown:
        raise ValueError(f"not constrained problems of the registry: {', '.join(unknown[:10])}")
    for name, kind in names:
        props = dict(constraint_properties(name), handling=kind)
        (store_dir() / name / "constraints.json").write_text(json.dumps(props, indent=1), encoding="utf-8")
    return len(names)


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(prog="mootation refs",
                                 description="the registry's reference data, built once per machine")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("handling", help="write a reading of constraint handling (a JSON file "
                       "{\"tested\": [...], \"untested\": [...], \"unread\": [...]}) into "
                       "constraints.json")
    s.add_argument("file")
    s.add_argument("--refs-dir", default="", help=f"the cache's root (or {ENV_DIR})")
    for cmd in ("build", "check"):
        s = sub.add_parser(cmd)
        s.add_argument("--problems", default="", help="comma-separated names (default: all "
                       "with an exact reference front)")
        s.add_argument("--n-ref", type=int, default=1000, help="reference size (default 1000)")
        s.add_argument("--refs-dir", default="", help=f"the cache's root (or {ENV_DIR})")
        if cmd == "build":
            s.add_argument("--workers", type=int, default=1)
            s.add_argument("--rebuild", action="store_true", help="build them again")
            s.add_argument("--extend", action="store_true",
                           help="entries of --n-ref points beside smaller ones (see the module)")
    args = ap.parse_args(argv)
    if args.refs_dir:
        os.environ[ENV_DIR] = args.refs_dir           # the workers inherit it
    if args.cmd == "handling":
        n = set_constraint_handling(json.loads(Path(args.file).read_text(encoding="utf-8")))
        print(f"{n} problems written ({store_dir()})")
        return 0
    names = [n for n in args.problems.split(",") if n] or shared_problems()
    unknown = [n for n in names if not shared(n)]
    if unknown:
        print(f"no reference of their own in the cache (no exact front, or not in the "
              f"registry): {', '.join(unknown)}", file=sys.stderr)
        return 2
    if args.cmd == "build":
        got = build(names, args.n_ref, workers=args.workers, rebuild=args.rebuild,
                    extend=args.extend)
        files, size = cache_size()
        print(f"cache {store_dir()}: {files} files, {size / 2**20:.1f} MiB")
        return 1 if any(v not in ("built", "ok") for v in got.values()) else 0
    st = check(names, args.n_ref)
    from .campaign import _code_stamp
    other = []
    for name in names:
        got = load_entry(name, args.n_ref)
        if got is not None and got[1].get("code_stamp") != _code_stamp():
            other.append(name)
    bad = {n: s for n, s in st.items() if s != "ok"}
    for n, s in sorted(bad.items()):
        print(f"{n}: {s}")
    print(f"{len(names) - len(bad)} of {len(names)} current at n_ref = {args.n_ref} "
          f"({store_dir()}); {len(other)} built by other benchmark code"
          + (f": {', '.join(other[:20])}{' ...' if len(other) > 20 else ''}" if other else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())

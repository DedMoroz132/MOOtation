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
MOOTATION_REFS_DIR (or --refs-dir) puts it anywhere. <problem>/n<n_ref>.npz.
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
# The families whose references take the longest (ZCAT's fronts and sets at
# ten objectives, DAS-CMOP's dense sample): built first, so that a pool of
# workers does not end on them.
SLOW_FIRST = ("rZCAT", "ZCAT", "DAS-CMOP", "rWFG", "WFG", "MW", "LIR-CMOP", "DOC")


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


def entry_path(name: str, n_ref: int) -> Path:
    return store_dir() / name / f"n{int(n_ref)}.npz"


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
        with np.load(path, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files if k != "meta"}
            meta = json.loads(str(z["meta"]))
    except (OSError, ValueError, KeyError):
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


def build_entry(name: str, n_ref: int, extend: bool = False) -> dict:
    """Build the entry of `name` at n_ref and write it (atomically: a copy built
    at the same time by another process gives the same bytes); its meta."""
    import numpy as np
    from .. import __version__
    from ..benchmarks import registry as R
    from .campaign import _code_stamp, reference_hash
    from .cover import igdp_floors
    t0 = time.perf_counter()
    p = R.get(name)
    ref = np.asarray(p.pareto_front(n_ref), float)
    ref3 = info3 = None
    if R.has_dense_reference(name):
        ref3, info3 = R.dense_reference(name, p.pop_size)
    pset = np.asarray(p.pareto_set(n_ref), float) if callable(p.pareto_set) else None
    floors = igdp_floors(name, n_ref, (p.pop_size,), ref=ref)
    pop = int(p.pop_size)
    meta = {
        "problem": name, "n_ref": int(n_ref), "n_obj": int(p.n_obj),
        "problem_version": R.problem_version(name), "reference_version": R.REFERENCE_VERSION,
        "code_stamp": _code_stamp(), "mootation": __version__,
        "reference_hash": reference_hash(p, ref, ref3, pset),
        "rows": floors["rows"], "distinct": floors["distinct"], "info3": info3,
        "floors": {str(pop): {"greedy": floors[pop][0], "dss": floors[pop][1]}},
    }
    if extend:
        meta["extends"] = _extends(name, n_ref, ref)
    arrays = dict(ideal=np.asarray(p.ideal, float), nadir=np.asarray(p.nadir, float), ref=ref)
    if ref3 is not None:
        arrays["ref3"] = np.asarray(ref3, float)
    if pset is not None:
        arrays["pset"] = pset
    meta["seconds"] = round(time.perf_counter() - t0, 3)
    path = entry_path(name, n_ref)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npz")
    np.savez_compressed(tmp, meta=np.asarray(json.dumps(meta)), **arrays)
    try:
        os.replace(tmp, path)
    except OSError:              # Windows: another process reads the copy it wrote
        tmp.unlink(missing_ok=True)
        if load_entry(name, n_ref) is None:
            raise
    return meta


def _extends(name: str, n_ref: int, ref) -> dict | None:
    """Against the largest smaller entry of `name`: whether its front is the
    first rows of `ref` (kept, extended) or not (built anew)."""
    import numpy as np
    sizes = sorted(int(f.stem[1:]) for f in (store_dir() / name).glob("n*.npz")
                   if f.stem[1:].isdigit() and int(f.stem[1:]) < n_ref)
    if not sizes:
        return None
    got = load_entry(name, sizes[-1])
    if got is None or "ref" not in got[0]:
        return None
    old = got[0]["ref"]
    kept = len(old) <= len(ref) and np.array_equal(old, ref[:len(old)])
    return {"n_ref": sizes[-1], "prefix": bool(kept)}


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


def _order(names) -> list:
    def key(n):
        from ..families import family
        fam = family(n)
        return (SLOW_FIRST.index(fam) if fam in SLOW_FIRST else len(SLOW_FIRST), n)
    return sorted(names, key=key)


def _build_task(task):
    name, n_ref, extend = task
    t0 = time.perf_counter()
    try:
        meta = build_entry(name, n_ref, extend=extend)
        return name, "built", meta, time.perf_counter() - t0
    except Exception as e:                       # noqa: BLE001
        return name, f"failed: {type(e).__name__}: {e}", None, time.perf_counter() - t0


def build(names, n_ref: int = 1000, workers: int = 1, rebuild: bool = False,
          extend: bool = False, log=print) -> dict:
    """Build the entries of `names` at n_ref that are missing or old (all of
    them with rebuild or extend), `workers` processes at a time. {name:
    "built" | "ok" | "failed: ..."}."""
    names = list(dict.fromkeys(names))
    now = check(names, n_ref)
    todo = _order([n for n in names if rebuild or extend or now[n] != "ok"])
    out = {n: "ok" for n in names if n not in todo}
    if not todo:
        return out
    log(f"references: {len(todo)} of {len(names)} to build at n_ref = {n_ref} -> {store_dir()}")
    tasks = [(n, int(n_ref), extend) for n in todo]
    t0 = time.perf_counter()
    if workers > 1 and len(tasks) > 1:
        import multiprocessing as mp
        from .campaign import single_threaded_blas
        single_threaded_blas()
        with mp.get_context("spawn").Pool(processes=min(workers, len(tasks))) as pool:
            results = pool.imap_unordered(_build_task, tasks)
            for i, (name, what, meta, sec) in enumerate(results, 1):
                out[name] = what
                log(f"  [{i}/{len(tasks)}] {name}: {what} ({sec:.1f} s)")
    else:
        for i, task in enumerate(tasks, 1):
            name, what, meta, sec = _build_task(task)
            out[name] = what
            log(f"  [{i}/{len(tasks)}] {name}: {what} ({sec:.1f} s)")
    again = [n for n in todo if out[n].startswith("failed")]
    if again and workers > 1:
        # one at a time in this process: a pool of builders can run out of memory
        # together (DAS-CMOP's dense samples) where one alone does not
        log(f"references: {len(again)} failed in the pool, built again one at a time")
        for name in again:
            name, what, meta, sec = _build_task((name, int(n_ref), extend))
            out[name] = what
            log(f"  {name}: {what} ({sec:.1f} s)")
    failed = sum(v.startswith("failed") for v in out.values())
    log(f"references: {len(todo) - failed} built, {failed} failed, "
        f"{time.perf_counter() - t0:.1f} s")
    return out


def cache_size() -> tuple:
    """(files, bytes) of this builder version's cache."""
    files = [f for f in store_dir().rglob("*.npz")] if store_dir().is_dir() else []
    return len(files), sum(f.stat().st_size for f in files)


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(prog="mootation refs",
                                 description="the registry's reference data, built once per machine")
    sub = ap.add_subparsers(dest="cmd", required=True)
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
        return 1 if any(v.startswith("failed") for v in got.values()) else 0
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

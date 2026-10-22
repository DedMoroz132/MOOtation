# SPDX-License-Identifier: Apache-2.0
"""What the search found, apart from how the answer was chosen: the oracle
(task 6 of 2026-10-09, item 13).

    python -m mootation.run.campaign camp.toml --oracle

Reads finished runs; runs nothing. CSV under <results>/_oracle/.

THE ORACLE of a run is the best N points of its archive under IGD+ against the
reference front, as far as a greedy search finds them (archive.greedy_igdp,
the candidates the archive's points, the reference points the front's): a
measure of the archive's content, which knows the reference and so is never
an answer a run could give. N is the size of the run's archive-scenario answer
(final_archive's n: the problem's population, or the whole archive when it is
smaller). Beside it, all in igdp_norm's frame and against the same reference:
the population (final), the archive scenario's answer as the run selected it
(final_archive: DSS, or answer_select's choice), the choice answer_select =
"igdp_self" would make from the same archive (archive.igdp_self_order, in the
frame the run's DSS used) and the whole archive, a bound no N of its points
beat. A run without an archive file, on a problem without a reference front,
or with an empty archive is left out and counted.

oracle.csv, one row per run: problem, algorithm, seed, n, archive_size and
the five igdp_norm values. The terminal gets the medians over the runs of
oracle / population, oracle / archive answer and igdp_self / archive answer,
and how often igdp_self beats the archive answer and the population.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

COLUMNS = ["problem", "algorithm", "seed", "n", "archive_size", "igdp_norm_population",
           "igdp_norm_archive", "igdp_norm_igdp_self", "igdp_norm_oracle",
           "igdp_norm_archive_whole"]


def run_values(run_dir: Path, n_ref: int = 1000) -> dict | str:
    """The row of one run (module docstring), or why it has none."""
    import numpy as np
    from ..benchmarks import get as bench_get
    from .archive import greedy_igdp_rows, igdp_self_order
    from .campaign import _read_points
    from .metrics import _normalised, igd_plus
    from ..store import as_run_dir
    run_dir = as_run_dir(run_dir)
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    if meta.get("status") != "done":
        return "not done"
    p = bench_get(meta["problem"])
    if not callable(p.pareto_front):
        return "no reference front"
    arc = meta.get("archive") or {}
    path = run_dir / (arc.get("file") or "archive.csv")
    if not path.is_file():
        return "no archive file"
    m, nv = int(meta.get("n_objs") or p.n_obj), int(meta.get("n_vars") or p.n_vars)
    F, _ = _read_points(path, m, nv)
    F = F[np.all(np.isfinite(F), axis=1)]
    if not len(F):
        return "empty archive"
    ref = np.asarray(p.pareto_front(int(meta.get("n_ref") or n_ref)), float)
    G, R = _normalised(F, ref, p.ideal, p.nadir)
    n = int((meta.get("final_archive") or {}).get("n") or min(p.pop_size, len(F)))
    oracle = greedy_igdp_rows(G, R, n)
    frame = arc.get("frame")
    lo, hi = (frame if frame and arc.get("normalization") != "archive" else (None, None))
    own = igdp_self_order(F, n, ideal=lo, nadir=hi)
    pop = (meta.get("final") or {}).get("igdp_norm")
    ans = (meta.get("final_archive") or {}).get("igdp_norm")
    return {"problem": meta["problem"], "algorithm": meta.get("algorithm"),
            "seed": meta.get("seed"), "n": n, "archive_size": int(len(F)),
            "igdp_norm_population": pop, "igdp_norm_archive": ans,
            "igdp_norm_igdp_self": igd_plus(G[own], R),
            "igdp_norm_oracle": igd_plus(G[oracle], R),
            "igdp_norm_archive_whole": igd_plus(G, R)}


def _task(args):
    run_dir, n_ref = args
    try:
        return run_values(run_dir, n_ref)
    except Exception as e:                          # noqa: BLE001 - one bad run is a row of its own
        return f"failed: {type(e).__name__}: {e}"


def _median(v) -> float:
    v = sorted(v)
    return (v[(len(v) - 1) // 2] + v[len(v) // 2]) / 2 if v else math.nan


def run(root: Path, *, n_ref: int = 1000, workers: int = 1, out_dir: Path | None = None) -> str:
    """The whole measure: text for the terminal; oracle.csv in out_dir."""
    from .campaign import main_rows, scan_results
    root = Path(root)
    out_dir = Path(out_dir) if out_dir else root / "_oracle"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [r for r in main_rows(scan_results(root)) if r["status"] == "done"]
    tasks = [(r["dir"], n_ref) for r in rows]
    if workers > 1 and len(tasks) > 1:
        import multiprocessing as mp
        from .campaign import single_threaded_blas
        single_threaded_blas()
        with mp.Pool(processes=workers) as pool:
            got = list(pool.imap(_task, tasks, chunksize=8))
    else:
        got = [_task(t) for t in tasks]
    done = [g for g in got if isinstance(g, dict)]
    left: dict = {}
    for g in got:
        if not isinstance(g, dict):
            left[g.split(":")[0]] = left.get(g.split(":")[0], 0) + 1
    with (out_dir / "oracle.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(COLUMNS)
        for g in sorted(done, key=lambda d: (d["problem"], str(d["algorithm"]), d["seed"] or 0)):
            w.writerow([f"{g[c]:.10g}" if isinstance(g[c], float) else
                        ("" if g[c] is None else g[c]) for c in COLUMNS])

    def ratios(a, b):
        return [g[a] / g[b] for g in done if g.get(a) is not None and g.get(b)
                and math.isfinite(g[a]) and math.isfinite(g[b]) and g[b] > 0]

    def share(a, b):
        pairs = [(g[a], g[b]) for g in done if g.get(a) is not None and g.get(b) is not None]
        return (sum(x < y for x, y in pairs) / len(pairs)) if pairs else math.nan

    lines = [f"oracle: {len(done)} runs measured, CSV in {out_dir / 'oracle.csv'}"
             + ("; left out: " + ", ".join(f"{k} {v}" for k, v in sorted(left.items()))
                if left else "")]
    if done:
        o_pop = _median(ratios("igdp_norm_oracle", "igdp_norm_population"))
        o_arc = _median(ratios("igdp_norm_oracle", "igdp_norm_archive"))
        s_arc = _median(ratios("igdp_norm_igdp_self", "igdp_norm_archive"))
        lines.append(f"  median oracle / population {o_pop:.3g}, oracle / archive answer "
                     f"{o_arc:.3g}, igdp_self / archive answer {s_arc:.3g}")
        lines.append(f"  igdp_self below the archive answer in "
                     f"{share('igdp_norm_igdp_self', 'igdp_norm_archive'):.1%} of the runs, "
                     f"below the population in "
                     f"{share('igdp_norm_igdp_self', 'igdp_norm_population'):.1%}")
    return "\n".join(lines)

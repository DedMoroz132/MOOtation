# SPDX-License-Identifier: Apache-2.0
"""The problems as points of a plane, and where each algorithm does well there
(task 4, item 5: instance space analysis).

    python -m mootation.run.campaign camp.toml --instance-space [--scenario archive]

Reads finished runs; runs nothing; needs SciPy. CSV and JSON under
<results>/_instance_space/ (_instance_space_archive/ for the archive).

After Smith-Miles & Muñoz, "Instance Space Analysis for Algorithm Testing", ACM
Computing Surveys 55(12), 2023, Section 3.2 (PRELIM, SIFTED, PILOT), and Muñoz
& Smith-Miles, "Performance Analysis of Continuous Black-Box Optimization
Algorithms via Footprints in Instance Space", Evolutionary Computation 25(4),
2017, Section 5 (the footprints). Written from the papers: the authors' MATLAB
toolkit (MATILDA) is under a non-commercial licence.

TWO SPACES, one per group of problems as in portfolio.py: "front", with a
reference front, and "bbob", bbob-biobj; their performance measures differ and
are never mixed.

FEATURES are the problem properties (benchmarks/properties.py), not landscape
measures: each shape of the front a column (1 when the front has it, 0.5 when
the front is unknown), every yes/no property 1 or 0 (0.5 when unknown), the
bbob-biobj function groups as counts (0, 1 or 2 of the pair), M, log2 D and the
front's dimension (M - 1 when unknown). Many problems share a feature vector
and so a point. Constant columns are dropped and the rest z-scored. Not
PRELIM's bounding (median +- 5 IQR) and Box-Cox transform: meant for measured
features, they would flatten a rare yes (deceptive: two problems) to a constant.

PERFORMANCE (PRELIM, the survey's Algorithm 1, relative): per problem the
median over the seeds at the full budget of igdp_norm ("front") or hv_h
("bbob"), as median / best - 1 or 1 - median / best, best the best median of
any algorithm there; zeros replaced by the machine epsilon, then Box-Cox (lambda
by maximum likelihood) and z-score per algorithm. The best algorithm of a
problem: the first by name with the best median.

SIFTED, its first step (Algorithm 2, lines 2-8): a feature stays when it is the
one most correlated with some algorithm's performance or correlates with some
algorithm's at 0.3 or more, correlations of p > 0.05 counted as 0. The second
step, k-means over the features and a random-forest search for one feature per
cluster, is meant for tens of measured features; the properties are a dozen.

PILOT (Algorithm 3): coordinates Z = A F such that linear models of Z predict
the features and the performances best, min ||F - B Z||^2 + ||Y - C Z||^2.
Numerically, as the toolkit does by default: BFGS from 30 random starts in
[-1, 1], keeping the solution of the highest topological preservation, the
Pearson correlation of the problems' distances in features and in the plane.
Beside it the analytical solution (lines 3-9: V the two leading eigenvectors of
Xbar Xbar^T, Xbar = [F; Y], A = V^T Xbar F^T (F F^T)^+, the pseudo-inverse
putting the problem in the subspace F spans when F is not of full row rank), its
loss and preservation reported, and the loss of the optimum itself as a check
that BFGS got there (pilot_optimum_loss; on stage 3 it did, and the analytical
solution fell 5-6 % short of it). The R^2 of every feature's and every
algorithm's linear model: is_projection.csv.

FOOTPRINTS (the 2017 paper's Algorithm 1): of the problems where the algorithm
is good, one of any two closer than delta is dropped; the rest are triangulated
(Delaunay); triangles with a side longer than Delta go; so do those whose
density (problems inside per unit area) is below rho or whose purity (the share
of good problems among those inside) is below pi. delta and Delta are 1 % and
25 % of the largest distance between two problems and pi = 0.75, as in the
paper; its rho = 10 problems per unit area was 10 / 233.7 of its known region's
density, and is taken so, relative to ours. The known region is the same
construction over all problems of the group, with no density or purity limit; a
footprint's area and density are given as shares of the known region's. GOOD
is --cover's criterion at a level and budget, reached in min_seeds seeds:
"igdp" (igdp_norm against the floor) and "gdp" (gdp_norm) on "front", the hv_h
gap to the campaign's best ("hv") on "bbob". The BEST footprints, of the best
algorithm of every problem at a budget, lose their contradictions as in the
paper's Algorithm 2: of two algorithms, a triangle of one goes when the
triangles of the other overlapping it are together larger than it.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

SHAPES = ("linear", "concave", "convex", "mixed", "disconnected", "degenerate")
YES_NO = ("multimodal", "deceptive", "bias", "scaled", "separable", "centre", "linkage")
BBOB_GROUPS = ("separable", "moderate", "ill-conditioned", "multimodal", "weakly-structured")
SIFTED_CORRELATION, SIFTED_P = 0.3, 0.05
PILOT_TRIES = 30
DELTA, BIG_DELTA, PURITY = 0.01, 0.25, 0.75
DENSITY_SHARE = 10 / 233.7463            # rho over the known region's density (2017)


# ── features and performance (PRELIM, SIFTED) ──────────────────────────────


def feature_names() -> list:
    return ([f"front={s}" for s in SHAPES] + list(YES_NO)
            + [f"bbob={g}" for g in BBOB_GROUPS] + ["n_obj", "log2_n_vars", "front_dim"])


def feature_rows(problems: list) -> list:
    """One row per feature_names(), one value per problem (FEATURES)."""
    from ..benchmarks import get as bench_get
    from ..benchmarks.properties import properties
    rows = [[] for _ in feature_names()]
    for p in problems:
        pr = properties(p) or {}
        spec = bench_get(p)
        front = pr.get("front")
        vals = [0.5 if front is None else float(s in front.split("+")) for s in SHAPES]
        vals += [0.5 if pr.get(k) is None else float(bool(pr[k])) for k in YES_NO]
        groups = (pr.get("bbob_groups") or "").split("+")
        vals += [float(groups.count(g)) for g in BBOB_GROUPS]
        m = float(spec.n_obj)
        vals += [m, math.log2(spec.n_vars),
                 float(pr["front_dim"]) if pr.get("front_dim") is not None else m - 1.0]
        for r, v in zip(rows, vals):
            r.append(v)
    return rows


def zscore(rows) -> tuple:
    """(kept row indices, the kept rows with mean 0 and standard deviation 1):
    constant rows are dropped."""
    import numpy as np
    X = np.asarray(rows, float)
    sd = X.std(axis=1)
    keep = np.flatnonzero(sd > 1e-12)
    return keep.tolist(), (X[keep] - X[keep].mean(axis=1, keepdims=True)) / sd[keep, None]


def relative_performance(med: dict, problems: list, algorithms: list, higher: bool) -> tuple:
    """(rows, best): rows[a][j] algorithm a's median on problem j relative to
    the best median there, median / best - 1 (lower better) or 1 - median /
    best (higher better), NaN without a median; best[j] the first algorithm by
    name with the best median (PERFORMANCE). med[p][a] as portfolio.medians."""
    rows = [[math.nan] * len(problems) for _ in algorithms]
    best = []
    for j, p in enumerate(problems):
        have = {a: med[p][a] for a in algorithms if a in med.get(p, {})}
        if not have:
            best.append(None)
            continue
        y = (max if higher else min)(have.values())
        best.append(next(a for a in algorithms if have.get(a) == y))
        for i, a in enumerate(algorithms):
            if a not in have:
                continue
            v = have[a]
            if higher:
                rows[i][j] = 1.0 - v / y if y > 0 else 0.0
            else:
                rows[i][j] = v / y - 1.0 if y > 0 else (0.0 if v == y else math.inf)
    return rows, best


def prelim_performance(rows) -> tuple:
    """PRELIM's normalization of relative performances, per row: a missing
    value takes the row's worst, zeros the machine epsilon, then Box-Cox and
    z-score. (kept row indices, rows); a constant row is dropped."""
    import numpy as np
    from scipy.stats import boxcox
    keep, out = [], []
    for i, r in enumerate(rows):
        x = np.asarray(r, float)
        finite = np.isfinite(x)
        if not finite.any():
            continue
        x[~finite] = x[finite].max()
        x[x <= 0.0] = np.finfo(float).eps
        if np.ptp(x) == 0.0:
            continue
        t = boxcox(x)[0]
        if not t.std() > 0.0:
            continue
        keep.append(i)
        out.append((t - t.mean()) / t.std())
    return keep, np.asarray(out)


def sifted(F, Y) -> tuple:
    """SIFTED's first step: (kept feature indices, R), R[f, a] the Pearson
    correlation of feature f with algorithm a's performance, 0 where p > 0.05.
    F and Y z-scored rows over the same problems."""
    import numpy as np
    from scipy.stats import t as student
    n = F.shape[1]
    R = (F @ Y.T) / n
    t = R * np.sqrt((n - 2) / np.maximum(1.0 - R ** 2, 1e-300))
    R = np.where(2.0 * student.sf(np.abs(t), n - 2) > SIFTED_P, 0.0, R)
    keep = set(np.flatnonzero((np.abs(R) >= SIFTED_CORRELATION).any(axis=1)).tolist())
    for a in range(R.shape[1]):
        if np.abs(R[:, a]).max() > 0.0:
            keep.add(int(np.argmax(np.abs(R[:, a]))))
    return sorted(keep), R


# ── PILOT ──────────────────────────────────────────────────────────────────


def preservation(F, Z) -> float:
    """Topological preservation: the Pearson correlation of the distances
    between the problems in features (F, q x n) and in the plane (Z, n x 2);
    NaN when either set of distances is constant (a plane collapsed to a point)."""
    import numpy as np
    from scipy.spatial.distance import pdist
    dh, dl = pdist(F.T), pdist(Z)
    if not (dh.std() > 0.0 and dl.std() > 0.0):
        return math.nan
    return float(np.corrcoef(dh, dl)[0, 1])


def _pilot_result(F, Y, A, B, C) -> dict:
    import numpy as np
    Z = A @ F
    Fh, Yh = B @ Z, C @ Z

    def r2(T, That):
        sst = ((T - T.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
        return 1.0 - ((T - That) ** 2).sum(axis=1) / np.where(sst > 0.0, sst, 1.0)

    return {"A": A, "B": B, "C": C, "Z": Z.T,
            "loss": float(((F - Fh) ** 2).sum() + ((Y - Yh) ** 2).sum()),
            "r2_features": r2(F, Fh), "r2_performance": r2(Y, Yh),
            "preservation": preservation(F, Z.T)}


def pilot_analytic(F, Y) -> dict:
    """PILOT's analytical solution (Algorithm 3, lines 3-9), the pseudo-inverse
    for (F F^T)^-1: {"A", "B", "C", "Z" (n x 2), "loss", "r2_features",
    "r2_performance", "preservation"}."""
    import numpy as np
    q = F.shape[0]
    Xbar = np.vstack([F, Y])
    vals, vecs = np.linalg.eigh(Xbar @ Xbar.T)
    V = vecs[:, np.argsort(vals)[::-1][:2]]
    for k in range(2):                           # each eigenvector's sign fixed
        if V[np.argmax(np.abs(V[:, k])), k] < 0.0:
            V[:, k] = -V[:, k]
    A = V.T @ Xbar @ F.T @ np.linalg.pinv(F @ F.T)
    return _pilot_result(F, Y, A, V[:q], V[q:])


def pilot_optimum_loss(F, Y) -> float:
    """The loss of PILOT's optimum. For a given Z the best B and C are least
    squares, which leave of Xbar = [F; Y] what lies outside Z's row space; Z's
    rows lie in F's, so the optimum keeps the most of Xbar P_F, P_F the
    projection on F's row space: ||Xbar||^2 less the two largest eigenvalues
    of (Xbar P_F)(Xbar P_F)^T."""
    import numpy as np
    Xbar = np.vstack([F, Y])
    XP = Xbar @ F.T @ np.linalg.pinv(F @ F.T) @ F
    return float((Xbar ** 2).sum() - np.sort(np.linalg.eigvalsh(XP @ XP.T))[-2:].sum())


def pilot_numerical(F, Y, tries: int = PILOT_TRIES, seed: int = 0) -> dict:
    """PILOT's numerical solution (Algorithm 3, lines 11-25): BFGS from `tries`
    random starts, the result of the highest topological preservation; the
    same dict as pilot_analytic."""
    import numpy as np
    from scipy.optimize import minimize
    q, a = F.shape[0], Y.shape[0]

    def split(x):
        return x[:2 * q].reshape(2, q), x[2 * q:4 * q].reshape(q, 2), x[4 * q:].reshape(a, 2)

    def loss(x):
        A, B, C = split(x)
        Z = A @ F
        E1, E2 = F - B @ Z, Y - C @ Z
        gZ = -2.0 * (B.T @ E1 + C.T @ E2)
        grad = np.concatenate([(gZ @ F.T).ravel(), (-2.0 * E1 @ Z.T).ravel(),
                               (-2.0 * E2 @ Z.T).ravel()])
        return float((E1 ** 2).sum() + (E2 ** 2).sum()), grad

    rng = np.random.default_rng(seed)
    best, best_rho = None, -math.inf
    for _ in range(tries):
        x = minimize(loss, rng.uniform(-1.0, 1.0, 4 * q + 2 * a), jac=True, method="BFGS").x
        res = _pilot_result(F, Y, *split(x))
        rho = res["preservation"] if math.isfinite(res["preservation"]) else -math.inf
        if best is None or rho > best_rho:
            best, best_rho = res, rho
    return best


# ── footprints ─────────────────────────────────────────────────────────────


def _area(Z, t) -> float:
    (x1, y1), (x2, y2), (x3, y3) = Z[list(t)]
    return abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)) / 2.0


def _inside(Z, t, tol: float = 1e-9):
    """Which points of Z lie in triangle t (indices into Z), its sides included."""
    import numpy as np
    (x1, y1), (x2, y2), (x3, y3) = Z[list(t)]
    den = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
    if den == 0.0:
        return np.zeros(len(Z), bool)
    l1 = ((y2 - y3) * (Z[:, 0] - x3) + (x3 - x2) * (Z[:, 1] - y3)) / den
    l2 = ((y3 - y1) * (Z[:, 0] - x3) + (x1 - x3) * (Z[:, 1] - y3)) / den
    return (l1 >= -tol) & (l2 >= -tol) & (1.0 - l1 - l2 >= -tol)


def overlap(P, Q, tol: float = 1e-12) -> bool:
    """Do triangles P and Q (3 x 2 vertex arrays) share some area? Touching
    does not count. The separating-axis test on the sides of both."""
    import numpy as np
    for tri in (P, Q):
        for i in range(3):
            edge = tri[(i + 1) % 3] - tri[i]
            normal = np.array([-edge[1], edge[0]])
            p, q = P @ normal, Q @ normal
            if p.max() <= q.min() + tol or q.max() <= p.min() + tol:
                return False
    return True


def summarize(Z, good, triangles: list) -> dict:
    """{"triangles", "area", "inside", "good_inside"}: the area of a set of
    non-overlapping triangles and the problems inside them, each counted once."""
    import numpy as np
    inside = np.zeros(len(Z), bool)
    for t in triangles:
        inside |= _inside(Z, t)
    return {"triangles": triangles, "area": float(sum(_area(Z, t) for t in triangles)),
            "inside": int(inside.sum()), "good_inside": int((inside & np.asarray(good)).sum())}


def footprint(Z, good, dmax: float, rho: float = 0.0, purity: float = 0.0) -> dict:
    """The 2017 paper's Algorithm 1 over the problems Z (n x 2) where `good`:
    summarize() of its triangles, each a triple of indices into Z."""
    import numpy as np
    from scipy.spatial import Delaunay, QhullError
    good = np.asarray(good, bool)
    kept: list = []                               # one of any two closer than delta
    for i in np.flatnonzero(good):
        if not kept or np.hypot(*(Z[kept] - Z[i]).T).min() >= DELTA * dmax:
            kept.append(int(i))
    if len(kept) < 3:
        return summarize(Z, good, [])
    try:
        simplices = Delaunay(Z[kept]).simplices
    except (QhullError, ValueError):              # the good problems on one line
        return summarize(Z, good, [])
    out = []
    for s in simplices:
        t = tuple(kept[v] for v in s)
        a = _area(Z, t)
        sides = [math.dist(Z[t[i]], Z[t[(i + 1) % 3]]) for i in range(3)]
        if a <= 1e-12 * dmax ** 2 or max(sides) > BIG_DELTA * dmax:
            continue
        inside = _inside(Z, t)
        n_in = int(inside.sum())
        if n_in == 0 or n_in / a < rho or (inside & good).sum() / n_in < purity:
            continue
        out.append(t)
    return summarize(Z, good, out)


def remove_contradictions(Z, fa: dict, fb: dict) -> tuple:
    """The 2017 paper's Algorithm 2 both ways: (the triangles of fa, of fb)
    left once each loses those the other's overlapping triangles outweigh."""
    def keep(base, test):
        out = []
        for t in base["triangles"]:
            clash = sum(_area(Z, u) for u in test["triangles"]
                        if overlap(Z[list(t)], Z[list(u)]))
            if clash <= _area(Z, t):
                out.append(t)
        return out
    return keep(fa, fb), keep(fb, fa)


# ── the whole analysis ─────────────────────────────────────────────────────


def run(root: Path, *, taus=None, floor_taus=None, budgets=None, min_seeds: int = 7,
        workers: int = 1, out_dir: Path | None = None, scenario: str = "final",
        n_ref: int = 1000, keep_caveats: bool = False, tries: int = PILOT_TRIES,
        seed: int = 0) -> str:
    """The whole analysis: text for the terminal; CSV and JSON in out_dir.
    tries = 0 takes PILOT's analytical solution instead of the numerical."""
    import time
    import numpy as np
    from .. import __version__
    from . import cover as V
    from .portfolio import medians
    from .provenance import revision
    try:
        import scipy  # noqa: F401
    except ImportError as e:                        # pragma: no cover - machine-dependent
        raise ImportError("--instance-space needs SciPy: pip install scipy") from e
    from scipy.spatial.distance import pdist
    taus = tuple(taus or V.TAUS)
    floor_taus = tuple(floor_taus or V.FLOOR_TAUS)
    budgets = tuple(budgets or V.BUDGETS)
    if len(floor_taus) != len(taus):
        raise ValueError(f"{len(floor_taus)} floor levels against {len(taus)} levels")
    root = Path(root)
    out_dir = (Path(out_dir) if out_dir else
               root / ("_instance_space_archive" if scenario == "archive" else "_instance_space"))
    out_dir.mkdir(parents=True, exist_ok=True)
    cover_dir = root / ("_cover_archive" if scenario == "archive" else "_cover")
    cover_dir.mkdir(parents=True, exist_ok=True)
    data = V.collect(root, budgets, workers=workers, scenario=scenario)
    if not data["values"]:
        return "instance space: no finished runs"
    prep = V.prepare(data, n_ref, cover_dir / "igdp_floor.csv", workers=workers,
                     keep_caveats=keep_caveats)
    best_hv = V.best_known_hv(data)
    kept = [p for p in prep["problems"] if p not in prep["excluded"]]
    groups = {"front": [p for p in kept if data["front"].get(p)],
              "bbob": [p for p in kept if not data["front"].get(p)]}
    algorithms = sorted({k[1] for k in data["values"]})
    full = max(budgets)
    # --cover's levels: "igdp" and "gdp" where there is a front, the hv_h gap
    # ("hv", which either of cover's criteria reads) where not
    hits = {}
    for k, tau in enumerate(taus):
        for b in budgets:
            hits[("igdp", k, b)] = V.reached(data, "igdp", (floor_taus[k], tau), b, best_hv,
                                             prep["floor"])
            hits[("gdp", k, b)] = hits[("hv", k, b)] = V.reached(data, "gdp", (tau, tau), b,
                                                                 best_hv)
    criteria = {"front": ("igdp", "gdp"), "bbob": ("hv",)}

    def level_text(crit, k):
        return (f"igdp_norm <= {1 + floor_taus[k]:g} x {prep['which']} floor" if crit == "igdp"
                else f"gdp_norm <= {taus[k]:g}" if crit == "gdp" else f"hv_h gap <= {taus[k]:g}")

    rev = revision()
    commit = (rev.get("commit") or "unknown")[:12] + ("+dirty" if rev.get("dirty") else "")
    names = feature_names()
    coords, projection, performance, fp_rows = [], [], [], []
    triangles, meta_groups = {}, {}
    lines = [f"instance space ({scenario}): analysis code {commit}; CSV in {out_dir}"]
    for group, problems in groups.items():
        if len(problems) < 4:
            if problems:
                lines.append(f"  {group}: {len(problems)} problems, too few for a plane")
            continue
        metric = "igdp_norm" if group == "front" else "hv_h"
        higher = group == "bbob"
        raw = feature_rows(problems)
        rel, best = relative_performance(medians(data, metric, full, set(problems)), problems,
                                         algorithms, higher)
        y_rows, Y = prelim_performance(rel)
        f_rows, F = zscore(raw)
        sel, R = sifted(F, Y)
        F = F[sel]
        used = [names[f_rows[i]] for i in sel]
        analytic = pilot_analytic(F, Y)
        optimum = pilot_optimum_loss(F, Y)
        proj = pilot_numerical(F, Y, tries, seed) if tries > 0 else analytic
        Z = proj["Z"]
        dmax = float(pdist(Z).max())
        known = footprint(Z, np.ones(len(Z), bool), dmax)
        density = known["inside"] / known["area"] if known["area"] > 0.0 else 0.0
        rho = DENSITY_SHARE * density

        for j, p in enumerate(problems):
            coords.append([group, p, f"{Z[j, 0]:.6g}", f"{Z[j, 1]:.6g}", best[j] or ""]
                          + [f"{r[j]:.6g}" for r in raw])
            for i, a in enumerate(algorithms):
                performance.append([group, p, a, "" if math.isnan(rel[i][j])
                                    else f"{rel[i][j]:.6g}"])
        for i, f in enumerate(f_rows):               # R's rows are f_rows', F's sel's
            s = sel.index(i) if i in sel else None
            projection.append([group, names[f], "feature", s is not None,
                               f"{np.abs(R[i]).max():.3f}"]
                              + ([f"{proj['A'][0, s]:.6g}", f"{proj['A'][1, s]:.6g}",
                                  f"{proj['B'][s, 0]:.6g}", f"{proj['B'][s, 1]:.6g}",
                                  f"{proj['r2_features'][s]:.4f}"] if s is not None
                                 else ["", "", "", "", ""]))
        for i, yi in enumerate(y_rows):
            projection.append([group, algorithms[yi], "performance", True,
                               f"{np.abs(R[:, i]).max():.3f}", "", "",
                               f"{proj['C'][i, 0]:.6g}", f"{proj['C'][i, 1]:.6g}",
                               f"{proj['r2_performance'][i]:.4f}"])

        shapes = {}

        def record(kind, crit, k, b, alg, fp, good):
            fp_rows.append([group, kind, crit, "" if k is None else k,
                            "" if k is None else level_text(crit, k), b, alg,
                            int(np.asarray(good).sum()), len(fp["triangles"]),
                            f"{fp['area'] / known['area']:.6g}",
                            f"{(fp['inside'] / fp['area'] / density) if fp['area'] else 0.0:.6g}",
                            f"{(fp['good_inside'] / fp['inside']) if fp['inside'] else 0.0:.6g}"])
            if fp["triangles"]:
                shapes[f"{kind}|{crit}|{'' if k is None else k}|{b}|{alg}"] = \
                    [list(t) for t in fp["triangles"]]

        # no known region (a few problems far apart: no triangle's sides all
        # within Delta), no footprint
        drawn = known["area"] > 0.0
        for crit in (criteria[group] if drawn else ()):
            for k in range(len(taus)):
                for b in budgets:
                    hit = hits[(crit, k, b)]
                    cov = V.successes({p: hit[p] for p in problems if p in hit}, min_seeds)
                    for alg in algorithms:
                        good = np.array([p in cov.get(alg, ()) for p in problems])
                        record("good", crit, k, b, alg, footprint(Z, good, dmax, rho, PURITY),
                               good)
        for b in (budgets if drawn else ()):
            _, best_b = relative_performance(medians(data, metric, b, set(problems)), problems,
                                             algorithms, higher)
            good_of = {a: np.array([x == a for x in best_b]) for a in algorithms}
            fps = {a: footprint(Z, good_of[a], dmax, rho, PURITY) for a in algorithms}
            some = [a for a in algorithms if fps[a]["triangles"]]
            for i, a in enumerate(some):
                for c in some[i + 1:]:
                    ta, tc = remove_contradictions(Z, fps[a], fps[c])
                    fps[a], fps[c] = summarize(Z, good_of[a], ta), summarize(Z, good_of[c], tc)
            for a in algorithms:
                record("best", "median", None, b, a, fps[a], good_of[a])
        triangles[group] = {"known": [list(t) for t in known["triangles"]], "footprints": shapes}

        points = len({tuple(r[j] for r in raw) for j in range(len(problems))})
        meta_groups[group] = {
            "problems": len(problems), "points": points, "metric": metric, "features": used,
            "features_constant": [n for i, n in enumerate(names) if i not in f_rows],
            "features_sifted_out": [names[f] for i, f in enumerate(f_rows) if i not in sel],
            "algorithms_constant": [a for i, a in enumerate(algorithms) if i not in y_rows],
            "pilot": {"method": "numerical" if tries > 0 else "analytical", "tries": tries,
                      "seed": seed, "loss": proj["loss"], "preservation": proj["preservation"],
                      "optimum_loss": optimum, "analytical_loss": analytic["loss"],
                      "analytical_preservation": analytic["preservation"]},
            "known_area": known["area"], "known_density": density, "rho": rho,
            "max_distance": dmax}
        r2f = sorted(zip(used, proj["r2_features"]), key=lambda t: -t[1])
        lines.append(f"  {group}: {len(problems)} problems at {points} points, {len(used)} "
                     f"features, {len(y_rows)} algorithms; PILOT "
                     f"{meta_groups[group]['pilot']['method']}: preservation "
                     f"{proj['preservation']:.3f} (analytical {analytic['preservation']:.3f}), "
                     f"loss {proj['loss']:.6g} (optimum {optimum:.6g}, analytical "
                     f"{analytic['loss']:.6g})")
        lines.append("    features best predicted from the plane (R^2): "
                     + ", ".join(f"{n} {r:.2f}" for n, r in r2f[:6]))
        lines.append(f"    performance R^2: median {float(np.median(proj['r2_performance'])):.2f}"
                     f", best {float(max(proj['r2_performance'])):.2f}")
        if not drawn:
            lines.append("    no known region: no triangle with every side within Delta "
                         f"({BIG_DELTA:.0%} of the largest distance); no footprints")
            continue
        for crit in criteria[group]:
            for k in (0, len(taus) - 1):
                top = sorted(((float(r[9]), r[6]) for r in fp_rows
                              if r[0] == group and r[1] == "good" and r[2] == crit
                              and r[3] == k and r[5] == full), reverse=True)[:5]
                lines.append(f"    largest footprints, {level_text(crit, k)} at {full}: "
                             + ", ".join(f"{a} {v:.0%}" for v, a in top))
        top = sorted(((float(r[9]), r[6]) for r in fp_rows if r[0] == group
                      and r[1] == "best" and r[5] == full), reverse=True)[:5]
        lines.append(f"    largest best footprints at {full}: "
                     + (", ".join(f"{a} {v:.0%}" for v, a in top if v > 0) or "none"))

    heads = {
        "is_coordinates.csv": ["group", "problem", "z1", "z2", "best_algorithm"] + names,
        "is_performance.csv": ["group", "problem", "algorithm", "relative"],
        "is_projection.csv": ["group", "name", "kind", "used", "max_abs_correlation", "a1",
                              "a2", "model1", "model2", "r2"],
        "is_footprints.csv": ["group", "kind", "criterion", "level", "level_text", "budget",
                              "algorithm", "problems_good", "triangles", "area_share",
                              "density_share", "purity"],
    }
    for name, rows_ in (("is_coordinates.csv", coords), ("is_performance.csv", performance),
                        ("is_projection.csv", projection), ("is_footprints.csv", fp_rows)):
        with (out_dir / name).open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(heads[name])
            w.writerows(rows_)
    (out_dir / "is_footprints.json").write_text(json.dumps(triangles, separators=(",", ":")),
                                                encoding="utf-8")
    meta = {"analysis": {"mootation": __version__, "commit": rev.get("commit"),
                         "dirty": rev.get("dirty"), "source": rev.get("source")},
            "results": str(root), "scenario": scenario,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "budgets": list(budgets),
            "full_budget": full, "taus": list(taus), "floor_taus": list(floor_taus),
            "min_seeds": min_seeds, "delta": DELTA, "big_delta": BIG_DELTA, "purity": PURITY,
            "density_share": DENSITY_SHARE, "sifted": [SIFTED_CORRELATION, SIFTED_P],
            "excluded": prep["excluded"], "groups": meta_groups,
            "triangles": "is_footprints.json: per group the known region's and every non-empty "
                         "footprint's triangles, as indices into the group's rows of "
                         "is_coordinates.csv; keys kind|criterion|level|budget|algorithm"}
    (out_dir / "is_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    text = "\n".join(lines)
    (out_dir / "is_report.txt").write_text(text + "\n", encoding="utf-8")
    return text

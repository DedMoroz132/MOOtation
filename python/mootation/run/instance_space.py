# SPDX-License-Identifier: Apache-2.0
"""The problems as points of a plane, and where each algorithm does well there
(task 4, item 5: instance space analysis).

    python -m mootation.run.campaign camp.toml --instance-space [--scenario archive]

Reads finished runs; runs nothing but the bbob-biobj samples (FEATURES FROM A
SAMPLE); needs SciPy. CSV and JSON under
<results>/_instance_space/ (_instance_space_archive/ for the archive).

After Smith-Miles & Muñoz, "Instance Space Analysis for Algorithm Testing", ACM
Computing Surveys 55(12), 2023, Section 3.2 (PRELIM, SIFTED, PILOT); Muñoz,
Villanova, Baatar & Smith-Miles, "Instance Spaces for Machine Learning
Classification", Machine Learning 107(1), 2018, Section 6.1 and Appendix A
(PILOT, there PBLDR, and its optimum); and Muñoz & Smith-Miles, "Performance
Analysis of Continuous Black-Box Optimization Algorithms via Footprints in
Instance Space", Evolutionary Computation 25(4), 2017, Section 5 (the
footprints). Written from the papers: the authors' toolkit (MATILDA) is under
the PolyForm Noncommercial licence (the MATLAB code since March 2026, the
Python one since July 2026; GPL v3 before).

TWO SPACES, one per group of problems as in portfolio.py: "front", with a
reference front, and "bbob", bbob-biobj; their performance measures differ and
are never mixed.

FEATURES are the problem properties (benchmarks/properties.py), not landscape
measures, for the problems with a reference front (bbob-biobj: FEATURES FROM A
SAMPLE below, or with bbob_features = "properties" these): each shape of the front a column (1 when the front has it, 0.5 when
the front is unknown), every yes/no property 1 or 0 (0.5 when unknown), the
bbob-biobj function groups as counts (0, 1 or 2 of the pair), M, log2 D and the
front's dimension (M - 1 when unknown). Many problems share a feature vector
and so a position in the plane (on stage 3 the 110 problems with a front have
77 vectors and 73 positions). Constant columns are dropped and the rest
z-scored. Not PRELIM's bounding (median +- 5 IQR), meant for measured features:
it would flatten a rare yes (deceptive: two problems) to a constant; nor its
Box-Cox transform, which after the z-score changes nothing on a yes/no column.

PERFORMANCE (PRELIM, the survey's Algorithm 1, relative): per problem the
median over the seeds at the full budget of igdp_norm ("front") or hv_h
("bbob"), as median / best - 1 or 1 - median / best, best the best median of
any algorithm there; zeros replaced by the machine epsilon, then Box-Cox (lambda
by maximum likelihood) and z-score per algorithm. A missing median takes the
worst value of the algorithm's row (the survey sets the instance to +-infinity;
stage 3 has none missing). The best algorithm of a problem: the first by name
with the best median (the survey breaks ties at random).

SIFTED, its first step (Algorithm 2, lines 2-8): a feature stays when it is the
one most correlated with some algorithm's performance or correlates with some
algorithm's at 0.3 or more, correlations of p > 0.05 counted as 0; when none
does, every feature stays (said in the report and is_meta.json). The second
step, k-means over the features and a random-forest search for one feature per
cluster, is meant for tens of measured features; the properties are a dozen.

PILOT (Algorithm 3): coordinates Z = A F such that linear models of Z predict
the features and the performances best, min ||F - B Z||^2 + ||Y - C Z||^2. The
problem is not convex, as the survey says: the loss is the same for (TA, B
T^-1, C T^-1), any invertible linear map T. The survey's analytical solution
(lines 3-9, the 2018 paper's Corollary 1, Eq. 13: V the two leading
eigenvectors of Xbar Xbar^T, Xbar = [F; Y], A = V^T Xbar F^T (F F^T)^+, the
pseudo-inverse putting the problem in the subspace F spans when F is not of
full row rank) is not its optimum: Appendix A's step from the relaxation (Z
free) to the problem (Eq. 11, "with the same objective value") holds only
when the relaxation's Z lies in the row space of F, which the part of it Y
adds in general does not. The 2018 paper's own numbers (its Section 6.1) show
the gap: Eq. 13 1.8749e3 against 1.8658e3 by CMA-ES, which it puts down to
numerical instability. The optimum is that of problem (D'), Eq. 15: a
reduced-rank regression of Xbar on F, Z's rows the two leading right singular
vectors of Xbar P_F (P_F the projection on F's row space).

THE PLANE (task 6 of 2026-10-09, item 13). That optimum is computed exactly
(pilot_exact), not as the toolkit does, by BFGS from 30 random starts keeping
the end of the highest topological preservation (the Pearson correlation of
the problems' distances in features and in the plane) among those at the
lowest loss; `tries` such starts are still run as a check (pilot_numerical;
is_meta.json: how many stopped short of the optimum, how many on a loss of
precision), and the analytical solution's loss is reported beside. The
optimum fixes the plane only up to T, which leaves the loss and every R^2 as
they are but not the preservation, the Delaunay triangulation or the
footprints: the toolkit's choice is one of tens of random T. So the axes are
fixed: whitened (Z's covariance the identity, the axes Z's principal
directions; a rotation or a common scale changes neither the triangulation
nor thresholds given as shares of the largest distance, so the footprints are
unique), the default, or "preserve": the whitened plane stretched — an angle
and a ratio, all that is left after whitening — to the highest preservation
(a grid, then Nelder-Mead). And the performance rows are weighted sqrt(q/a),
q features and a algorithms (weight "blocks", the default; "none" is the
survey's): with tens of algorithms against a dozen features the performances
would outweigh the features and the plane be fitted almost wholly to them.
The other weighting's preservation and median R^2 are reported beside. The
R^2 of every feature's and every algorithm's linear model: is_projection.csv
(C there of Y itself, the weight taken out).

FOOTPRINT SHARES. Every footprint's area share is given on the main plane and
as a range over six calibrations of the same optimum: the whitened plane, it
halved across the directions 0, 45, 90 and 135 degrees, and the plane of the
highest preservation (is_footprints.csv: area_share_min, area_share_max). At a
strict level a share of a few per cent can move by as much as itself: read it
with its range.

FEATURES FROM A SAMPLE (bbob-biobj; bbob_features = "sample", the default).
The properties put the bbob-biobj problems on a few tens of points: their
function groups and dimension. Instead, 19 features of a sample of
SAMPLE_PER_POP x N points drawn uniformly from the box (N the problem's
population; a seed of its own per problem), in the manner of exploratory
landscape analysis, per objective normalised to [0, 1] on the sample and
averaged over the objectives: nd_share and dom_pairs (non-dominated points,
dominated pairs), rho_mean and rho_min (Spearman between objectives), the
scale ratios log_range_ratio and log_iqr_ratio (on the raw objectives),
skew_mean and kurt_mean, lin_r2 and quad_r2 (adjusted R^2 of f ~ 1 + x and of
f ~ 1 + x + x^2, mean and least), fdc_mean (f against the distance to the
sample's best), lm_share_mean (points better than their 10 nearest
neighbours), nbc_ratio_mean and nb_fit_cor_mean (nearest neighbour against
nearest better), lnd_share (points no neighbour dominates), mo_fdc (Spearman
of the non-domination rank against the distance to the first front) and
nd_xspread (the non-dominated points' spread in x against everyone's). Fixed
transforms (log10 of nd_share, kurt_mean + 3, lm_share_mean + 0.001 and
nbc_ratio_mean; asinh of skew_mean), then PRELIM for measured features:
bounded to the median +- 5 IQR, shifted to a least value of 1, Box-Cox and
z-score; a row equal to an earlier one goes (rho_min is rho_mean at two
objectives). The problems with a reference front keep their properties.

FOOTPRINTS (the 2017 paper's Algorithm 1): of the problems where the algorithm
is good, one of any two closer than delta is dropped; the rest are triangulated
(Delaunay); triangles with a side longer than Delta go; so do those whose
density (problems inside per unit area) is below rho or whose purity (the share
of good problems among those inside) is below pi. delta and Delta are 1 % and
25 % of the largest distance between two problems and pi = 0.75, as in the 2017
and 2018 papers (the authors' 2018 code takes delta and Delta as the 1st and
25th percentiles of the distances between the good problems instead). Both
papers set rho = 10 problems per unit area, an absolute number, in planes of
different scale: 10 / 233.7 of the known region's density in 2017, 10 / 19.88 of
the convex hull's in 2018. PILOT's plane has no scale of its own, so rho is the
2017 share of the known region's density; with Delta so short it never binds —
a triangle holds at least its three vertices, and with sides within Delta its
area is at most (3^0.5 / 4) Delta^2. The known region is the same construction
over all problems of the group, with no density or purity limit; a footprint's
area and density are given as shares of the known region's (2017), its area
also as a share of the convex hull of all problems (2018; Yap, Muñoz &
Smith-Miles, IEEE TEVC 26(6), 2022). GOOD is --cover's criterion at a level
and budget, reached in min_seeds seeds: "igdp" (igdp_norm against the floor)
and "gdp" (gdp_norm) on "front", the hv_h gap to the campaign's best ("hv") on
"bbob". The BEST footprints, of the best algorithm of every problem at a
budget, lose their contradictions as in the paper's Algorithm 2: of two
algorithms, a triangle of one goes when the triangles of the other overlapping
it are together larger than it — both ways at once and in the order of the
names (the authors' code takes the ordered pairs in turn), a tie leaving both.
The 2017 paper also suggests Algorithm 2 between an algorithm's good footprint
and its unsolved problems taken as a second algorithm; that is not done here.

ONE RUN EACH (task 5, 2026-10-01), where the campaign has a deterministic
algorithm (cover.py, DETERMINISTIC ALGORITHMS: DMS, whose ten seeds are one
run): every good footprint once more per seed, good = the level in
that one seed, and the mean over the seeds of the problems good, the triangles
and the two area shares (kind "good_one_run" in is_footprints.csv; no
triangles kept). The gdp levels count only an answer of at least N/2 points
(cover.py, THE ANSWER'S SIZE).
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
CALIBRATIONS = ("whiten", "preserve")     # THE PLANE: the main plane's axes
WEIGHTS = ("blocks", "none")              # the performance rows weighted sqrt(q/a), or not
BBOB_FEATURES = ("sample", "properties")  # FEATURES FROM A SAMPLE
STRETCH, STRETCH_ANGLES = 0.5, (0, 45, 90, 135)   # the family's stretched planes
SAMPLE_PER_POP, SAMPLE_SEED, NEIGHBOURS = 10, 20261010, 10
SAMPLE_FEATURES = ("nd_share", "dom_pairs", "rho_mean", "rho_min", "log_range_ratio",
                   "log_iqr_ratio", "skew_mean", "kurt_mean", "lin_r2_mean", "lin_r2_min",
                   "quad_r2_mean", "quad_r2_min", "fdc_mean", "lm_share_mean", "nbc_ratio_mean",
                   "nb_fit_cor_mean", "lnd_share", "mo_fdc", "nd_xspread")


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
    of (Xbar P_F)(Xbar P_F)^T — the reduced-rank regression of Xbar on F, which
    is problem (D'), Eq. 15, of the 2018 paper. The analytical solution takes
    the eigenvectors of Xbar Xbar^T instead, the same only when Y lies in F's
    row space."""
    import numpy as np
    Xbar = np.vstack([F, Y])
    XP = Xbar @ F.T @ np.linalg.pinv(F @ F.T) @ F
    return float((Xbar ** 2).sum() - np.sort(np.linalg.eigvalsh(XP @ XP.T))[-2:].sum())


def pilot_numerical(F, Y, tries: int = PILOT_TRIES, seed: int = 0) -> dict:
    """PILOT's numerical solution (Algorithm 3, lines 11-25): BFGS from `tries`
    random starts; of the results at the lowest loss, the one of the highest
    topological preservation. The survey takes every BFGS result for a global
    optimum; on stage 3 up to a quarter stopped short of it, by up to 1 %, and
    one of those had the highest preservation. The same dict as pilot_analytic,
    and "short": how many starts stopped above the lowest loss, "imprecise":
    how many ended on BFGS's loss of precision rather than its tolerance."""
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
    results, imprecise = [], 0
    for _ in range(tries):
        res = minimize(loss, rng.uniform(-1.0, 1.0, 4 * q + 2 * a), jac=True, method="BFGS")
        imprecise += not res.success
        results.append(_pilot_result(F, Y, *split(res.x)))
    lowest = min(r["loss"] for r in results)
    at_optimum = [r for r in results if r["loss"] <= lowest + 1e-6 * abs(lowest)]
    best = max(at_optimum, key=lambda r: (r["preservation"] if math.isfinite(r["preservation"])
                                          else -math.inf))
    return dict(best, short=len(results) - len(at_optimum), imprecise=imprecise)


def pilot_exact(F, Y) -> dict:
    """PILOT's optimum itself (pilot_optimum_loss; THE PLANE): Z's rows the two
    leading right singular vectors of Xbar P_F scaled by their singular values,
    each sign fixed by its largest entry; A = Z F^T (F F^T)^+, which gives A F
    = Z, Z lying in F's row space; B and C least squares. The same dict as
    pilot_analytic."""
    import numpy as np
    q = F.shape[0]
    Xbar = np.vstack([F, Y])
    Fp = F.T @ np.linalg.pinv(F @ F.T)
    _, s, Vt = np.linalg.svd(Xbar @ Fp @ F, full_matrices=False)
    Zt = s[:2, None] * Vt[:2]
    for k in range(2):
        if Zt[k, np.argmax(np.abs(Zt[k]))] < 0.0:
            Zt[k] = -Zt[k]
    M = Xbar @ Zt.T @ np.linalg.pinv(Zt @ Zt.T)    # pinv: one feature gives a line
    return _pilot_result(F, Y, Zt @ Fp, M[:q], M[q:])


def transformed(F, Y, proj, T) -> dict:
    """The same plane in other axes, Z -> Z T (T invertible, 2 x 2): A -> T^T A,
    B -> B T^-T, C -> C T^-T; the loss and every R^2 stay, the preservation and
    the footprints do not."""
    import numpy as np
    Ti = np.linalg.inv(T)
    return _pilot_result(F, Y, T.T @ proj["A"], proj["B"] @ Ti.T, proj["C"] @ Ti.T)


def whitening(Z):
    """T with Z T of covariance the identity, its axes Z's principal directions
    in the order of their variance, each sign fixed by its largest entry."""
    import numpy as np
    Zc = Z - Z.mean(axis=0)
    vals, vecs = np.linalg.eigh(Zc.T @ Zc / len(Z))
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vals = np.where(vals > 1e-12 * max(vals[0], 1e-300), vals, 1.0)   # a line stays a line
    T = vecs[:, order] / np.sqrt(vals)
    W = Zc @ T
    for k in range(2):
        if W[np.argmax(np.abs(W[:, k])), k] < 0.0:
            T[:, k] = -T[:, k]
    return T


def stretch(angle: float, s: float):
    """R diag(1, s) R^T, R the rotation by `angle` (radians): the plane
    stretched by s across the direction `angle`."""
    import numpy as np
    c, si = math.cos(angle), math.sin(angle)
    R = np.array([[c, -si], [si, c]])
    return R @ np.diag([1.0, s]) @ R.T


def preserving(F, Zw):
    """The stretch of the whitened plane Zw of the highest topological
    preservation: after whitening only an angle and a ratio are left (a
    rotation and a common scale change nothing), searched on a grid of 36
    angles by 25 ratios 2^-3 .. 2^3, then by Nelder-Mead from the grid's best."""
    import numpy as np
    from scipy.optimize import minimize

    def neg(v):
        r = preservation(F, Zw @ stretch(v[0], 2.0 ** v[1]))
        return -r if math.isfinite(r) else 1.0

    grid = [(a, b) for a in np.linspace(0.0, math.pi, 36, endpoint=False)
            for b in np.linspace(-3.0, 3.0, 25)]
    start = min(grid, key=neg)
    res = minimize(neg, np.array(start), method="Nelder-Mead",
                   options={"xatol": 1e-4, "fatol": 1e-8})
    best = res.x if res.fun <= neg(start) else np.array(start)
    return stretch(best[0], 2.0 ** best[1])


def planes(F, Y, proj, calibration: str = CALIBRATIONS[0]) -> dict:
    """The calibrations of one optimal plane (THE PLANE): {"whitened",
    "stretched 0", "stretched 45", "stretched 90", "stretched 135" (the whitened
    plane halved across those directions, in degrees), "preserving"} -> the
    PILOT dict, and under "main" the one `calibration` names."""
    if calibration not in CALIBRATIONS:
        raise ValueError(f"calibration {calibration!r}: one of {', '.join(CALIBRATIONS)}")
    Tw = whitening(proj["Z"])
    out = {"whitened": transformed(F, Y, proj, Tw)}
    for deg in STRETCH_ANGLES:
        out[f"stretched {deg}"] = transformed(F, Y, proj, Tw @ stretch(math.radians(deg),
                                                                        STRETCH))
    out["preserving"] = transformed(F, Y, proj, Tw @ preserving(F, out["whitened"]["Z"]))
    out["main"] = out["whitened" if calibration == "whiten" else "preserving"]
    return out


# ── features from a sample (bbob-biobj) ────────────────────────────────────


def sample_points(problem: str, seed: int = SAMPLE_SEED) -> tuple:
    """(X, F): SAMPLE_PER_POP times the problem's population of points drawn
    uniformly from its box, X in the unit cube, F the raw objectives (a noisy
    copy's noise-free ones); rows with a value not finite dropped."""
    import zlib
    import numpy as np
    from ..benchmarks import get as bench_get
    p = bench_get(problem)
    lo = np.array([b[0] for b in p.bounds], float)
    hi = np.array([b[1] for b in p.bounds], float)
    rng = np.random.default_rng([seed, zlib.crc32(problem.encode())])
    U = rng.random((SAMPLE_PER_POP * p.pop_size, p.n_vars))
    f = p.true_objectives or p.evaluate
    F = np.array([f(list(lo + u * (hi - lo))) for u in U], float)
    ok = np.all(np.isfinite(F), axis=1)
    return U[ok], F[ok]


def sample_features(X, F) -> dict:
    """The features of a sample (FEATURES FROM A SAMPLE), before their fixed
    transforms: X in the unit cube, F raw. A correlation with a constant input
    (every point non-dominated: mo_fdc) is NaN, without SciPy's warning."""
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*constant.*")
        return _sample_features(X, F)


def _sample_features(X, F) -> dict:
    import numpy as np
    from scipy.spatial import cKDTree
    from scipy.spatial.distance import cdist
    from scipy.stats import kurtosis, pearsonr, skew, spearmanr
    n, m = F.shape
    span = np.ptp(F, axis=0)
    G = (F - F.min(axis=0)) / np.where(span > 0.0, span, 1.0)
    dom = np.all(G[:, None, :] <= G[None, :, :], axis=2) & np.any(G[:, None, :] < G[None, :, :],
                                                                  axis=2)
    nd = ~dom.any(axis=0)
    rank = np.zeros(n, int)
    left = np.ones(n, bool)
    r = 0
    while left.any():
        r += 1
        idx = np.flatnonzero(left)
        first = idx[~dom[np.ix_(idx, idx)].any(axis=0)]
        rank[first] = r
        left[first] = False
    rho = [spearmanr(G[:, a], G[:, b])[0] for a in range(m) for b in range(a + 1, m)]
    q75, q25 = np.percentile(F, 75, axis=0), np.percentile(F, 25, axis=0)
    iqr = q75 - q25

    def ratio(v):
        return math.log10(v.max() / v.min()) if v.min() > 0.0 else math.nan

    def adj_r2(design, y):
        A = np.column_stack([np.ones(n), design])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        sse = float(((y - A @ coef) ** 2).sum())
        sst = float(((y - y.mean()) ** 2).sum())
        k = design.shape[1]
        return 1.0 - (sse / sst) * (n - 1) / (n - k - 1) if sst > 0.0 else math.nan

    tree = cKDTree(X)
    dist, nb = tree.query(X, k=NEIGHBOURS + 1)
    dist, nb = dist[:, 1:], nb[:, 1:]
    DX = cdist(X, X)
    lin, quad, fdc, lm, nbc, nbcor = [], [], [], [], [], []
    for j in range(m):
        y = G[:, j]
        lin.append(adj_r2(X, y))
        quad.append(adj_r2(np.column_stack([X, X * X]), y))
        fdc.append(pearsonr(y, DX[np.argmin(y)])[0])
        lm.append(float(np.mean(np.all(y[:, None] < y[nb], axis=1))))
        better = y[None, :] < y[:, None]                     # [i, k]: k better than i
        has = better.any(axis=1)
        nbd = np.where(better, DX, np.inf).min(axis=1)[has]
        nnd = dist[has, 0]
        nbc.append(float(nnd.mean() / nbd.mean()))
        nbcor.append(pearsonr(nbd / np.where(nnd > 0.0, nnd, 1e-300), y[has])[0])
    lnd = float(np.mean(~np.any(dom[nb, np.arange(n)[:, None]], axis=1)))
    ndi = np.flatnonzero(nd)
    to_first = DX[:, ndi].min(axis=1)
    pairs = lambda idx: DX[np.ix_(idx, idx)][np.triu_indices(len(idx), 1)]  # noqa: E731
    return {"nd_share": float(nd.mean()),
            "dom_pairs": float(dom.sum()) / (n * (n - 1) / 2.0),
            "rho_mean": float(np.mean(rho)), "rho_min": float(np.min(rho)),
            "log_range_ratio": ratio(span), "log_iqr_ratio": ratio(iqr),
            "skew_mean": float(np.mean(skew(G, axis=0))),
            "kurt_mean": float(np.mean(kurtosis(G, axis=0))),
            "lin_r2_mean": float(np.mean(lin)), "lin_r2_min": float(np.min(lin)),
            "quad_r2_mean": float(np.mean(quad)), "quad_r2_min": float(np.min(quad)),
            "fdc_mean": float(np.mean(fdc)), "lm_share_mean": float(np.mean(lm)),
            "nbc_ratio_mean": float(np.mean(nbc)), "nb_fit_cor_mean": float(np.mean(nbcor)),
            "lnd_share": lnd,
            "mo_fdc": float(spearmanr(rank, to_first)[0]),
            "nd_xspread": (float(pairs(ndi).mean() / pairs(np.arange(n)).mean())
                           if len(ndi) >= 2 else 0.0)}


# the fixed transforms (chosen before any fit): the rest as they are
SAMPLE_TRANSFORMS = {"nd_share": math.log10, "skew_mean": math.asinh,
                     "kurt_mean": lambda v: math.log10(v + 3.0),
                     "lm_share_mean": lambda v: math.log10(v + 0.001),
                     "nbc_ratio_mean": math.log10}


def sample_feature_rows(problems: list, seed: int = SAMPLE_SEED) -> list:
    """One row per SAMPLE_FEATURES, one value per problem, after the fixed
    transforms (SAMPLE_TRANSFORMS); NaN where a value is undefined."""
    rows = [[] for _ in SAMPLE_FEATURES]
    for p in problems:
        got = sample_features(*sample_points(p, seed))
        for r, name in zip(rows, SAMPLE_FEATURES):
            v = got[name]
            if name in SAMPLE_TRANSFORMS and math.isfinite(v):
                try:
                    v = SAMPLE_TRANSFORMS[name](v)
                except ValueError:
                    v = math.nan
            r.append(float(v))
    return rows


def prelim_features(rows) -> tuple:
    """PRELIM for measured features (the survey's Algorithm 1), per row: NaN
    to the median, bounded to the median +- 5 IQR, shifted to a least value of
    1, Box-Cox (lambda by maximum likelihood), z-score. (kept row indices,
    rows): a constant row, and a row equal to an earlier one (rho_min at two
    objectives), dropped."""
    import numpy as np
    from scipy.stats import boxcox
    keep, out, seen = [], [], []
    for i, r in enumerate(rows):
        x = np.asarray(r, float)
        if not np.isfinite(x).any():
            continue
        x[~np.isfinite(x)] = np.median(x[np.isfinite(x)])
        if any(np.array_equal(x, s) for s in seen):
            continue
        seen.append(x.copy())
        med = np.median(x)
        iqr = np.subtract(*np.percentile(x, [75, 25]))
        if iqr > 0.0:
            x = np.clip(x, med - 5.0 * iqr, med + 5.0 * iqr)
        if np.ptp(x) == 0.0:
            continue
        t = boxcox(x - x.min() + 1.0)[0]
        if not t.std() > 0.0:
            continue
        keep.append(i)
        out.append((t - t.mean()) / t.std())
    return keep, np.asarray(out)


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
        seed: int = 0, calibration: str = CALIBRATIONS[0], weight: str = WEIGHTS[0],
        bbob_features: str = BBOB_FEATURES[0]) -> str:
    """The whole analysis: text for the terminal; CSV and JSON in out_dir.
    The plane is PILOT's optimum (pilot_exact) under `calibration` and
    `weight`; tries > 0 BFGS starts check that it is the optimum (0: none);
    `bbob_features` "sample" or "properties" (THE PLANE, FEATURES)."""
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
    from scipy.spatial import ConvexHull, QhullError
    from scipy.spatial.distance import pdist
    if weight not in WEIGHTS or bbob_features not in BBOB_FEATURES \
            or calibration not in CALIBRATIONS:
        raise ValueError(f"calibration {calibration!r} (one of {CALIBRATIONS}), weight "
                         f"{weight!r} (one of {WEIGHTS}), bbob_features {bbob_features!r} "
                         f"(one of {BBOB_FEATURES})")
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
    fixed = V.deterministic(data["values"])
    seeds = sorted({k[2] for k in data["values"]})
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
    axes = {"whiten": "whitened", "preserve": "of the highest preservation"}[calibration]
    lines = [f"instance space ({scenario}): analysis code {commit}; CSV in {out_dir}; the "
             f"plane PILOT's optimum, its axes {axes}, weight {weight}"]
    if fixed:
        lines.append(f"  deterministic, every seed the same run on every problem: "
                     f"{', '.join(fixed)}; footprints 'good_one_run': good in one seed, the "
                     f"mean over the seeds")
    for group, problems in groups.items():
        if len(problems) < 4:
            if problems:
                lines.append(f"  {group}: {len(problems)} problems, too few for a plane")
            continue
        metric = "igdp_norm" if group == "front" else "hv_h"
        higher = group == "bbob"
        raw = feature_rows(problems)
        sampled = group == "bbob" and bbob_features == "sample"
        raw_sample = sample_feature_rows(problems) if sampled else None
        rel, best = relative_performance(medians(data, metric, full, set(problems)), problems,
                                         algorithms, higher)
        y_rows, Y = prelim_performance(rel)
        if sampled:
            f_rows, F = prelim_features(raw_sample)
            f_names = list(SAMPLE_FEATURES)
        else:
            f_rows, F = zscore(raw)
            f_names = names
        sel, R = sifted(F, Y)
        none_sifted = not sel
        if none_sifted:                     # no feature tracks a performance: all of them
            sel = list(range(F.shape[0]))
        F = F[sel]
        used = [f_names[f_rows[i]] for i in sel]
        q, a = F.shape[0], Y.shape[0]
        w = math.sqrt(q / a) if weight == "blocks" else 1.0
        Yw = w * Y
        exact = pilot_exact(F, Yw)
        optimum = pilot_optimum_loss(F, Yw)
        analytic = pilot_analytic(F, Yw)
        check = pilot_numerical(F, Yw, tries, seed) if tries > 0 else None
        family = planes(F, Yw, exact, calibration)
        proj = family["main"]
        # the other weighting, for comparison (the survey's own: none)
        w2 = 1.0 if weight == "blocks" else math.sqrt(q / a)
        other = planes(F, w2 * Y, pilot_exact(F, w2 * Y), calibration)["main"]
        Z = proj["Z"]

        def geometry(Zp):
            """(dmax, known region, rho, convex hull area) of one plane."""
            dmax_p = float(pdist(Zp).max())
            known_p = footprint(Zp, np.ones(len(Zp), bool), dmax_p)
            dens = known_p["inside"] / known_p["area"] if known_p["area"] > 0.0 else 0.0
            try:                                    # the 2018 paper's and Yap et al.'s reference
                hull_p = float(ConvexHull(Zp).volume)
            except (QhullError, ValueError):
                hull_p = 0.0
            return dmax_p, known_p, DENSITY_SHARE * dens, hull_p, dens

        geo = {k: geometry(v["Z"]) for k, v in family.items() if k != "main"}
        dmax, known, rho, hull, density = geometry(Z)

        def shares(good):
            """The footprint's share of the known region on every plane of the
            family (FOOTPRINT SHARES): (min, max)."""
            vals = []
            for k, (dm, kn, rh, _, _) in geo.items():
                if kn["area"] > 0.0:
                    vals.append(footprint(family[k]["Z"], good, dm, rh, PURITY)["area"]
                                / kn["area"])
            return (min(vals), max(vals)) if vals else (0.0, 0.0)

        for j, p in enumerate(problems):
            coords.append([group, p, f"{Z[j, 0]:.6g}", f"{Z[j, 1]:.6g}", best[j] or ""]
                          + [f"{r[j]:.6g}" for r in raw]
                          + ([f"{r[j]:.6g}" for r in raw_sample] if sampled
                             else [""] * len(SAMPLE_FEATURES)))
            for i, alg in enumerate(algorithms):
                performance.append([group, p, alg, "" if math.isnan(rel[i][j])
                                    else f"{rel[i][j]:.6g}"])
        for i, f in enumerate(f_rows):               # R's rows are f_rows', F's sel's
            s = sel.index(i) if i in sel else None
            projection.append([group, f_names[f], "feature", s is not None,
                               f"{np.abs(R[i]).max():.3f}"]
                              + ([f"{proj['A'][0, s]:.6g}", f"{proj['A'][1, s]:.6g}",
                                  f"{proj['B'][s, 0]:.6g}", f"{proj['B'][s, 1]:.6g}",
                                  f"{proj['r2_features'][s]:.4f}"] if s is not None
                                 else ["", "", "", "", ""]))
        for i, yi in enumerate(y_rows):               # C of Y itself: C / w
            projection.append([group, algorithms[yi], "performance", True,
                               f"{np.abs(R[:, i]).max():.3f}", "", "",
                               f"{proj['C'][i, 0] / w:.6g}", f"{proj['C'][i, 1] / w:.6g}",
                               f"{proj['r2_performance'][i]:.4f}"])

        shapes = {}

        def record(kind, crit, k, b, alg, fp, good, span=None):
            fp_rows.append([group, kind, crit, "" if k is None else k,
                            "" if k is None else level_text(crit, k), b, alg,
                            int(np.asarray(good).sum()), len(fp["triangles"]),
                            f"{fp['area'] / known['area']:.6g}",
                            f"{(fp['area'] / hull) if hull else 0.0:.6g}",
                            f"{(fp['inside'] / fp['area'] / density) if fp['area'] else 0.0:.6g}",
                            f"{(fp['good_inside'] / fp['inside']) if fp['inside'] else 0.0:.6g}"]
                           + ([f"{span[0]:.6g}", f"{span[1]:.6g}"] if span else ["", ""]))
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
                               good, shares(good))
                    per_seed = ([V.successes({p: hit[p] for p in problems if p in hit}, 1,
                                             seeds=[s]) for s in seeds] if fixed else [])
                    for alg in (algorithms if fixed else ()):     # ONE RUN EACH
                        figs = []
                        for cov_s in per_seed:
                            good = np.array([p in cov_s.get(alg, ()) for p in problems])
                            fp = footprint(Z, good, dmax, rho, PURITY)
                            figs.append((int(good.sum()), len(fp["triangles"]),
                                         fp["area"] / known["area"],
                                         (fp["area"] / hull) if hull else 0.0))
                        mean = [sum(f[i] for f in figs) / len(figs) for i in range(4)]
                        fp_rows.append([group, "good_one_run", crit, k, level_text(crit, k), b,
                                        alg] + [f"{v:.6g}" for v in mean] + ["", "", "", ""])

        def best_footprints(Zp, dm, rh, best_b):
            good_of = {a_: np.array([x == a_ for x in best_b]) for a_ in algorithms}
            fps = {a_: footprint(Zp, good_of[a_], dm, rh, PURITY) for a_ in algorithms}
            some = [a_ for a_ in algorithms if fps[a_]["triangles"]]
            for i, a_ in enumerate(some):
                for c in some[i + 1:]:
                    ta, tc = remove_contradictions(Zp, fps[a_], fps[c])
                    fps[a_], fps[c] = summarize(Zp, good_of[a_], ta), summarize(Zp, good_of[c], tc)
            return good_of, fps

        for b in (budgets if drawn else ()):
            _, best_b = relative_performance(medians(data, metric, b, set(problems)), problems,
                                             algorithms, higher)
            good_of, fps = best_footprints(Z, dmax, rho, best_b)
            spans = {a_: [] for a_ in algorithms}
            for k_, (dm, kn, rh, _, _) in geo.items():
                if kn["area"] > 0.0:
                    _, f_k = best_footprints(family[k_]["Z"], dm, rh, best_b)
                    for a_ in algorithms:
                        spans[a_].append(f_k[a_]["area"] / kn["area"])
            for a_ in algorithms:
                record("best", "median", None, b, a_, fps[a_], good_of[a_],
                       (min(spans[a_]), max(spans[a_])) if spans[a_] else None)
        triangles[group] = {"known": [list(t) for t in known["triangles"]], "footprints": shapes}

        points = len({tuple(r[j] for r in (raw_sample if sampled else raw))
                      for j in range(len(problems))})
        positions = len({(round(float(z[0]), 6), round(float(z[1]), 6)) for z in Z})

        def med(v):
            return float(np.median(v))

        meta_groups[group] = {
            "problems": len(problems), "points": points, "positions": positions,
            "metric": metric, "features_kind": "sample" if sampled else "properties",
            "features": used,
            "features_constant": [n for i, n in enumerate(f_names) if i not in f_rows],
            "features_sifted_out": [f_names[f] for i, f in enumerate(f_rows) if i not in sel],
            "sifted_none": none_sifted,
            "algorithms_constant": [a_ for i, a_ in enumerate(algorithms) if i not in y_rows],
            "pilot": {"method": "exact", "calibration": calibration, "weight": weight,
                      "w": w, "loss": proj["loss"], "optimum_loss": optimum,
                      "preservation": proj["preservation"],
                      "r2_features_median": med(proj["r2_features"]),
                      "r2_performance_median": med(proj["r2_performance"]),
                      "analytical_loss": analytic["loss"],
                      "bfgs": None if check is None else {
                          "tries": tries, "seed": seed, "loss": check["loss"],
                          "starts_short": check["short"],
                          "starts_imprecise": check["imprecise"]}},
            "planes": {k_: v["preservation"] for k_, v in family.items() if k_ != "main"},
            "other_weight": {"weight": "none" if weight == "blocks" else "blocks", "w": w2,
                             "preservation": other["preservation"],
                             "r2_features_median": med(other["r2_features"]),
                             "r2_performance_median": med(other["r2_performance"])},
            "known_area": known["area"], "known_density": density, "rho": rho,
            "convex_hull_area": hull, "max_distance": dmax}
        r2f = sorted(zip(used, proj["r2_features"]), key=lambda t: -t[1])
        lines.append(f"  {group}: {len(problems)} problems ({points} feature vectors, "
                     f"{positions} positions), {len(used)} features "
                     f"({'from a sample' if sampled else 'properties'}"
                     f"{'; SIFTED kept none, so all' if none_sifted else ''}), {len(y_rows)} "
                     f"algorithms; preservation {proj['preservation']:.3f} (planes of the same "
                     f"optimum {min(meta_groups[group]['planes'].values()):.3f}-"
                     f"{max(meta_groups[group]['planes'].values()):.3f}; weight "
                     f"{meta_groups[group]['other_weight']['weight']}: "
                     f"{other['preservation']:.3f}), loss {proj['loss']:.6g} (optimum "
                     f"{optimum:.6g}, analytical {analytic['loss']:.6g})"
                     + (f"; of {tries} BFGS starts {check['short']} stopped short of the "
                        f"optimum, {check['imprecise']} on a loss of precision" if check else ""))
        lines.append("    features best predicted from the plane (R^2): "
                     + ", ".join(f"{n} {r:.2f}" for n, r in r2f[:6]))
        lines.append(f"    performance R^2: median {med(proj['r2_performance']):.2f}, best "
                     f"{float(max(proj['r2_performance'])):.2f} (weight "
                     f"{meta_groups[group]['other_weight']['weight']}: median "
                     f"{med(other['r2_performance']):.2f})")
        if not drawn:
            lines.append("    no known region: no triangle with every side within Delta "
                         f"({BIG_DELTA:.0%} of the largest distance); no footprints")
            continue
        for crit in criteria[group]:
            for k in (0, len(taus) - 1):
                for kind in ("good", "good_one_run") if fixed else ("good",):
                    order = sorted(((float(r[9]), r[6], r[13], r[14]) for r in fp_rows
                                    if r[0] == group and r[1] == kind and r[2] == crit
                                    and r[3] == k and r[5] == full), reverse=True)

                    def show(v, a_, lo, hi):
                        return f"{a_} {v:.0%}" + (f" ({float(lo):.0%}-{float(hi):.0%})"
                                                  if lo != "" else "")
                    places = "".join(f"; {a_} {v:.0%} (place {i + 1})"
                                     for i, (v, a_, _, _) in enumerate(order) if a_ in fixed)
                    lines.append((f"    largest footprints, {level_text(crit, k)} at {full}: "
                                  if kind == "good" else "      one run each: ")
                                 + ", ".join(show(*o) for o in order[:5]) + places)
        top = sorted(((float(r[9]), r[6], r[13], r[14]) for r in fp_rows if r[0] == group
                      and r[1] == "best" and r[5] == full), reverse=True)[:5]
        lines.append(f"    largest best footprints at {full}: "
                     + (", ".join(f"{a_} {v:.0%} ({float(lo):.0%}-{float(hi):.0%})"
                                  for v, a_, lo, hi in top if v > 0) or "none"))

    heads = {
        "is_coordinates.csv": ["group", "problem", "z1", "z2", "best_algorithm"] + names
                              + list(SAMPLE_FEATURES),
        "is_performance.csv": ["group", "problem", "algorithm", "relative"],
        "is_projection.csv": ["group", "name", "kind", "used", "max_abs_correlation", "a1",
                              "a2", "model1", "model2", "r2"],
        "is_footprints.csv": ["group", "kind", "criterion", "level", "level_text", "budget",
                              "algorithm", "problems_good", "triangles", "area_share",
                              "area_share_hull", "density_share", "purity", "area_share_min",
                              "area_share_max"],
    }
    for name, rows_ in (("is_coordinates.csv", coords), ("is_performance.csv", performance),
                        ("is_projection.csv", projection), ("is_footprints.csv", fp_rows)):
        with (out_dir / name).open("w", newline="", encoding="utf-8") as fh:
            w_ = csv.writer(fh)
            w_.writerow(heads[name])
            w_.writerows(rows_)
    (out_dir / "is_footprints.json").write_text(json.dumps(triangles, separators=(",", ":")),
                                                encoding="utf-8")
    meta = {"analysis": {"mootation": __version__, "commit": rev.get("commit"),
                         "dirty": rev.get("dirty"), "source": rev.get("source")},
            "results": str(root), "scenario": scenario,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "budgets": list(budgets),
            "full_budget": full, "taus": list(taus), "floor_taus": list(floor_taus),
            "min_seeds": min_seeds, "deterministic": fixed,
            "calibration": calibration, "weight": weight, "bbob_features": bbob_features,
            "sample": {"per_population": SAMPLE_PER_POP, "seed": SAMPLE_SEED,
                       "neighbours": NEIGHBOURS},
            "family": ["whitened"] + [f"stretched {d}" for d in STRETCH_ANGLES] + ["preserving"],
            "delta": DELTA, "big_delta": BIG_DELTA, "purity": PURITY,
            "density_share": DENSITY_SHARE, "sifted": [SIFTED_CORRELATION, SIFTED_P],
            "excluded": prep["excluded"], "groups": meta_groups,
            "triangles": "is_footprints.json: per group the known region's and every non-empty "
                         "footprint's triangles, as indices into the group's rows of "
                         "is_coordinates.csv; keys kind|criterion|level|budget|algorithm"}
    (out_dir / "is_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    text = "\n".join(lines)
    (out_dir / "is_report.txt").write_text(text + "\n", encoding="utf-8")
    return text

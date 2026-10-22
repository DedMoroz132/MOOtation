# SPDX-License-Identifier: Apache-2.0
"""The run archive, and the subset selection shared by every place that thins to N.

GridArchive keeps every nondominated point a run evaluates, with points that
are too close to each other pruned; `dss_order` selects N of them, and of any
other point set, the same way everywhere (reference fronts, random search, the
"archive" reading of a run).

`dss_order` is distance-based subset selection (DSS): start from the best point
of every objective, then repeatedly add the candidate that the selected set
covers WORST, until the set is full. "Covers" is measured with the IGD+
distance, d+(c, s) = ||max(s − c, 0)||, the amount by which a selected point s
is worse than candidate c (c in the place of IGD+'s reference point); a
candidate dominated by (or equal to) a selected point is covered completely.
Greedy maximin on that distance picks the point contributing most to IGD+ of
the selection against the whole set, without optimizing IGD+ (a mean) or the
hypervolume themselves, where selecting by one indicator would flatter that
indicator afterwards. It does lower, greedily, the LARGEST d+ from the set to
the selection, a norm analogue of the additive epsilon.

The order is INCREMENTAL: the first n points of dss_order(F) are exactly the
DSS selection of size n from F. A set stored in this order is thinned by
slicing.

Everything is computed in normalized coordinates, (f − ideal)/(nadir − ideal),
so that no objective dominates the distance by its units alone; pass the frame
explicitly where one is known (a registry problem), otherwise the set's own
minimum and maximum are used.

ALMOST-DOMINATED POINTS GO LAST (2026-09-27). A point p is set aside when some
other point q is better than p by more than DRS_LOSS in some objective while p
is better than q by at most DRS_GAIN in every objective — in normalized
coordinates, a trade-off worse than 100 to 1 over a tenth of the range. Such
points (dominance-resistant: the best value of one objective, bought with a
much worse value of another) are exactly what the per-objective seeds picked
first, and far from everything else, the max-min steps picked them next.
Without a given frame the set's minimum and maximum are taken again from the
points kept, and the test repeated until nothing more is set aside, so an
outlier does not stretch the frame the rest is measured in. The points set
aside follow all the others in the order, in the same greedy order among
themselves; a selection of k no larger than the rest contains none of them.

NO APPROXIMATION GUARANTEE. No bound holds for the covering radius DSS leaves
— the largest d+ from a point of the set to the selection — against the best
achievable with the same k: d+ is not symmetric and the seeds are forced, so
the argument that bounds greedy max-min selection under a metric does not
apply. It is a spread heuristic, not an optimal subset.

Sources: DSS is Singh, Bhattacharjee & Ray, "Distance based subset selection
for benchmarking in evolutionary multi/many-objective optimization" (IEEE
TEVC, 2019), Algo. 1. The procedure above is the one specified for MOOtation
on 2026-09-22 and departs from Algo. 1 three times: the seeds are the M
per-objective minima (the rule of Tanabe, Ishibuchi & Oyama 2017, which Singh
et al. replace by ONE extreme point); the distance is d+ (Ishibuchi, Masuda,
Tanigaki & Nojima, EMO 2015, Eq. 18; in DSS after Chen, Ishibuchi & Shang
2020), not the Euclidean; the almost-dominated points go
last (Singh et al. filter nothing: they select from reference sets). Singh et
al.'s results on the spacing of the selected points assume the Euclidean
distance and are not claimed here. include/mootation/dss.hpp is the same
selection in C++, index for index.

SELECTING FOR IGD+ AGAINST THE ARCHIVE ITSELF ([campaign] answer_select =
"igdp_self", task 6 of 2026-10-09, item 13). DSS chooses for coverage, not
for IGD+; the best N points of the same archive under IGD+ against the
reference (the oracle, campaign --oracle) bound what any choice from it can
reach. Without a reference, igdp_self_order takes the archive thinned by DSS to
SELF_REFERENCE = 1 000 points as one (the size of the reference fronts), and
from the whole archive greedily the N points of the smallest IGD+ against it:
first the point of the smallest mean d+, then each the one that lowers the
mean of the reference's smallest d+ most, ties to the lowest index
(greedy_igdp) — no swaps after. DSS stays the default.

THE COMPILED HALF (task 10 of 2026-10-18). almost_dominated, set_aside,
dss_order, igdp_self_order, greedy_igdp_rows and the archive's points are
computed in C++ when the extension is built (include/mootation/
archive_select.hpp and grid_archive.hpp), by the same expressions and in
NumPy's order of summation: the same values, the same indices, the same
points. The NumPy code here stays as the reference the tests compare with (the
functions named *_numpy and PyGridArchive) and runs where the extension is not
built.
"""

from __future__ import annotations

import numpy as np


def _compiled(name: str):
    """The extension's function or class `name`, or None where the extension
    is not built (or is older than this module)."""
    try:
        from .. import _core
    except ImportError:                              # pragma: no cover - no extension
        return None
    return getattr(_core, name, None)


def _frame(v):
    return None if v is None else np.asarray(v, float)


def normalized(F, ideal=None, nadir=None):
    """(F − ideal)/(nadir − ideal), a zero range counting as 1."""
    F = np.asarray(F, float)
    lo = F.min(0) if ideal is None else np.asarray(ideal, float)
    hi = F.max(0) if nadir is None else np.asarray(nadir, float)
    span = hi - lo
    span = np.where(span > 0.0, span, 1.0)
    return (F - lo) / span


# Almost-dominated points (module docstring): p is set aside when some q beats
# it by more than DRS_LOSS in an objective while p beats q by at most DRS_GAIN
# in every objective, in normalized coordinates.
DRS_GAIN = 1e-3
DRS_LOSS = 0.1


def almost_dominated(G, rows=None) -> np.ndarray:
    """Which rows p of the normalized set G another row q almost dominates:
    q better than p by more than DRS_LOSS somewhere, p better than q by at
    most DRS_GAIN everywhere. All rows, or only `rows` (indices into G),
    compared with every row, a block of rows at a time."""
    G = np.asarray(G, float)
    fn = _compiled("almost_dominated")
    if fn is not None and G.ndim == 2 and G.shape[1]:
        return fn(G, None if rows is None else np.asarray(rows, dtype=np.intp).tolist())
    return _almost_dominated_numpy(G, rows)


def _almost_dominated_numpy(G, rows=None) -> np.ndarray:
    """almost_dominated() in NumPy: the reference."""
    G = np.asarray(G, float)
    rows = np.arange(len(G)) if rows is None else np.asarray(rows, dtype=np.intp)
    out = np.zeros(len(rows), bool)
    step = max(1, 500_000 // max(1, G.size))            # about 4 MB of differences a block
    for a in range(0, len(rows), step):
        r = rows[a:a + step]
        diff = G[r][:, None, :] - G[None, :, :]         # p − q; > 0 where p is worse
        # p better than q by at most DRS_GAIN everywhere (-diff <= DRS_GAIN), and
        # only for those q, worse by more than DRS_LOSS somewhere
        near = np.all(diff >= -DRS_GAIN, axis=2)
        near[np.arange(len(r)), r] = False
        for b in range(len(r)):
            q = np.flatnonzero(near[b])
            if len(q):
                out[a + b] = bool((diff[b, q].max(axis=1) > DRS_LOSS).any())
    return out


def set_aside(F, ideal=None, nadir=None):
    """(G, kept): F normalized, and the points DSS seeds from and picks first.

    A missing ideal or nadir is the minimum or maximum of the points kept, taken
    again after every round that sets points aside, until a round sets none
    aside. A round never sets aside every point left.
    """
    F = np.asarray(F, float)
    fn = _compiled("set_aside")
    if fn is not None and F.ndim == 2 and len(F) and F.shape[1]:
        return fn(F, _frame(ideal), _frame(nadir))
    return _set_aside_numpy(F, ideal, nadir)


def _set_aside_numpy(F, ideal=None, nadir=None):
    """set_aside() in NumPy: the reference."""
    F = np.asarray(F, float)
    kept = np.ones(len(F), bool)
    if not len(F):
        return F.copy(), kept
    while True:
        lo = F[kept].min(0) if ideal is None else ideal
        hi = F[kept].max(0) if nadir is None else nadir
        G = normalized(F, lo, hi)
        idx = np.flatnonzero(kept)
        drop = _almost_dominated_numpy(G[idx])
        if not drop.any() or drop.all():
            return G, kept
        kept[idx[drop]] = False


def dss_order(F, k=None, ideal=None, nadir=None) -> np.ndarray:
    """Indices of F in DSS order (all of them, or the first k).

    The points set_aside() keeps come first: the best of every objective, then
    the greedy max-min steps; the points it sets aside follow in the same
    greedy order. Ties are broken by the lowest index, so the order is
    deterministic. A set with nothing to set aside is ordered exactly as
    before the rule existed.
    """
    F = np.asarray(F, float)
    n = len(F)
    k = n if k is None else min(int(k), n)
    if k <= 0:
        return np.empty(0, dtype=np.intp)
    fn = _compiled("dss_select")
    if fn is not None and F.ndim == 2 and F.shape[1]:
        return fn(F, k, _frame(ideal), _frame(nadir))
    return _dss_order_numpy(F, k, ideal, nadir)


def _dss_order_numpy(F, k=None, ideal=None, nadir=None) -> np.ndarray:
    """dss_order() in NumPy: the reference."""
    F = np.asarray(F, float)
    n = len(F)
    k = n if k is None else min(int(k), n)
    if k <= 0:
        return np.empty(0, dtype=np.intp)
    # state: 1 kept, -1 set aside, 0 not known yet
    if ideal is not None and nadir is not None:
        # In a given frame whether a point is set aside does not depend on the
        # others' fate, so it is asked only of the points DSS is about to take:
        # the same order as asking all first, at k checks instead of n (a
        # 12 758-point archive at five objectives: 13 s -> well under one).
        G = normalized(F, ideal, nadir)
        state = np.zeros(n, np.int8)
    else:
        G, kept = _set_aside_numpy(F, ideal, nadir)
        state = np.where(kept, 1, -1).astype(np.int8)

    def is_kept(i):
        if state[i] == 0:
            state[i] = -1 if _almost_dominated_numpy(G, [i])[0] else 1
        return state[i] == 1

    order = []
    taken = np.zeros(n, bool)
    cover = np.full(n, np.inf)                 # min over selected of d+(s, c)

    def take(i):
        taken[i] = True
        order.append(int(i))
        d = np.sqrt(np.sum(np.maximum(G[i] - G, 0.0) ** 2, axis=1))
        np.minimum(cover, d, out=cover)

    for j in range(G.shape[1]):               # the best kept point of every objective
        if len(order) >= k:
            break
        for i in np.lexsort((np.arange(n), G[:, j])):   # by value, ties by index
            if is_kept(int(i)):
                if not taken[i]:
                    take(int(i))
                break
    while len(order) < k:
        pool = ~taken & (state >= 0)           # not taken, not known to be set aside
        if not pool.any():
            pool = ~taken                      # then the points set aside
        i = int(np.argmax(np.where(pool, cover, -1.0)))
        if taken[i]:
            break
        if state[i] >= 0 and not is_kept(i):
            continue                           # set aside now: look again
        take(i)
    return np.asarray(order, dtype=np.intp)


SELF_REFERENCE = 1000      # igdp_self: the archive thinned by DSS to this many points
ANSWER_SELECT = ("dss", "igdp_self")


def dplus_matrix(C, R) -> np.ndarray:
    """D[c, z] = d+(z, c) = ||max(C_c − R_z, 0)||, how far candidate c leaves
    reference point z uncovered, a block of candidates at a time."""
    C, R = np.asarray(C, float), np.asarray(R, float)
    D = np.empty((len(C), len(R)))
    step = max(1, 2_000_000 // max(1, len(R) * C.shape[1]))
    for a in range(0, len(C), step):
        D[a:a + step] = np.sqrt((np.maximum(C[a:a + step, None, :] - R[None, :, :], 0.0) ** 2)
                                .sum(axis=2))
    return D


def greedy_igdp(D, k: int) -> np.ndarray:
    """Indices of k rows of D (candidates x reference points) chosen greedily
    for the smallest IGD+ against the reference (module docstring): the row of
    the smallest mean first, then each the row whose gain — the mean over the
    reference of max(current − d+, 0) — is largest, ties to the lowest index.
    Lazily: a row's gain only shrinks as the selection grows, so a stale gain
    bounds the fresh one and only the top of the heap is recomputed — the same
    rows as recomputing every gain each step."""
    import heapq
    D = np.asarray(D, float)
    n = len(D)
    k = min(int(k), n)
    if k <= 0:
        return np.empty(0, dtype=np.intp)
    first = int(np.argmin(D.mean(axis=1)))
    chosen, cur = [first], D[first].copy()
    heap = [(-float(np.mean(np.maximum(cur - D[c], 0.0))), c) for c in range(n) if c != first]
    heapq.heapify(heap)
    while len(chosen) < k and heap:
        _, c = heapq.heappop(heap)
        fresh = (-float(np.mean(np.maximum(cur - D[c], 0.0))), c)
        if not heap or fresh <= heap[0]:
            chosen.append(c)
            np.minimum(cur, D[c], out=cur)
        else:
            heapq.heappush(heap, fresh)
    return np.asarray(chosen, dtype=np.intp)


def greedy_igdp_rows(C, R, k: int) -> np.ndarray:
    """greedy_igdp(dplus_matrix(C, R), k), the matrix's rows computed as they
    are needed where the extension is built (an archive times a reference of a
    thousand points does not always fit)."""
    C, R = np.asarray(C, float), np.asarray(R, float)
    fn = _compiled("greedy_igdp")
    if fn is not None and C.ndim == 2 and R.ndim == 2 and len(R) and C.shape[1]:
        return fn(C, R, int(max(k, 0)))
    return greedy_igdp(dplus_matrix(C, R), k)


def igdp_self_order(F, k: int, ideal=None, nadir=None) -> np.ndarray:
    """Indices of k points of F chosen for IGD+ against F thinned by DSS to
    SELF_REFERENCE points (module docstring), in the frame DSS uses: the given
    one, or that of the points set_aside() keeps. All of F when k >= len(F)."""
    F = np.asarray(F, float)
    if len(F) <= k:
        return np.arange(len(F))
    fn = _compiled("igdp_self_order")
    if fn is not None and F.ndim == 2 and F.shape[1]:
        return fn(F, int(k), _frame(ideal), _frame(nadir))
    return _igdp_self_order_numpy(F, k, ideal, nadir)


def _igdp_self_order_numpy(F, k: int, ideal=None, nadir=None) -> np.ndarray:
    """igdp_self_order() in NumPy: the reference."""
    F = np.asarray(F, float)
    if len(F) <= k:
        return np.arange(len(F))
    if ideal is not None and nadir is not None:
        G = normalized(F, ideal, nadir)
    else:
        G, _ = _set_aside_numpy(F)
    ref = _dss_order_numpy(F, k=SELF_REFERENCE, ideal=ideal, nadir=nadir)
    return greedy_igdp(dplus_matrix(G, G[ref]), k)


def default_delta(n_obj: int) -> float:
    """Grid step in normalized units: 1e-3, and 1e-2 from five objectives up."""
    return 1e-2 if n_obj >= 5 else 1e-3


class PyGridArchive:
    """GridArchive in Python (GridArchive's docstring says what it keeps): the
    reference the compiled archive is compared with, and the archive where the
    extension is not built."""

    def __init__(self, n_obj: int, n_vars: int = 0, *, ideal=None, nadir=None,
                 delta: float | None = None):
        self.m = int(n_obj)
        self.n_vars = int(n_vars)
        self.delta = float(delta) if delta else default_delta(self.m)
        self.normalization = "problem" if ideal is not None and nadir is not None else "archive"
        if self.normalization == "problem":
            self._lo = np.asarray(ideal, float)
            span = np.asarray(nadir, float) - self._lo
            self._span = np.where(span > 0.0, span, 1.0)
        else:
            self._lo = None
            self._span = None
        cap = 256
        self._F = np.empty((cap, self.m))
        self._X = np.empty((cap, self.n_vars))
        self._alive = np.zeros(cap, bool)
        self._n = 0
        self._cell: dict = {}          # cell key -> slot
        self._slot_cell: dict = {}     # slot -> cell key
        self._ext = [-1] * self.m      # slot of each objective's best point
        self.offered = 0
        self.accepted = 0

    # ── bookkeeping ──────────────────────────────────────────────────────────
    def __len__(self) -> int:
        return int(self._alive[:self._n].sum())

    def _append(self, f, x) -> int:
        if self._n == len(self._F):
            if self._n > 2 * len(self) + 64:
                self._compact()
            if self._n == len(self._F):
                grow = len(self._F)
                self._F = np.vstack([self._F, np.empty((grow, self.m))])
                self._X = np.vstack([self._X, np.empty((grow, self.n_vars))])
                self._alive = np.concatenate([self._alive, np.zeros(grow, bool)])
        s = self._n
        self._F[s] = f
        if self.n_vars:
            self._X[s] = x
        self._alive[s] = True
        self._n += 1
        return s

    def _remove(self, s: int) -> None:
        self._alive[s] = False
        c = self._slot_cell.pop(s, None)
        if c is not None and self._cell.get(c) == s:
            del self._cell[c]
        for j in range(self.m):
            if self._ext[j] == s:
                self._ext[j] = -1

    def _compact(self) -> None:
        live = np.flatnonzero(self._alive[:self._n])
        remap = {int(old): new for new, old in enumerate(live)}
        k = len(live)
        self._F[:k] = self._F[live]
        if self.n_vars:
            self._X[:k] = self._X[live]
        self._alive[:] = False
        self._alive[:k] = True
        self._n = k
        self._slot_cell = {remap[s]: c for s, c in self._slot_cell.items() if s in remap}
        self._cell = {c: s for s, c in self._slot_cell.items()}
        self._ext = [remap.get(e, -1) if e >= 0 else -1 for e in self._ext]

    # ── the grid ─────────────────────────────────────────────────────────────
    def _key(self, f):
        return tuple(np.floor((f - self._lo) / self._span / self.delta).astype(np.int64).tolist())

    def _corner(self, f, key) -> float:
        g = (f - self._lo) / self._span
        return float(np.sum((g - np.asarray(key, float) * self.delta) ** 2))

    def _place(self, s: int) -> bool:
        """Put stored slot s into its cell; False (and s removed) if it loses."""
        f = self._F[s]
        key = self._key(f)
        q = self._cell.get(key)
        if q is None:
            self._cell[key] = s
            self._slot_cell[s] = key
            return True
        if self._corner(f, key) < self._corner(self._F[q], key):
            self._remove(q)
            self._cell[key] = s
            self._slot_cell[s] = key
            return True
        self._remove(s)
        return False

    def _regrid(self, exclude: int = -1) -> None:
        """Re-estimate the frame from the stored points and rebuild every cell.

        `exclude` is a slot the caller will place itself (the point being added).
        """
        live = np.flatnonzero(self._alive[:self._n])
        F = self._F[live]
        self._lo = F.min(axis=0)
        span = F.max(axis=0) - self._lo
        self._span = np.where(span > 0.0, span, 1.0)
        self._cell.clear()
        self._slot_cell.clear()
        ext = set(e for e in self._ext if e >= 0)
        for s in live:
            s = int(s)
            if s != exclude and s not in ext and self._alive[s]:
                self._place(s)

    def _frame_drifted(self, f) -> bool:
        if self._lo is None:
            return True
        tol = 0.25 * self._span
        return bool(np.any(f < self._lo - tol) or np.any(f > self._lo + self._span + tol))

    # ── the interface ────────────────────────────────────────────────────────
    def add(self, f, x=None) -> bool:
        """Offer one evaluated point; True if it is stored."""
        f = np.asarray(f, float)
        self.offered += 1
        if f.shape != (self.m,) or not np.all(np.isfinite(f)):
            return False
        n = self._n
        if n:
            A = self._F[:n]
            alive = self._alive[:n]
            if np.any(alive & np.all(A <= f, axis=1)):     # dominated or a duplicate
                return False
            beaten = np.flatnonzero(alive & np.all(f <= A, axis=1))
            for s in beaten:
                self._remove(int(s))
        s = self._append(f, x if x is not None else np.zeros(self.n_vars))
        self.accepted += 1
        if self.normalization == "archive" and self._frame_drifted(f):
            self._regrid(exclude=s)                         # s is placed below
        new_ext = [j for j in range(self.m)
                   if self._ext[j] < 0 or f[j] < self._F[self._ext[j], j]]
        if new_ext:
            old = {self._ext[j] for j in new_ext if self._ext[j] >= 0}
            for j in new_ext:
                self._ext[j] = s
            for o in old:                                   # a dethroned extreme rejoins the grid
                if self._alive[o] and o not in self._ext:
                    self._place(o)
            return True
        return self._place(s)

    def add_many(self, F, X=None) -> np.ndarray:
        """add() for every row of F (and X) in order; whether each was stored."""
        F = np.asarray(F, float)
        return np.array([self.add(F[i], None if X is None else X[i]) for i in range(len(F))],
                        bool)

    def points(self):
        """(F, X, is_extreme) of the stored points, in insertion order."""
        live = np.flatnonzero(self._alive[:self._n])
        ext = set(e for e in self._ext if e >= 0)
        return (self._F[live].copy(), self._X[live].copy(),
                np.array([int(s) in ext for s in live], bool))

    def grid_frame(self):
        """(lo, lo + span) of the grid, or None before the archive's own frame
        is first estimated."""
        if self._lo is None:
            return None
        return self._lo.copy(), self._lo + self._span

    def frame(self):
        """(ideal, nadir) the grid uses — the problem's, or the current estimate."""
        if self._lo is None:
            F = self.points()[0]
            return (F.min(axis=0), F.max(axis=0)) if len(F) else (None, None)
        return self._lo.copy(), self._lo + self._span

    def select(self, k: int, method: str = "dss"):
        """Indices into points() of a selection of k (all if fewer): DSS, or
        with method "igdp_self" the points of the smallest IGD+ against the
        archive thinned by DSS (igdp_self_order).

        In the problem's frame when there is one; otherwise in the frame of the
        points DSS keeps (set_aside), not in the grid's running estimate, which
        an outlier stretches.
        """
        if method not in ANSWER_SELECT:
            raise ValueError(f"answer selection {method!r}: one of {', '.join(ANSWER_SELECT)}")
        F = self.points()[0]
        if len(F) <= k:
            return np.arange(len(F))
        lo = hi = None
        if self.normalization == "problem":
            lo, hi = self.frame()
        if method == "igdp_self":
            return igdp_self_order(F, k, ideal=lo, nadir=hi)
        return dss_order(F, k=k, ideal=lo, nadir=hi)

    def info(self) -> dict:
        return {"size": len(self), "delta": self.delta, "normalization": self.normalization,
                "offered": self.offered, "accepted": self.accepted}


class GridArchive:
    """Every nondominated point seen, thinned to one per δ-cell, plus the extremes.

    add() accepts a point unless a stored point weakly dominates it (dominates
    or equals it), and removes the stored points it dominates — dominance is
    checked against the CURRENT archive only, so the archive holds what beats
    everything still in it, not a certificate against all history. Accepted
    points then compete for their cell of an additive grid of step δ in
    normalized coordinates, (f − ideal)/(nadir − ideal): one point per cell,
    the one nearer the cell's lower corner, as in the ε-archives of Laumanns,
    Thiele, Deb & Zitzler (2002) but without their box-dominance — the grid
    only prunes points too close to matter, it does not bound the archive to a
    fixed ε-approximation.

    THE EXTREMES LIVE OUTSIDE THE GRID. The best point of each objective is
    kept whatever its cell holds; a grid alone loses the ends of the front,
    measured on HypE populations even at δ = 1e-3, and the ends are exactly
    what IGD and the hypervolume weigh most. A replaced extreme goes back
    through the grid like any other point.

    The frame is the problem's ideal and nadir when given ("problem"). Without
    them it is estimated from the archive itself ("archive"): the current
    minimum and maximum of the stored points, re-gridding everything when the
    estimate drifts by more than a quarter of its span, so the cells stay
    meaningful while the front is still moving.

    The points are kept by the compiled archive (include/mootation/
    grid_archive.hpp) where the extension is built, by PyGridArchive
    otherwise: the same points in the same order, the same extremes, the same
    counts. add_many offers a batch, as add() one point after the other would.
    """

    def __init__(self, n_obj: int, n_vars: int = 0, *, ideal=None, nadir=None,
                 delta: float | None = None):
        self.m = int(n_obj)
        self.n_vars = int(n_vars)
        self.delta = float(delta) if delta else default_delta(self.m)
        self.normalization = "problem" if ideal is not None and nadir is not None else "archive"
        compiled = _compiled("GridArchive")
        if compiled is not None:
            self._a = compiled(self.m, self.n_vars, ideal=_frame(ideal), nadir=_frame(nadir),
                               delta=self.delta)
        else:
            self._a = PyGridArchive(self.m, self.n_vars, ideal=ideal, nadir=nadir,
                                    delta=self.delta)

    @property
    def offered(self) -> int:
        return int(self._a.offered)

    @property
    def accepted(self) -> int:
        return int(self._a.accepted)

    def __len__(self) -> int:
        return len(self._a)

    def add(self, f, x=None) -> bool:
        """Offer one evaluated point; True if it is stored."""
        return bool(self._a.add(f, x))

    def add_many(self, F, X=None) -> np.ndarray:
        """Offer the rows of F (and their variables X) in order; whether each was stored."""
        F = np.asarray(F, float).reshape(-1, self.m)
        if X is not None:
            X = np.asarray(X, float).reshape(len(F), self.n_vars)
        return np.asarray(self._a.add_many(F, X), bool)

    def points(self):
        """(F, X, is_extreme) of the stored points, in insertion order."""
        return self._a.points()

    def frame(self):
        """(ideal, nadir) the grid uses — the problem's, or the current estimate."""
        got = self._a.grid_frame()
        if got is None:
            F = self.points()[0]
            return (F.min(axis=0), F.max(axis=0)) if len(F) else (None, None)
        return got

    def select(self, k: int, method: str = "dss"):
        """Indices into points() of a selection of k (all if fewer): DSS, or
        with method "igdp_self" the points of the smallest IGD+ against the
        archive thinned by DSS (igdp_self_order).

        In the problem's frame when there is one; otherwise in the frame of the
        points DSS keeps (set_aside), not in the grid's running estimate, which
        an outlier stretches.
        """
        if method not in ANSWER_SELECT:
            raise ValueError(f"answer selection {method!r}: one of {', '.join(ANSWER_SELECT)}")
        F = self.points()[0]
        if len(F) <= k:
            return np.arange(len(F))
        lo = hi = None
        if self.normalization == "problem":
            lo, hi = self.frame()
        if method == "igdp_self":
            return igdp_self_order(F, k, ideal=lo, nadir=hi)
        return dss_order(F, k=k, ideal=lo, nadir=hi)

    def info(self) -> dict:
        return {"size": len(self), "delta": self.delta, "normalization": self.normalization,
                "offered": self.offered, "accepted": self.accepted}


# ── The second answer: for the indicators in the decision space ─────────────

class SetArchive:
    """Every point evaluated that no other point evaluated dominates, points
    with equal objectives and other variables kept apart (task 15, §6, the
    owner's word of 2026-10-22). An archive in the objective space (GridArchive,
    even with the finest grid) keeps one point per objective vector, so the
    copies of a Pareto set that map onto the same front — SYM-PART's nine —
    glue together there; this one keeps them, for the decision-space
    indicators' answer (spread_select). No bound on its size.

    The points are kept by the compiled archive (include/mootation/
    set_archive.hpp: a sorted list at two objectives, an ND-tree from three;
    task 16, supplement 1, §3) where the extension is built, by PySetArchive
    otherwise: the same points in the same order. F and X: the stored points'
    objectives and variables, in the order they were offered."""

    def __init__(self, n_obj: int, n_vars: int):
        compiled = _compiled("SetArchive")
        self._a = compiled(int(n_obj), int(n_vars)) if compiled is not None else PySetArchive(n_obj, n_vars)
        self._points = None

    def __len__(self) -> int:
        return len(self._a)

    def add_many(self, F, X) -> None:
        """Offer the rows of F and X, in order."""
        F, X = np.asarray(F, float), np.asarray(X, float)
        if len(F):
            self._a.add_many(F.reshape(len(F), -1), X.reshape(len(F), -1))
            self._points = None

    def _read(self):
        if self._points is None:
            self._points = self._a.points()
        return self._points

    @property
    def F(self):
        return self._read()[0]

    @property
    def X(self):
        return self._read()[1]


class PySetArchive:
    """SetArchive in NumPy: the reference (a batch compared at once)."""

    def __init__(self, n_obj: int, n_vars: int):
        self.F = np.empty((0, n_obj))
        self.X = np.empty((0, n_vars))

    def points(self):
        return self.F, self.X

    def __len__(self) -> int:
        return len(self.F)

    def add_many(self, F, X) -> None:
        """The points of a batch, at once: those no point of the batch or the
        archive strictly dominates join (in the batch's order), the archive's
        that one of them dominates leave — the set and the order of adding the
        points one at a time."""
        F, X = np.asarray(F, float), np.asarray(X, float)
        if not len(F):
            return

        def dominated_by(A, B):        # row i of A strictly dominated by some row of B
            # one objective at a time: 2-D arrays, ten times faster than one
            # 3-D comparison (task 16, §2.1)
            out = np.zeros(len(A), bool)
            for a in range(0, len(A), 256):
                C = A[a:a + 256]
                le = B[None, :, 0] <= C[:, 0, None]
                lt = B[None, :, 0] < C[:, 0, None]
                for j in range(1, A.shape[1]):
                    le &= B[None, :, j] <= C[:, j, None]
                    lt |= B[None, :, j] < C[:, j, None]
                out[a:a + 256] = np.any(le & lt, axis=1)
            return out
        keep = ~dominated_by(F, F)
        if len(self.F):
            keep &= ~dominated_by(F, self.F)
        F, X = F[keep], X[keep]
        if not len(F):
            return
        if len(self.F):
            stay = ~dominated_by(self.F, F)
            self.F, self.X = self.F[stay], self.X[stay]
        self.F = np.vstack([self.F, F])
        self.X = np.vstack([self.X, X])


# The second answer's version (meta["set_answer"]["version"]): 2 from task 17,
# supplement 1 (§3, the owner's decision of 2026-10-23), the first point
# drawn by the library's generator (spread_first) instead of NumPy's
# default_rng(seed).integers.
SET_ANSWER_VERSION = 2

# decision_indicators.hpp, kSecondAnswerStream
_SECOND_ANSWER_STREAM = 0x2C9277B5


def spread_first(seed: int, rows: int) -> int:
    """The first point of spread_select among `rows`: rnd::below of a
    std::mt19937 seeded with the seed's low 32 bits XOR the second answer's
    stream (decision_indicators.hpp, spread_first) — the extension's, or the
    same draw here: NumPy's MT19937 seeded as std::mt19937 is (init_genrand),
    its 32-bit words, Lemire's multiply-and-reject (random.hpp, below)."""
    try:
        from .. import _core
        if hasattr(_core, "spread_first"):
            return int(_core.spread_first(int(seed) & 0xFFFFFFFFFFFFFFFF, int(rows)))
    except ImportError:                              # pragma: no cover - no extension
        pass
    bg = np.random.MT19937()
    bg._legacy_seeding((int(seed) & 0xFFFFFFFF) ^ _SECOND_ANSWER_STREAM)
    n = int(rows)
    if n > 0xFFFFFFFF:
        raise ValueError("spread_first: more than 2^32 rows")
    m = int(bg.random_raw()) * n
    low = m & 0xFFFFFFFF
    if low < n:
        t = (2 ** 32 - n) % n
        while low < t:
            m = int(bg.random_raw()) * n
            low = m & 0xFFFFFFFF
    return m >> 32


def spread_select(X, n: int, bounds, seed: int, first: int | None = None) -> np.ndarray:
    """n of the rows of X spread out in the decision space: Tanabe & Ishibuchi's
    selection for IGDX (IEEE TEVC 2020, §III-H, after [17]): a random first
    point (spread_first, from seed; or `first`), then each time the point
    farthest — its distance to the nearest point chosen, in the variables
    normalized by the bounds — until n. Ties go to the lowest index. The
    extension's where it is built, the same indices."""
    X = np.asarray(X, float)
    if len(X) <= n:
        return np.arange(len(X))
    lo = np.array([b[0] for b in bounds], float)
    hi = np.array([b[1] for b in bounds], float)
    if first is None:
        first = spread_first(seed, len(X))
    fn = _compiled("spread_select")
    if fn is not None and X.ndim == 2 and X.shape[1]:
        return np.asarray(fn(np.ascontiguousarray(X), int(n), lo, hi, int(first)), dtype=np.intp)
    return _spread_select_numpy(X, n, lo, hi, int(first))


def _spread_select_numpy(X, n: int, lo, hi, first: int) -> np.ndarray:
    """spread_select's NumPy reference, from `first`."""
    Z = (X - lo) / np.where(hi > lo, hi - lo, 1.0)
    chosen = [first]
    d = np.sqrt(((Z - Z[first]) ** 2).sum(axis=1))
    d[first] = -1.0
    while len(chosen) < n:
        c = int(np.argmax(d))
        chosen.append(c)
        d = np.minimum(d, np.sqrt(((Z - Z[c]) ** 2).sum(axis=1)))
        d[chosen] = -1.0
    return np.asarray(chosen)

# SPDX-License-Identifier: Apache-2.0
"""The run archive, and the subset selection shared by every place that thins to N.

GridArchive keeps every nondominated point a run evaluates, with points that
are too close to each other pruned; `dss_order` selects N of them, and of any
other point set, the same way everywhere (reference fronts, random search, the
"archive" reading of a run).

`dss_order` is distance-based subset selection (DSS): start from the best point
of every objective, then repeatedly add the candidate that the selected set
covers WORST, until the set is full. "Covers" is measured with the IGD+
distance, d+(s, c) = ||max(s − c, 0)||, the amount by which a selected point s
is worse than candidate c; a candidate dominated by (or equal to) a selected
point is covered completely. Greedy maximin on that distance picks the point
contributing most to IGD+ of the selection against the whole set, without
optimizing IGD+ or the hypervolume themselves — an indicator-neutral choice,
where selecting by one indicator would flatter that indicator afterwards.

The order is INCREMENTAL: the first n points of dss_order(F) are exactly the
DSS selection of size n from F. A set stored in this order is thinned by
slicing.

Everything is computed in normalized coordinates, (f − ideal)/(nadir − ideal),
so that no objective dominates the distance by its units alone; pass the frame
explicitly where one is known (a registry problem), otherwise the set's own
minimum and maximum are used.

Sources: DSS is Singh, Bhattacharjee & Ray (IEEE TEVC 23(5):904-912, 2019);
the IGD+ distance in its place is Chen, Ishibuchi & Shang (2020). Neither paper
is in this project's corpus yet; the procedure above is the one specified for
MOOtation on 2026-09-22 and is written from that specification.
"""

from __future__ import annotations

import numpy as np


def normalized(F, ideal=None, nadir=None):
    """(F − ideal)/(nadir − ideal), a zero range counting as 1."""
    F = np.asarray(F, float)
    lo = F.min(0) if ideal is None else np.asarray(ideal, float)
    hi = F.max(0) if nadir is None else np.asarray(nadir, float)
    span = hi - lo
    span = np.where(span > 0.0, span, 1.0)
    return (F - lo) / span


def dss_order(F, k=None, ideal=None, nadir=None) -> np.ndarray:
    """Indices of F in DSS order (all of them, or the first k).

    Ties are broken by the lowest index, so the order is deterministic.
    """
    G = normalized(F, ideal, nadir)
    n = len(G)
    k = n if k is None else min(int(k), n)
    if k <= 0:
        return np.empty(0, dtype=np.intp)
    order = []
    taken = np.zeros(n, bool)
    cover = np.full(n, np.inf)                 # min over selected of d+(s, c)

    def take(i):
        taken[i] = True
        order.append(int(i))
        d = np.sqrt(np.sum(np.maximum(G[i] - G, 0.0) ** 2, axis=1))
        np.minimum(cover, d, out=cover)

    for j in range(G.shape[1]):               # the best point of every objective
        if len(order) >= k:
            break
        i = int(np.argmin(G[:, j]))
        if not taken[i]:
            take(i)
    while len(order) < k:
        c = np.where(taken, -1.0, cover)
        i = int(np.argmax(c))
        if taken[i]:
            break
        take(i)
    return np.asarray(order, dtype=np.intp)


def default_delta(n_obj: int) -> float:
    """Grid step in normalized units: 1e-3, and 1e-2 from five objectives up."""
    return 1e-2 if n_obj >= 5 else 1e-3


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
    """

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

    def points(self):
        """(F, X, is_extreme) of the stored points, in insertion order."""
        live = np.flatnonzero(self._alive[:self._n])
        ext = set(e for e in self._ext if e >= 0)
        return (self._F[live].copy(), self._X[live].copy(),
                np.array([int(s) in ext for s in live], bool))

    def frame(self):
        """(ideal, nadir) the grid uses — the problem's, or the current estimate."""
        if self._lo is None:
            F = self.points()[0]
            return (F.min(axis=0), F.max(axis=0)) if len(F) else (None, None)
        return self._lo.copy(), self._lo + self._span

    def select(self, k: int):
        """Indices into points() of a DSS selection of k (all if fewer)."""
        F = self.points()[0]
        if len(F) <= k:
            return np.arange(len(F))
        lo, hi = self.frame()
        return dss_order(F, k=k, ideal=lo, nadir=hi)

    def info(self) -> dict:
        return {"size": len(self), "delta": self.delta, "normalization": self.normalization,
                "offered": self.offered, "accepted": self.accepted}

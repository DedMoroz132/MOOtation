# SPDX-License-Identifier: Apache-2.0
"""The scalarized subproblems.

Every scalarization here is linear in the normalized objectives
F̄ = (F - z*) / (z^N - z*) and in at most one extra variable t, so one form
holds them all: minimize  a0·F̄ + b0·t  subject to rows  a·F̄ + b·t + c >= 0
(inequalities) and  a·F̄ + b·t + c = 0  (equalities), beside the problem's own
constraints. A gradient-based solver (sqp) takes them as they are, with t a
variable of its own; a derivative-free one (nm, cmaes) gets t eliminated — the
least t the inequalities allow, or the least-squares t of the equalities — and
what the rows still violate as a violation (merit).

The normalization is the anchors' (Frame): z* the utopia point (each
objective's minimum among the anchors) and z^N the pseudo-nadir (each
objective's maximum among them), as NBI's pay-off matrix and NNC's normalized
space take it (Das & Dennis 1998, §3; Messac et al., as gnc_messac2026 §2.2
gives it). Before the anchors are known the frame is the identity.

  eps   ε-constraint: min F̄_k  s.t.  F̄_j <= ε̄_j (j != k)
  tch   Chebyshev, smooth form: min t  s.t.  w_i F̄_i <= t; the lattice gives a
        direction d on the simplex and w_i = 1 / max(d_i, 1e-6), so that the
        solution's ray F̄ ∝ d is what the lattice spreads (SCALAR-3); with
        tch_weights = lattice, w = d (the plain weight grid)
  nbi   NBI (Das & Dennis, SIAM J. Optim. 8(3), 1998, Eq. (1) and §4.3):
        max t  s.t.  Φ̄β + t·n̂ = F̄,  n̂ = -Φ̄e the quasi-normal
  nnc   NNC (Messac, Ismail-Yahaya & Mattson, Struct. Multidisc. Optim.
        25(2), 2003, Problem P3, Eqs. 36-41): min F̄_m  s.t.
        (Ā_m - Ā_i)ᵀ(F̄ - X̄) <= 0, i = 1..m-1, X̄ = Σ α_i Ā_i (SCALAR-4)
  ps    Pascoletti-Serafini (J. Optim. Theory Appl. 42(4), 1984, Problem
        P(p, q): max ξ s.t. F(x) = p + ξq + λ, λ in the cone, written for
        maximization; here its minimization form): min t  s.t.
        ā + t·r - F̄ >= 0 (the ordering cone R^m_+), ā = Φ̄β on the anchors'
        simplex and r = Φ̄e: NBI's targets and its quasi-normal reversed, with
        an inequality where NBI has an equality (SCALAR-5). ε-constraint
        (r = e_k, ā_j = ε_j) and Chebyshev (ā = 0, r_i = 1/w_i) are its
        special cases.
"""

from __future__ import annotations

import math

import numpy as np


class Frame:
    """The normalization the anchors give."""

    def __init__(self, m: int, z=None, s=None, anchors=None):
        self.m = int(m)
        self.z = np.zeros(m) if z is None else np.asarray(z, float)
        self.s = np.ones(m) if s is None else np.asarray(s, float)
        # anchors[i] = F̄ of the i-th anchor point (row i)
        self.A = None if anchors is None else np.asarray(anchors, float)

    @classmethod
    def from_anchors(cls, F_anchors):
        """z* = each objective's best among the anchors, z^N its worst; a range
        that vanishes is taken as 1 (a constant objective)."""
        F = np.asarray(F_anchors, float)
        z, n = F.min(axis=0), F.max(axis=0)
        s = n - z
        s = np.where(s > 1e-12 * np.maximum(1.0, np.abs(z)), s, 1.0)
        fr = cls(F.shape[1], z, s)
        fr.A = fr.bar(F)
        return fr

    def bar(self, F):
        return (np.asarray(F, float) - self.z) / self.s

    @property
    def Phi(self):
        """NBI's pay-off matrix, normalized: column i = Ā_i (the origin at z*)."""
        return self.A.T

    @property
    def quasi_normal(self):
        return -self.Phi @ np.ones(self.m)


class Sub:
    """One scalarized subproblem: minimize a0·F̄ + b0·t + c0 under the rows."""

    def __init__(self, kind: str, a0, b0: float = 0.0, c0: float = 0.0, aux: bool = False,
                 ineq=(), eq=(), param=None):
        self.kind = kind
        self.a0 = np.asarray(a0, float)
        self.b0, self.c0, self.aux = float(b0), float(c0), bool(aux)
        # rows: (a vector, b, c)
        self.ineq = [(np.asarray(a, float), float(b), float(c)) for a, b, c in ineq]
        self.eq = [(np.asarray(a, float), float(b), float(c)) for a, b, c in eq]
        self.param = None if param is None else np.asarray(param, float)

    # the rows as matrices, for the solvers
    def mats(self, rows):
        if not rows:
            return np.zeros((0, len(self.a0))), np.zeros(0), np.zeros(0)
        return (np.array([r[0] for r in rows]), np.array([r[1] for r in rows]),
                np.array([r[2] for r in rows]))

    def t_star(self, Fb) -> float:
        """The t a point implies: least squares on the equalities if there are
        any with t, else the least t every inequality with t allows."""
        if not self.aux:
            return 0.0
        Ae, be, ce = self.mats(self.eq)
        if len(be) and np.any(be != 0.0):
            r = Ae @ Fb + ce
            return float(-(be @ r) / (be @ be))
        Ai, bi, ci = self.mats(self.ineq)
        lo = [-(Ai[k] @ Fb + ci[k]) / bi[k] for k in range(len(bi)) if bi[k] > 0.0]
        return float(max(lo)) if lo else 0.0

    def merit(self, Fb):
        """(value, violation) of a point with t eliminated."""
        t = self.t_star(Fb)
        value = float(self.a0 @ Fb + self.b0 * t + self.c0)
        viol = 0.0
        for a, b, c in self.ineq:
            viol += max(0.0, -(a @ Fb + b * t + c))
        for a, b, c in self.eq:
            viol += abs(a @ Fb + b * t + c)
        return value, viol

    def merits(self, FB):
        """merit() of many rows at once (FB: N × m), for picking a start."""
        FB = np.asarray(FB, float)
        if not len(FB):
            return np.zeros(0), np.zeros(0)
        if self.aux:
            Ae, be, ce = self.mats(self.eq)
            if len(be) and np.any(be != 0.0):
                t = -((FB @ Ae.T + ce) @ be) / (be @ be)
            else:
                Ai, bi, ci = self.mats(self.ineq)
                pos = bi > 0.0
                t = (np.max(-(FB @ Ai[pos].T + ci[pos]) / bi[pos], axis=1)
                     if pos.any() else np.zeros(len(FB)))
        else:
            t = np.zeros(len(FB))
        value = FB @ self.a0 + self.b0 * t + self.c0
        viol = np.zeros(len(FB))
        Ai, bi, ci = self.mats(self.ineq)
        if len(bi):
            viol += np.maximum(0.0, -(FB @ Ai.T + np.outer(t, bi) + ci)).sum(axis=1)
        Ae, be, ce = self.mats(self.eq)
        if len(be):
            viol += np.abs(FB @ Ae.T + np.outer(t, be) + ce).sum(axis=1)
        return value, viol


# ── the subproblems ─────────────────────────────────────────────────────────
def eps(m: int, k: int, levels) -> Sub:
    """min F̄_k  s.t.  F̄_j <= ε̄_j for j != k (levels: m - 1 values, in order)."""
    a0 = np.zeros(m)
    a0[k] = 1.0
    rows, others = [], [j for j in range(m) if j != k]
    for j, e in zip(others, levels):
        if math.isfinite(e):
            a = np.zeros(m)
            a[j] = -1.0
            rows.append((a, 0.0, float(e)))
    return Sub("eps", a0, ineq=rows, param=np.asarray(levels, float))


def tch(m: int, d, weights: str = "inverse") -> Sub:
    """min t  s.t.  t - w_i F̄_i >= 0: w_i = 1 / max(d_i, 1e-6) (inverse), or
    the lattice point itself, zeros as 1e-6 (lattice: the plain weight grid)."""
    d = np.asarray(d, float)
    w = 1.0 / np.maximum(d, 1e-6) if weights == "inverse" else np.maximum(d, 1e-6)
    w /= w.sum()
    rows = []
    for i in range(m):
        a = np.zeros(m)
        a[i] = -w[i]
        rows.append((a, 1.0, 0.0))
    return Sub("tch", np.zeros(m), b0=1.0, aux=True, ineq=rows, param=d)


def nbi(frame: Frame, beta) -> Sub:
    """max t  s.t.  Φ̄β + t·n̂ - F̄ = 0."""
    m = frame.m
    beta = np.asarray(beta, float)
    target, nh = frame.Phi @ beta, frame.quasi_normal
    rows = []
    for i in range(m):
        a = np.zeros(m)
        a[i] = -1.0
        rows.append((a, nh[i], target[i]))
    return Sub("nbi", np.zeros(m), b0=-1.0, aux=True, eq=rows, param=beta)


def ps(frame: Frame, beta) -> Sub:
    """min t  s.t.  ā + t·r - F̄ >= 0,  ā = Φ̄β,  r = Φ̄e."""
    m = frame.m
    beta = np.asarray(beta, float)
    target, r = frame.Phi @ beta, -frame.quasi_normal
    rows = []
    for i in range(m):
        a = np.zeros(m)
        a[i] = -1.0
        rows.append((a, r[i], target[i]))
    return Sub("ps", np.zeros(m), b0=1.0, aux=True, ineq=rows, param=beta)


def nnc(frame: Frame, alpha) -> Sub:
    """min F̄_m  s.t.  (Ā_m - Ā_i)ᵀ(F̄ - X̄) <= 0, i < m,  X̄ = Σ α_i Ā_i."""
    m = frame.m
    alpha = np.asarray(alpha, float)
    A = frame.A
    X = alpha @ A
    a0 = np.zeros(m)
    a0[m - 1] = 1.0
    rows = []
    for i in range(m - 1):
        N = A[m - 1] - A[i]
        rows.append((-N, 0.0, float(N @ X)))
    return Sub("nnc", a0, ineq=rows, param=alpha)


# ── where a found point sits in a plan's parameter space (sub_plan = gap) ───
def coords(kind: str, frame: Frame, Fb, k: int = 0):
    """The parameter a subproblem of this kind would need to aim at F̄: the
    constrained objectives' values (eps), the direction on the simplex (tch),
    the barycentric coordinates of the projection onto the anchors' simplex —
    along the quasi-normal (nbi, ps) or along e (nnc)."""
    Fb = np.asarray(Fb, float)
    m = frame.m
    if kind == "eps":
        return np.delete(Fb, k, axis=-1)
    if kind == "tch":
        P = np.maximum(Fb, 0.0)
        s = P.sum(axis=-1, keepdims=True)
        return np.where(s > 0, P / np.where(s > 0, s, 1.0), 1.0 / m)
    direction = frame.quasi_normal if kind in ("nbi", "ps") else np.ones(m)
    M = np.zeros((m + 1, m + 1))
    M[:m, :m] = frame.A.T
    M[:m, m] = direction
    M[m, :m] = 1.0
    rhs = np.concatenate([np.atleast_2d(Fb), np.ones((np.atleast_2d(Fb).shape[0], 1))], axis=1)
    sol = np.linalg.lstsq(M, rhs.T, rcond=None)[0].T
    out = sol[:, :m]
    return out[0] if Fb.ndim == 1 else out

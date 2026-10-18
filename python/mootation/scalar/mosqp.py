# SPDX-License-Identifier: Apache-2.0
"""MOSQP: multiobjective SQP on a set of points, without scalarization.

J. Fliege, A.I.F. Vaz, "A method for constrained multiobjective optimization
based on SQP techniques", SIAM J. Optim. 26(4):2091-2119, 2016,
doi:10.1137/15M1016424 (the preprint, Optimization Online 2015:
mosqp_fliege2016_draft; §2.1, §3 problems (8)-(9), Algorithm 4.1, §6).

  stage 1  the extreme points: min f_i under the constraints, from the
           box's centre (10) — SLSQP with finite differences (solvers.py);
  stage 2  "spread", at most mosqp_spread (20, §6) iterations over a set X of
           n̄ = pop_size points, first on the line between the bounds (§6,
           "line"): every point not yet stopped gives, for every objective i,
           the direction d of the QP (8) — min ∇f_iᵀd + ½dᵀH_i d under the
           linearized constraints and the box — and, when ‖d‖ >= τ^(1/4), the
           point x + αd, α the first of 1, β, β², ... with Armijo's decrease of
           the ℓ1 merit φ_i(x, σ) = f_i + σ(Σ g⁺ + Σ|h|) (§2.1) and α >= τ^(1/4);
           the point is then stopped; X becomes the non-dominated points of X
           and the new ones (dominance extended to (f, g⁺, |h|), §2.1), cut to
           n̄ by crowding distance (§4: ten of the most crowded at a time);
  stage 3  "refining", on X and the extreme points: every point not yet
           stopped moves along the solution v of (9) — min Σζ_i,
           ∇f_iᵀv + ½vᵀH_i v <= ζ_i, the linearized constraints, the box — by
           Armijo on the merit Φ = Σf_i + σ(Σg⁺ + Σ|h|), until ‖v‖ < τ^(1/2)
           or α < τ^(1/2); points dominated by feasible points leave X.
A QP without a feasible point sends the point to a restoration: min
Σ(g⁺)² + Σh² + 10⁻⁶‖x − x_l‖² from x_l (SLSQP, finite differences); a feasible
result replaces x_l (stage 3) or joins T (stage 2).

Every evaluation goes through the evaluator (budget, archive, answer).

READINGS (the preprint gives no numbers for τ, μ, β, n̄; §6 is MATLAB-specific):
  MS-1. τ = 1e-8 (mosqp_tau), μ = 1e-4 (mosqp_mu, Armijo), β = 1/2
    (mosqp_beta); n̄ = pop_size.
  MS-2. H_i = I in both stages: the first of §6's three cases (the others
    need second derivatives the problems here do not give).
  MS-3. Gradients and Jacobians by forward differences (fd_step in the box
    normalized to [0, 1], as sqp in solvers.py), n evaluations a point.
  MS-4. σ: the first of 1, 10, 100, ... that makes the merit's directional
    derivative ∇fᵀd − σ‖c(x)‖₁ negative (Algorithm 4.1, 3(a)iD), at most 10⁸.
  MS-5. The line of stage 2: x_l = ℓ + l(u − ℓ)/(n̄ + 1), l = 1..n̄.
  MS-6. The QPs (8) and (9) are solved by SciPy's SLSQP with exact
    derivatives of the quadratic models (no evaluation of fn); "no feasible
    solution" is SLSQP's report of incompatible constraints.
"""

from __future__ import annotations

import math


def run_mosqp(ev, *, n_objs: int, n_eq: int, pop_size: int, knobs: dict, fd_step: float):
    """MOSQP on the Evaluator ev; returns nothing (the answer is the archive's)."""
    import numpy as np
    from scipy.optimize import minimize as sp_minimize
    from . import solvers as S
    from . import subproblems as P
    from .evaluator import BudgetSpent, SubBudgetSpent

    tau = float(knobs.get("mosqp_tau", 1e-8))
    mu = float(knobs.get("mosqp_mu", 1e-4))
    beta = float(knobs.get("mosqp_beta", 0.5))
    spread_its = int(knobs.get("mosqp_spread", 20))
    m, n = int(n_objs), len(ev.lo)
    width = ev.hi - ev.lo
    k_in = ev.n_cons - n_eq
    nbar = int(pop_size)

    def at(y):
        return ev.index(ev.lo + np.clip(y, 0.0, 1.0) * width)

    def parts(i):
        g = ev.G[i]
        return ev.F[i], g[:k_in], g[k_in:]

    def viol(i):
        _, g, h = parts(i)
        return float(np.sum(np.maximum(g, 0.0)) + np.sum(np.abs(h)))

    def jac(y, i0):
        """(JF, Jg, Jh) at y (normalized) by forward differences."""
        F0, g0, h0 = parts(i0)
        JF = np.zeros((m, n)); Jg = np.zeros((len(g0), n)); Jh = np.zeros((len(h0), n))
        for j in range(n):
            hstep = fd_step if y[j] + fd_step <= 1.0 else -fd_step
            yj = y.copy(); yj[j] += hstep
            i = at(yj)
            if ev.failed[i]:
                return None
            F1, g1, h1 = parts(i)
            JF[:, j] = (F1 - F0) / hstep
            if len(g0):
                Jg[:, j] = (g1 - g0) / hstep
            if len(h0):
                Jh[:, j] = (h1 - h0) / hstep
        return JF, Jg, Jh

    def qp(y, i0, J, which):
        """(8) for objective `which`, or (9) for which = None; (direction, ok)."""
        JF, Jg, Jh = J
        _, g0, h0 = parts(i0)
        lo, hi = -y, 1.0 - y                          # the box in steps
        cons = []
        if len(g0):
            cons.append({"type": "ineq", "fun": lambda z: -(g0 + Jg @ z[:n]),
                         "jac": lambda z: np.hstack([-Jg, np.zeros((len(g0), len(z) - n))])})
        if len(h0):
            cons.append({"type": "eq", "fun": lambda z: h0 + Jh @ z[:n],
                         "jac": lambda z: np.hstack([Jh, np.zeros((len(h0), len(z) - n))])})
        if which is not None:
            gi = JF[which]
            r = sp_minimize(lambda d: gi @ d + 0.5 * d @ d, np.zeros(n), jac=lambda d: gi + d,
                            method="SLSQP", bounds=list(zip(lo, hi)), constraints=cons,
                            options={"maxiter": 500, "ftol": 1e-14})
            return r.x, r.status != 4                # 4: incompatible constraints
        z0 = np.zeros(n + m)
        for i in range(m):
            cons.append({"type": "ineq",
                         "fun": (lambda z, i=i: z[n + i] - (JF[i] @ z[:n] + 0.5 * z[:n] @ z[:n])),
                         "jac": (lambda z, i=i: np.concatenate([-(JF[i] + z[:n]),
                                                                np.eye(m)[i]]))})
        r = sp_minimize(lambda z: np.sum(z[n:]), z0,
                        jac=lambda z: np.concatenate([np.zeros(n), np.ones(m)]),
                        method="SLSQP", bounds=list(zip(lo, hi)) + [(None, None)] * m,
                        constraints=cons, options={"maxiter": 500, "ftol": 1e-14})
        return r.x[:n], r.status != 4

    def armijo(y, i0, d, slope_f, merit):
        """The first α of 1, β, β², ... with merit(y + αd) <= merit(y) +
        μα(slope_f − σ‖c‖₁), σ by MS-4; (point index, α) or (None, α)."""
        c0 = viol(i0)
        sigma = 1.0
        while slope_f - sigma * c0 >= 0.0 and c0 > 0.0 and sigma < 1e8:
            sigma *= 10.0
        slope = slope_f - sigma * c0
        m0 = merit(i0, sigma)
        a = 1.0
        while a >= min_step:
            i = at(y + a * d)
            if not ev.failed[i] and merit(i, sigma) <= m0 + mu * a * slope:
                return i, a
            a *= beta
        return None, a

    def restore(y, i0):
        """min Σ(g⁺)² + Σh² + 1e-6‖x − x_l‖² from x_l; a feasible point or None."""
        if ev.n_cons == 0:
            return None
        def phi(z):
            i = at(z)
            if ev.failed[i]:
                return 1e300
            _, g, h = parts(i)
            return float(np.sum(np.maximum(g, 0.0) ** 2) + np.sum(h ** 2) +
                         1e-6 * np.sum((z - y) ** 2))
        try:
            r = sp_minimize(phi, y, method="SLSQP", bounds=[(0.0, 1.0)] * n,
                            options={"maxiter": 100, "eps": fd_step})
        except (SubBudgetSpent,):
            return None
        i = at(r.x)
        return i if ev.cv(i) <= 0.0 else None

    def dominates_ext(a, b):
        """x_a dominates x_b in (f, g⁺, |h|) (§2.1)."""
        fa = np.concatenate([ev.F[a], np.maximum(parts(a)[1], 0.0), np.abs(parts(a)[2])])
        fb = np.concatenate([ev.F[b], np.maximum(parts(b)[1], 0.0), np.abs(parts(b)[2])])
        return bool(np.all(fa <= fb) and np.any(fa < fb))

    def nondominated(idx):
        out = []
        for a in idx:
            if a in out or ev.failed[a]:
                continue
            if any(dominates_ext(b, a) for b in idx if b != a):
                continue
            out.append(a)
        return out

    def cleanup(idx):
        """Ten of the lowest crowding distance out at a time, down to n̄ (§4)."""
        while len(idx) > nbar:
            F = np.array([ev.F[i] for i in idx])
            cd = np.zeros(len(idx))
            for j in range(m):
                o = np.argsort(F[:, j], kind="stable")
                span = F[o[-1], j] - F[o[0], j]
                cd[o[0]] = cd[o[-1]] = np.inf
                if span > 0:
                    cd[o[1:-1]] += (F[o[2:], j] - F[o[:-2], j]) / span
            drop = set(np.argsort(cd, kind="stable")[:min(10, len(idx) - nbar)].tolist())
            idx = [i for k, i in enumerate(idx) if k not in drop]
        return idx

    try:
        # stage 1: the extreme points, from the box's centre (10)
        centre = at(np.full(n, 0.5))
        identity = P.Frame(m)
        extremes = []
        for i in range(m):
            a0 = np.zeros(m); a0[i] = 1.0
            run_ = S.Run(P.Sub("extreme", a0), ev, identity)
            run_.touched.append(centre)
            S.sqp(run_, np.full(n, 0.5), 0.0, fd_step=fd_step, tol=1e-12)
            extremes.append(S.best_of(run_, run_.touched))
        # stage 2: spread
        tol2 = tau ** 0.25
        min_step = tol2
        X = [at(np.full(n, (l + 1) / (nbar + 1))) for l in range(nbar)]          # MS-5
        stopped = set()
        for _ in range(spread_its):
            T = []
            todo = [x for x in X if x not in stopped]
            if not todo:
                break
            for xl in todo:
                y = (ev.X[xl] - ev.lo) / width
                J = jac(y, xl)
                if J is not None:
                    for i in range(m):
                        d, ok = qp(y, xl, J, i)
                        if not ok:
                            r = restore(y, xl)
                            if r is not None:
                                T.append(r)
                            continue
                        if np.linalg.norm(d) < tol2:
                            continue
                        j, a = armijo(y, xl, d, float(J[0][i] @ d),
                                      lambda k, s, i=i: float(ev.F[k][i]) + s * viol(k))
                        if j is not None:
                            T.append(j)
                stopped.add(xl)
            X = cleanup(nondominated(X + T))
        # stage 3: refining
        X = list(dict.fromkeys(X + [e for e in extremes if e is not None]))
        stopped = set()
        tol3 = math.sqrt(tau)
        min_step = tol3
        while any(x not in stopped for x in X):
            newX = []
            for xl in X:
                if xl in stopped:
                    newX.append(xl)
                    continue
                y = (ev.X[xl] - ev.lo) / width
                J = jac(y, xl)
                if J is None:
                    stopped.add(xl); newX.append(xl); continue
                v, ok = qp(y, xl, J, None)
                if not ok:
                    r = restore(y, xl)
                    if r is None:
                        stopped.add(xl); newX.append(xl)
                    else:
                        newX.append(r)
                    continue
                if np.linalg.norm(v) < tol3:
                    stopped.add(xl); newX.append(xl); continue
                j, a = armijo(y, xl, v, float(np.sum(J[0] @ v)),
                              lambda k, s: float(np.sum(ev.F[k])) + s * viol(k))
                if j is None:
                    stopped.add(xl); newX.append(xl)
                else:
                    newX.append(j)
            feas = [x for x in newX if ev.cv(x) <= 0.0]
            X = [x for x in dict.fromkeys(newX)
                 if not any(f != x and dominates_ext(f, x) for f in feas)]
            stopped &= set(X)
    except (BudgetSpent, SubBudgetSpent):
        pass

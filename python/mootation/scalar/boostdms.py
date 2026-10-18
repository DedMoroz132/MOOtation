# SPDX-License-Identifier: Apache-2.0
"""BoostDMS: direct multisearch with a search step on quadratic models.

C.P. Brás, A.L. Custódio, "On the use of polynomial models in multiobjective
directional direct search", Comput. Optim. Appl. 77(3):897-918, 2020,
doi:10.1007/s10589-020-00233-8 (the preprint, boostdms_bras2020_draft: §4,
Algorithm 2, §5); the minimization of the models by the adapted ε-constraint
of A.L. Custódio, M. Pozzi, E.J. Silva, "Exploring polynomial models in the
Search Step of Direct Multisearch", arXiv:2607.04902, 2026 (§4.2, Eq. 7),
which found it the most promising of the ways tried.

DMS (Custódio et al., SIAM J. Optim. 21(3), 2011, as dms.hpp follows it):
a list of non-dominated points with a step size each, in the box normalized to
[0, 1]^n; the poll centre the first point of the list still worth polling
(α >= 10⁻³); the coordinate directions [I −I], all of them; success when the
list changes — α kept (γ = 1), else halved (β = 1/2); the centre moves to the
end of the list. One step() is one iteration.

THE SEARCH STEP, before every poll:
  1. up to (n + 1)(n + 2) points of the cache (every evaluation), 80 % of
     them from the ball B(x_k; 3Δ_k) and 20 % outside it (§5), the nearest
     first; Δ_k = σ_k α_k max‖d‖ with σ_k = 1/β_1 = 2 (§4);
  2. a quadratic model of every objective centred at x_k: by regression with
     more points than the (n + 1)(n + 2)/2 coefficients, by minimum Frobenius
     norm (the KKT system of Conn, Scheinberg & Vicente) with more than n + 1
     and fewer, by linear interpolation (minimum norm) with at most n + 1;
  3. boost_scalarization = eps (default; Custódio-Pozzi-Silva Eq. 7): for
     every objective i, min m_i(x) s.t. m_j(x) <= f_j(x_k) (j != i),
     ‖x − x_k‖ <= Δ_k, x in the box — m candidates, evaluated together;
     boost_scalarization = minmax (Brás & Custódio, Algorithm 2): by levels
     l = 1..m, for every set I of l objectives, min ζ s.t. m_i(x) <= ζ
     (i ∈ I), ‖x − x_k‖ <= Δ_k — each level's candidates evaluated together;
  4. success when the list changes: the poll is skipped (Algorithm 2, "Check
     for success").
The model subproblems are solved by SciPy's SLSQP with the models' exact
gradients: no evaluation of fn. The run ends when every α is below 10⁻³ or
at the budget; the answer is the list, reduced to pop_size by DSS.

READINGS (declared):
  BD-1. The box normalized to [0, 1]^n (as dms, DMS-1); α_0 = 1; the start
    DMS(n, line), §5.
  BD-2. The candidates are accepted by simple decrease (the list's filter),
    as dms; the papers' integer-lattice or sufficient-decrease globalization
    of the search points is not imposed.
  BD-3. "80 % ... in B(x_k; 3Δ_k)": the nearest points of the ball, then the
    nearest outside it, up to their shares of (n + 1)(n + 2); fewer when the
    cache holds fewer.
  BD-4. A candidate equal to a point of the cache is not evaluated again.
  BD-5. Constraints: the extreme barrier, as dms (an infeasible point never
    enters the list; a start without a feasible point keeps the least
    violating).
"""

from __future__ import annotations

import itertools
import math


def _quad_basis(S):
    """[1, s_1..s_n, ½ s_i², s_i s_j (i < j)] for the rows of S (offsets)."""
    import numpy as np
    S = np.atleast_2d(S)
    n = S.shape[1]
    cols = [np.ones(len(S))] + [S[:, i] for i in range(n)]
    cols += [0.5 * S[:, i] ** 2 for i in range(n)]
    cols += [S[:, i] * S[:, j] for i in range(n) for j in range(i + 1, n)]
    return np.column_stack(cols)


def _unpack(alpha, n):
    """(c, g, H) of the coefficients over _quad_basis."""
    import numpy as np
    c, g = alpha[0], alpha[1:1 + n]
    H = np.zeros((n, n))
    d = alpha[1 + n:1 + 2 * n]
    H[np.diag_indices(n)] = d
    k = 1 + 2 * n
    for i in range(n):
        for j in range(i + 1, n):
            H[i, j] = H[j, i] = alpha[k]
            k += 1
    return c, g, H


def models(S, F):
    """[(c, g, H)] per column of F: quadratic models of the values at the
    offsets S (regression, minimum Frobenius norm or linear, by the number of
    points), every objective from one factorization."""
    import numpy as np
    S = np.atleast_2d(np.asarray(S, float))
    F = np.asarray(F, float).reshape(len(S), -1)
    p, n = S.shape
    q = (n + 1) * (n + 2) // 2
    if p >= q:                                     # regression (interpolation at p = q)
        A = np.linalg.lstsq(_quad_basis(S), F, rcond=None)[0]
        return [_unpack(A[:, i], n) for i in range(F.shape[1])]
    if p > n + 1:                                  # minimum Frobenius norm
        ML = np.column_stack([np.ones(p), S])
        MQ = _quad_basis(S)[:, 1 + n:]
        # off-diagonal terms scaled so that ‖α_Q‖ is the Frobenius norm of H
        w = np.concatenate([np.ones(n), np.full(MQ.shape[1] - n, math.sqrt(2.0))])
        MQs = MQ / w
        K = np.block([[MQs @ MQs.T, ML], [ML.T, np.zeros((n + 1, n + 1))]])
        rhs = np.vstack([F, np.zeros((n + 1, F.shape[1]))])
        sol = np.linalg.lstsq(K, rhs, rcond=None)[0]
        lam, aL = sol[:p], sol[p:]
        aQ = (MQs.T @ lam) / w[:, None]
        return [_unpack(np.concatenate([aL[:, i], aQ[:, i]]), n) for i in range(F.shape[1])]
    ML = np.column_stack([np.ones(p), S])           # linear, minimum norm
    aL = np.linalg.lstsq(ML, F, rcond=None)[0]
    return [(aL[0, i], aL[1:, i], np.zeros((n, n))) for i in range(F.shape[1])]


def run_boostdms(ev, *, n_objs: int, pop_size: int, how: str = "eps", start=None):
    """BoostDMS on the Evaluator ev (evaluator.py), from the archive indices
    start (a warm start or a design) or DMS's line; the answer's indices."""
    import numpy as np
    from scipy.optimize import minimize as sp_minimize
    from .evaluator import BudgetSpent, SubBudgetSpent

    m = int(n_objs)
    n = len(ev.lo)
    width = ev.hi - ev.lo
    tol = 1e-3
    lst = []                                        # [index, α]

    def x_of(u):
        return ev.lo + np.clip(u, 0.0, 1.0) * width

    def u_of(i):
        return (ev.X[i] - ev.lo) / width

    def feasible(i):
        return not ev.failed[i] and ev.cv(i) <= 0.0

    def dominates(i, j):
        a, b = ev.F[i], ev.F[j]
        return bool(np.all(a <= b) and np.any(a < b))

    def filter_in(new, alpha):
        """Add the feasible new points the list does not dominate (BD-2); the
        list changes or not."""
        changed = False
        for i in new:
            if not feasible(i) or any(e[0] == i for e in lst):
                continue
            if any(dominates(e[0], i) or np.array_equal(ev.F[e[0]], ev.F[i]) for e in lst):
                continue
            lst[:] = [e for e in lst if not dominates(i, e[0])]
            lst.append([i, alpha])
            changed = True
        return changed

    def evaluate(U):
        out = []
        for u in U:
            out.append(ev.index(x_of(u)))
        return out

    # the start: DMS(n, line), or the points given
    S0 = [np.full(n, 0.5)] if n < 2 else [np.full(n, i / (n - 1)) for i in range(n)]
    try:
        first = list(start) if start else evaluate(S0)
        if not any(feasible(i) for i in first):     # BD-5
            ok = [i for i in first if not ev.failed[i]] or first
            best = min(ok, key=lambda i: ev.cv(i))
            lst.append([best, 1.0])
        else:
            filter_in(first, 1.0)
        ev.record()
        while True:
            open_ = [e for e in lst if e[1] >= tol]
            if not open_:
                break
            centre = open_[0]
            ic, alpha = centre[0], centre[1]
            xc = u_of(ic)
            success = search(ev, lst, ic, alpha, xc, how, m, n, x_of, u_of, filter_in, np,
                             sp_minimize)
            if not success:
                P = []
                for sign in (1.0, -1.0):
                    for j in range(n):
                        u = xc.copy()
                        u[j] += sign * alpha
                        if 0.0 <= u[j] <= 1.0:
                            P.append(u)
                success = filter_in(evaluate(P), alpha)
            for e in lst:                           # the centre to the end of the list
                if e[0] == ic:
                    if not success:
                        e[1] = alpha * 0.5
                    lst.remove(e)
                    lst.append(e)
                    break
    except (BudgetSpent, SubBudgetSpent):
        pass
    ev.over = True
    idx = [e[0] for e in lst]
    if len(idx) > pop_size:
        from ..run.archive import dss_order
        idx = [idx[k] for k in dss_order(np.array([ev.F[i] for i in idx]), pop_size)]
    return idx


def search(ev, lst, ic, alpha, xc, how, m, n, x_of, u_of, filter_in, np, sp_minimize):
    """The search step (Algorithm 2 / Eq. 7); True on success."""
    delta = 2.0 * alpha                                   # Δ_k = α_k max‖d‖ / β_1
    pmax = (n + 1) * (n + 2)
    I, U = ev.ok_points()                                 # the cache, failures left out
    U = (U - ev.lo) / (ev.hi - ev.lo)
    if len(I) < 2:
        return False
    d = np.linalg.norm(U - xc, axis=1)
    order = np.argsort(d, kind="stable")
    inside = [k for k in order if d[k] <= 3.0 * delta]
    outside = [k for k in order if d[k] > 3.0 * delta]
    n_in = min(len(inside), int(0.8 * pmax))                                       # BD-3
    n_out = min(len(outside), pmax - n_in, int(math.ceil(0.2 * pmax)))
    take = inside[:n_in] + outside[:n_out]
    if len(take) < 2:
        return False
    S = U[take] - xc
    F = np.array([ev.F[I[k]] for k in take])
    mod = models(S, F)
    fk = ev.F[ic]

    def mval(i, s):
        c, g, H = mod[i]
        return c + g @ s + 0.5 * s @ H @ s

    def mgrad(i, s):
        c, g, H = mod[i]
        return g + H @ s

    lo, hi = -xc, 1.0 - xc                        # the box in offsets
    bounds = list(zip(lo, hi))
    tr = {"type": "ineq", "fun": lambda s: delta ** 2 - s @ s, "jac": lambda s: -2.0 * s}

    def solve_eps(i):
        cons = [tr]
        for j in range(m):
            if j != i:
                cons.append({"type": "ineq", "fun": (lambda s, j=j: fk[j] - mval(j, s)),
                             "jac": (lambda s, j=j: -mgrad(j, s))})
        r = sp_minimize(lambda s: mval(i, s), np.zeros(n), jac=lambda s: mgrad(i, s),
                        method="SLSQP", bounds=bounds, constraints=cons,
                        options={"maxiter": 200, "ftol": 1e-12})
        return r.x

    def solve_minmax(Iset):
        z0 = np.append(np.zeros(n), max(fk[i] for i in Iset))
        cons = [{"type": "ineq", "fun": lambda z: delta ** 2 - z[:n] @ z[:n],
                 "jac": lambda z: np.append(-2.0 * z[:n], 0.0)}]
        for i in Iset:
            cons.append({"type": "ineq", "fun": (lambda z, i=i: z[n] - mval(i, z[:n])),
                         "jac": (lambda z, i=i: np.append(-mgrad(i, z[:n]), 1.0))})
        r = sp_minimize(lambda z: z[n], z0, jac=lambda z: np.append(np.zeros(n), 1.0),
                        method="SLSQP", bounds=bounds + [(None, None)], constraints=cons,
                        options={"maxiter": 200, "ftol": 1e-12})
        return r.x[:n]

    def try_points(offsets):
        cand = []
        for s in offsets:
            u = np.clip(xc + s, 0.0, 1.0)
            x = x_of(u)
            if np.ascontiguousarray(x).tobytes() in ev.where:          # BD-4
                continue
            if any(np.array_equal(u, c) for c in cand):
                continue
            cand.append(u)
        if not cand:
            return False
        idx = [ev.index(x_of(u)) for u in cand]
        return filter_in(idx, alpha)

    if how == "eps":
        return try_points([solve_eps(i) for i in range(m)])
    for level in range(1, m + 1):
        if try_points([solve_minmax(Iset) for Iset in itertools.combinations(range(m), level)]):
            return True
    return False

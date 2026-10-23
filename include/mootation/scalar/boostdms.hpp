#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// BoostDMS: direct multisearch with a search step on quadratic models.
//
// C.P. Brás, A.L. Custódio, "On the use of polynomial models in multiobjective
// directional direct search", Comput. Optim. Appl. 77(3):897-918, 2020,
// doi:10.1007/s10589-020-00233-8 (the preprint, boostdms_bras2020_draft: §4,
// Algorithm 2, §5); the minimization of the models by the adapted
// ε-constraint of A.L. Custódio, M. Pozzi, E.J. Silva, "Exploring polynomial
// models in the Search Step of Direct Multisearch", arXiv:2607.04902, 2026
// (§4.2, Eq. 7), which found it the most promising of the ways tried.
//
// DMS (Custódio et al., SIAM J. Optim. 21(3), 2011, as dms.hpp follows it):
// a list of non-dominated points with a step size each, in the box
// normalized to [0, 1]^n; the poll centre the first point of the list still
// worth polling (α >= 10⁻³); the coordinate directions [I −I], all of them;
// success when the list changes — α kept (γ = 1), else halved (β = 1/2); the
// centre moves to the end of the list. One pass of the loop is one iteration.
//
// THE SEARCH STEP, before every poll:
//   1. up to (n + 1)(n + 2) points of the cache (every evaluation), 80 % of
//      them from the ball B(x_k; 3Δ_k) and 20 % outside it (§5), the nearest
//      first; Δ_k = σ_k α_k max‖d‖ with σ_k = 1/β_1 = 2 (§4);
//   2. a quadratic model of every objective centred at x_k: by regression
//      with more points than the (n + 1)(n + 2)/2 coefficients, by minimum
//      Frobenius norm (the KKT system of Conn, Scheinberg & Vicente) with more
//      than n + 1 and fewer, by linear interpolation (minimum norm) with at
//      most n + 1 (least squares by lsq::hfti, the solution of least length);
//   3. boost_scalarization = eps (default; Custódio-Pozzi-Silva Eq. 7): for
//      every objective i, min m_i(x) s.t. m_j(x) <= f_j(x_k) (j != i),
//      ‖x − x_k‖ <= Δ_k, x in the box — m candidates, evaluated together;
//      boost_scalarization = minmax (Brás & Custódio, Algorithm 2): by levels
//      l = 1..m, for every set I of l objectives, min ζ s.t. m_i(x) <= ζ
//      (i ∈ I), ‖x − x_k‖ <= Δ_k — each level's candidates evaluated
//      together;
//   4. success when the list changes: the poll is skipped (Algorithm 2,
//      "Check for success").
// The model subproblems are solved by slsqp.hpp with the models' exact
// gradients: no evaluation of fn (200 iterations, tol 1e-12). The run ends
// when every α is below 10⁻³ or at the budget; the answer is the list,
// reduced to pop_size by DSS.
//
// READINGS (declared):
//   BD-1. The box normalized to [0, 1]^n (as dms, DMS-1); α_0 = 1; the start
//     DMS(n, line), §5.
//   BD-2. The candidates are accepted by simple decrease (the list's filter),
//     as dms; the papers' integer-lattice or sufficient-decrease
//     globalization of the search points is not imposed.
//   BD-3. "80 % ... in B(x_k; 3Δ_k)": the nearest points of the ball, then the
//     nearest outside it, up to their shares of (n + 1)(n + 2); fewer when the
//     cache holds fewer.
//   BD-4. A candidate equal to a point of the cache is not evaluated again.
//   BD-5. Constraints: the extreme barrier, as dms (an infeasible point never
//     enters the list; a start without a feasible point keeps the least
//     violating).
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <vector>

#include "../dss.hpp"
#include "../fp_contract.hpp"
#include "solvers.hpp"

namespace mootation::scalar {

// A quadratic model c + gᵀs + ½ sᵀHs.
struct Model {
    double c = 0.0;
    Vec g;
    lsq::Matrix H;

    double value(const Vec& s) const
    {
        double v = c, q = 0.0;
        for (std::size_t i = 0; i < s.size(); ++i) {
            v += g[i] * s[i];
            double h = 0.0;
            for (std::size_t j = 0; j < s.size(); ++j) h += H(i, j) * s[j];
            q += s[i] * h;
        }
        return v + 0.5 * q;
    }

    Vec grad(const Vec& s) const
    {
        Vec out = g;
        for (std::size_t i = 0; i < s.size(); ++i)
            for (std::size_t j = 0; j < s.size(); ++j) out[i] += H(i, j) * s[j];
        return out;
    }
};

namespace boost_detail {

// [1, s_1..s_n, ½ s_i², s_i s_j (i < j)] for one offset.
inline Vec quad_basis(const Vec& s)
{
    const std::size_t n = s.size();
    Vec b{1.0};
    for (std::size_t i = 0; i < n; ++i) b.push_back(s[i]);
    for (std::size_t i = 0; i < n; ++i) b.push_back(0.5 * s[i] * s[i]);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = i + 1; j < n; ++j) b.push_back(s[i] * s[j]);
    return b;
}

// The model of the coefficients over quad_basis.
inline Model unpack(const Vec& alpha, std::size_t n)
{
    Model md;
    md.c = alpha[0];
    md.g.assign(alpha.begin() + 1, alpha.begin() + 1 + static_cast<std::ptrdiff_t>(n));
    md.H = lsq::Matrix(n, n);
    for (std::size_t i = 0; i < n; ++i) md.H(i, i) = alpha[1 + n + i];
    std::size_t k = 1 + 2 * n;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = i + 1; j < n; ++j) {
            md.H(i, j) = md.H(j, i) = alpha[k];
            ++k;
        }
    return md;
}

}   // namespace boost_detail

// One model per column of F, of the values at the offsets S (regression,
// minimum Frobenius norm or linear, by the number of points).
inline std::vector<Model> models(const std::vector<Vec>& S, const std::vector<Vec>& F)
{
    using namespace boost_detail;
    const std::size_t p = S.size(), n = S[0].size(), cols = F[0].size();
    const std::size_t q = (n + 1) * (n + 2) / 2;
    std::vector<Model> out;
    auto column = [&](std::size_t c) {
        Vec v(p);
        for (std::size_t i = 0; i < p; ++i) v[i] = F[i][c];
        return v;
    };
    if (p >= q) {                                     // regression (interpolation at p = q)
        lsq::Matrix A(p, q);
        for (std::size_t i = 0; i < p; ++i) {
            const Vec b = quad_basis(S[i]);
            for (std::size_t j = 0; j < q; ++j) A(i, j) = b[j];
        }
        for (std::size_t c = 0; c < cols; ++c) out.push_back(unpack(lsq::hfti(A, column(c)), n));
        return out;
    }
    if (p > n + 1) {                                  // minimum Frobenius norm
        const std::size_t nq = q - 1 - n;             // the quadratic coefficients
        // off-diagonal terms scaled so that ‖α_Q‖ is the Frobenius norm of H
        Vec w(nq, std::sqrt(2.0));
        for (std::size_t j = 0; j < n; ++j) w[j] = 1.0;
        lsq::Matrix MQs(p, nq), K(p + n + 1, p + n + 1);
        for (std::size_t i = 0; i < p; ++i) {
            const Vec b = quad_basis(S[i]);
            for (std::size_t j = 0; j < nq; ++j) MQs(i, j) = b[1 + n + j] / w[j];
        }
        for (std::size_t i = 0; i < p; ++i) {
            for (std::size_t j = 0; j < p; ++j) {
                double s = 0.0;
                for (std::size_t k = 0; k < nq; ++k) s += MQs(i, k) * MQs(j, k);
                K(i, j) = s;
            }
            K(i, p) = K(p, i) = 1.0;                  // ML = [1, S]
            for (std::size_t j = 0; j < n; ++j) K(i, p + 1 + j) = K(p + 1 + j, i) = S[i][j];
        }
        for (std::size_t c = 0; c < cols; ++c) {
            Vec rhs = column(c);
            rhs.resize(p + n + 1, 0.0);
            const Vec sol = lsq::hfti(K, rhs);
            Vec alpha(q);
            for (std::size_t j = 0; j <= n; ++j) alpha[j] = sol[p + j];
            for (std::size_t k = 0; k < nq; ++k) {
                double s = 0.0;
                for (std::size_t i = 0; i < p; ++i) s += MQs(i, k) * sol[i];
                alpha[1 + n + k] = s / w[k];
            }
            out.push_back(unpack(alpha, n));
        }
        return out;
    }
    lsq::Matrix ML(p, n + 1);                         // linear, minimum norm
    for (std::size_t i = 0; i < p; ++i) {
        ML(i, 0) = 1.0;
        for (std::size_t j = 0; j < n; ++j) ML(i, 1 + j) = S[i][j];
    }
    for (std::size_t c = 0; c < cols; ++c) {
        const Vec a = lsq::hfti(ML, column(c));
        Model md;
        md.c = a[0];
        md.g.assign(a.begin() + 1, a.end());
        md.H = lsq::Matrix(n, n);
        out.push_back(md);
    }
    return out;
}

// BoostDMS on the Evaluator ev, from the archive indices start (a warm start
// or a design; empty: DMS's line); the answer's indices.
inline std::vector<std::size_t> run_boostdms(Evaluator& ev, int n_objs, int pop_size, bool minmax,
                                             const std::vector<std::size_t>& start)
{
    const std::size_t m = static_cast<std::size_t>(n_objs), n = ev.lo.size();
    std::vector<double> width(n);
    for (std::size_t j = 0; j < n; ++j) width[j] = ev.hi[j] - ev.lo[j];
    const double tol = 1e-3;
    struct Entry {
        std::size_t i;
        double alpha;
    };
    std::vector<Entry> lst;

    auto x_of = [&](const Vec& u) {
        Vec x(n);
        for (std::size_t j = 0; j < n; ++j) x[j] = ev.lo[j] + std::min(1.0, std::max(0.0, u[j])) * width[j];
        return x;
    };
    auto u_of = [&](std::size_t i) {
        Vec u(n);
        for (std::size_t j = 0; j < n; ++j) u[j] = (ev.X[i][j] - ev.lo[j]) / width[j];
        return u;
    };
    auto feasible = [&](std::size_t i) { return !ev.failed[i] && ev.cv(i) <= 0.0; };
    auto dominates = [&](std::size_t a, std::size_t b) {
        bool le = true, lt = false;
        for (std::size_t j = 0; j < m; ++j) {
            le = le && ev.F[a][j] <= ev.F[b][j];
            lt = lt || ev.F[a][j] < ev.F[b][j];
        }
        return le && lt;
    };
    // add the feasible new points the list does not dominate (BD-2); did the list change?
    auto filter_in = [&](const std::vector<std::size_t>& fresh, double alpha) {
        bool changed = false;
        for (std::size_t i : fresh) {
            if (!feasible(i)) continue;
            bool skip = false;
            for (const Entry& e : lst)
                if (e.i == i || dominates(e.i, i) || ev.F[e.i] == ev.F[i]) { skip = true; break; }
            if (skip) continue;
            std::vector<Entry> keep;
            for (const Entry& e : lst)
                if (!dominates(i, e.i)) keep.push_back(e);
            lst = keep;
            lst.push_back({i, alpha});
            changed = true;
        }
        return changed;
    };
    auto evaluate = [&](const std::vector<Vec>& U) {
        std::vector<std::size_t> out;
        for (const Vec& u : U) out.push_back(ev.index(x_of(u)));
        return out;
    };

    // the search step (Algorithm 2 / Eq. 7); true on success
    auto search = [&](std::size_t ic, double alpha, const Vec& xc) {
        const double delta = 2.0 * alpha;               // Δ_k = α_k max‖d‖ / β_1
        const std::size_t pmax = (n + 1) * (n + 2);
        std::vector<std::size_t> I;                       // the cache, failures left out
        for (std::size_t i = 0; i < ev.X.size(); ++i)
            if (!ev.failed[i]) I.push_back(i);
        if (I.size() < 2) return false;
        std::vector<Vec> U(I.size());
        Vec d(I.size());
        for (std::size_t k = 0; k < I.size(); ++k) {
            U[k] = u_of(I[k]);
            double s = 0.0;
            for (std::size_t j = 0; j < n; ++j) s += (U[k][j] - xc[j]) * (U[k][j] - xc[j]);
            d[k] = std::sqrt(s);
        }
        std::vector<std::size_t> order(I.size());
        std::iota(order.begin(), order.end(), std::size_t{0});
        std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return d[a] < d[b]; });
        std::vector<std::size_t> inside, outside;
        for (std::size_t k : order) (d[k] <= 3.0 * delta ? inside : outside).push_back(k);
        const std::size_t n_in = std::min(inside.size(), static_cast<std::size_t>(0.8 * static_cast<double>(pmax)));
        const std::size_t n_out = std::min({outside.size(), pmax - n_in,
                                            static_cast<std::size_t>(std::ceil(0.2 * static_cast<double>(pmax)))});
        std::vector<std::size_t> take(inside.begin(), inside.begin() + static_cast<std::ptrdiff_t>(n_in));
        take.insert(take.end(), outside.begin(), outside.begin() + static_cast<std::ptrdiff_t>(n_out));
        if (take.size() < 2) return false;
        std::vector<Vec> S, Fm;
        for (std::size_t k : take) {
            Vec s(n);
            for (std::size_t j = 0; j < n; ++j) s[j] = U[k][j] - xc[j];
            S.push_back(s);
            Fm.push_back(ev.F[I[k]]);
        }
        const std::vector<Model> mod = models(S, Fm);
        const Vec fk = ev.F[ic];

        // the trust region and the box on the offsets (the first n variables)
        auto base_nlp = [&](std::size_t extra) {
            slsqp::NLP nlp;
            nlp.n = n + extra;
            nlp.lo.assign(n + extra, -std::numeric_limits<double>::infinity());
            nlp.hi.assign(n + extra, std::numeric_limits<double>::infinity());
            for (std::size_t j = 0; j < n; ++j) {
                nlp.lo[j] = -xc[j];
                nlp.hi[j] = 1.0 - xc[j];
            }
            return nlp;
        };
        auto head = [&](const Vec& z) { return Vec(z.begin(), z.begin() + static_cast<std::ptrdiff_t>(n)); };

        auto solve_eps = [&](std::size_t i) {
            slsqp::NLP nlp = base_nlp(0);
            nlp.n_ineq = m;                               // the trust region, then m_j <= f_j(x_k)
            nlp.values = [&](const Vec& s, Vec& c) {
                c.assign(m, 0.0);
                c[0] = delta * delta - dot(s, s);
                std::size_t r = 1;
                for (std::size_t j = 0; j < m; ++j)
                    if (j != i) c[r++] = fk[j] - mod[j].value(s);
                return mod[i].value(s);
            };
            nlp.derivatives = [&](const Vec& s, Vec& g, lsq::Matrix& A) {
                g = mod[i].grad(s);
                A = lsq::Matrix(m, n);
                for (std::size_t j = 0; j < n; ++j) A(0, j) = -2.0 * s[j];
                std::size_t r = 1;
                for (std::size_t j = 0; j < m; ++j)
                    if (j != i) {
                        const Vec gr = mod[j].grad(s);
                        for (std::size_t c = 0; c < n; ++c) A(r, c) = -gr[c];
                        ++r;
                    }
            };
            return slsqp::minimize(nlp, Vec(n, 0.0), {1e-12, 200}).x;
        };

        auto solve_minmax = [&](const std::vector<std::size_t>& Iset) {
            slsqp::NLP nlp = base_nlp(1);
            nlp.n_ineq = 1 + Iset.size();
            nlp.values = [&](const Vec& z, Vec& c) {
                const Vec s = head(z);
                c.assign(1 + Iset.size(), 0.0);
                c[0] = delta * delta - dot(s, s);
                for (std::size_t r = 0; r < Iset.size(); ++r) c[1 + r] = z[n] - mod[Iset[r]].value(s);
                return z[n];
            };
            nlp.derivatives = [&](const Vec& z, Vec& g, lsq::Matrix& A) {
                const Vec s = head(z);
                g.assign(n + 1, 0.0);
                g[n] = 1.0;
                A = lsq::Matrix(1 + Iset.size(), n + 1);
                for (std::size_t j = 0; j < n; ++j) A(0, j) = -2.0 * s[j];
                for (std::size_t r = 0; r < Iset.size(); ++r) {
                    const Vec gr = mod[Iset[r]].grad(s);
                    for (std::size_t j = 0; j < n; ++j) A(1 + r, j) = -gr[j];
                    A(1 + r, n) = 1.0;
                }
            };
            Vec z0(n, 0.0);
            double top = fk[Iset[0]];
            for (std::size_t i : Iset) top = std::max(top, fk[i]);
            z0.push_back(top);
            return head(slsqp::minimize(nlp, z0, {1e-12, 200}).x);
        };

        auto try_points = [&](const std::vector<Vec>& offsets) {
            std::vector<Vec> cand;
            for (const Vec& s : offsets) {
                Vec u(n);
                for (std::size_t j = 0; j < n; ++j) u[j] = std::min(1.0, std::max(0.0, xc[j] + s[j]));
                if (ev.known(x_of(u))) continue;          // BD-4
                if (std::find(cand.begin(), cand.end(), u) != cand.end()) continue;
                cand.push_back(u);
            }
            if (cand.empty()) return false;
            return filter_in(evaluate(cand), alpha);
        };

        if (!minmax) {
            std::vector<Vec> offs;
            for (std::size_t i = 0; i < m; ++i) offs.push_back(solve_eps(i));
            return try_points(offs);
        }
        for (std::size_t level = 1; level <= m; ++level) {
            // every set of `level` objectives, in itertools.combinations order
            std::vector<Vec> offs;
            std::vector<std::size_t> comb(level);
            std::iota(comb.begin(), comb.end(), std::size_t{0});
            for (;;) {
                offs.push_back(solve_minmax(comb));
                std::size_t r = level;
                while (r > 0 && comb[r - 1] == m - level + r - 1) --r;
                if (r == 0) break;
                ++comb[r - 1];
                for (std::size_t t = r; t < level; ++t) comb[t] = comb[t - 1] + 1;
            }
            if (try_points(offs)) return true;
        }
        return false;
    };

    // the start: DMS(n, line), or the points given
    std::vector<Vec> S0;
    if (n < 2) S0.push_back(Vec(n, 0.5));
    else
        for (std::size_t i = 0; i < n; ++i)
            S0.push_back(Vec(n, static_cast<double>(i) / static_cast<double>(n - 1)));
    try {
        const std::vector<std::size_t> first = !start.empty() ? start : evaluate(S0);
        bool any_feasible = false;
        for (std::size_t i : first) any_feasible = any_feasible || feasible(i);
        if (!any_feasible) {                              // BD-5
            std::vector<std::size_t> ok;
            for (std::size_t i : first)
                if (!ev.failed[i]) ok.push_back(i);
            if (ok.empty()) ok = first;
            std::size_t best = ok[0];
            for (std::size_t i : ok)
                if (ev.cv(i) < ev.cv(best)) best = i;
            lst.push_back({best, 1.0});
        } else {
            filter_in(first, 1.0);
        }
        ev.record();
        for (;;) {
            long long c = -1;
            for (std::size_t k = 0; k < lst.size(); ++k)
                if (lst[k].alpha >= tol) { c = static_cast<long long>(k); break; }
            if (c < 0) break;
            const std::size_t ic = lst[static_cast<std::size_t>(c)].i;
            const double alpha = lst[static_cast<std::size_t>(c)].alpha;
            const Vec xc = u_of(ic);
            bool success = search(ic, alpha, xc);
            if (!success) {
                std::vector<Vec> P;
                for (double sign : {1.0, -1.0})
                    for (std::size_t j = 0; j < n; ++j) {
                        Vec u = xc;
                        u[j] += sign * alpha;
                        if (0.0 <= u[j] && u[j] <= 1.0) P.push_back(u);
                    }
                success = filter_in(evaluate(P), alpha);
            }
            for (std::size_t k = 0; k < lst.size(); ++k)  // the centre to the end of the list
                if (lst[k].i == ic) {
                    Entry e = lst[k];
                    if (!success) e.alpha = alpha * 0.5;
                    lst.erase(lst.begin() + static_cast<std::ptrdiff_t>(k));
                    lst.push_back(e);
                    break;
                }
        }
    } catch (const BudgetSpent&) {
    } catch (const SubBudgetSpent&) {
    }
    ev.over = true;
    std::vector<std::size_t> idx;
    for (const Entry& e : lst) idx.push_back(e.i);
    if (idx.size() > static_cast<std::size_t>(pop_size)) {
        Rows Fn;
        for (std::size_t i : idx) Fn.push_back(ev.F[i]);
        std::vector<std::size_t> order = dss::order(Fn, static_cast<std::size_t>(pop_size));
        std::vector<std::size_t> out;
        for (std::size_t k : order) out.push_back(idx[k]);
        idx = out;
    }
    return idx;
}

}   // namespace mootation::scalar

#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// MOSQP: multiobjective SQP on a set of points, without scalarization.
//
// J. Fliege, A.I.F. Vaz, "A method for constrained multiobjective optimization
// based on SQP techniques", SIAM J. Optim. 26(4):2091-2119, 2016,
// doi:10.1137/15M1016424 (the preprint, Optimization Online 2015:
// mosqp_fliege2016_draft; §2.1, §3 problems (8)-(9), Algorithm 4.1, §6).
//
//   stage 1  the extreme points: min f_i under the constraints, from the
//            box's centre ((10), MS-7) — SQP with finite differences
//            (solvers.hpp);
//   stage 2  "spread", at most mosqp_spread (20, §6) iterations over a set X
//            of n̄ = pop_size points, first on the line between the bounds
//            (§6, "line"): every point not yet stopped gives, for every
//            objective i, the direction d of the QP (8) — min ∇f_iᵀd + ½dᵀH_i d
//            under the linearized constraints and the box — and, when
//            ‖d‖ >= τ^(1/4), the point x + αd, α the first of 1, β, β², ...
//            with Armijo's decrease of the ℓ1 merit φ_i(x, σ) = f_i + σ(Σ g⁺ +
//            Σ|h|) (§2.1) and α >= τ^(1/2); the point is then stopped; X
//            becomes the non-dominated points of X and the new ones
//            (dominance extended to (f, g⁺, |h|), §2.1), cut to n̄ by crowding
//            distance (§4: ten of the most crowded at a time);
//   stage 3  "refining", on X and the extreme points: every point not yet
//            stopped moves along the solution v of (9) — min Σζ_i,
//            ∇f_iᵀv + ½vᵀH_i v <= ζ_i, the linearized constraints, the box
//            (Lemma 3.2, item 6: min Σ_i (∇f_iᵀv + ½vᵀH_i v) under the rows) —
//            by Armijo on the merit Φ = Σf_i + σ(Σg⁺ + Σ|h|), until ‖v‖ < τ
//            or α < τ^(1/2); points dominated by feasible points leave X.
// A QP without a feasible point sends the point to a restoration: min
// Σ(g⁺)² + Σh² + 10⁻⁶‖x − x_l‖² from x_l (SQP, finite differences); a
// feasible result replaces x_l (stage 3) or joins T (stage 2).
//
// Every evaluation goes through the evaluator (budget, archive, answer).
//
// READINGS (the preprint gives no numbers for τ, μ, β, n̄; §6 is
// MATLAB-specific):
//   MS-1. τ = 1e-8 (mosqp_tau), μ = 1e-4 (mosqp_mu, Armijo), β = 1/2
//     (mosqp_beta); n̄ = pop_size.
//   MS-2. H_i = I in both stages: the first of §6's three cases (the others
//     need second derivatives the problems here do not give).
//   MS-3. Gradients and Jacobians by forward differences (fd_step in the box
//     normalized to [0, 1], as sqp in solvers.hpp), n evaluations a point.
//   MS-4. σ: the first of 1, 10, 100, ... that makes the merit's directional
//     derivative ∇fᵀd − σ‖c(x)‖₁ negative (Algorithm 4.1, 3(a)iD), at most 10⁸.
//   MS-5. The line of stage 2: x_l = ℓ + l(u − ℓ)/(n̄ + 1), l = 1..n̄.
//   MS-6. With H_i = I, (8) is the least-squares problem min ‖d + ∇f_i‖
//     under the linear rows, and (9), which Lemma 3.2 (item 6) writes as
//     min Σ_i (∇f_iᵀv + ½vᵀH_i v) under the same rows, is min
//     ‖√m v + Σ_i ∇f_i / √m‖: one lsq::lsei each, exact, no evaluation of fn;
//     "no feasible solution" is lsq's report that the linear rows are
//     incompatible. The restoration is solved by slsqp.hpp with finite
//     differences; it stops at 100 iterations or tol 1e-6.
//   MS-7. (10) prints (u_j − ℓ_j)/2 for a finite box, which lies outside
//     [ℓ_j, u_j] whenever ℓ_j > (u_j − ℓ_j)/2; read as the centre (u_j + ℓ_j)/2.
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <set>
#include <vector>

#include "../detmath.hpp"
#include "../fp_contract.hpp"
#include "solvers.hpp"

namespace mootation::scalar {

struct MOSQPKnobs {
    double tau = 1e-8, mu = 1e-4, beta = 0.5, spread = 20, fd_step = 1.4901161193847656e-08;
};

inline void run_mosqp(Evaluator& ev, int n_objs, int n_eq, int pop_size, const MOSQPKnobs& kn)
{
    const double tau = kn.tau, mu = kn.mu, beta = kn.beta, fd_step = kn.fd_step;
    const int spread_its = static_cast<int>(kn.spread);
    const std::size_t m = static_cast<std::size_t>(n_objs), n = ev.lo.size();
    const std::size_t k_in = static_cast<std::size_t>(ev.n_cons - n_eq);
    const std::size_t ne = static_cast<std::size_t>(n_eq);
    const std::size_t nbar = static_cast<std::size_t>(pop_size);
    std::vector<double> width(n);
    for (std::size_t j = 0; j < n; ++j) width[j] = ev.hi[j] - ev.lo[j];

    auto at = [&](const Vec& y) {
        Vec x(n);
        for (std::size_t j = 0; j < n; ++j) x[j] = ev.lo[j] + std::min(1.0, std::max(0.0, y[j])) * width[j];
        return ev.index(x);
    };
    auto y_of = [&](std::size_t i) {
        Vec y(n);
        for (std::size_t j = 0; j < n; ++j) y[j] = (ev.X[i][j] - ev.lo[j]) / width[j];
        return y;
    };
    auto viol = [&](std::size_t i) {
        double a = 0.0, b = 0.0;
        for (std::size_t j = 0; j < k_in; ++j) a += std::max(ev.G[i][j], 0.0);
        for (std::size_t j = 0; j < ne; ++j) b += std::fabs(ev.G[i][k_in + j]);
        return a + b;
    };
    auto norm = [](const Vec& v) {
        double s = 0.0;
        for (double t : v) s += t * t;
        return std::sqrt(s);
    };

    struct Jac {
        lsq::Matrix JF, Jg, Jh;
    };
    // (JF, Jg, Jh) at y (normalized) by forward differences; false at a failure
    auto jac = [&](const Vec& y, std::size_t i0, Jac& J) {
        const Vec F0 = ev.F[i0];         // copies: the archive grows below
        const Vec G0 = ev.G[i0];
        J.JF = lsq::Matrix(m, n);
        J.Jg = lsq::Matrix(k_in, n);
        J.Jh = lsq::Matrix(ne, n);
        for (std::size_t j = 0; j < n; ++j) {
            const double hs = y[j] + fd_step <= 1.0 ? fd_step : -fd_step;
            Vec yj = y;
            yj[j] += hs;
            const std::size_t i = at(yj);
            if (ev.failed[i]) return false;
            for (std::size_t r = 0; r < m; ++r) J.JF(r, j) = (ev.F[i][r] - F0[r]) / hs;
            for (std::size_t r = 0; r < k_in; ++r) J.Jg(r, j) = (ev.G[i][r] - G0[r]) / hs;
            for (std::size_t r = 0; r < ne; ++r) J.Jh(r, j) = (ev.G[i][k_in + r] - G0[k_in + r]) / hs;
        }
        return true;
    };

    // The linear rows of (8) and (9) in v (n columns, `extra` more of zeros):
    // h0 + Jh v = 0; -(g0 + Jg v) >= 0; -y <= v <= 1 - y.
    auto linear_rows = [&](const Vec& y, std::size_t i0, const Jac& J, std::size_t extra,
                           lsq::Matrix& C, Vec& d, lsq::Matrix& G, Vec& h) {
        const std::size_t k = n + extra;
        C = lsq::Matrix(ne, k);
        d.assign(ne, 0.0);
        for (std::size_t r = 0; r < ne; ++r) {
            for (std::size_t j = 0; j < n; ++j) C(r, j) = J.Jh(r, j);
            d[r] = -ev.G[i0][k_in + r];
        }
        G = lsq::Matrix(k_in + 2 * n, k);
        h.assign(k_in + 2 * n, 0.0);
        for (std::size_t r = 0; r < k_in; ++r) {
            for (std::size_t j = 0; j < n; ++j) G(r, j) = -J.Jg(r, j);
            h[r] = ev.G[i0][r];
        }
        for (std::size_t j = 0; j < n; ++j) {
            G(k_in + 2 * j, j) = 1.0;
            h[k_in + 2 * j] = -y[j];
            G(k_in + 2 * j + 1, j) = -1.0;
            h[k_in + 2 * j + 1] = y[j] - 1.0;
        }
    };

    // (8) for objective `which`, or (9) for which = m; (direction, ok). With
    // H_i = I, (8) is min ||d + grad f_i|| and (9) — by Lemma 3.2, item 6, the
    // problem min sum_i (grad f_iᵀv + ½vᵀv) — is min ||sqrt(m) v + s / sqrt(m)||,
    // s = sum_i grad f_i: one least-squares problem each under the linear rows
    // (MS-6).
    auto qp = [&](const Vec& y, std::size_t i0, const Jac& J, std::size_t which, Vec& dir) {
        lsq::Matrix C, G;
        Vec d, h;
        linear_rows(y, i0, J, 0, C, d, G, h);
        const double w = which < m ? 1.0 : std::sqrt(static_cast<double>(m));
        lsq::Matrix E(n, n);
        Vec f(n, 0.0);
        for (std::size_t j = 0; j < n; ++j) {
            E(j, j) = w;
            if (which < m) f[j] = -J.JF(which, j);
            else {
                double sj = 0.0;
                for (std::size_t i = 0; i < m; ++i) sj += J.JF(i, j);
                f[j] = -sj / w;
            }
        }
        lsq::LSIResult s = lsq::lsei(E, f, C, d, G, h);
        if (s.status == lsq::Status::incompatible) return false;
        dir = s.status == lsq::Status::ok ? s.x : Vec(n, 0.0);
        return true;
    };

    double min_step = std::sqrt(tau);
    // The first α of 1, β, β², ... with merit(y + αd) <= merit(y) + μα(slope_f −
    // σ‖c‖₁), σ by MS-4; the point's index, or -1.
    auto armijo = [&](const Vec& y, std::size_t i0, const Vec& d, double slope_f, auto merit) -> long long {
        const double c0 = viol(i0);
        double sigma = 1.0;
        while (slope_f - sigma * c0 >= 0.0 && c0 > 0.0 && sigma < 1e8) sigma *= 10.0;
        const double slope = slope_f - sigma * c0;
        const double m0 = merit(i0, sigma);
        double a = 1.0;
        while (a >= min_step) {
            Vec yt(n);
            for (std::size_t j = 0; j < n; ++j) yt[j] = y[j] + a * d[j];
            const std::size_t i = at(yt);
            if (!ev.failed[i] && merit(i, sigma) <= m0 + mu * a * slope) return static_cast<long long>(i);
            a *= beta;
        }
        return -1;
    };

    // min Σ(g⁺)² + Σh² + 1e-6‖x − x_l‖² from x_l; a feasible point or -1.
    auto restore = [&](const Vec& y) -> long long {
        if (ev.n_cons == 0) return -1;
        auto phi = [&](const Vec& z) {
            const std::size_t i = at(z);
            if (ev.failed[i]) return 1e300;
            double s = 0.0, t = 0.0, u = 0.0;
            for (std::size_t j = 0; j < k_in; ++j) s += std::max(ev.G[i][j], 0.0) * std::max(ev.G[i][j], 0.0);
            for (std::size_t j = 0; j < ne; ++j) t += ev.G[i][k_in + j] * ev.G[i][k_in + j];
            for (std::size_t j = 0; j < n; ++j) u += (z[j] - y[j]) * (z[j] - y[j]);
            return s + t + 1e-6 * u;
        };
        slsqp::NLP nlp;
        nlp.n = n;
        nlp.lo.assign(n, 0.0);
        nlp.hi.assign(n, 1.0);
        nlp.values = [&](const Vec& z, Vec&) { return phi(z); };
        nlp.derivatives = [&](const Vec& z, Vec& g, lsq::Matrix&) {   // MS-3
            const double p0 = phi(z);
            g.assign(n, 0.0);
            for (std::size_t j = 0; j < n; ++j) {
                const double hs = z[j] + fd_step <= 1.0 ? fd_step : -fd_step;
                Vec zj = z;
                zj[j] += hs;
                g[j] = (phi(zj) - p0) / hs;
            }
        };
        Vec x = y;
        try {
            x = slsqp::minimize(nlp, y, {1e-6, 100}).x;
        } catch (const SubBudgetSpent&) {
            return -1;
        }
        const std::size_t i = at(x);
        return ev.cv(i) <= 0.0 ? static_cast<long long>(i) : -1;
    };

    // x_a dominates x_b in (f, g⁺, |h|) (§2.1)
    auto ext = [&](std::size_t i) {
        Vec v = ev.F[i];
        for (std::size_t j = 0; j < k_in; ++j) v.push_back(std::max(ev.G[i][j], 0.0));
        for (std::size_t j = 0; j < ne; ++j) v.push_back(std::fabs(ev.G[i][k_in + j]));
        return v;
    };
    auto dominates_ext = [&](std::size_t a, std::size_t b) {
        const Vec fa = ext(a), fb = ext(b);
        bool all_le = true, any_lt = false;
        for (std::size_t j = 0; j < fa.size(); ++j) {
            all_le = all_le && fa[j] <= fb[j];
            any_lt = any_lt || fa[j] < fb[j];
        }
        return all_le && any_lt;
    };
    auto contains = [](const std::vector<std::size_t>& v, std::size_t x) {
        return std::find(v.begin(), v.end(), x) != v.end();
    };
    auto unique = [&](const std::vector<std::size_t>& v) {         // dict.fromkeys
        std::vector<std::size_t> out;
        for (std::size_t x : v)
            if (!contains(out, x)) out.push_back(x);
        return out;
    };
    auto nondominated = [&](const std::vector<std::size_t>& idx) {
        std::vector<std::size_t> out;
        for (std::size_t a : idx) {
            if (contains(out, a) || ev.failed[a]) continue;
            bool dom = false;
            for (std::size_t b : idx)
                if (b != a && dominates_ext(b, a)) { dom = true; break; }
            if (!dom) out.push_back(a);
        }
        return out;
    };
    // ten of the lowest crowding distance out at a time, down to n̄ (§4)
    auto cleanup = [&](std::vector<std::size_t> idx) {
        while (idx.size() > nbar) {
            const std::size_t N = idx.size();
            Vec cd(N, 0.0);
            for (std::size_t j = 0; j < m; ++j) {
                std::vector<std::size_t> o(N);
                std::iota(o.begin(), o.end(), std::size_t{0});
                std::stable_sort(o.begin(), o.end(), [&](std::size_t a, std::size_t b) {
                    return ev.F[idx[a]][j] < ev.F[idx[b]][j];
                });
                const double span = ev.F[idx[o[N - 1]]][j] - ev.F[idx[o[0]]][j];
                cd[o[0]] = cd[o[N - 1]] = std::numeric_limits<double>::infinity();
                if (span > 0)
                    for (std::size_t k = 1; k + 1 < N; ++k)
                        cd[o[k]] += (ev.F[idx[o[k + 1]]][j] - ev.F[idx[o[k - 1]]][j]) / span;
            }
            std::vector<std::size_t> o(N);
            std::iota(o.begin(), o.end(), std::size_t{0});
            std::stable_sort(o.begin(), o.end(), [&](std::size_t a, std::size_t b) { return cd[a] < cd[b]; });
            const std::size_t k_drop = std::min<std::size_t>(10, N - nbar);
            std::vector<char> drop(N, 0);
            for (std::size_t k = 0; k < k_drop; ++k) drop[o[k]] = 1;
            std::vector<std::size_t> keep;
            for (std::size_t k = 0; k < N; ++k)
                if (!drop[k]) keep.push_back(idx[k]);
            idx = keep;
        }
        return idx;
    };

    try {
        // stage 1: the extreme points, from the box's centre (10)
        const std::size_t centre = at(Vec(n, 0.5));
        const Frame identity(m);
        std::vector<long long> extremes;
        for (std::size_t i = 0; i < m; ++i) {
            Sub s;
            s.kind = "extreme";
            s.a0 = unit(m, i);
            Run run(s, ev, identity);
            run.touched.push_back(centre);
            sqp(run, Vec(n, 0.5), 0.0, fd_step, 1e-12);
            extremes.push_back(best_of(run, run.touched));
        }
        // stage 2: spread
        const double tol2 = dm::pow(tau, 0.25);                  // 3(a)iC
        min_step = std::sqrt(tau);                               // 3(a)iE
        std::vector<std::size_t> X;
        for (std::size_t l = 0; l < nbar; ++l)                   // MS-5
            X.push_back(at(Vec(n, static_cast<double>(l + 1) / static_cast<double>(nbar + 1))));
        std::vector<std::size_t> stopped;
        for (int it = 0; it < spread_its; ++it) {
            std::vector<std::size_t> T, todo;
            for (std::size_t x : X)
                if (!contains(stopped, x)) todo.push_back(x);
            if (todo.empty()) break;
            for (std::size_t xl : todo) {
                const Vec y = y_of(xl);
                Jac J;
                if (jac(y, xl, J)) {
                    for (std::size_t i = 0; i < m; ++i) {
                        Vec d;
                        if (!qp(y, xl, J, i, d)) {
                            const long long r = restore(y);
                            if (r >= 0) T.push_back(static_cast<std::size_t>(r));
                            continue;
                        }
                        if (norm(d) < tol2) continue;
                        double slope = 0.0;
                        for (std::size_t j = 0; j < n; ++j) slope += J.JF(i, j) * d[j];
                        const long long jdx = armijo(y, xl, d, slope, [&](std::size_t k, double s) {
                            return ev.F[k][i] + s * viol(k);
                        });
                        if (jdx >= 0) T.push_back(static_cast<std::size_t>(jdx));
                    }
                }
                stopped.push_back(xl);
            }
            std::vector<std::size_t> both = X;
            both.insert(both.end(), T.begin(), T.end());
            X = cleanup(nondominated(both));
        }
        // stage 3: refining
        for (long long e : extremes)
            if (e >= 0) X.push_back(static_cast<std::size_t>(e));
        X = unique(X);
        stopped.clear();
        const double tol3 = tau;                                 // 4(a)iii
        min_step = std::sqrt(tau);                               // 4(a)v
        for (;;) {
            bool any_open = false;
            for (std::size_t x : X) any_open = any_open || !contains(stopped, x);
            if (!any_open) break;
            std::vector<std::size_t> newX;
            for (std::size_t xl : X) {
                if (contains(stopped, xl)) { newX.push_back(xl); continue; }
                const Vec y = y_of(xl);
                Jac J;
                if (!jac(y, xl, J)) { stopped.push_back(xl); newX.push_back(xl); continue; }
                Vec v;
                if (!qp(y, xl, J, m, v)) {
                    const long long r = restore(y);
                    if (r < 0) { stopped.push_back(xl); newX.push_back(xl); }
                    else newX.push_back(static_cast<std::size_t>(r));
                    continue;
                }
                if (norm(v) < tol3) { stopped.push_back(xl); newX.push_back(xl); continue; }
                double slope = 0.0;
                for (std::size_t i = 0; i < m; ++i)
                    for (std::size_t j = 0; j < n; ++j) slope += J.JF(i, j) * v[j];
                const long long jdx = armijo(y, xl, v, slope, [&](std::size_t k, double s) {
                    double sum = 0.0;
                    for (double t : ev.F[k]) sum += t;
                    return sum + s * viol(k);
                });
                if (jdx < 0) { stopped.push_back(xl); newX.push_back(xl); }
                else newX.push_back(static_cast<std::size_t>(jdx));
            }
            std::vector<std::size_t> feas;
            for (std::size_t x : newX)
                if (ev.cv(x) <= 0.0) feas.push_back(x);
            std::vector<std::size_t> next;
            for (std::size_t x : unique(newX)) {
                bool dom = false;
                for (std::size_t f : feas)
                    if (f != x && dominates_ext(f, x)) { dom = true; break; }
                if (!dom) next.push_back(x);
            }
            X = next;
            std::vector<std::size_t> st;
            for (std::size_t s : stopped)
                if (contains(X, s) && !contains(st, s)) st.push_back(s);
            stopped = st;
        }
    } catch (const BudgetSpent&) {
    } catch (const SubBudgetSpent&) {
    }
}

}   // namespace mootation::scalar

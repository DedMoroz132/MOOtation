#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// Stacking (task 28, §28.3): the weights of K members' responses that fit the
// data best, nonnegative — M. LeBlanc, R. Tibshirani, "Combining estimates
// in regression and classification", JASA 91(436) 1996, 1641-1650, and L.
// Breiman, "Stacked regressions", Machine Learning 24 1996, 49-64 (both in
// the reading of 11.11, 05-поиск-отжиг-смеси-ансамбли.md): the stacked
// response sum_k alpha_k yhat_k with alpha >= 0, the alpha of
//
//   min || W^(1/2) (Yhat alpha - y) ||   subject to alpha >= 0
//                                         [and sum_k alpha_k = 1]
//
// Yhat (m x K) the members' responses on m points none of them was fitted
// to (their out-of-sample responses, from bootstrap.hpp's out-of-bag blocks
// or a cross-validation by blocks), y the data there, W the points' weights.
// NNLS (lsq.hpp, Lawson & Hanson's Algorithm 23.10); with sum_to_one, lsq.hpp's
// lsei (the equality eliminated, then LSI). Nonnegativity is required (both
// papers: without it the weights of correlated members run to large values
// of both signs); sum to one does not lose accuracy (LeBlanc & Tibshirani)
// and keeps the stack a mean of devices.
//
// Readings (OURS):
//   STK-1. One member: alpha = 1 and no fit (the stack of one is the member).
//   STK-2. sum_to_one with Yhat not of full column rank (two members alike;
//     a diagonal of its QR within max(m, K) eps |R_00|, lsq.hpp's LSQ-6, or
//     lsei refusing it: LSI needs E of full column rank): then NNLS
//     on Yhat with the row sqrt(rho) (1, ..., 1 | 1) added, rho = 1e6
//     max(1, ||W^(1/2) Yhat||_F^2), the equality as a penalty: sum alpha = 1
//     within about 1 / rho of its scale (Result::method says which).
//   STK-3. A member whose column is all zero has weight 0 (NNLS leaves it out).
//   STK-4. Refused: K = 0, m = 0, rows of unequal length, y not m long, a
//     weight < 0 or not finite, a value not finite.
// ============================================================================

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "../fp_contract.hpp"
#include "lsq.hpp"

namespace mootation::scalar::stacking {

using Vec = std::vector<double>;

struct Options {
    bool sum_to_one = false;
};

struct Result {
    Vec alpha;                      // one weight a member, >= 0
    double rss = 0.0;               // || W^(1/2) (Yhat alpha - y) ||^2
    std::string method;             // "one" (STK-1), "nnls", "lsei", "nnls_penalty" (STK-2)
};

// Yhat: m rows of K members' responses; y: the m data; w: the rows' weights (empty: 1)
inline Result stack(const std::vector<Vec>& Yhat, const Vec& y, const Vec& w = {}, const Options& o = {})
{
    const std::size_t m = Yhat.size();
    if (m == 0) throw std::invalid_argument("stack: no points (STK-4)");
    const std::size_t K = Yhat[0].size();
    if (K == 0) throw std::invalid_argument("stack: no members (STK-4)");
    if (y.size() != m) throw std::invalid_argument("stack: y needs one value a point (STK-4)");
    if (!w.empty() && w.size() != m) throw std::invalid_argument("stack: w needs one weight a point (STK-4)");
    for (std::size_t i = 0; i < m; ++i) {
        if (Yhat[i].size() != K) throw std::invalid_argument("stack: every row needs K responses (STK-4)");
        if (!std::isfinite(y[i])) throw std::invalid_argument("stack: a value not finite (STK-4)");
        if (!w.empty() && !(w[i] >= 0.0 && std::isfinite(w[i])))
            throw std::invalid_argument("stack: a weight < 0 or not finite (STK-4)");
        for (double v : Yhat[i])
            if (!std::isfinite(v)) throw std::invalid_argument("stack: a response not finite (STK-4)");
    }
    Result r;
    auto rss_of = [&](const Vec& a) {
        double s = 0.0;
        for (std::size_t i = 0; i < m; ++i) {
            double e = -y[i];
            for (std::size_t k = 0; k < K; ++k) e += Yhat[i][k] * a[k];
            s += (w.empty() ? 1.0 : w[i]) * e * e;
        }
        return s;
    };
    if (K == 1) {                                     // STK-1
        r.alpha = {1.0};
        r.rss = rss_of(r.alpha);
        r.method = "one";
        return r;
    }
    lsq::Matrix E(m, K);
    Vec f(m);
    double norm2 = 0.0;
    for (std::size_t i = 0; i < m; ++i) {
        const double s = w.empty() ? 1.0 : std::sqrt(w[i]);
        for (std::size_t k = 0; k < K; ++k) {
            E(i, k) = s * Yhat[i][k];
            norm2 += E(i, k) * E(i, k);
        }
        f[i] = s * y[i];
    }
    if (!o.sum_to_one) {
        r.alpha = lsq::nnls(E, f).x;
        r.method = "nnls";
    } else {
        lsq::Matrix C(1, K), G(K, K);
        for (std::size_t k = 0; k < K; ++k) {
            C(0, k) = 1.0;
            G(k, k) = 1.0;
        }
        bool full = m >= K;                           // STK-2: E's column rank, as lsq.hpp's LSQ-6
        if (full) {
            const lsq::QR qr(E);
            const double tol = static_cast<double>(m > K ? m : K) * lsq::kEps * std::fabs(qr.A(0, 0));
            for (std::size_t j = 0; j < K; ++j) full = full && std::fabs(qr.A(j, j)) > tol;
        }
        const lsq::LSIResult e = full ? lsq::lsei(E, f, C, {1.0}, G, Vec(K, 0.0)) : lsq::LSIResult{};
        if (full && e.status == lsq::Status::ok) {
            r.alpha = e.x;
            for (double& a : r.alpha) a = a < 0.0 ? 0.0 : a;   // round-off below the bound
            r.method = "lsei";
        } else {                                      // STK-2
            const double rho = 1e6 * (norm2 > 1.0 ? norm2 : 1.0), s = std::sqrt(rho);
            lsq::Matrix Ea(m + 1, K);
            Vec fa(m + 1);
            for (std::size_t i = 0; i < m; ++i) {
                for (std::size_t k = 0; k < K; ++k) Ea(i, k) = E(i, k);
                fa[i] = f[i];
            }
            for (std::size_t k = 0; k < K; ++k) Ea(m, k) = s;
            fa[m] = s;
            r.alpha = lsq::nnls(Ea, fa).x;
            r.method = "nnls_penalty";
        }
    }
    r.rss = rss_of(r.alpha);
    return r;
}

}   // namespace mootation::scalar::stacking

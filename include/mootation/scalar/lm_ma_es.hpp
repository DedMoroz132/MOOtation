#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// The Limited-Memory Matrix Adaptation Evolution Strategy, LM-MA-ES (task 25,
// item 24.3.1): I. Loshchilov, T. Glasmachers, H.-G. Beyer, "Limited-memory
// matrix adaptation for large scale black-box optimization", 2018 (source
// large-scale-black_loshchilov2018; "L18" below), Algorithm 1, the lines
// marked LM-MA-ES. O(n m) a sample and
// O(n m) memory, m = 4 + floor(3 ln n) direction vectors: for problems with
// hundreds or thousands of variables, where the full CMA-ES (cmaes.hpp) costs
// O(n^2) memory and O(n^3) a decomposition.
//
// An iteration (L18 Algorithm 1):
//   z_i ~ N(0, I), d_i = z_i, then for j = 1, ..., min(t, m):
//     d_i <- (1 - c_d,j) d_i + c_d,j m_j (m_j^T d_i)                   (line 10)
//   f_i = f(y + sigma d_i), i = 1..lambda
//   y <- y + sigma sum_{i<=mu} w_i d_{i:lambda}                         (line 12)
//   p_sigma <- (1 - c_sigma) p_sigma + sqrt(mu_w c_sigma (2 - c_sigma))
//              sum w_i z_{i:lambda}                                     (line 13)
//   m_i <- (1 - c_c,i) m_i + sqrt(mu_w c_c,i (2 - c_c,i)) sum w_j z_{j:lambda}
//                                                                       (line 18)
//   sigma <- sigma exp[c_sigma / 2 (||p_sigma||^2 / n - 1)]             (line 19)
// with lambda = 4 + floor(3 ln n), mu = floor(lambda / 2), w_i = (ln(mu +
// 1/2) - ln i) / sum_j (ln(mu + 1/2) - ln j), mu_w = 1 / sum w_i^2, m = 4 +
// floor(3 ln n), c_sigma = 2 lambda / n, c_d,i = 1 / (1.5^(i-1) n), c_c,i =
// lambda / (4^(i-1) n) (line 1).
//
// The search is unconstrained: f is what the caller gives (a box is the
// caller's: clipping and a penalty, as scalar.hpp does for cmaes).
//
// READINGS:
//   LMMA-1. c_sigma = 2 lambda / n is meant for large n (L18's experiments: n
//     = 128 ... 8192); at 2 lambda >= n it is 1 or more and line 13 breaks
//     down (n = 10: c_sigma = 2, sqrt(mu_w c (2 - c)) = 0). There the
//     MA-ES value of line 1 is taken, (mu_w + 2) / (n + mu_w + 5). The c_c,i
//     are capped at 1 for the same reason (c_c,1 = lambda / n > 1 when
//     lambda > n).
//   LMMA-2. Algorithm 1 leaves the stopping criterion open ("until stopping
//     criterion is met"): here the budget, f <= f_target, sigma below
//     tol_sigma times its start, and max_iter iterations.
//   LMMA-3. The ranking is a stable sort of f (ties in sampling order); a NaN
//     ranks last.
//   LMMA-4. The best point evaluated is the answer (the mean y is not
//     evaluated, as in Algorithm 1).
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

#include "../detmath.hpp"
#include "../fp_contract.hpp"
#include "../random.hpp"

namespace mootation::lm_ma_es {

struct Options {
    double sigma0 = 0.3;
    int lambda = 0;                 // 0: 4 + floor(3 ln n)
    int m = 0;                      // 0: 4 + floor(3 ln n)
    long long max_evals = std::numeric_limits<long long>::max();
    long long max_iter = std::numeric_limits<long long>::max();
    double f_target = -std::numeric_limits<double>::infinity();
    double tol_sigma = 1e-12;       // LMMA-2: relative to sigma0
};

enum class Stop { budget, target, sigma, max_iter };

struct Result {
    std::vector<double> x;          // the best point evaluated (LMMA-4)
    double f = std::numeric_limits<double>::infinity();
    std::vector<double> mean;       // y at the end
    double sigma = 0.0;
    long long evaluations = 0;
    long long iterations = 0;
    Stop stop = Stop::budget;
};

inline Result minimize(const std::function<double(const std::vector<double>&)>& f, const std::vector<double>& y0,
                       std::mt19937& rng, const Options& o = {})
{
    const std::size_t n = y0.size();
    if (n == 0) throw std::invalid_argument("lm_ma_es: no variables");
    const double dn = static_cast<double>(n);
    const int lambda = o.lambda > 0 ? o.lambda : 4 + static_cast<int>(std::floor(3.0 * dm::log(dn)));
    const int mu = lambda / 2;
    const int m = o.m > 0 ? o.m : 4 + static_cast<int>(std::floor(3.0 * dm::log(dn)));
    std::vector<double> w(static_cast<std::size_t>(mu));
    double sw = 0.0, sw2 = 0.0;
    for (int i = 0; i < mu; ++i) {
        w[static_cast<std::size_t>(i)] = dm::log(mu + 0.5) - dm::log(static_cast<double>(i + 1));
        sw += w[static_cast<std::size_t>(i)];
    }
    for (double& v : w) {
        v /= sw;
        sw2 += v * v;
    }
    const double mu_w = 1.0 / sw2;
    double c_sigma = 2.0 * lambda / dn;
    if (c_sigma >= 1.0) c_sigma = (mu_w + 2.0) / (dn + mu_w + 5.0);       // LMMA-1
    std::vector<double> c_d(static_cast<std::size_t>(m)), c_c(static_cast<std::size_t>(m));
    for (int i = 0; i < m; ++i) {
        c_d[static_cast<std::size_t>(i)] = 1.0 / (dm::pow(1.5, i) * dn);
        c_c[static_cast<std::size_t>(i)] = std::min(1.0, lambda / (dm::pow(4.0, i) * dn));
    }

    Result res;
    std::vector<double> y = y0, p_sigma(n, 0.0);
    std::vector<std::vector<double>> M(static_cast<std::size_t>(m), std::vector<double>(n, 0.0));
    std::vector<std::vector<double>> Z(static_cast<std::size_t>(lambda), std::vector<double>(n));
    std::vector<std::vector<double>> D(static_cast<std::size_t>(lambda), std::vector<double>(n));
    std::vector<double> F(static_cast<std::size_t>(lambda)), x(n), zw(n), dw(n);
    std::vector<int> order(static_cast<std::size_t>(lambda));
    double sigma = o.sigma0;

    for (long long t = 0;; ++t) {
        if (t >= o.max_iter) {
            res.stop = Stop::max_iter;
            break;
        }
        if (res.evaluations + lambda > o.max_evals) {
            res.stop = Stop::budget;
            break;
        }
        const long long used = std::min<long long>(t, m);
        for (int i = 0; i < lambda; ++i) {
            auto& z = Z[static_cast<std::size_t>(i)];
            auto& d = D[static_cast<std::size_t>(i)];
            for (std::size_t k = 0; k < n; ++k) z[k] = rnd::normal01(rng);
            d = z;
            for (long long j = 0; j < used; ++j) {                         // line 10
                const auto& mj = M[static_cast<std::size_t>(j)];
                double s = 0.0;
                for (std::size_t k = 0; k < n; ++k) s += mj[k] * d[k];
                const double cd = c_d[static_cast<std::size_t>(j)];
                for (std::size_t k = 0; k < n; ++k) d[k] = (1.0 - cd) * d[k] + cd * mj[k] * s;
            }
            for (std::size_t k = 0; k < n; ++k) x[k] = y[k] + sigma * d[k];
            const double fx = f(x);
            ++res.evaluations;
            F[static_cast<std::size_t>(i)] = fx;
            if (fx < res.f || res.x.empty()) {
                res.f = fx;
                res.x = x;
            }
        }
        ++res.iterations;
        for (int i = 0; i < lambda; ++i) order[static_cast<std::size_t>(i)] = i;
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) {             // LMMA-3
            const double fa = F[static_cast<std::size_t>(a)], fb = F[static_cast<std::size_t>(b)];
            if (std::isnan(fb)) return !std::isnan(fa);
            return fa < fb;
        });
        std::fill(zw.begin(), zw.end(), 0.0);
        std::fill(dw.begin(), dw.end(), 0.0);
        for (int i = 0; i < mu; ++i) {
            const auto& z = Z[static_cast<std::size_t>(order[static_cast<std::size_t>(i)])];
            const auto& d = D[static_cast<std::size_t>(order[static_cast<std::size_t>(i)])];
            const double wi = w[static_cast<std::size_t>(i)];
            for (std::size_t k = 0; k < n; ++k) {
                zw[k] += wi * z[k];
                dw[k] += wi * d[k];
            }
        }
        for (std::size_t k = 0; k < n; ++k) y[k] += sigma * dw[k];                   // line 12
        const double cs = std::sqrt(mu_w * c_sigma * (2.0 - c_sigma));
        double ps2 = 0.0;
        for (std::size_t k = 0; k < n; ++k) {                                         // line 13
            p_sigma[k] = (1.0 - c_sigma) * p_sigma[k] + cs * zw[k];
            ps2 += p_sigma[k] * p_sigma[k];
        }
        for (int i = 0; i < m; ++i) {                                                 // line 18
            const double cc = c_c[static_cast<std::size_t>(i)];
            const double s = std::sqrt(mu_w * cc * (2.0 - cc));
            auto& mi = M[static_cast<std::size_t>(i)];
            for (std::size_t k = 0; k < n; ++k) mi[k] = (1.0 - cc) * mi[k] + s * zw[k];
        }
        sigma *= dm::exp(0.5 * c_sigma * (ps2 / dn - 1.0));                         // line 19
        if (res.f <= o.f_target) {
            res.stop = Stop::target;
            break;
        }
        if (sigma < o.tol_sigma * o.sigma0) {
            res.stop = Stop::sigma;
            break;
        }
    }
    res.mean = y;
    res.sigma = sigma;
    return res;
}

}   // namespace mootation::lm_ma_es

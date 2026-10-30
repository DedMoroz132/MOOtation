#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// sep-CMA-ES, as R. Ros, N. Hansen, "A Simple Modification in CMA-ES
// Achieving Linear Time and Space Complexity", PPSN X, LNCS 5199, pp.
// 296-305, 2008, §2 and Algorithm 1 (source sep-cma-es_ros2008): the
// (mu/mu_W, lambda)-CMA-ES of Algorithm 1 with its parameter setting, the
// covariance matrix kept diagonal — D^2 = diag(C), B = I (Eq. 1) — and the
// learning rate c_cov = (n + 2)/3 c_cov^def.
//
// A separate core, not a knob of cmaes.hpp: that one is the 2016 tutorial's
// CMA-ES (active weights, c_1 and c_mu), this one Algorithm 1's (c_cov,
// mu_cov), whose parameters the paper's tables were run with.
//
// The caller asks for lambda points, evaluates them and tells their values f
// and violations v. Minimization; random numbers are rnd:: of the caller's
// generator, exp and log by dm:: (detmath.hpp).
//
// Readings (OURS):
//   SEP-1. Line 7's "sqrt(1 - (1 - c_sigma)^2 g)" is (1 - c_sigma)^(2g), as
//     in the CMA-ES literature it comes from.
//   SEP-2. E||N(0, I)|| = sqrt(n) (1 - 1/(4n) + 1/(21 n^2)).
//   SEP-3. Ranking: the comparison of evo.hpp (NSGA-II's constrained-
//     domination at M = 1 for constraints g <= 0, not in the paper), ties in
//     the order told (stable).
//   SEP-4. No box: Algorithm 1 has none (as cmaes.hpp); the caller's problem
//     takes x as it comes.
//   SEP-5. The draws: lambda vectors of n normals, point by point.
//   SEP-6. c_cov = (n + 2)/3 c_cov^def is held at 1 at most (line 9 needs
//     1 - c_cov >= 0); below n = 2 nothing changes, the paper's n are larger.
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

#include "../detmath.hpp"
#include "../random.hpp"
#include "evo.hpp"

namespace mootation::scalar::sep_cmaes {

using evo::Rows;
using evo::Vec;

class SepCMAES {
public:
    SepCMAES(Vec mean, double sigma, std::mt19937& rng, std::size_t lambda = 0)
        : m_(std::move(mean)), n_(m_.size()), sigma_(sigma), rng_(rng)
    {
        if (n_ == 0 || !(sigma_ > 0.0)) throw std::invalid_argument("sep_cmaes: mean, sigma > 0");
        const double n = static_cast<double>(n_);
        lam_ = lambda > 0 ? lambda : 4 + static_cast<std::size_t>(std::floor(3.0 * dm::log(n)));
        mu_ = lam_ / 2;
        if (mu_ < 1) throw std::invalid_argument("sep_cmaes: lambda >= 2");
        // Parameter setting of Algorithm 1
        w_.resize(mu_);
        double sw = 0.0;
        for (std::size_t i = 0; i < mu_; ++i) {
            w_[i] = dm::log(static_cast<double>(mu_) + 1.0) - dm::log(static_cast<double>(i + 1));
            sw += w_[i];
        }
        double sw2 = 0.0;
        for (double& w : w_) {
            w /= sw;
            sw2 += w * w;
        }
        muw_ = 1.0 / sw2;
        cs_ = (muw_ + 2.0) / (n + muw_ + 3.0);
        ds_ = 1.0 + 2.0 * std::max(0.0, std::sqrt((muw_ - 1.0) / (n + 1.0)) - 1.0) + cs_;
        cc_ = 4.0 / (n + 4.0);
        mucov_ = muw_;
        const double def = 1.0 / mucov_ * 2.0 / ((n + std::sqrt(2.0)) * (n + std::sqrt(2.0))) +
                           (1.0 - 1.0 / mucov_) * std::min(1.0, (2.0 * mucov_ - 1.0) / ((n + 2.0) * (n + 2.0) + mucov_));
        ccov_ = std::min(1.0, (n + 2.0) / 3.0 * def);                    // §2
        chiN_ = std::sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n));   // SEP-2
        ps_.assign(n_, 0.0);
        pc_.assign(n_, 0.0);
        C_.assign(n_, 1.0);
        D_.assign(n_, 1.0);
    }

    std::size_t lambda() const { return lam_; }
    double sigma() const { return sigma_; }
    double c_cov() const { return ccov_; }
    const Vec& mean() const { return m_; }

    Rows ask()                                                           // lines 2-3, SEP-5
    {
        z_.assign(lam_, Vec(n_));
        Rows x(lam_, Vec(n_));
        for (std::size_t k = 0; k < lam_; ++k)
            for (std::size_t j = 0; j < n_; ++j) {
                z_[k][j] = rnd::normal01(rng_);
                x[k][j] = m_[j] + sigma_ * D_[j] * z_[k][j];
            }
        return x;
    }

    void tell(const Vec& f, const Vec& v = {})
    {
        if (f.size() != lam_ || (!v.empty() && v.size() != lam_)) throw std::invalid_argument("sep_cmaes: tell sizes");
        const Vec vv = v.empty() ? Vec(lam_, 0.0) : v;
        const double n = static_cast<double>(n_);
        std::vector<std::size_t> order(lam_);
        std::iota(order.begin(), order.end(), std::size_t{0});
        std::stable_sort(order.begin(), order.end(),
                         [&](std::size_t a, std::size_t b) { return evo::better(f[a], vv[a], f[b], vv[b]); });   // SEP-3
        Vec zw(n_, 0.0);
        for (std::size_t i = 0; i < mu_; ++i)                            // lines 4-5
            for (std::size_t j = 0; j < n_; ++j) zw[j] += w_[i] * z_[order[i]][j];
        for (std::size_t j = 0; j < n_; ++j) m_[j] += sigma_ * D_[j] * zw[j];
        ++g_;
        const double a = std::sqrt(cs_ * (2.0 - cs_) * muw_);
        double norm = 0.0;
        for (std::size_t j = 0; j < n_; ++j) {                           // line 6
            ps_[j] = (1.0 - cs_) * ps_[j] + a * zw[j];
            norm += ps_[j] * ps_[j];
        }
        norm = std::sqrt(norm);
        const bool hs = norm / std::sqrt(1.0 - dm::pow(1.0 - cs_, 2.0 * static_cast<double>(g_))) <
                        (1.4 + 2.0 / (n + 1.0)) * chiN_;                // line 7, SEP-1
        const double b = std::sqrt(cc_ * (2.0 - cc_) * muw_);
        for (std::size_t j = 0; j < n_; ++j) pc_[j] = (1.0 - cc_) * pc_[j] + (hs ? b * D_[j] * zw[j] : 0.0);   // line 8
        for (std::size_t j = 0; j < n_; ++j) {                           // line 9, the diagonal
            double rank_mu = 0.0;
            for (std::size_t i = 0; i < mu_; ++i) {
                const double y = D_[j] * z_[order[i]][j];
                rank_mu += w_[i] * y * y;
            }
            C_[j] = (1.0 - ccov_) * C_[j] + ccov_ / mucov_ * pc_[j] * pc_[j] + ccov_ * (1.0 - 1.0 / mucov_) * rank_mu;
        }
        sigma_ *= dm::exp(cs_ / ds_ * (norm / chiN_ - 1.0));             // line 10
        for (std::size_t j = 0; j < n_; ++j) D_[j] = std::sqrt(C_[j]);   // line 11, Eq. (1)
    }

    // sigma times the largest standard deviation: the scale of the next step.
    double spread() const { return sigma_ * *std::max_element(D_.begin(), D_.end()); }

private:
    Vec m_;
    std::size_t n_;
    double sigma_;
    std::mt19937& rng_;
    std::size_t lam_ = 0, mu_ = 0;
    double muw_ = 0.0, cs_ = 0.0, ds_ = 0.0, cc_ = 0.0, mucov_ = 0.0, ccov_ = 0.0, chiN_ = 0.0;
    Vec w_, ps_, pc_, C_, D_;
    Rows z_;
    long long g_ = 0;
};

}   // namespace mootation::scalar::sep_cmaes

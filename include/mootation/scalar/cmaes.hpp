#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// Single-objective CMA-ES, as N. Hansen, "The CMA Evolution Strategy: A
// Tutorial", arXiv:1604.00772 (2016), Figure 6 (Eqs. 38-47) with the default
// parameters of its Table 1 (Eqs. 48-58), negative weights (active CMA)
// included. Written from the tutorial, not from a library.
//
// The caller asks for λ points, evaluates them and tells their values; the
// strategy does the rest. Minimization. The normal samples are rnd::normal01
// (random.hpp), the eigendecomposition of C eigen.hpp's; log, exp and pow are
// dm:: (detmath.hpp).
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <random>
#include <vector>

#include "../detmath.hpp"
#include "../fp_contract.hpp"
#include "../random.hpp"
#include "eigen.hpp"

namespace mootation::scalar {

class CMAES {
public:
    CMAES(std::vector<double> mean, double sigma, std::mt19937& rng, int popsize = 0)
        : m_(std::move(mean)), n_(m_.size()), sigma_(sigma), rng_(rng)
    {
        const double n = static_cast<double>(n_);
        // Table 1: selection and recombination
        lam_ = popsize > 0 ? static_cast<std::size_t>(popsize)
                           : 4 + static_cast<std::size_t>(std::floor(3.0 * dm::log(n)));
        mu_ = lam_ / 2;
        std::vector<double> w(lam_);
        for (std::size_t i = 0; i < lam_; ++i)                                 // (49)
            w[i] = dm::log((static_cast<double>(lam_) + 1.0) / 2.0) - dm::log(static_cast<double>(i + 1));
        double sp = 0.0, sp2 = 0.0, sn = 0.0, sn2 = 0.0;
        for (std::size_t i = 0; i < lam_; ++i) {
            if (i < mu_) { sp += w[i]; sp2 += w[i] * w[i]; }
            else { sn += w[i]; sn2 += w[i] * w[i]; }
        }
        mueff_ = sp * sp / sp2;
        const double mueff_neg = (lam_ > mu_ && sn2 > 0.0) ? sn * sn / sn2 : 0.0;
        // step-size control (55)
        cs_ = (mueff_ + 2.0) / (n + mueff_ + 5.0);
        ds_ = 1.0 + 2.0 * std::max(0.0, std::sqrt((mueff_ - 1.0) / (n + 1.0)) - 1.0) + cs_;
        // covariance matrix adaptation (56)-(58)
        cc_ = (4.0 + mueff_ / n) / (n + 4.0 + 2.0 * mueff_ / n);
        const double alpha_cov = 2.0;
        c1_ = alpha_cov / ((n + 1.3) * (n + 1.3) + mueff_);
        cmu_ = std::min(1.0 - c1_, alpha_cov * (0.25 + mueff_ + 1.0 / mueff_ - 2.0) /
                                       ((n + 2.0) * (n + 2.0) + alpha_cov * mueff_ / 2.0));
        // (50)-(53): positive weights sum to one, negative ones scaled
        const double alpha_mu = 1.0 + c1_ / cmu_;
        const double alpha_mueff = 1.0 + 2.0 * mueff_neg / (mueff_ + 2.0);
        const double alpha_posdef = (1.0 - c1_ - cmu_) / (n * cmu_);
        const double sum_neg = -sn;
        w_.resize(lam_);
        const double neg_scale = std::min(alpha_mu, std::min(alpha_mueff, alpha_posdef));
        for (std::size_t i = 0; i < lam_; ++i)
            w_[i] = w[i] >= 0.0 ? w[i] / sp : neg_scale * w[i] / std::max(sum_neg, 1e-300);
        cm_ = 1.0;                                                             // (54)
        chiN_ = std::sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n));
        ps_.assign(n_, 0.0);
        pc_.assign(n_, 0.0);
        C_ = lsq::Matrix(n_, n_);
        B_ = lsq::Matrix(n_, n_);
        for (std::size_t i = 0; i < n_; ++i) C_(i, i) = B_(i, i) = 1.0;
        D_.assign(n_, 1.0);
    }

    std::size_t lambda() const { return lam_; }

    // λ points: x_k = m + σ B D z_k (38)-(40).
    std::vector<std::vector<double>> ask()
    {
        y_.assign(lam_, std::vector<double>(n_, 0.0));
        std::vector<std::vector<double>> x(lam_, std::vector<double>(n_));
        for (std::size_t k = 0; k < lam_; ++k) {
            std::vector<double> z(n_);
            for (std::size_t j = 0; j < n_; ++j) z[j] = rnd::normal01(rng_);
            for (std::size_t i = 0; i < n_; ++i) {
                double s = 0.0;
                for (std::size_t j = 0; j < n_; ++j) s += z[j] * D_[j] * B_(i, j);
                y_[k][i] = s;
                x[k][i] = m_[i] + sigma_ * s;
            }
        }
        return x;
    }

    void tell(const std::vector<double>& values)
    {
        const double n = static_cast<double>(n_);
        std::vector<std::size_t> order(lam_);
        std::iota(order.begin(), order.end(), std::size_t{0});
        std::stable_sort(order.begin(), order.end(),
                         [&](std::size_t a, std::size_t b) { return values[a] < values[b]; });
        std::vector<std::vector<double>> y(lam_);
        for (std::size_t k = 0; k < lam_; ++k) y[k] = y_[order[k]];
        std::vector<double> yw(n_, 0.0);                                        // (41)
        for (std::size_t k = 0; k < mu_; ++k)
            for (std::size_t i = 0; i < n_; ++i) yw[i] += w_[k] * y[k][i];
        for (std::size_t i = 0; i < n_; ++i) m_[i] += cm_ * sigma_ * yw[i];     // (42)
        // C^-1/2 = B D^-1 B^T
        lsq::Matrix Ci(n_, n_);
        for (std::size_t i = 0; i < n_; ++i)
            for (std::size_t j = 0; j < n_; ++j) {
                double s = 0.0;
                for (std::size_t k = 0; k < n_; ++k) s += B_(i, k) / D_[k] * B_(j, k);
                Ci(i, j) = s;
            }
        auto times_ci = [&](const std::vector<double>& v) {
            std::vector<double> out(n_, 0.0);
            for (std::size_t i = 0; i < n_; ++i)
                for (std::size_t j = 0; j < n_; ++j) out[i] += Ci(i, j) * v[j];
            return out;
        };
        const std::vector<double> ciyw = times_ci(yw);
        const double cps = std::sqrt(cs_ * (2.0 - cs_) * mueff_);
        double norm_ps = 0.0;
        for (std::size_t i = 0; i < n_; ++i) {                                  // (43)
            ps_[i] = (1.0 - cs_) * ps_[i] + cps * ciyw[i];
            norm_ps += ps_[i] * ps_[i];
        }
        norm_ps = std::sqrt(norm_ps);
        ++g_;
        const bool hs = norm_ps / std::sqrt(1.0 - dm::pow(1.0 - cs_, 2.0 * static_cast<double>(g_))) <
                        (1.4 + 2.0 / (n + 1.0)) * chiN_;
        const double cpc = std::sqrt(cc_ * (2.0 - cc_) * mueff_);
        for (std::size_t i = 0; i < n_; ++i)                                    // (45)
            pc_[i] = (1.0 - cc_) * pc_[i] + (hs ? cpc * yw[i] : 0.0);
        std::vector<double> wo = w_;                                             // (46)
        for (std::size_t k = 0; k < lam_; ++k)
            if (wo[k] < 0.0) {
                const std::vector<double> v = times_ci(y[k]);
                double q = 0.0;
                for (double t : v) q += t * t;
                wo[k] *= n / std::max(q, 1e-300);
            }
        const double delta_h = (hs ? 0.0 : 1.0) * cc_ * (2.0 - cc_);
        double wsum = 0.0;
        for (double t : w_) wsum += t;
        const double keep = 1.0 + c1_ * delta_h - c1_ - cmu_ * wsum;
        lsq::Matrix Cn(n_, n_);                                                  // (47)
        for (std::size_t i = 0; i < n_; ++i)
            for (std::size_t j = 0; j < n_; ++j) {
                double rank_mu = 0.0;
                for (std::size_t k = 0; k < lam_; ++k) rank_mu += y[k][i] * wo[k] * y[k][j];
                Cn(i, j) = keep * C_(i, j) + c1_ * pc_[i] * pc_[j] + cmu_ * rank_mu;
            }
        sigma_ *= dm::exp(cs_ / ds_ * (norm_ps / chiN_ - 1.0));                  // (44)
        for (std::size_t i = 0; i < n_; ++i)
            for (std::size_t j = 0; j < n_; ++j) C_(i, j) = (Cn(i, j) + Cn(j, i)) / 2.0;
        std::vector<double> ev;
        eigen::symmetric(C_, ev, B_);
        for (std::size_t i = 0; i < n_; ++i) D_[i] = std::sqrt(std::max(ev[i], 1e-300));
    }

    // σ times the largest standard deviation: the scale of the next step.
    double spread() const { return sigma_ * *std::max_element(D_.begin(), D_.end()); }

private:
    std::vector<double> m_;
    std::size_t n_;
    double sigma_;
    std::mt19937& rng_;
    std::size_t lam_ = 0, mu_ = 0;
    double mueff_ = 0.0, cs_ = 0.0, ds_ = 0.0, cc_ = 0.0, c1_ = 0.0, cmu_ = 0.0, cm_ = 1.0, chiN_ = 0.0;
    std::vector<double> w_, ps_, pc_, D_;
    lsq::Matrix C_, B_;
    long long g_ = 0;
    std::vector<std::vector<double>> y_;
};

}   // namespace mootation::scalar

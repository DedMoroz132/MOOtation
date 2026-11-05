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
//
// Injection (task 19, §19.4.3): external points, as N. Hansen, "Injecting
// external solutions into CMA-ES", INRIA RR-7748, arXiv:1110.4181, 2011
// (source cmaes-injection_hansen2011_arxiv), Figure 1. inject(points): the
// next ask() returns them in place of its first draws (1); in tell, the step
// y_i = (x_i - m) / σ of an injected point is clipped to the Mahalanobis
// length c_y, y_i <- α_clip(c_y, ||C^-1/2 y_i||) y_i, α_clip(c, x) = min(1,
// c/x) (3), (12). inject_mean(x_m): the mean step is (x_m - m) / σ (4),
// clipped to c_y^m by α_clip(c_y^m, sqrt(μ_w) ||C^-1/2 Δm||) (6), before the
// mean moves (the order of (5) and (6) exchanged, as the report allows on p.
// 6). σ's change is bounded: σ <- σ exp(min(Δσ_max, (c_σ/d_σ)(||p_σ|| / E||N||
// - 1))) (11). elitist(true): the best point told so far is injected at
// every ask (the report's "elitist algorithm with comma selection", §3). With
// c_y = c_y^m = Δσ_max = inf, the state before any injection, the strategy is
// the one above unchanged.
//   INJ-1. The strategy's own parameters stay the tutorial's (Table 1 of the
//     report lists those of [5], Hansen & Kern 2004; the report itself
//     advises keeping the original strategy parameters, p. 7).
//   INJ-2. Δσ_max = 1.0 as the report's Table 1 has it (its text on p. 5
//     says the factor exp(0.6)); c_y = sqrt(n) + 2n/(n + 2), c_y^m =
//     sqrt(2n) + 2n/(n + 2) (Table 1). They take effect at the first
//     injection (inject, inject_mean or elitist) and hold from then on.
//   INJ-3. The tutorial's negative weights: an injected point among the worst
//     is clipped by (3) as well; the report expects no harm (p. 9: the
//     clipping of long steps even helps the negative update).
//   INJ-4. An injected mean takes the place of the weighted mean of (41) for
//     the mean's move and for p_σ and p_c, the y_i (clipped) still serve the
//     rank-μ update (Figure 1).
//
// The CMA-ES of BIPOP-CMA-ES (task 18, stage 3, owner's В4): N. Hansen,
// "Benchmarking a BI-Population CMA-ES on the BBOB-2009 Function Testbed",
// GECCO 2009 (source bipop-cma-es_hansen2009, H09), §2. Two options, the
// tutorial's the default (the runs above unchanged):
//   - weights = positive: μ = ⌊λ/2⌋, w_i ∝ ln(μ + 1) − ln i for i ≤ μ, summing
//     to one, none negative (H09 §2); μ_w from these;
//   - cmu = hansen2009: c_μ = min(1 − c_1, 2(μ_w − 2 + 1/μ_w)/((D + 2)² + μ_w))
//     (H09 §2). c_σ, d_σ, c_c and c_1 are the tutorial's, which H09's are.
// With both, tell() is H09's Table 1 (1)-(6). stop() names the first of H09's
// termination criteria (§2.2) that holds after the last tell:
//   MaxIter      t ≥ 100 + 50(D + 3)²/√λ;
//   TolHistFun   the best values of the last 10 + ⌈30D/λ⌉ iterations span
//                less than 1e-12;
//   EqualFunVals in more than a third of the last D iterations f(x_1:λ) =
//                f(x_k:λ), k = 1 + ⌈0.1 + λ/4⌉;
//   TolX         every |p_c,i| and every √C_ii, times σ/σ⁰, below 1e-12;
//   TolUpSigma   σ/σ⁰ > 1e20 √l, l the largest eigenvalue of C;
//   Stagnation   over the last ⌈0.2t + 120 + 30D/λ⌉ iterations, the median of
//                the 20 newest values not below that of the 20 oldest;
//   ConditionCov the condition number of C above 1e14;
//   NoEffectAxis m unchanged by adding 0.1σ√l v, l the (1 + t mod D)-largest
//                eigenvalue of C and v its unit eigenvector;
//   NoEffectCoor some m_i unchanged by adding 0.2σ√C_ii.
//   CMA-1. t counts the tells; the criteria are tested after the update of
//     iteration t, on the state it left (m, σ, C, p_c of t + 1 in H09's Table
//     1), each iteration, as a production code tests them before sampling.
//   CMA-2. Stagnation holds when it holds in both arrays, the best and the
//     median values ("respectively"), and is tested only once the iterations
//     reach the window's length (with fewer, 20 newest and 20 oldest overlap
//     or are not there). The median of an even λ is the mean of the two middle
//     values.
//   CMA-3. TolHistFun and EqualFunVals are tested only once there are as many
//     iterations as their window holds (10 + ⌈30D/λ⌉, D).
//   CMA-4. "Numerically constant": m_i + δ_i == m_i in double for every i
//     (NoEffectAxis), for some i (NoEffectCoor).
//   CMA-5. The eigendecomposition of C at every iteration, where H09 makes it
//     every (c_1 + c_μ)^-1/(10D) iterations (§4): that saves time and changes
//     no update but the sampling's B and D between decompositions.
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

#include "../detmath.hpp"
#include "../fp_contract.hpp"
#include "../random.hpp"
#include "eigen.hpp"

namespace mootation::scalar {

// The weights and c_μ: the tutorial's (the default) or H09's.
struct CMAESOptions {
    enum class Weights { active, positive } weights = Weights::active;
    enum class Cmu { tutorial2016, hansen2009 } cmu = Cmu::tutorial2016;
};

class CMAES {
public:
    // H09's termination criteria (§2.2), in the order stop() tests them
    enum class Stop {
        none, max_iter, tol_hist_fun, equal_fun_vals, tol_x, tol_up_sigma, stagnation,
        condition_cov, no_effect_axis, no_effect_coor
    };
    static const char* stop_name(Stop s)
    {
        switch (s) {
        case Stop::none: return "none";
        case Stop::max_iter: return "MaxIter";
        case Stop::tol_hist_fun: return "TolHistFun";
        case Stop::equal_fun_vals: return "EqualFunVals";
        case Stop::tol_x: return "TolX";
        case Stop::tol_up_sigma: return "TolUpSigma";
        case Stop::stagnation: return "Stagnation";
        case Stop::condition_cov: return "ConditionCov";
        case Stop::no_effect_axis: return "NoEffectAxis";
        case Stop::no_effect_coor: return "NoEffectCoor";
        }
        return "none";
    }

    CMAES(std::vector<double> mean, double sigma, std::mt19937& rng, int popsize = 0,
          const CMAESOptions& options = CMAESOptions())
        : m_(std::move(mean)), n_(m_.size()), sigma_(sigma), sigma0_(sigma), rng_(rng)
    {
        const double n = static_cast<double>(n_);
        const bool positive = options.weights == CMAESOptions::Weights::positive;
        // Table 1: selection and recombination
        lam_ = popsize > 0 ? static_cast<std::size_t>(popsize)
                           : 4 + static_cast<std::size_t>(std::floor(3.0 * dm::log(n)));
        mu_ = lam_ / 2;
        std::vector<double> w(lam_);
        for (std::size_t i = 0; i < lam_; ++i)                                 // (49)
            w[i] = dm::log((static_cast<double>(lam_) + 1.0) / 2.0) - dm::log(static_cast<double>(i + 1));
        if (positive)                                                          // H09 §2
            for (std::size_t i = 0; i < lam_; ++i)
                w[i] = i < mu_ ? dm::log(static_cast<double>(mu_) + 1.0) - dm::log(static_cast<double>(i + 1)) : 0.0;
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
        cmu_ = options.cmu == CMAESOptions::Cmu::hansen2009
                   ? std::min(1.0 - c1_, 2.0 * (mueff_ - 2.0 + 1.0 / mueff_) / ((n + 2.0) * (n + 2.0) + mueff_))
                   : std::min(1.0 - c1_, alpha_cov * (0.25 + mueff_ + 1.0 / mueff_ - 2.0) /
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
    const std::vector<double>& mean() const { return m_; }
    double sigma() const { return sigma_; }
    const std::vector<double>& weights() const { return w_; }
    double mu_eff() const { return mueff_; }
    double c_mu() const { return cmu_; }

    // Injection's parameters (INJ-2), 0: Table 1's; the first injection
    // switches them on with Table 1's where none were set
    struct Injection {
        double c_y = 0.0, c_ym = 0.0, dsigma_max = 1.0;
    };
    void injection(const Injection& p)
    {
        const double n = static_cast<double>(n_);
        cy_ = p.c_y > 0.0 ? p.c_y : std::sqrt(n) + 2.0 * n / (n + 2.0);
        cym_ = p.c_ym > 0.0 ? p.c_ym : std::sqrt(2.0 * n) + 2.0 * n / (n + 2.0);
        dsmax_ = p.dsigma_max;
        injecting_ = true;
    }
    // points for the next ask, in place of its first draws (at most λ)
    void inject(std::vector<std::vector<double>> points)
    {
        if (!injecting_) injection(Injection());
        for (auto& p : points)
            if (injected_.size() < lam_) injected_.push_back(std::move(p));
    }
    // the next tell's mean step from x_m (4)
    void inject_mean(std::vector<double> xm)
    {
        if (!injecting_) injection(Injection());
        mean_x_ = std::move(xm);
    }
    // the best point told so far injected at every ask
    void elitist(bool on)
    {
        if (on && !injecting_) injection(Injection());
        elitist_ = on;
    }

    // λ points: x_k = m + σ B D z_k (38)-(40), the injected ones first (1).
    std::vector<std::vector<double>> ask()
    {
        if (elitist_ && !best_x_.empty() && injected_.size() < lam_) injected_.insert(injected_.begin(), best_x_);
        y_.assign(lam_, std::vector<double>(n_, 0.0));
        std::vector<std::vector<double>> x(lam_, std::vector<double>(n_));
        is_injected_.assign(lam_, false);
        for (std::size_t k = 0; k < lam_; ++k) {
            if (k < injected_.size()) {                                         // (1), (2)
                is_injected_[k] = true;
                x[k] = injected_[k];
                for (std::size_t i = 0; i < n_; ++i) y_[k][i] = (x[k][i] - m_[i]) / sigma_;
                continue;
            }
            std::vector<double> z(n_);
            for (std::size_t j = 0; j < n_; ++j) z[j] = rnd::normal01(rng_);
            for (std::size_t i = 0; i < n_; ++i) {
                double s = 0.0;
                for (std::size_t j = 0; j < n_; ++j) s += z[j] * D_[j] * B_(i, j);
                y_[k][i] = s;
                x[k][i] = m_[i] + sigma_ * s;
            }
        }
        injected_.clear();
        if (elitist_) x_ = x;
        return x;
    }

    void tell(const std::vector<double>& values)
    {
        const double n = static_cast<double>(n_);
        std::vector<std::size_t> order(lam_);
        std::iota(order.begin(), order.end(), std::size_t{0});
        std::stable_sort(order.begin(), order.end(),
                         [&](std::size_t a, std::size_t b) { return values[a] < values[b]; });
        if (elitist_)                                                            // the best told so far
            for (std::size_t k = 0; k < lam_; ++k)
                if (values[k] < best_f_) {
                    best_f_ = values[k];
                    best_x_ = x_[k];
                }
        std::vector<std::vector<double>> y(lam_);
        for (std::size_t k = 0; k < lam_; ++k) y[k] = y_[order[k]];
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
        auto length = [](const std::vector<double>& v) {
            double q = 0.0;
            for (double t : v) q += t * t;
            return std::sqrt(q);
        };
        if (injecting_)                                                         // (3), (12)
            for (std::size_t k = 0; k < lam_; ++k) {
                if (is_injected_.empty() || !is_injected_[order[k]]) continue;
                const double l = length(times_ci(y[k]));
                if (l > cy_)
                    for (double& t : y[k]) t *= cy_ / l;
            }
        std::vector<double> yw(n_, 0.0);                                        // (41)
        for (std::size_t k = 0; k < mu_; ++k)
            for (std::size_t i = 0; i < n_; ++i) yw[i] += w_[k] * y[k][i];
        if (!mean_x_.empty()) {                                                 // (4), (6) (INJ-4)
            for (std::size_t i = 0; i < n_; ++i) yw[i] = (mean_x_[i] - m_[i]) / sigma_;
            const double l = std::sqrt(mueff_) * length(times_ci(yw));
            if (l > cym_)
                for (double& t : yw) t *= cym_ / l;
            mean_x_.clear();
        }
        for (std::size_t i = 0; i < n_; ++i) m_[i] += cm_ * sigma_ * yw[i];     // (42)
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
        const double log_step = cs_ / ds_ * (norm_ps / chiN_ - 1.0);            // (44)
        sigma_ *= dm::exp(injecting_ ? std::min(dsmax_, log_step) : log_step);   // (11)
        for (std::size_t i = 0; i < n_; ++i)
            for (std::size_t j = 0; j < n_; ++j) C_(i, j) = (Cn(i, j) + Cn(j, i)) / 2.0;
        eigen::symmetric(C_, ev_, B_);
        for (std::size_t i = 0; i < n_; ++i) D_[i] = std::sqrt(std::max(ev_[i], 1e-300));
        // what H09's criteria read (CMA-1, CMA-2)
        const std::size_t k = std::min(lam_, 1 + static_cast<std::size_t>(std::ceil(0.1 + static_cast<double>(lam_) / 4.0)));
        best_hist_.push_back(values[order[0]]);
        median_hist_.push_back(lam_ % 2 == 1 ? values[order[lam_ / 2]]
                                             : (values[order[lam_ / 2 - 1]] + values[order[lam_ / 2]]) / 2.0);
        equal_hist_.push_back(values[order[0]] == values[order[k - 1]]);
    }

    // σ times the largest standard deviation: the scale of the next step.
    double spread() const { return sigma_ * *std::max_element(D_.begin(), D_.end()); }

    // the number of tells so far
    long long iterations() const { return g_; }

    // The first of H09's termination criteria that holds after the last tell
    // (CMA-1 ... CMA-4), Stop::none if none does.
    Stop stop() const
    {
        if (g_ == 0) return Stop::none;
        const double n = static_cast<double>(n_), lam = static_cast<double>(lam_), t = static_cast<double>(g_);
        const std::size_t len = best_hist_.size();
        if (t >= 100.0 + 50.0 * (n + 3.0) * (n + 3.0) / std::sqrt(lam)) return Stop::max_iter;
        const std::size_t hist_fun = 10 + static_cast<std::size_t>(std::ceil(30.0 * n / lam));
        if (len >= hist_fun) {                                                    // CMA-3
            const auto first = best_hist_.end() - static_cast<std::ptrdiff_t>(hist_fun);
            const auto [lo, hi] = std::minmax_element(first, best_hist_.end());
            if (*hi - *lo < 1e-12) return Stop::tol_hist_fun;
        }
        if (len >= n_) {                                                          // CMA-3
            std::size_t equal = 0;
            for (std::size_t i = len - n_; i < len; ++i) equal += equal_hist_[i] ? 1 : 0;
            if (3.0 * static_cast<double>(equal) > n) return Stop::equal_fun_vals;
        }
        const double scale = sigma_ / sigma0_;
        bool small = true;
        for (std::size_t i = 0; i < n_ && small; ++i)
            small = std::fabs(pc_[i]) * scale < 1e-12 && std::sqrt(C_(i, i)) * scale < 1e-12;
        if (small) return Stop::tol_x;
        const double l_max = ev_.back(), l_min = ev_.front();
        if (scale > 1e20 * std::sqrt(l_max)) return Stop::tol_up_sigma;
        const std::size_t window = static_cast<std::size_t>(std::ceil(0.2 * t + 120.0 + 30.0 * n / lam));
        if (len >= window && window >= 20) {                                      // CMA-2
            auto median20 = [](std::vector<double> v) {
                std::sort(v.begin(), v.end());
                return (v[9] + v[10]) / 2.0;
            };
            auto stagnant = [&](const std::vector<double>& h) {
                const auto start = h.end() - static_cast<std::ptrdiff_t>(window);
                const std::vector<double> oldest(start, start + 20), newest(h.end() - 20, h.end());
                return !(median20(newest) < median20(oldest));
            };
            if (stagnant(best_hist_) && stagnant(median_hist_)) return Stop::stagnation;
        }
        if (!(l_min > 0.0) || l_max / l_min > 1e14) return Stop::condition_cov;
        const std::size_t axis = n_ - 1 - static_cast<std::size_t>(g_ % static_cast<long long>(n_));
        bool unchanged = true;                                                    // CMA-4
        for (std::size_t i = 0; i < n_ && unchanged; ++i)
            unchanged = m_[i] + 0.1 * sigma_ * std::sqrt(std::max(ev_[axis], 0.0)) * B_(i, axis) == m_[i];
        if (unchanged) return Stop::no_effect_axis;
        for (std::size_t i = 0; i < n_; ++i)
            if (m_[i] + 0.2 * sigma_ * std::sqrt(C_(i, i)) == m_[i]) return Stop::no_effect_coor;
        return Stop::none;
    }

private:
    std::vector<double> m_;
    std::size_t n_;
    double sigma_, sigma0_;
    std::mt19937& rng_;
    std::vector<double> ev_ = std::vector<double>(n_, 1.0);                   // C's eigenvalues, increasing
    std::vector<double> best_hist_, median_hist_;                              // per iteration (CMA-1)
    std::vector<bool> equal_hist_;
    std::size_t lam_ = 0, mu_ = 0;
    double mueff_ = 0.0, cs_ = 0.0, ds_ = 0.0, cc_ = 0.0, c1_ = 0.0, cmu_ = 0.0, cm_ = 1.0, chiN_ = 0.0;
    std::vector<double> w_, ps_, pc_, D_;
    lsq::Matrix C_, B_;
    long long g_ = 0;
    std::vector<std::vector<double>> y_;
    // injection (INJ-2): off until the first injection, the strategy unchanged
    double cy_ = std::numeric_limits<double>::infinity(), cym_ = std::numeric_limits<double>::infinity(),
           dsmax_ = std::numeric_limits<double>::infinity();
    bool injecting_ = false, elitist_ = false;
    std::vector<std::vector<double>> injected_, x_;
    std::vector<bool> is_injected_;
    std::vector<double> mean_x_, best_x_;
    double best_f_ = std::numeric_limits<double>::infinity();
};

}   // namespace mootation::scalar

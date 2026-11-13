#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// CatCMA (task 24, §24.3.4): R. Hamano, S. Saito, M. Nomura, K. Uchida, S.
// Shirakawa, "CatCMA: Stochastic Optimization for Mixed-Category Problems",
// GECCO 2024 (source catcma-stochastic-optimization_hamano2024): the joint
// distribution of a multivariate Gaussian on the N_co continuous variables
// and independent categorical distributions on the N_ca categorical ones (K_n
// categories each), Algorithm 1, Table 1:
//
//   sample   y_i ~ N(0, C), x_i = m + sigma y_i (16)-(17); c_i ~ Cat(q) (5)
//   rank     the lambda pairs by f(x, c); weights w_i / lambda of Table 1
//   Gaussian m (13); p_sigma, p_c (18)-(19); C (20); sigma (21); the floor
//            sigma >= sqrt(Lambda_min / min eig(C)) (27)
//   categories  G = sum w_i/lambda (c_i:lambda - q) (23); q += delta G /
//            ||G||_F (22), F the Fisher matrix of the reduced q_n (q_{n,1..K_n
//            - 1}; §3.2); the trust region delta by s, gamma (24)-(26), beta =
//            delta / sqrt(sum (K_n - 1)), alpha = 1.5; the margin q_{n,k} >=
//            q_n^min (28)-(29)
//
// The class asks for lambda points and is told their values, as CMAES
// (cmaes.hpp). It is a standalone solver: no registration yet (task 26's slot,
// with var_types in the single-objective path).
//
// Readings (OURS; checked on the PDF's pages 3-5 themselves, Table 1 and
// (18)-(29)):
//   CC-1. c_1 = 2 / ((N_co + 1.3)^2 + mu_w): Table 1 prints N_ca, but every
//     other rate of the continuous part has N_co and Table 1 follows [9,
//     Table 2] (Hansen's tutorial), whose n is the Gaussian's.
//   CC-2 (not a deviation). q_n^min = (1 - 0.73^(1/N_ca)) / (K_n - 1), as
//     Table 1, (31) and §4.1 print it.
//   CC-3 (not a deviation). (18)-(19) carry sqrt(c (2 - c) mu_w), as printed.
//   CC-4 (not a deviation). h_sigma = 1 where ||p_sigma|| < sqrt(1 - (1 -
//     c_sigma)^(2(t+1))) (1.4 + 2 / (N_co + 1)) E||N(0, I)||, as printed.
//   CC-5. F^(1/2)(q) in (24): the exact symmetric square root of each block,
//     by its eigendecomposition (the ASNG approximates it, footnote 3 and [4,
//     App. B], not in the corpus; the blocks are K_n - 1 wide).
//   CC-6. ||G||_F = 0 (the selected categories average to q exactly): (22)
//     is skipped, (24)-(26) applied with a zero direction (s decays, gamma
//     takes its beta (2 - beta), delta follows).
//   CC-7. Ties in f: the earlier candidate ranks first (stable).
//   CC-8. Draws, candidate by candidate: N_co normals for y_i, then one
//     uniform per categorical variable, the category the first k with u <
//     q_{n,1} + ... + q_{n,k} (the last where rounding leaves none).
//   CC-9. N_co >= 1 (the rates of Table 1 divide by it), N_ca >= 0, K_n >= 2.
//   CC-10. enhanced = false is §5.3's ablation: no rank-one update, no step
//     size adaptation and no (27) (sigma stays sigma^(0)), c_m = 1, c_mu as
//     Table 1, and the categories by (15) with eta_n = 1 / (N_ca (K_n - 1)),
//     then the margin.
//   CC-11. (24)-(25) accumulate the unit step: s gets sqrt(beta (2 - beta))
//     F^(1/2) G / ||G||_F, the direction (22) actually moves q along, and
//     gamma gets beta (2 - beta). The PDF prints F^(1/2) G and beta (2 -
//     beta) ||G||_F^2, unnormalized. Both keep E||s||^2 = gamma when the
//     ranking is random, but printed so the exponent of (26) scales with
//     ||G||_F^2, whose expectation under a random ranking is sum (K_n - 1) /
//     mu_w (about 24 at N_ca = K_n = 10): delta then jumps by e^several in a
//     generation, the step of (22) throws q onto the margin, and the
//     categories fix on what one generation favoured.
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

#include "../detmath.hpp"
#include "../fp_contract.hpp"
#include "../random.hpp"
#include "eigen.hpp"

namespace mootation::scalar::catcma {

using Vec = std::vector<double>;

struct Options {
    int popsize = 0;                      // 0: Table 1's lambda
    double q_min = -1.0;                  // < 0: CC-2's margin; else this one for every n
    bool enhanced = true;                 // CC-10
    double lambda_min = 1e-30;            // (27)
};

// a candidate: its continuous part and its categories (0-based, one a variable)
struct Point {
    Vec x;
    std::vector<int> c;
};

// CC-2
inline double default_margin(std::size_t n_ca, int K)
{
    if (n_ca == 0) return 0.0;
    return (1.0 - dm::pow(0.73, 1.0 / static_cast<double>(n_ca))) / static_cast<double>(K - 1);
}

class CatCMA {
public:
    CatCMA(Vec mean, double sigma, std::vector<int> categories, std::mt19937& rng, const Options& o = Options())
        : m_(std::move(mean)), K_(std::move(categories)), n_(m_.size()), nca_(K_.size()), sigma_(sigma),
          rng_(rng), o_(o)
    {
        if (n_ == 0) throw std::invalid_argument("catcma: at least one continuous variable (CC-9)");
        for (int k : K_)
            if (k < 2) throw std::invalid_argument("catcma: every categorical variable needs K_n >= 2 (CC-9)");
        if (!(sigma > 0.0)) throw std::invalid_argument("catcma: sigma > 0");
        const double n = static_cast<double>(n_), nca = static_cast<double>(nca_);
        // Table 1
        lam_ = o_.popsize > 0 ? static_cast<std::size_t>(o_.popsize)
                              : 4 + static_cast<std::size_t>(std::floor(3.0 * dm::log(n + nca)));
        mu_ = lam_ / 2;
        w_.assign(lam_, 0.0);
        double s = 0.0;
        for (std::size_t i = 0; i < mu_; ++i) {
            w_[i] = dm::log((static_cast<double>(lam_) + 1.0) / 2.0) - dm::log(static_cast<double>(i + 1));
            s += w_[i];
        }
        double s2 = 0.0;
        for (std::size_t i = 0; i < mu_; ++i) {
            w_[i] /= s;
            s2 += w_[i] * w_[i];
        }
        muw_ = 1.0 / s2;
        cs_ = (muw_ + 2.0) / (n + muw_ + 5.0);
        ds_ = 1.0 + cs_ + 2.0 * std::max(0.0, std::sqrt((muw_ - 1.0) / (n + 1.0)) - 1.0);
        cc_ = (4.0 + muw_ / n) / (n + 4.0 + 2.0 * muw_ / n);
        c1_ = 2.0 / ((n + 1.3) * (n + 1.3) + muw_);                              // CC-1
        cmu_ = std::min(1.0 - c1_, 2.0 * (muw_ - 2.0 + 1.0 / muw_) / ((n + 2.0) * (n + 2.0) + muw_));
        chi_ = std::sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n));
        ps_.assign(n_, 0.0);
        pc_.assign(n_, 0.0);
        C_ = lsq::Matrix(n_, n_);
        B_ = lsq::Matrix(n_, n_);
        for (std::size_t i = 0; i < n_; ++i) C_(i, i) = B_(i, i) = 1.0;
        D_.assign(n_, 1.0);
        // the categories: uniform q, the margins, the trust region
        q_.resize(nca_);
        qmin_.resize(nca_);
        dims_ = 0;
        for (std::size_t v = 0; v < nca_; ++v) {
            q_[v].assign(static_cast<std::size_t>(K_[v]), 1.0 / static_cast<double>(K_[v]));
            qmin_[v] = o_.q_min >= 0.0 ? o_.q_min : default_margin(nca_, K_[v]);
            if (qmin_[v] * static_cast<double>(K_[v]) > 1.0)
                throw std::invalid_argument("catcma: q_min K_n must not exceed 1");
            dims_ += static_cast<std::size_t>(K_[v] - 1);
        }
        s_.assign(dims_, 0.0);
    }

    std::size_t lambda() const { return lam_; }
    const Vec& mean() const { return m_; }
    double sigma() const { return sigma_; }
    double delta() const { return delta_; }
    const std::vector<Vec>& q() const { return q_; }
    const std::vector<double>& eigenvalues() const { return ev_; }
    double margin(std::size_t v) const { return qmin_[v]; }

    // lambda candidates (16), (17), (5); CC-8
    std::vector<Point> ask()
    {
        y_.assign(lam_, Vec(n_));
        pts_.assign(lam_, Point{});
        for (std::size_t k = 0; k < lam_; ++k) {
            Vec z(n_);
            for (double& t : z) t = rnd::normal01(rng_);
            Point& p = pts_[k];
            p.x.assign(n_, 0.0);
            for (std::size_t i = 0; i < n_; ++i) {
                double t = 0.0;
                for (std::size_t j = 0; j < n_; ++j) t += B_(i, j) * D_[j] * z[j];
                y_[k][i] = t;
                p.x[i] = m_[i] + sigma_ * t;
            }
            p.c.assign(nca_, 0);
            for (std::size_t v = 0; v < nca_; ++v) {
                const double u = rnd::uniform01(rng_);
                double cum = 0.0;
                int pick = K_[v] - 1;
                for (int j = 0; j < K_[v]; ++j) {
                    cum += q_[v][static_cast<std::size_t>(j)];
                    if (u < cum) {
                        pick = j;
                        break;
                    }
                }
                p.c[v] = pick;
            }
        }
        return pts_;
    }

    // the values of the last ask()'s candidates, in its order
    void tell(const Vec& f)
    {
        if (f.size() != lam_) throw std::invalid_argument("catcma: tell needs lambda values");
        std::vector<std::size_t> order(lam_);
        std::iota(order.begin(), order.end(), std::size_t{0});
        std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return f[a] < f[b]; });  // CC-7
        update_gaussian(order);
        update_categories(order);
        ++t_;
    }

private:
    Vec m_;
    std::vector<int> K_;
    std::size_t n_, nca_, lam_ = 0, mu_ = 0, dims_ = 0;
    double sigma_;
    std::mt19937& rng_;
    Options o_;
    Vec w_;
    double muw_ = 0.0, cs_ = 0.0, ds_ = 0.0, cc_ = 0.0, c1_ = 0.0, cmu_ = 0.0, chi_ = 0.0;
    Vec ps_, pc_, D_, ev_;
    lsq::Matrix C_, B_;
    std::vector<Vec> q_;
    Vec qmin_, s_;
    double delta_ = 1.0, gamma_ = 0.0;
    long long t_ = 0;
    std::vector<Vec> y_;
    std::vector<Point> pts_;

    void update_gaussian(const std::vector<std::size_t>& order)
    {
        const double n = static_cast<double>(n_);
        Vec yw(n_, 0.0);
        for (std::size_t k = 0; k < mu_; ++k)
            for (std::size_t i = 0; i < n_; ++i) yw[i] += w_[k] * y_[order[k]][i];
        for (std::size_t i = 0; i < n_; ++i) m_[i] += sigma_ * yw[i];                       // (13), c_m = 1
        if (o_.enhanced) {
            Vec ciyw(n_, 0.0);                                                              // C^(-1/2) yw = B D^-1 B^T yw
            for (std::size_t i = 0; i < n_; ++i)
                for (std::size_t j = 0; j < n_; ++j) {
                    double t = 0.0;
                    for (std::size_t k = 0; k < n_; ++k) t += B_(i, k) / D_[k] * B_(j, k);
                    ciyw[i] += t * yw[j];
                }
            const double a = std::sqrt(cs_ * (2.0 - cs_) * muw_);                          // (18), CC-3
            double norm = 0.0;
            for (std::size_t i = 0; i < n_; ++i) {
                ps_[i] = (1.0 - cs_) * ps_[i] + a * ciyw[i];
                norm += ps_[i] * ps_[i];
            }
            norm = std::sqrt(norm);
            const bool hs = norm / std::sqrt(1.0 - dm::pow(1.0 - cs_, 2.0 * static_cast<double>(t_ + 1))) <
                            (1.4 + 2.0 / (n + 1.0)) * chi_;                                  // CC-4
            const double b = std::sqrt(cc_ * (2.0 - cc_) * muw_);
            for (std::size_t i = 0; i < n_; ++i) pc_[i] = (1.0 - cc_) * pc_[i] + (hs ? b * yw[i] : 0.0);   // (19)
            const double keep = 1.0 + (hs ? 0.0 : 1.0) * c1_ * cc_ * (2.0 - cc_) - c1_ - cmu_;
            lsq::Matrix Cn(n_, n_);                                                         // (20)
            for (std::size_t i = 0; i < n_; ++i)
                for (std::size_t j = 0; j < n_; ++j) {
                    double r = 0.0;
                    for (std::size_t k = 0; k < mu_; ++k) r += w_[k] * y_[order[k]][i] * y_[order[k]][j];
                    Cn(i, j) = keep * C_(i, j) + c1_ * pc_[i] * pc_[j] + cmu_ * r;
                }
            for (std::size_t i = 0; i < n_; ++i)
                for (std::size_t j = 0; j < n_; ++j) C_(i, j) = (Cn(i, j) + Cn(j, i)) / 2.0;
            sigma_ *= dm::exp(cs_ / ds_ * (norm / chi_ - 1.0));                             // (21)
        } else {
            lsq::Matrix Cn(n_, n_);                                                         // (14)
            for (std::size_t i = 0; i < n_; ++i)
                for (std::size_t j = 0; j < n_; ++j) {
                    double r = 0.0;
                    for (std::size_t k = 0; k < mu_; ++k) r += w_[k] * y_[order[k]][i] * y_[order[k]][j];
                    Cn(i, j) = (1.0 - cmu_) * C_(i, j) + cmu_ * r;
                }
            for (std::size_t i = 0; i < n_; ++i)
                for (std::size_t j = 0; j < n_; ++j) C_(i, j) = (Cn(i, j) + Cn(j, i)) / 2.0;
        }
        eigen::symmetric(C_, ev_, B_);
        for (std::size_t i = 0; i < n_; ++i) D_[i] = std::sqrt(std::max(ev_[i], 0.0));
        if (o_.enhanced) {                                                                  // (27)
            const double low = *std::min_element(ev_.begin(), ev_.end());
            if (low > 0.0) sigma_ = std::max(sigma_, std::sqrt(o_.lambda_min / low));
        }
    }

    // the natural gradient's block of variable v, reduced to its first K - 1 (23)
    Vec gradient(std::size_t v, const std::vector<std::size_t>& order) const
    {
        const std::size_t k = static_cast<std::size_t>(K_[v] - 1);
        Vec g(k, 0.0);
        for (std::size_t r = 0; r < mu_; ++r) {
            const int c = pts_[order[r]].c[v];
            for (std::size_t j = 0; j < k; ++j) g[j] += w_[r] * ((static_cast<std::size_t>(c) == j ? 1.0 : 0.0) - q_[v][j]);
        }
        return g;
    }

    // F(q_v) = diag(q_v)^-1 + (1 / q_{v,K}) 1 1^T on the reduced block (§3.2)
    lsq::Matrix fisher(std::size_t v) const
    {
        const std::size_t k = static_cast<std::size_t>(K_[v] - 1);
        lsq::Matrix F(k, k);
        const double last = q_[v][k];
        for (std::size_t i = 0; i < k; ++i)
            for (std::size_t j = 0; j < k; ++j) F(i, j) = (i == j ? 1.0 / q_[v][i] : 0.0) + 1.0 / last;
        return F;
    }

    void update_categories(const std::vector<std::size_t>& order)
    {
        if (nca_ == 0) return;
        std::vector<Vec> G(nca_);
        for (std::size_t v = 0; v < nca_; ++v) G[v] = gradient(v, order);
        if (!o_.enhanced) {                                                                  // (15), CC-10
            for (std::size_t v = 0; v < nca_; ++v) {
                const double eta = 1.0 / (static_cast<double>(nca_) * static_cast<double>(K_[v] - 1));
                step(v, G[v], eta);
            }
            margins();
            return;
        }
        // ||G||_F^2 = sum_v G_v^T F_v G_v, and F^(1/2) G block by block (CC-5)
        double gf2 = 0.0;
        Vec root;                                                                            // F^(1/2) G, all blocks
        root.reserve(dims_);
        for (std::size_t v = 0; v < nca_; ++v) {
            const lsq::Matrix F = fisher(v);
            const std::size_t k = G[v].size();
            for (std::size_t i = 0; i < k; ++i)
                for (std::size_t j = 0; j < k; ++j) gf2 += G[v][i] * F(i, j) * G[v][j];
            Vec w;
            lsq::Matrix V;
            eigen::symmetric(F, w, V);
            for (std::size_t i = 0; i < k; ++i) {
                double t = 0.0;
                for (std::size_t j = 0; j < k; ++j) {
                    double r = 0.0;                                                          // (V diag(sqrt w) V^T)_ij
                    for (std::size_t l = 0; l < k; ++l) r += V(i, l) * std::sqrt(std::max(w[l], 0.0)) * V(j, l);
                    t += r * G[v][j];
                }
                root.push_back(t);
            }
        }
        const double gf = std::sqrt(gf2);
        if (gf > 0.0)                                                                        // (22), CC-6
            for (std::size_t v = 0; v < nca_; ++v) step(v, G[v], delta_ / gf);
        const double beta = delta_ / std::sqrt(static_cast<double>(dims_));
        const double unit = gf > 0.0 ? 1.0 / gf : 0.0;                                      // CC-11
        double s2 = 0.0;
        for (std::size_t i = 0; i < dims_; ++i) {                                           // (24)
            s_[i] = (1.0 - beta) * s_[i] + std::sqrt(beta * (2.0 - beta)) * root[i] * unit;
            s2 += s_[i] * s_[i];
        }
        gamma_ = (1.0 - beta) * (1.0 - beta) * gamma_ + beta * (2.0 - beta);                // (25), CC-11
        delta_ *= dm::exp(beta * (s2 / 1.5 - gamma_));                                      // (26), alpha = 1.5
        margins();
    }

    // q_v's first K - 1 moved by `scale` G, the last the rest
    void step(std::size_t v, const Vec& g, double scale)
    {
        const std::size_t k = g.size();
        double sum = 0.0;
        for (std::size_t j = 0; j < k; ++j) {
            q_[v][j] += scale * g[j];
            sum += q_[v][j];
        }
        q_[v][k] = 1.0 - sum;
    }

    // (28), (29)
    void margins()
    {
        for (std::size_t v = 0; v < nca_; ++v) {
            const double lo = qmin_[v];
            double sum = 0.0, over = 0.0;
            for (double& t : q_[v]) {
                t = std::max(t, lo);
                sum += t;
                over += t - lo;
            }
            if (over > 0.0)
                for (double& t : q_[v]) t += (1.0 - sum) / over * (t - lo);
        }
    }
};

}   // namespace mootation::scalar::catcma

#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// Standard PSO 2011, as M. Zambrano-Bigiarini, M. Clerc, R. Rojas, "Standard
// Particle Swarm Optimisation 2011 at CEC-2013: A baseline for future PSO
// improvements", IEEE CEC 2013, §II-B, §II-C, §III (source
// spso2011_zambrano-bigiarini2013):
//   p'_i = X_i + c1 U1 (x) (P_i - X_i), l'_i = X_i + c2 U2 (x) (L_i - X_i)
//   (2a-b); G_i = (X_i + p'_i + l'_i) / 3 (2c); x' a random point of the
//   hypersphere H(G_i, ||G_i - X_i||); V_i = w V_i + x' - X_i (3); X_i = X_i
//   + V_i (1b); P_i = X_i where f(X_i) < f(P_i);
//   the adaptive random topology: every particle informs K random particles
//   (the same one may be drawn more than once) and itself; the links are
//   drawn again when the best found shows no improvement (§II-B);
//   X_i^0 = U(min, max), V_i^0 = (U(min, max) - X_i^0) / 2 (4a-b); at a
//   boundary the particle stops on it with zero velocity (absorbing wall).
// Defaults are §III's: 40 particles, K = 3, c1 = c2 = 0.5 + ln 2, w = 1 / (2
// ln 2), the velocity within [-X^max, X^max].
//
// The caller asks for points, evaluates them and tells their values f and
// violations v; the first ask() gives the initial swarm, uniform in the box.
// Minimization; random numbers are rnd:: of the caller's generator, ln by
// dm:: (detmath.hpp).
//
// Readings (OURS):
//   SPSO-1. Synchronous: the whole swarm is evaluated, then every P_i, the
//     links and every L_i are updated, then every particle moves (§II-C's
//     loop updates P_i and L particle by particle).
//   SPSO-2. L_i: the best P_j over j = i and every j that informs i. Ties: the
//     lowest index.
//   SPSO-3. x' is uniform in the ball's volume: a normal direction, the radius
//     ||G - X|| U^(1/D). (2c) is used also where L_i = P_i (the 2011 code's
//     two-point centre is not in the paper).
//   SPSO-4. "the global optimum shows no improvement": the best P after a
//     generation is not better than the one before it; the links are drawn
//     again then, and at the start.
//   SPSO-5. "[-X^max, X^max]" (§II-A: V^max = X^max, "the limits of the
//     search space"): |V_id| <= (max_d - min_d) / 2, which is X^max on the
//     symmetric [-100, 100] of §III.
//   SPSO-6. The absorbing wall acts coordinate by coordinate: a coordinate
//     outside its range is put on the bound it crossed and its velocity is 0.
//   SPSO-7. Draws, particle by particle: U1 for every d, U2 for every d, the
//     direction's D normals, the radius's U; the initial velocity's U after
//     all positions; links: informer by informer, K draws each.
//   SPSO-8. Constraints g <= 0: NSGA-II's constrained-domination at M = 1
//     (evo.hpp) for "f(X_i) < f(P_i)", L_i and SPSO-4; not in the paper.
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <random>
#include <stdexcept>
#include <vector>

#include "../detmath.hpp"
#include "../random.hpp"
#include "evo.hpp"

namespace mootation::scalar::spso2011 {

using evo::Rows;
using evo::Vec;

struct Options {
    std::size_t n = 40;                                    // §III
    std::size_t K = 3;                                     // §III
    double c = 0.5 + 0.6931471805599453;                   // 0.5 + ln 2
    double w = 1.0 / (2.0 * 0.6931471805599453);           // 1 / (2 ln 2)
};

class SPSO2011 {
public:
    SPSO2011(Vec lo, Vec hi, const Options& o, std::mt19937& rng) : lo_(std::move(lo)), hi_(std::move(hi)), o_(o), rng_(rng)
    {
        if (lo_.empty() || lo_.size() != hi_.size()) throw std::invalid_argument("spso2011: lo, hi");
        if (o_.n < 2 || o_.K < 1) throw std::invalid_argument("spso2011: n >= 2, K >= 1");
    }

    Rows ask()
    {
        const std::size_t d = lo_.size();
        rnd::uniform_real_distribution<double> U01(0.0, 1.0);
        if (x_.empty()) {                                                // (4a-b)
            x_ = evo::uniform(lo_, hi_, o_.n, rng_);
            v_.assign(o_.n, Vec(d));
            for (std::size_t i = 0; i < o_.n; ++i)
                for (std::size_t j = 0; j < d; ++j)
                    v_[i][j] = (lo_[j] + U01(rng_) * (hi_[j] - lo_[j]) - x_[i][j]) / 2.0;
            return x_;
        }
        for (std::size_t i = 0; i < o_.n; ++i) {
            const Vec& P = p_[i];
            const Vec& L = p_[local_[i]];
            Vec G(d);
            for (std::size_t j = 0; j < d; ++j) {                        // (2a)
                const double pp = x_[i][j] + o_.c * U01(rng_) * (P[j] - x_[i][j]);
                G[j] = x_[i][j] + pp;
            }
            for (std::size_t j = 0; j < d; ++j) {                        // (2b), (2c)
                const double ll = x_[i][j] + o_.c * U01(rng_) * (L[j] - x_[i][j]);
                G[j] = (G[j] + ll) / 3.0;
            }
            double radius = 0.0;
            for (std::size_t j = 0; j < d; ++j) radius += (G[j] - x_[i][j]) * (G[j] - x_[i][j]);
            radius = std::sqrt(radius);
            Vec dir(d);                                                  // SPSO-3
            double norm = 0.0;
            for (std::size_t j = 0; j < d; ++j) {
                dir[j] = rnd::normal01(rng_);
                norm += dir[j] * dir[j];
            }
            norm = std::sqrt(norm);
            const double r = radius * dm::pow(U01(rng_), 1.0 / static_cast<double>(d));
            for (std::size_t j = 0; j < d; ++j) {
                const double xp = G[j] + (norm > 0.0 ? r * dir[j] / norm : 0.0);
                const double vmax = (hi_[j] - lo_[j]) / 2.0;             // SPSO-5
                v_[i][j] = std::clamp(o_.w * v_[i][j] + xp - x_[i][j], -vmax, vmax);   // (3)
                x_[i][j] += v_[i][j];                                    // (1b)
                if (x_[i][j] < lo_[j] || x_[i][j] > hi_[j]) {            // SPSO-6
                    x_[i][j] = x_[i][j] < lo_[j] ? lo_[j] : hi_[j];
                    v_[i][j] = 0.0;
                }
            }
        }
        return x_;
    }

    void tell(const Vec& f, const Vec& v = {})
    {
        if (f.size() != o_.n || (!v.empty() && v.size() != o_.n)) throw std::invalid_argument("spso2011: tell sizes");
        const Vec vv = v.empty() ? Vec(o_.n, 0.0) : v;
        bool improved = false;
        if (p_.empty()) {
            p_ = x_;
            pf_ = f;
            pv_ = vv;
        } else {
            const std::size_t b = evo::best(pf_, pv_);
            const double fb = pf_[b], vb = pv_[b];
            for (std::size_t i = 0; i < o_.n; ++i)
                if (evo::better(f[i], vv[i], pf_[i], pv_[i])) {
                    p_[i] = x_[i];
                    pf_[i] = f[i];
                    pv_[i] = vv[i];
                }
            const std::size_t a = evo::best(pf_, pv_);
            improved = evo::better(pf_[a], pv_[a], fb, vb);
        }
        if (!improved) links();                                         // SPSO-4
        local_.assign(o_.n, 0);
        for (std::size_t i = 0; i < o_.n; ++i) {                         // SPSO-2
            std::size_t b = i;
            for (std::size_t j : informed_by_[i])
                if (evo::better(pf_[j], pv_[j], pf_[b], pv_[b]) ||
                    (j < b && !evo::better(pf_[b], pv_[b], pf_[j], pv_[j])))
                    b = j;
            local_[i] = b;
        }
    }

    const Rows& population() const { return p_; }   // the particles' bests
    const Vec& values() const { return pf_; }
    const Vec& violations() const { return pv_; }
    std::size_t best() const { return evo::best(pf_, pv_); }
    const Rows& positions() const { return x_; }
    const Rows& velocities() const { return v_; }

private:
    void links()
    {
        rnd::uniform_int_distribution<std::size_t> pick(0, o_.n - 1);
        informed_by_.assign(o_.n, {});
        for (std::size_t i = 0; i < o_.n; ++i)                           // i informs K random particles
            for (std::size_t k = 0; k < o_.K; ++k) informed_by_[pick(rng_)].push_back(i);
    }

    Vec lo_, hi_;
    Options o_;
    std::mt19937& rng_;
    Rows x_, v_, p_;
    Vec pf_, pv_;
    std::vector<std::vector<std::size_t>> informed_by_;
    std::vector<std::size_t> local_;
};

}   // namespace mootation::scalar::spso2011

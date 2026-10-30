#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// Particle swarm with the Type 1'' constriction, as M. Clerc, J. Kennedy, "The
// Particle Swarm — Explosion, Stability, and Convergence in a
// Multidimensional Complex Space", IEEE TEVC 6(1):58-73, 2002, §VI-VII
// (source pso-constriction_clerc2002), on the particle swarm of J. Kennedy,
// R. Eberhart, "Particle swarm optimization", ICNN 1995 (source
// pso_kennedy1995):
//   v_id = chi (v_id + phi_1 (p_id - x_id) + phi_2 (p_gd - x_id)),
//   x_id = x_id + v_id, phi_1, phi_2 = rand() phi_max / 2 for every d;
//   p_i = x_i where f(x_i) < f(p_i); p_g the best p of i's neighbours.
//   chi = 2 kappa / (phi - 2 + sqrt(phi^2 - 4 phi)) for phi > 4, kappa
//   otherwise (4.17), phi = phi_max.
// Defaults are §VII's: 20 particles, phi_max = 4.1, kappa = 1 (chi = 0.7298),
// no V_max; the neighbours a ring (PSO-4).
//
// The caller asks for points, evaluates them and tells their values f and
// violations v; the first ask() gives the initial swarm, uniform in the box.
// Minimization; random numbers are rnd:: of the caller's generator.
//
// Readings (OURS):
//   PSO-1. §VI computes chi "as shown in (4.15)"; (4.15) is the modulus of
//     the eigenvalues, sqrt(chi). The coefficient is (4.17), the one whose
//     modulus (4.15) is.
//   PSO-2. Synchronous: the whole swarm is evaluated, then every p_i and p_g
//     are updated, then every particle moves (the pseudocode updates p_g
//     particle by particle).
//   PSO-3. "Initialize population: random x_i, v_i": v_i uniform in
//     [-(hi - lo)/2, (hi - lo)/2] coordinate by coordinate, drawn after all
//     positions; `v0_zero` starts at 0 (SMPSO-5 of smpso.hpp).
//   PSO-4. "p_g = min(p_neighbors)": §VII names no topology. Taken: the ring
//     of radius 1, x_i-1, x_i, x_i+1 (indices mod n; ties to i, then i - 1),
//     which gives Table V's Type 1'' Griewank (0.0039) and Ackley (0.13
//     against 0.205); the global best (`ring = false`) converges early there
//     (0.17 and 4.5) — the TASK18 report, §14.
//   PSO-5. Draws: particle by particle, phi_1 then phi_2 for every d.
//   PSO-6. Bounds: the paper's box is the initial range; a position outside
//     the box is repaired by ops::repair_value (reflect by default; `native`
//     leaves it — the paper's), the velocity kept.
//   PSO-7. Constraints g <= 0: NSGA-II's constrained-domination at M = 1
//     (evo.hpp) for "f(x_i) < f(p_i)" and for p_g; not in the papers.
//   PSO-8. done() is true once every velocity is 0 and every x_i = p_i = p_g.
// ============================================================================

#include <cmath>
#include <cstddef>
#include <random>
#include <stdexcept>
#include <vector>

#include "../operators/bound_repair.hpp"
#include "../random.hpp"
#include "evo.hpp"

namespace mootation::scalar::pso {

using evo::Rows;
using evo::Vec;

struct Options {
    std::size_t n = 20;                                    // §VII
    double phi_max = 4.1;                                  // §VII
    double kappa = 1.0;                                    // Type 1''
    bool v0_zero = false;                                  // PSO-3
    bool ring = true;                                      // PSO-4
    ops::BoundRepair repair = ops::BoundRepair::Reflect;   // PSO-6
};

// (4.17)
inline double constriction(double phi, double kappa)
{
    return phi > 4.0 ? 2.0 * kappa / (phi - 2.0 + std::sqrt(phi * phi - 4.0 * phi)) : kappa;
}

class PSO {
public:
    PSO(Vec lo, Vec hi, const Options& o, std::mt19937& rng)
        : lo_(std::move(lo)), hi_(std::move(hi)), o_(o), rng_(rng), chi_(constriction(o.phi_max, o.kappa))
    {
        if (lo_.empty() || lo_.size() != hi_.size()) throw std::invalid_argument("pso: lo, hi");
        if (o_.n < 1 || !(o_.phi_max > 0.0) || !(o_.kappa > 0.0)) throw std::invalid_argument("pso: n, phi_max, kappa");
        if (o_.repair == ops::BoundRepair::Resample) throw std::invalid_argument("pso: resample (PSO-6)");
    }

    double chi() const { return chi_; }

    Rows ask()
    {
        const std::size_t d = lo_.size();
        if (x_.empty()) {
            x_ = evo::uniform(lo_, hi_, o_.n, rng_);
            v_.assign(o_.n, Vec(d, 0.0));
            if (!o_.v0_zero) {                                           // PSO-3
                rnd::uniform_real_distribution<double> U01(0.0, 1.0);
                for (Vec& v : v_)
                    for (std::size_t j = 0; j < d; ++j) v[j] = (U01(rng_) - 0.5) * (hi_[j] - lo_[j]);
            }
            return x_;
        }
        rnd::uniform_real_distribution<double> U01(0.0, 1.0);
        std::vector<std::size_t> nb(o_.n, g_);                           // PSO-4
        if (o_.ring)
            for (std::size_t i = 0; i < o_.n; ++i) {
                std::size_t b = i;
                for (std::size_t k : {(i + o_.n - 1) % o_.n, (i + 1) % o_.n})
                    if (evo::better(pf_[k], pv_[k], pf_[b], pv_[b])) b = k;
                nb[i] = b;
            }
        for (std::size_t i = 0; i < o_.n; ++i)
            for (std::size_t j = 0; j < d; ++j) {                        // PSO-5
                const Vec& g = p_[nb[i]];
                const double phi1 = U01(rng_) * o_.phi_max / 2.0, phi2 = U01(rng_) * o_.phi_max / 2.0;
                v_[i][j] = chi_ * (v_[i][j] + phi1 * (p_[i][j] - x_[i][j]) + phi2 * (g[j] - x_[i][j]));
                double y = x_[i][j] + v_[i][j];
                if (o_.repair != ops::BoundRepair::Native && (y < lo_[j] || y > hi_[j]))
                    y = ops::repair_value(y, lo_[j], hi_[j], x_[i][j], o_.repair, rng_);   // PSO-6
                x_[i][j] = y;
            }
        return x_;
    }

    void tell(const Vec& f, const Vec& v = {})
    {
        if (f.size() != o_.n || (!v.empty() && v.size() != o_.n)) throw std::invalid_argument("pso: tell sizes");
        const Vec vv = v.empty() ? Vec(o_.n, 0.0) : v;
        if (p_.empty()) {
            p_ = x_;
            pf_ = f;
            pv_ = vv;
        } else {
            for (std::size_t i = 0; i < o_.n; ++i)                       // PSO-2, PSO-7
                if (evo::better(f[i], vv[i], pf_[i], pv_[i])) {
                    p_[i] = x_[i];
                    pf_[i] = f[i];
                    pv_[i] = vv[i];
                }
        }
        g_ = evo::best(pf_, pv_);                                        // PSO-4
    }

    bool done() const                                                    // PSO-8
    {
        if (p_.empty()) return false;
        for (std::size_t i = 0; i < o_.n; ++i) {
            if (x_[i] != p_[g_] || p_[i] != p_[g_]) return false;
            for (double t : v_[i]) if (t != 0.0) return false;
        }
        return true;
    }

    const Rows& population() const { return p_; }   // the particles' bests
    const Vec& values() const { return pf_; }
    const Vec& violations() const { return pv_; }
    std::size_t best() const { return g_; }
    const Rows& positions() const { return x_; }

private:
    Vec lo_, hi_;
    Options o_;
    std::mt19937& rng_;
    double chi_;
    Rows x_, v_, p_;
    Vec pf_, pv_;
    std::size_t g_ = 0;
};

}   // namespace mootation::scalar::pso

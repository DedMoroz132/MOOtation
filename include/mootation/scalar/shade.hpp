#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// SHADE, as R. Tanabe, A. Fukunaga, "Success-History Based Parameter
// Adaptation for Differential Evolution", IEEE CEC 2013, pp. 71-78, §V,
// Algorithm 1 (source shade_tanabe2013):
//   memories M_CR, M_F of H entries, all 0.5, and a position k = 1;
//   for every x_i: r_i from [1, H]; CR_i = randn(M_CR,r_i, 0.1) (15), F_i =
//   randc(M_F,r_i, 0.1) (16), handled as JADE's; p_i = rand[p_min, 0.2],
//   p_min = 2/N (20); the trial by current-to-pbest/1/bin with the archive,
//   x_pbest one of the N p_i best (pbest.hpp), a coordinate outside the box
//   to the middle of bound and parent (21);
//   the trial replaces the parent when f(u) <= f(x) (6); when f(u) < f(x) the
//   parent goes to A and CR_i, F_i to S_CR, S_F, with weight dF = |f(u) -
//   f(x)|; A is cut at random to |P|;
//   when S is not empty: M_CR,k = mean_WA(S_CR) (13)-(14), M_F,k =
//   mean_WL(S_F) (19), k = k + 1, back to 1 past H (17)-(18).
// Defaults are §VI's: N = 100, H = 100, the archive on.
//
// The caller asks for points, evaluates them and tells their values f and
// violations v; the first ask() gives the initial population, uniform in the
// box. Minimization; random numbers are rnd:: of the caller's generator.
//
// Readings (OURS):
//   SHADE-1. "the top N p_i members": the ceil(N p_i) best (p_i >= 2/N gives
//     at least 2); ranking and the trial's draws as pbest.hpp (PB-1, PB-2).
//   SHADE-2. The draws for x_i, in Algorithm 1's order: r_i, CR_i, F_i, p_i,
//     then the trial's.
//   SHADE-3. Constraints g <= 0: NSGA-II's constrained-domination at M = 1
//     (evo.hpp) for "<=" and "<" of Algorithm 1 and for the ranking; not in
//     the paper. The weight of a success is |f(u) - f(x)| when both points
//     are feasible, the drop of the violation otherwise. Where the weights
//     do not sum to a finite positive number (a NaN or infinite parent
//     value), every success weighs the same.
//   SHADE-4. N >= 4 so that x_i, x_r1, x~_r2 can differ; refused otherwise.
//   SHADE-5. All N trials of a generation are built from P and A at its
//     start (Algorithm 1 builds them all before any selection).
//   SHADE-6. done() is true once all of P and A are one point.
// ============================================================================

#include <cmath>
#include <cstddef>
#include <random>
#include <stdexcept>
#include <vector>

#include "../random.hpp"
#include "evo.hpp"
#include "pbest.hpp"

namespace mootation::scalar::shade {

using evo::Rows;
using evo::Vec;

struct Options {
    std::size_t np = 100;               // §VI
    std::size_t H = 100;                // §VI
    double p_max = 0.2;                 // (20)
    bool archive = true;
};

namespace detail {

// the weight of a success (SHADE-3)
inline double gain(double fu, double vu, double fx, double vx)
{
    if (evo::key(vu) <= 0.0 && evo::key(vx) <= 0.0) return std::fabs(evo::key(fu) - evo::key(fx));
    return std::fabs(evo::key(vx) - evo::key(vu));
}

// weights w_k = dF_k / sum dF (14), or all equal (SHADE-3)
inline Vec weights(const Vec& gains)
{
    double s = 0.0;
    for (double g : gains) s += g;
    Vec w(gains.size(), 1.0 / static_cast<double>(gains.size()));
    if (std::isfinite(s) && s > 0.0)
        for (std::size_t k = 0; k < gains.size(); ++k) w[k] = gains[k] / s;
    return w;
}

inline double mean_wa(const Vec& w, const Vec& S)                       // (13)
{
    double s = 0.0;
    for (std::size_t k = 0; k < S.size(); ++k) s += w[k] * S[k];
    return s;
}

inline double mean_wl(const Vec& w, const Vec& S)                       // (19)
{
    double a = 0.0, b = 0.0;
    for (std::size_t k = 0; k < S.size(); ++k) {
        a += w[k] * S[k] * S[k];
        b += w[k] * S[k];
    }
    return a / b;
}

}   // namespace detail

class SHADE {
public:
    SHADE(Vec lo, Vec hi, const Options& o, std::mt19937& rng)
        : lo_(std::move(lo)), hi_(std::move(hi)), o_(o), rng_(rng), m_cr_(o.H, 0.5), m_f_(o.H, 0.5)
    {
        if (lo_.empty() || lo_.size() != hi_.size()) throw std::invalid_argument("shade: lo, hi");
        if (o_.np < 4) throw std::invalid_argument("shade: N >= 4 (SHADE-4)");
        if (o_.H < 1) throw std::invalid_argument("shade: H >= 1");
        if (!(o_.p_max >= 2.0 / static_cast<double>(o_.np) && o_.p_max <= 1.0))
            throw std::invalid_argument("shade: p_max in [2/N, 1]");
    }

    std::size_t np() const { return o_.np; }
    const Vec& memory_cr() const { return m_cr_; }
    const Vec& memory_f() const { return m_f_; }
    const Rows& archive() const { return a_; }

    Rows ask()
    {
        if (x_.empty()) return u_ = evo::uniform(lo_, hi_, o_.np, rng_);
        const std::size_t n = o_.np;
        const double nn = static_cast<double>(n);
        const std::vector<std::size_t> order = pbest::ranking(f_, v_);
        rnd::uniform_int_distribution<std::size_t> pick_h(0, o_.H - 1);
        rnd::uniform_real_distribution<double> pick_p(2.0 / nn, o_.p_max);
        u_.assign(n, Vec());
        cr_.assign(n, 0.0);
        F_.assign(n, 0.0);
        for (std::size_t i = 0; i < n; ++i) {                            // SHADE-2
            const std::size_t r = pick_h(rng_);
            cr_[i] = pbest::draw_CR(rng_, m_cr_[r]);                     // (15)
            F_[i] = pbest::draw_F(rng_, m_f_[r]);                        // (16)
            const double p = pick_p(rng_);                               // (20)
            const std::size_t top = static_cast<std::size_t>(std::ceil(nn * p));   // SHADE-1
            u_[i] = pbest::trial(x_, a_, i, order, std::max<std::size_t>(top, 2), F_[i], cr_[i], lo_, hi_, rng_);
        }
        return u_;
    }

    // f and v of the points of the last ask(); v empty: no constraints.
    void tell(const Vec& f, const Vec& v = {})
    {
        const std::size_t n = o_.np;
        if (f.size() != n || (!v.empty() && v.size() != n)) throw std::invalid_argument("shade: tell sizes");
        if (x_.empty()) {
            x_ = u_;
            f_ = f;
            v_ = v.empty() ? Vec(n, 0.0) : v;
            return;
        }
        Vec s_cr, s_f, gains;
        for (std::size_t i = 0; i < n; ++i) {
            const double vi = v.empty() ? 0.0 : v[i];
            if (!evo::not_worse(f[i], vi, f_[i], v_[i])) continue;      // (6)
            if (evo::better(f[i], vi, f_[i], v_[i])) {                   // Algorithm 1, lines 21-23
                if (o_.archive) a_.push_back(x_[i]);
                s_cr.push_back(cr_[i]);
                s_f.push_back(F_[i]);
                gains.push_back(detail::gain(f[i], vi, f_[i], v_[i]));
            }
            x_[i] = u_[i];
            f_[i] = f[i];
            v_[i] = vi;
        }
        pbest::cut(a_, n, rng_);
        if (!s_cr.empty()) {                                             // (17)-(18)
            const Vec w = detail::weights(gains);
            m_cr_[k_] = detail::mean_wa(w, s_cr);
            m_f_[k_] = detail::mean_wl(w, s_f);
            k_ = (k_ + 1) % o_.H;
        }
    }

    bool done() const { return pbest::collapsed(x_, a_); }               // SHADE-6

    const Rows& population() const { return x_; }
    const Vec& values() const { return f_; }
    const Vec& violations() const { return v_; }
    std::size_t best() const { return evo::best(f_, v_); }

private:
    Vec lo_, hi_;
    Options o_;
    std::mt19937& rng_;
    Rows x_, u_, a_;
    Vec f_, v_, cr_, F_;
    Vec m_cr_, m_f_;
    std::size_t k_ = 0;
};

}   // namespace mootation::scalar::shade

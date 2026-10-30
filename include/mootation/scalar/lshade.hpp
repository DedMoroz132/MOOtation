#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// L-SHADE, as R. Tanabe, A.S. Fukunaga, "Improving the Search Performance of
// SHADE Using Linear Population Size Reduction", IEEE CEC 2014, pp.
// 1658-1665, §II, Algorithms 1-2 (source lshade_tanabe2014): SHADE 1.1 with
// linear population size reduction (LPSR).
//   memories M_CR, M_F of H entries, all 0.5; for every x_i: r_i from [1, H];
//   CR_i = 0 when M_CR,r_i is the terminal value ⊥, randn(M_CR,r_i, 0.1)
//   otherwise (1); F_i = randc(M_F,r_i, 0.1) (2), both handled as JADE's;
//   the trial by current-to-pbest/1/bin with the archive, x_pbest one of the
//   N p best (pbest.hpp), a coordinate outside the box to the middle of
//   bound and parent (4);
//   the trial replaces the parent when f(u) <= f(x) (6); when f(u) < f(x) the
//   parent goes to A and CR_i, F_i to S_CR, S_F with weight |f(u) - f(x)|
//   (9); A is cut at random to its size;
//   Algorithm 1: when S is not empty, M_CR,k = ⊥ if it is ⊥ already or
//   max(S_CR) = 0, else mean_WL(S_CR); M_F,k = mean_WL(S_F) (7)-(8); k = k +
//   1, back to 1 past H;
//   LPSR: N_G+1 = round((N_min - N_init) / MAX_NFE * NFE + N_init) (10); the
//   N_G - N_G+1 worst are deleted and |A| follows the new N.
// Defaults are Table II's tuned settings: N_init = round(18 D), |A| =
// round(2.6 N), p = 0.11, H = 6; N_min = 4 (§II-F); MAX_NFE = 10000 D, the
// CEC 2014 budget of §III.
//
// The caller asks for points, evaluates them and tells their values f and
// violations v; the first ask() gives the initial population, uniform in the
// box. Minimization; random numbers are rnd:: of the caller's generator.
//
// Readings (OURS):
//   LSHADE-1. "the top N p members": the ceil(N p) best, at least 2 (SHADE's
//     p_min = 2/N); ranking and the trial's draws as pbest.hpp (PB-1, PB-2).
//     (5) takes v_j where rand[0, 1) <= CR_i; PB-1's "<" differs only on a
//     draw of exactly CR_i.
//   LSHADE-2. The draws for x_i, in Algorithm 2's order: r_i, CR_i (none
//     when M_CR,r_i = ⊥), F_i, then the trial's.
//   LSHADE-3. Algorithm 2, line 25 reads "if N_G < N_G+1"; the text says
//     "whenever N_G+1 < N_G, the worst-ranking individuals are deleted" —
//     taken. NFE counts every point told, the initial population included;
//     N_G+1 is at least N_min. The deleted are the last of the ranking
//     (pbest.hpp, PB-2); the others keep their order. Then |A| =
//     round(2.6 N_G+1) (line 28: "according to new |P|") and A is cut.
//   LSHADE-4. Constraints g <= 0: NSGA-II's constrained-domination at M = 1
//     (evo.hpp) for "<=", "<", the ranking and the deletion; the weight of a
//     success as SHADE-3 (shade.hpp). Not in the paper.
//   LSHADE-5. All N trials of a generation are built from P and A at its
//     start (Algorithm 2 builds them all before any selection).
//   LSHADE-6. done() is true once all of P and A are one point.
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <random>
#include <stdexcept>
#include <vector>

#include "../random.hpp"
#include "evo.hpp"
#include "pbest.hpp"
#include "shade.hpp"

namespace mootation::scalar::lshade {

using evo::Rows;
using evo::Vec;

struct Options {
    double r_ninit = 18.0;              // Table II
    double r_arc = 2.6;                 // Table II
    double p = 0.11;                    // Table II
    std::size_t H = 6;                  // Table II
    std::size_t n_min = 4;              // §II-F
    long long max_nfe = 0;              // 0: 10000 D (§III)
};

class LSHADE {
public:
    LSHADE(Vec lo, Vec hi, const Options& o, std::mt19937& rng)
        : lo_(std::move(lo)), hi_(std::move(hi)), o_(o), rng_(rng), m_cr_(o.H, 0.5), m_f_(o.H, 0.5),
          terminal_(o.H, false)
    {
        if (lo_.empty() || lo_.size() != hi_.size()) throw std::invalid_argument("lshade: lo, hi");
        const double d = static_cast<double>(lo_.size());
        n_init_ = static_cast<std::size_t>(std::llround(o_.r_ninit * d));
        if (o_.max_nfe == 0) o_.max_nfe = 10000 * static_cast<long long>(lo_.size());
        if (o_.n_min < 4 || n_init_ < o_.n_min) throw std::invalid_argument("lshade: 4 <= N_min <= N_init");
        if (o_.H < 1 || !(o_.p > 0.0 && o_.p <= 1.0) || !(o_.r_arc >= 0.0) || o_.max_nfe < 1)
            throw std::invalid_argument("lshade: H, p, r_arc, max_nfe");
        n_ = n_init_;
        a_size_ = arc_size(n_);
    }

    std::size_t np() const { return n_; }
    long long nfe() const { return nfe_; }
    const Vec& memory_cr() const { return m_cr_; }       // an entry whose terminal() is true is ⊥
    const Vec& memory_f() const { return m_f_; }
    bool terminal(std::size_t k) const { return terminal_[k]; }
    const Rows& archive() const { return a_; }

    Rows ask()
    {
        if (x_.empty()) return u_ = evo::uniform(lo_, hi_, n_, rng_);
        const std::vector<std::size_t> order = pbest::ranking(f_, v_);
        const std::size_t top = std::max<std::size_t>(                   // LSHADE-1
            2, static_cast<std::size_t>(std::ceil(o_.p * static_cast<double>(n_))));
        rnd::uniform_int_distribution<std::size_t> pick_h(0, o_.H - 1);
        u_.assign(n_, Vec());
        cr_.assign(n_, 0.0);
        F_.assign(n_, 0.0);
        for (std::size_t i = 0; i < n_; ++i) {                           // LSHADE-2
            const std::size_t r = pick_h(rng_);
            cr_[i] = terminal_[r] ? 0.0 : pbest::draw_CR(rng_, m_cr_[r]);   // (1)
            F_[i] = pbest::draw_F(rng_, m_f_[r]);                         // (2)
            u_[i] = pbest::trial(x_, a_, i, order, std::min(top, n_), F_[i], cr_[i], lo_, hi_, rng_);
        }
        return u_;
    }

    // f and v of the points of the last ask(); v empty: no constraints.
    void tell(const Vec& f, const Vec& v = {})
    {
        if (f.size() != n_ || (!v.empty() && v.size() != n_)) throw std::invalid_argument("lshade: tell sizes");
        nfe_ += static_cast<long long>(n_);
        if (x_.empty()) {
            x_ = u_;
            f_ = f;
            v_ = v.empty() ? Vec(n_, 0.0) : v;
            return;
        }
        Vec s_cr, s_f, gains;
        for (std::size_t i = 0; i < n_; ++i) {
            const double vi = v.empty() ? 0.0 : v[i];
            if (!evo::not_worse(f[i], vi, f_[i], v_[i])) continue;      // (6)
            if (evo::better(f[i], vi, f_[i], v_[i])) {                   // Algorithm 2, lines 18-20
                a_.push_back(x_[i]);
                s_cr.push_back(cr_[i]);
                s_f.push_back(F_[i]);
                gains.push_back(shade::detail::gain(f[i], vi, f_[i], v_[i]));
            }
            x_[i] = u_[i];
            f_[i] = f[i];
            v_[i] = vi;
        }
        pbest::cut(a_, a_size_, rng_);
        if (!s_cr.empty()) {                                             // Algorithm 1
            const Vec w = shade::detail::weights(gains);
            if (terminal_[k_] || *std::max_element(s_cr.begin(), s_cr.end()) == 0.0) terminal_[k_] = true;
            else m_cr_[k_] = shade::detail::mean_wl(w, s_cr);
            m_f_[k_] = shade::detail::mean_wl(w, s_f);
            k_ = (k_ + 1) % o_.H;
        }
        reduce();
    }

    bool done() const { return pbest::collapsed(x_, a_); }               // LSHADE-6

    const Rows& population() const { return x_; }
    const Vec& values() const { return f_; }
    const Vec& violations() const { return v_; }
    std::size_t best() const { return evo::best(f_, v_); }

private:
    std::size_t arc_size(std::size_t n) const
    {
        return static_cast<std::size_t>(std::llround(o_.r_arc * static_cast<double>(n)));
    }

    void reduce()                                                        // (10), LSHADE-3
    {
        const double ni = static_cast<double>(n_init_), nm = static_cast<double>(o_.n_min);
        const double next = std::round((nm - ni) / static_cast<double>(o_.max_nfe) * static_cast<double>(nfe_) + ni);
        const std::size_t n_next = std::max(o_.n_min, static_cast<std::size_t>(std::max(next, 0.0)));
        if (n_next >= n_) return;
        const std::vector<std::size_t> order = pbest::ranking(f_, v_);
        std::vector<bool> keep(n_, true);
        for (std::size_t r = n_next; r < n_; ++r) keep[order[r]] = false;
        Rows x;
        Vec f, v;
        for (std::size_t i = 0; i < n_; ++i)
            if (keep[i]) {
                x.push_back(x_[i]);
                f.push_back(f_[i]);
                v.push_back(v_[i]);
            }
        x_ = std::move(x);
        f_ = std::move(f);
        v_ = std::move(v);
        n_ = n_next;
        a_size_ = arc_size(n_);
        pbest::cut(a_, a_size_, rng_);
    }

    Vec lo_, hi_;
    Options o_;
    std::mt19937& rng_;
    std::size_t n_init_ = 0, n_ = 0, a_size_ = 0, k_ = 0;
    long long nfe_ = 0;
    Rows x_, u_, a_;
    Vec f_, v_, cr_, F_;
    Vec m_cr_, m_f_;
    std::vector<bool> terminal_;
};

}   // namespace mootation::scalar::lshade

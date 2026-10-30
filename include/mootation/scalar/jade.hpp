#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// JADE, as J. Zhang, A.C. Sanderson, "JADE: Adaptive Differential Evolution
// With Optional External Archive", IEEE TEVC 13(5):945-958, 2009, §II and §IV,
// Table I (source jade_zhang2009):
//   CR_i = randn(mu_CR, 0.1) truncated to [0, 1] (8); F_i = randc(mu_F, 0.1),
//   1 where it is >= 1, drawn again where it is <= 0 (10);
//   v_i = x_i + F_i (x_pbest - x_i) + F_i (x_r1 - x~_r2) (7): x_pbest one of
//   the 100p % best of P, x_r1 != x_i from P, x~_r2 != x_r1, x_i from P u A;
//   a mutant coordinate outside the box becomes the middle of the bound and
//   the parent's coordinate (§II); binomial crossover with j_rand (4); the
//   trial replaces the parent when strictly better (5), and then the parent
//   goes to the archive A and F_i, CR_i to S_F, S_CR;
//   A is cut at random to NP; mu_CR = (1 - c) mu_CR + c mean_A(S_CR) (9),
//   mu_F = (1 - c) mu_F + c mean_L(S_F), the Lehmer mean (11)-(12).
// Defaults are §V's: p = 0.05, c = 0.1, the archive on (archive = false is
// the paper's "JADE without archive", (6)).
//
// The caller asks for points, evaluates them and tells their values f and
// violations v; the first ask() gives the initial population, uniform in the
// box (§II). Minimization; random numbers are rnd:: of the caller's generator.
//
// Readings (OURS):
//   JADE-1. "the 100p % best": the ceil(p NP) best of P (at least 1);
//     ranking and draws as pbest.hpp (PB-1, PB-2).
//   JADE-2. All NP trials of a generation are built from P and A as they are
//     at its start; the replaced parents join A at its end (§IV-A: "after
//     each generation"), and then A is cut. Table I adds x_i to A inside the
//     loop, where a later trial of the same generation could pick it; with
//     ask/tell the whole generation is evaluated at once.
//   JADE-3. When no trial succeeds, S_F and S_CR are empty and mu_F, mu_CR
//     keep their values (mean of an empty set is not defined).
//   JADE-4. NP defaults to §V's sizes: 30 for D <= 10, 100 for D <= 30, 400
//     above; NP >= 4 so that x_i, x_r1, x~_r2 can differ, refused otherwise.
//   JADE-5. The draws for x_i, in Table I's order: CR_i, F_i, then the
//     trial's (pbest.hpp, PB-1). Table I tests j = j_rand first and needs no
//     number there; PB-1 draws one for every coordinate.
//   JADE-6. The box: §V calls f8 "the only bound-constrained function" of its
//     set and gives the others an initial range; here the box is both, and
//     the middle rule of §II holds for every problem.
//   JADE-7. Constraints g <= 0: NSGA-II's constrained-domination at M = 1
//     (evo.hpp), for "better" in (5) and in the ranking of JADE-1; not in the
//     paper.
//   JADE-8. done() is true once all of P and A are one point.
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <random>
#include <stdexcept>
#include <vector>

#include "evo.hpp"
#include "pbest.hpp"

namespace mootation::scalar::jade {

using evo::Rows;
using evo::Vec;

struct Options {
    std::size_t np = 0;                 // 0: JADE-4
    double p = 0.05;                    // §V
    double c = 0.1;                     // §V
    bool archive = true;
};

class JADE {
public:
    JADE(Vec lo, Vec hi, const Options& o, std::mt19937& rng)
        : lo_(std::move(lo)), hi_(std::move(hi)), o_(o), rng_(rng)
    {
        if (lo_.empty() || lo_.size() != hi_.size()) throw std::invalid_argument("jade: lo, hi");
        if (o_.np == 0) o_.np = lo_.size() <= 10 ? 30 : lo_.size() <= 30 ? 100 : 400;
        if (o_.np < 4) throw std::invalid_argument("jade: NP >= 4 (JADE-4)");
        if (!(o_.p > 0.0 && o_.p <= 1.0)) throw std::invalid_argument("jade: p in (0, 1]");
        if (!(o_.c >= 0.0 && o_.c <= 1.0)) throw std::invalid_argument("jade: c in [0, 1]");
    }

    std::size_t np() const { return o_.np; }
    double mu_cr() const { return mu_cr_; }
    double mu_f() const { return mu_f_; }
    const Rows& archive() const { return a_; }

    Rows ask()
    {
        if (x_.empty()) return u_ = evo::uniform(lo_, hi_, o_.np, rng_);
        const std::size_t n = o_.np;
        const std::vector<std::size_t> order = pbest::ranking(f_, v_);
        const std::size_t top =                                          // JADE-1
            std::max<std::size_t>(1, static_cast<std::size_t>(std::ceil(o_.p * static_cast<double>(n))));
        u_.assign(n, Vec());
        cr_.assign(n, 0.0);
        F_.assign(n, 0.0);
        for (std::size_t i = 0; i < n; ++i) {                            // JADE-5
            cr_[i] = pbest::draw_CR(rng_, mu_cr_);                       // (8)
            F_[i] = pbest::draw_F(rng_, mu_f_);                          // (10)
            u_[i] = pbest::trial(x_, a_, i, order, top, F_[i], cr_[i], lo_, hi_, rng_);   // (7), (4), §II
        }
        return u_;
    }

    // f and v of the points of the last ask(); v empty: no constraints.
    void tell(const Vec& f, const Vec& v = {})
    {
        const std::size_t n = o_.np;
        if (f.size() != n || (!v.empty() && v.size() != n)) throw std::invalid_argument("jade: tell sizes");
        if (x_.empty()) {
            x_ = u_;
            f_ = f;
            v_ = v.empty() ? Vec(n, 0.0) : v;
            return;
        }
        double s_cr = 0.0, s_f = 0.0, s_f2 = 0.0;
        std::size_t wins = 0;
        for (std::size_t i = 0; i < n; ++i) {
            const double vi = v.empty() ? 0.0 : v[i];
            if (!evo::better(f[i], vi, f_[i], v_[i])) continue;           // (5), JADE-7
            if (o_.archive) a_.push_back(x_[i]);                          // JADE-2
            x_[i] = u_[i];
            f_[i] = f[i];
            v_[i] = vi;
            s_cr += cr_[i];
            s_f += F_[i];
            s_f2 += F_[i] * F_[i];
            ++wins;
        }
        pbest::cut(a_, n, rng_);                                         // A to NP at random
        if (wins > 0) {                                                  // JADE-3
            mu_cr_ = (1.0 - o_.c) * mu_cr_ + o_.c * s_cr / static_cast<double>(wins);   // (9)
            mu_f_ = (1.0 - o_.c) * mu_f_ + o_.c * s_f2 / s_f;                            // (11)-(12)
        }
    }

    bool done() const { return pbest::collapsed(x_, a_); }               // JADE-8

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
    double mu_cr_ = 0.5, mu_f_ = 0.5;                                    // Table I, line 02
};

}   // namespace mootation::scalar::jade

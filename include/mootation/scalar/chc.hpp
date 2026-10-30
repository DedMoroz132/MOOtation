#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// CHC with BLX-alpha, the `chc` preset of the GA family, as L.J. Eshelman,
// "The CHC Adaptive Search Algorithm", FOGA 1991, pp. 265-283, §2, §4, §5
// (source chc_eshelman1991), with the real-valued crossover of L.J. Eshelman,
// J.D. Schaffer, "Real-Coded Genetic Algorithms and Interval-Schemata", FOGA
// 2, 1993, §4 (source blx-alpha_eshelman1993):
//   every member of P is paired at random without replacement; a pair mates
//   only when half its Hamming distance exceeds the difference threshold,
//   L/4 at the start, and gives two children by BLX-0.5 (ops::
//   blx_alpha_crossover, p_c = 1); parents and children are merged and the
//   best M stay (cross-generational elitist selection); when no child is
//   accepted the threshold drops by one; no mutation — when the threshold
//   has dropped below zero, the population is diverged: the best stays and
//   M - 1 copies of it have 35 % of their bits flipped, chosen at random
//   without replacement, and the search goes on.
// Defaults: M = 50, divergence rate 0.35 (1991, §6; 1993, §4), alpha = 0.5.
//
// The caller asks for points, evaluates them and tells their values f and
// violations v; the first ask() gives the initial population, uniform in the
// box. Minimization; random numbers are rnd:: of the caller's generator.
//
// Readings (OURS):
//   CHC-1. The bits. Both papers run CHC on bit strings, BLX-0.5 on the
//     decoded parameters; here the points are real and each variable has a
//     b-bit code, round((x - lo) / (hi - lo) (2^b - 1)), for the Hamming
//     distance and the divergence; L = n b. b = 10 by default (most of the
//     1993 Table 2 functions).
//   CHC-2. Divergence: a flipped bit changes its variable to the decoded
//     value lo + code / (2^b - 1) (hi - lo) of the flipped code; a variable
//     without a flipped bit keeps the best point's value exactly.
//   CHC-3. "the threshold has dropped to zero and there have been several
//     generations without any new offspring accepted": the population is
//     diverged when a generation without acceptance would take the threshold
//     below zero; after it the threshold starts again at L/4.
//   CHC-4. A generation where no pair may mate makes no point: the threshold
//     drops at once and the pairing is drawn again, inside ask(), until a
//     pair mates or the population is diverged — ask() always returns points.
//   CHC-5. A child is accepted when it is among the M best of parents and
//     children (stably by the comparison of evo.hpp, parents first); M odd:
//     the last of the shuffled pairing does not mate.
//   CHC-6. BLX-0.5 children outside the box are repaired by the operator,
//     ops::BoundRepair `reflect` by default (real_crossover.hpp).
//   CHC-7. Constraints g <= 0: NSGA-II's constrained-domination at M = 1
//     (evo.hpp) in the selection; not in the papers.
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include "../operators/bound_repair.hpp"
#include "../operators/real_crossover.hpp"
#include "../random.hpp"
#include "evo.hpp"

namespace mootation::scalar::chc {

using evo::Rows;
using evo::Vec;

struct Options {
    std::size_t m = 50;                                    // 1991 §6, 1993 §4
    double divergence = 0.35;                              // 1991 §5-6
    double alpha = 0.5;                                    // 1993 §4
    unsigned bits = 10;                                    // CHC-1
    ops::BoundRepair repair = ops::BoundRepair::Reflect;   // CHC-6
};

class CHC {
public:
    CHC(Vec lo, Vec hi, const Options& o, std::mt19937& rng) : lo_(std::move(lo)), hi_(std::move(hi)), o_(o), rng_(rng)
    {
        if (lo_.empty() || lo_.size() != hi_.size()) throw std::invalid_argument("chc: lo, hi");
        if (o_.m < 2 || o_.bits < 1 || o_.bits > 31 || !(o_.divergence > 0.0 && o_.divergence <= 1.0))
            throw std::invalid_argument("chc: M >= 2, 1 <= bits <= 31, divergence in (0, 1]");
        ops::require_repair(o_.repair, "chc", true, false);
        L_ = static_cast<long long>(lo_.size()) * o_.bits;
        threshold_ = L_ / 4;
    }

    long long threshold() const { return threshold_; }
    long long restarts() const { return restarts_; }

    Rows ask()
    {
        if (x_.empty()) return u_ = evo::uniform(lo_, hi_, o_.m, rng_);
        if (pending_) return diverge();
        diverged_ = false;
        const auto b = ops_bounds();
        while (true) {                                                   // CHC-4
            std::vector<std::size_t> order(o_.m);
            std::iota(order.begin(), order.end(), std::size_t{0});
            rnd::shuffle(order.begin(), order.end(), rng_);
            u_.clear();
            Vec c1, c2;
            for (std::size_t k = 0; k + 1 < o_.m; k += 2) {
                const Vec& p1 = x_[order[k]];
                const Vec& p2 = x_[order[k + 1]];
                if (hamming(p1, p2) / 2 <= threshold_) continue;        // incest prevention
                ops::blx_alpha_crossover(p1, p2, c1, c2, b, o_.alpha, 1.0, o_.repair, rng_);
                u_.push_back(c1);
                u_.push_back(c2);
            }
            if (!u_.empty()) return u_;
            if (--threshold_ < 0) return diverge();                     // CHC-3
        }
    }

    void tell(const Vec& f, const Vec& v = {})
    {
        if (f.size() != u_.size() || (!v.empty() && v.size() != u_.size())) throw std::invalid_argument("chc: tell sizes");
        const Vec vv = v.empty() ? Vec(f.size(), 0.0) : v;
        if (x_.empty() || diverged_) {
            if (x_.empty()) {
                x_ = u_;
                f_ = f;
                v_ = vv;
            } else {
                for (std::size_t i = 0; i < u_.size(); ++i) {           // slot 0 keeps the best
                    x_[i + 1] = u_[i];
                    f_[i + 1] = f[i];
                    v_[i + 1] = vv[i];
                }
            }
            diverged_ = false;
            return;
        }
        Rows x = x_;                                                     // CHC-5
        Vec ff = f_, vs = v_;
        x.insert(x.end(), u_.begin(), u_.end());
        ff.insert(ff.end(), f.begin(), f.end());
        vs.insert(vs.end(), vv.begin(), vv.end());
        std::vector<std::size_t> order(x.size());
        std::iota(order.begin(), order.end(), std::size_t{0});
        std::stable_sort(order.begin(), order.end(),
                         [&](std::size_t a, std::size_t b) { return evo::better(ff[a], vs[a], ff[b], vs[b]); });
        bool accepted = false;
        for (std::size_t i = 0; i < o_.m; ++i) {
            accepted = accepted || order[i] >= o_.m;
            x_[i] = x[order[i]];
            f_[i] = ff[order[i]];
            v_[i] = vs[order[i]];
        }
        if (!accepted && --threshold_ < 0) pending_ = true;
    }

    const Rows& population() const { return x_; }
    const Vec& values() const { return f_; }
    const Vec& violations() const { return v_; }
    std::size_t best() const { return evo::best(f_, v_); }

    // a pending divergence happens at the next ask()
    bool diverging() const { return pending_; }

private:
    std::vector<std::pair<std::optional<double>, std::optional<double>>> ops_bounds() const
    {
        std::vector<std::pair<std::optional<double>, std::optional<double>>> b(lo_.size());
        for (std::size_t j = 0; j < lo_.size(); ++j) b[j] = {lo_[j], hi_[j]};
        return b;
    }

    std::uint32_t code(double x, std::size_t j) const                    // CHC-1
    {
        const double top = static_cast<double>((1u << o_.bits) - 1u);
        const double t = hi_[j] > lo_[j] ? (x - lo_[j]) / (hi_[j] - lo_[j]) : 0.0;
        return static_cast<std::uint32_t>(std::llround(std::clamp(t, 0.0, 1.0) * top));
    }

    double decode(std::uint32_t c, std::size_t j) const                  // CHC-2
    {
        const double top = static_cast<double>((1u << o_.bits) - 1u);
        return lo_[j] + static_cast<double>(c) / top * (hi_[j] - lo_[j]);
    }

    long long hamming(const Vec& a, const Vec& b) const
    {
        long long h = 0;
        for (std::size_t j = 0; j < a.size(); ++j) {
            std::uint32_t d = code(a[j], j) ^ code(b[j], j);
            while (d) {
                h += d & 1u;
                d >>= 1;
            }
        }
        return h;
    }

    Rows diverge()
    {
        const std::size_t b = evo::best(f_, v_);
        std::swap(x_[0], x_[b]);
        std::swap(f_[0], f_[b]);
        std::swap(v_[0], v_[b]);
        const std::size_t flips =
            static_cast<std::size_t>(std::llround(o_.divergence * static_cast<double>(L_)));
        std::vector<std::size_t> bits(static_cast<std::size_t>(L_));
        u_.clear();
        for (std::size_t k = 1; k < o_.m; ++k) {
            std::iota(bits.begin(), bits.end(), std::size_t{0});
            std::vector<std::uint32_t> c(lo_.size());
            for (std::size_t j = 0; j < lo_.size(); ++j) c[j] = code(x_[0][j], j);
            std::vector<bool> touched(lo_.size(), false);
            for (std::size_t t = 0; t < flips; ++t) {                    // without replacement
                rnd::uniform_int_distribution<std::size_t> any(t, bits.size() - 1);
                std::swap(bits[t], bits[any(rng_)]);
                const std::size_t j = bits[t] / o_.bits, q = bits[t] % o_.bits;
                c[j] ^= 1u << q;
                touched[j] = true;
            }
            Vec y = x_[0];
            for (std::size_t j = 0; j < lo_.size(); ++j)
                if (touched[j]) y[j] = decode(c[j], j);
            u_.push_back(std::move(y));
        }
        threshold_ = L_ / 4;
        pending_ = false;
        diverged_ = true;
        ++restarts_;
        return u_;
    }

    Vec lo_, hi_;
    Options o_;
    std::mt19937& rng_;
    Rows x_, u_;
    Vec f_, v_;
    long long L_ = 0, threshold_ = 0, restarts_ = 0;
    bool diverged_ = false, pending_ = false;
};

}   // namespace mootation::scalar::chc

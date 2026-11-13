#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// Real-coded genetic algorithms on the library's operators (operators/, read
// only): `GA`, a (mu + lambda) GA with tournament selection, and `G3`, the
// generalized generation gap model with PCX.
//
// GA. mu parents; lambda offspring a generation, made in pairs: two parents
// by tournaments, the pair crossed, the children mutated; then the mu best of
// parents and offspring stay. Presets:
//   ga_sbx — K. Deb, D. Deb, "Analysing mutation schemes for real-parameter
//     genetic algorithms", IJAISC 4(1):1-28, 2014, §5 (source polynomial-
//     mutation_deb2014): N = 150, binary tournament, SBX (Deb & Agrawal 1995,
//     source sbx_deb1995) with p_c = 0.9, eta_c = 2, polynomial mutation with
//     p_m = 1/n, eta_m = 20 — ops::sbx and ops::polynomial_mutation;
//   ssga — a steady-state GA: (mu + 1), two parents by tournament, one
//     child, the worst of the mu + 1 goes (no paper: this header is its
//     description).
// G3. K. Deb, A. Anand, D. Joshi, "A Computationally Efficient Evolutionary
// Algorithm for Real-Parameter Optimization", Evol. Comput. 10(4):371-395,
// 2002, §3.2 (source pcx_deb2002): (1) the best of P and mu - 1 others at
// random are the parents; (2) lambda offspring by PCX (ops::pcx_child); (3)
// two members of P at random; (4) the best two of those two and the lambda
// offspring take their places. `modified` is §3.2.2's: one member in (3),
// replaced by the best of it and the offspring. N = 100, mu = 3, lambda = 2,
// sigma_zeta = sigma_eta = 0.1 (§3.2.1).
//
// The caller asks for points, evaluates them and tells their values f and
// violations v; the first ask() gives the initial population, uniform in the
// box. Minimization; random numbers are rnd:: of the caller's generator.
//
// Readings (OURS):
//   GA-1. Tournament: k members drawn with replacement, the best by the
//     comparison of evo.hpp, a tie to the later draw; k = 1 is a uniform draw.
//   GA-2. (mu + lambda): parents first, then offspring in the order made,
//     stably sorted by the comparison; the first mu stay, in that order.
//   GA-3. lambda odd: the second child of the last pair is dropped after its
//     mutation (nsga2 does the same). lambda = 1 with `coin`: one child of the
//     pair kept by a fair coin drawn between crossover and mutation, the
//     other neither mutated nor kept (sms_emoa's order); the child takes the
//     place of the worst of the mu + 1 — the first of equals, so a child equal
//     to the worst member stays — or is dropped when it alone is the worst.
//   GA-4. Bounds: SBX and the polynomial mutation stay in the box by
//     construction; BLX-alpha and PCX children are repaired coordinate by
//     coordinate by ops::repair_value (reflect by default; `native` leaves
//     them — Deb et al. 2002 start in [-10, -5]^n and have no box).
//   GA-5. Constraints g <= 0: NSGA-II's constrained-domination at M = 1
//     (evo.hpp) in tournaments, truncation and G3's choices; not in the
//     papers.
//   GA-6. sym (task 19, symmetry.hpp SYM-7; off by default): the second
//     parent aligned to the first (align), or both canonical (canon), after
//     the tournaments and before the crossover; no draw changes, and sym =
//     off is GA as before.
//   GA-7. block_uniform (task 19, §19.2.3; off by default): with probability
//     o.block_uniform a pair, after its crossover (which made the free
//     coordinates), every block of sym's groups whole from one parent: a
//     fair coin a block, group by group, heads the first child's from the
//     first parent and the second's from the second, tails crosswise. The
//     coins follow the crossover's draws; o.block_uniform = 0 draws nothing.
//   GA-8. block_mutation (§19.2.3; off by default): with probability
//     o.block_mutation a child, after its mutation, one block of all the
//     groups' blocks drawn uniformly and each of its coordinates mutated by
//     the polynomial mutation (eta_m, p_m = 1) within its bounds — the
//     operator on the block alone (its per-variable eta, the MO runs' integer
//     variables, is not set at M = 1). o.block_mutation = 0 draws nothing.
//   GA-9. The user's genome (task 24, §24.1.2; off by default: the builtin
//     operators): ga_crossover = user calls Problem::crossover(p1, p2, c1,
//     c2, rng) in SBX's place, ga_mutation = user Problem::mutate(c, rng) in
//     the polynomial mutation's (each child, GA-3's order kept), ga_init =
//     user Problem::sample(x, rng) for each member of the first population
//     (the seeds of POP-7 then over its first rows); rng is the run's, drawn
//     through rnd:: only. The block operators (GA-7, GA-8) are the builtin's
//     and are refused beside the user's.
//   GA-10. Off by default. A row of the user's operators is repaired
//     (ga_repair: builtin clips each coordinate to the box — SBX and the
//     polynomial mutation stay in it, so the builtin run is unchanged; user
//     calls Problem::repair; off leaves it) and then must have n finite values
//     in the box, or the run is refused with the row's number and generation;
//     so is an exception of a hook (sample, crossover, mutate, repair,
//     decode), with its message.
//   GA-11. Off by default. With Problem::decode every row is decoded before it
//     is evaluated, its canonical genome written back (Lamarckian; checked as
//     GA-10); a row that does not decode (valid = false) is not evaluated, costs
//     nothing, and is told +inf in f and v (the worst), counted in
//     Result::decode_invalid.
//   G3-1. PCX's index parent x_p is the best of the parents (best_centre, on
//     by default: §3.2.1, "For a faster convergence, the best parent is
//     always used"), not the operator's own pick "with equal probability"
//     (§2.2), which best_centre = false gives (ops::pcx_child's default; the
//     owner's word, 31.10).
//   G3-2. Step (1)'s mu - 1 others differ from the best and from each other;
//     step (3)'s two differ. Each of the lambda offspring is one pcx_child
//     call on the same parents.
//   G3-3. Step (4) keeps the two best of the family in its order (the two
//     chosen, then the offspring), stably; the better goes to the first
//     chosen place.
// With mu = N, lambda = N, k = 2, SBX (p_c = 0.9, eta_c = 20) and PM (eta_m =
// 20, p_m = 1/n) a GA draws its random numbers as nsga2 does at n_objs = 1;
// with lambda = 1, coin, k = 1, SBX (p_c = 1, eta_c = 15) as sms_emoa. Ties
// in f are where they may part: nsga2 orders them by crowding distance;
// sms_emoa's contributions of tied worst points are all 0 at n_objs = 1 and
// it removes the first of them, as here.
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <numeric>
#include <string>
#include <optional>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include "../operators/bound_repair.hpp"
#include "../operators/multi_parent.hpp"
#include "../operators/poly_mutation.hpp"
#include "../operators/real_crossover.hpp"
#include "../operators/sbx.hpp"
#include "../random.hpp"
#include "evo.hpp"
#include "genome.hpp"
#include "symmetry.hpp"

namespace mootation::scalar::ga {

using evo::Rows;
using evo::Vec;
using Bounds = std::vector<std::pair<std::optional<double>, std::optional<double>>>;

enum class Crossover { SBX, BLX };

struct Options {
    std::size_t mu = 150;                                  // ga_sbx
    std::size_t lambda = 0;                                // 0: mu
    std::size_t tournament = 2;
    Crossover crossover = Crossover::SBX;
    double pc = 0.9, eta_c = 2.0, alpha = 0.5;
    bool mutation = true;
    double eta_m = 20.0, pm = 0.0;                         // pm 0: 1/n
    bool coin = false;                                     // GA-3
    ops::BoundRepair repair = ops::BoundRepair::Reflect;   // GA-4 (BLX)
    symmetry::Sym sym;                                     // GA-6
    double block_uniform = 0.0, block_mutation = 0.0;      // GA-7, GA-8 (sym's groups)
    Rows seeds;                         // the first population's first rows (task 24, §24.4.1; POP-7)
    // the user's genome (task 24, §24.1.2; GA-9 ... GA-11): empty, the builtin
    std::function<void(const Vec&, const Vec&, Vec&, Vec&, std::mt19937&)> user_crossover;
    std::function<void(Vec&, std::mt19937&)> user_mutate, user_sample;
    std::function<void(Vec&)> user_repair;
    bool clip = true;                   // repair = builtin after a user operator: clipped to the box
    std::function<Decoded(const Vec&)> decode;
};

inline Options ga_sbx() { return Options{}; }

inline Options ssga()
{
    Options o;
    o.mu = 100;
    o.lambda = 1;
    o.coin = true;
    return o;
}

namespace detail {

inline Bounds bounds(const Vec& lo, const Vec& hi)
{
    Bounds b(lo.size());
    for (std::size_t j = 0; j < lo.size(); ++j) b[j] = {lo[j], hi[j]};
    return b;
}

}   // namespace detail

// The children c1, c2 of the parents p1, p2: GA-6, the crossover, GA-7.
inline void cross(const Vec& p1, const Vec& p2, Vec& c1, Vec& c2, const Bounds& b, const Options& o,
                  std::mt19937& rng)
{
    Vec k1, k2;                                                          // GA-6's copies
    const Vec& a = o.sym.mode == symmetry::Mode::canon ? o.sym.view(p1, p1, k1) : p1;
    const Vec& c = o.sym.view(p2, p1, k2);
    if (o.crossover == Crossover::SBX) ops::sbx(a, c, c1, c2, b, o.eta_c, o.pc, rng);
    else ops::blx_alpha_crossover(a, c, c1, c2, b, o.alpha, o.pc, o.repair, rng);
    if (o.block_uniform == 0.0) return;
    rnd::uniform_real_distribution<double> U01(0.0, 1.0);
    if (!(U01(rng) < o.block_uniform)) return;
    rnd::uniform_int_distribution<int> coin(0, 1);
    for (const symmetry::Blocks& g : o.sym.groups)                       // GA-7
        for (std::size_t k = 0; k < g.count; ++k) {
            const bool heads = coin(rng) == 1;
            for (std::size_t j = g.offset + k * g.size; j < g.offset + (k + 1) * g.size; ++j) {
                c1[j] = heads ? a[j] : c[j];
                c2[j] = heads ? c[j] : a[j];
            }
        }
}

// One block of the child c mutated, with probability o.block_mutation (GA-8).
inline void block_mutation(Vec& c, const Bounds& b, const Options& o, std::mt19937& rng)
{
    if (o.block_mutation == 0.0) return;
    rnd::uniform_real_distribution<double> U01(0.0, 1.0);
    if (!(U01(rng) < o.block_mutation)) return;
    std::size_t total = 0;
    for (const symmetry::Blocks& g : o.sym.groups) total += g.count;
    rnd::uniform_int_distribution<std::size_t> pick(0, total - 1);
    std::size_t k = pick(rng);
    for (const symmetry::Blocks& g : o.sym.groups) {
        if (k >= g.count) {
            k -= g.count;
            continue;
        }
        const std::size_t first = g.offset + k * g.size;
        Vec y(c.begin() + static_cast<std::ptrdiff_t>(first), c.begin() + static_cast<std::ptrdiff_t>(first + g.size));
        const Bounds bb(b.begin() + static_cast<std::ptrdiff_t>(first), b.begin() + static_cast<std::ptrdiff_t>(first + g.size));
        ops::polynomial_mutation(y, bb, o.eta_m, 1.0, rng);
        std::copy(y.begin(), y.end(), c.begin() + static_cast<std::ptrdiff_t>(first));
        return;
    }
}

class GA {
public:
    GA(Vec lo, Vec hi, const Options& o, std::mt19937& rng)
        : lo_(std::move(lo)), hi_(std::move(hi)), o_(o), rng_(rng), b_(detail::bounds(lo_, hi_))
    {
        if (lo_.empty() || lo_.size() != hi_.size()) throw std::invalid_argument("ga: lo, hi");
        if (o_.lambda == 0) o_.lambda = o_.mu;
        if (o_.pm == 0.0) o_.pm = 1.0 / static_cast<double>(lo_.size());
        if (o_.mu < 2 || o_.tournament < 1) throw std::invalid_argument("ga: mu >= 2, tournament >= 1");
        if (o_.coin && o_.lambda != 1) throw std::invalid_argument("ga: coin needs lambda = 1 (GA-3)");
        if (o_.crossover == Crossover::BLX) ops::require_repair(o_.repair, "ga", true, false);
        if (!(o_.block_uniform >= 0.0 && o_.block_uniform <= 1.0 && o_.block_mutation >= 0.0 && o_.block_mutation <= 1.0))
            throw std::invalid_argument("ga: block_uniform and block_mutation in [0, 1]");
        if ((o_.block_uniform > 0.0 || o_.block_mutation > 0.0) && o_.sym.groups.empty())
            throw std::invalid_argument("ga: the block operators need the problem's blocks (GA-7, GA-8)");
    }

    std::size_t mu() const { return o_.mu; }

    Rows ask()
    {
        ++gen_;
        if (x_.empty()) {
            if (o_.user_sample) {                                        // GA-9
                u_.assign(o_.mu, Vec(lo_.size(), 0.0));
                for (std::size_t k = 0; k < u_.size(); ++k) guarded("sample", k, [&] { o_.user_sample(u_[k], rng_); });
                for (std::size_t i = 0; i < o_.seeds.size() && i < o_.mu; ++i) u_[i] = o_.seeds[i];
            } else {
                u_ = evo::uniform(lo_, hi_, o_.mu, rng_, o_.seeds);
            }
            prepare();
            return u_;
        }
        u_.clear();
        Vec c1, c2;
        rnd::uniform_int_distribution<int> coin(0, 1);
        while (u_.size() < o_.lambda) {
            const Vec& p1 = x_[tournament()];
            const Vec& p2 = x_[tournament()];
            if (o_.user_crossover)                                       // GA-9
                guarded("crossover", u_.size(), [&] { o_.user_crossover(p1, p2, c1, c2, rng_); });
            else cross(p1, p2, c1, c2, b_, o_, rng_);
            const int keep = o_.coin ? coin(rng_) : -1;                  // GA-3
            if (keep != 1) mutate(c1, u_.size());
            if (keep != 0) mutate(c2, u_.size() + (keep != 1 ? 1 : 0));
            if (keep != 1) u_.push_back(c1);
            if (keep != 0 && u_.size() < o_.lambda) u_.push_back(c2);
        }
        prepare();
        return u_;
    }

    // GA-11: the rows of the last ask() that do not decode, not to be evaluated
    bool decoding() const { return static_cast<bool>(o_.decode); }
    const std::vector<char>& invalid() const { return invalid_; }

    void tell(const Vec& f, const Vec& v = {})
    {
        if (f.size() != u_.size() || (!v.empty() && v.size() != u_.size())) throw std::invalid_argument("ga: tell sizes");
        const Vec vv = v.empty() ? Vec(f.size(), 0.0) : v;
        if (x_.empty()) {
            x_ = u_;
            f_ = f;
            v_ = vv;
            return;
        }
        if (o_.coin) {                                                   // GA-3: the worst of mu + 1 goes
            auto fa = [&](std::size_t i) { return i < o_.mu ? f_[i] : f[0]; };
            auto va = [&](std::size_t i) { return i < o_.mu ? v_[i] : vv[0]; };
            std::size_t w = 0;                                           // the first of the worst
            for (std::size_t i = 1; i <= o_.mu; ++i)
                if (evo::better(fa(w), va(w), fa(i), va(i))) w = i;
            if (w < o_.mu) {
                x_[w] = u_[0];
                f_[w] = f[0];
                v_[w] = vv[0];
            }
            return;
        }
        Rows x = x_;                                                     // GA-2
        Vec ff = f_, vs = v_;
        x.insert(x.end(), u_.begin(), u_.end());
        ff.insert(ff.end(), f.begin(), f.end());
        vs.insert(vs.end(), vv.begin(), vv.end());
        std::vector<std::size_t> order(x.size());
        std::iota(order.begin(), order.end(), std::size_t{0});
        std::stable_sort(order.begin(), order.end(),
                         [&](std::size_t a, std::size_t b) { return evo::better(ff[a], vs[a], ff[b], vs[b]); });
        for (std::size_t i = 0; i < o_.mu; ++i) {
            x_[i] = x[order[i]];
            f_[i] = ff[order[i]];
            v_[i] = vs[order[i]];
        }
    }

    bool done() const
    {
        if (x_.empty()) return false;
        for (const Vec& x : x_) if (x != x_[0]) return false;
        return !o_.mutation;
    }

    const Rows& population() const { return x_; }
    const Vec& values() const { return f_; }
    const Vec& violations() const { return v_; }
    std::size_t best() const { return evo::best(f_, v_); }
    // a migrant (pap, task 19): the worst member's place where it is better
    void migrate(const Vec& x, double f, double v) { evo::replace_worst(x_, f_, v_, x, f, v); }

private:
    void mutate(Vec& c, std::size_t k)
    {
        if (o_.user_mutate) {                                            // GA-9
            guarded("mutate", k, [&] { o_.user_mutate(c, rng_); });
            return;
        }
        if (o_.mutation) ops::polynomial_mutation(c, b_, o_.eta_m, o_.pm, rng_);
        block_mutation(c, b_, o_, rng_);                                 // GA-8
    }

    bool user() const { return o_.user_crossover || o_.user_mutate || o_.user_sample; }

    // GA-10: an exception of the user's hook, refused with the row's number
    // and generation (the budget's end passes through)
    template <class Fn>
    void guarded(const char* hook, std::size_t k, Fn&& fn) const
    {
        try {
            fn();
        } catch (const std::exception& e) {
            throw std::invalid_argument("ga: the user's " + std::string(hook) + ", individual " + std::to_string(k) +
                                        " of generation " + std::to_string(gen_ - 1) + ": " + e.what());
        }
    }

    // GA-10: a row of the user's operators or decoder of the right length,
    // finite, in the box; else the run is refused with the row's number
    void checked(const Vec& x, std::size_t k, const char* what) const
    {
        const std::string where = "ga: " + std::string(what) + ", individual " + std::to_string(k) +
                                  " of generation " + std::to_string(gen_ - 1);
        if (x.size() != lo_.size())
            throw std::invalid_argument(where + ": " + std::to_string(x.size()) + " values for " +
                                        std::to_string(lo_.size()) + " variables");
        for (std::size_t j = 0; j < x.size(); ++j) {
            if (!std::isfinite(x[j])) throw std::invalid_argument(where + ": a value not finite (variable " +
                                                                  std::to_string(j) + ")");
            if (x[j] < lo_[j] || x[j] > hi_[j])
                throw std::invalid_argument(where + ": outside the box (variable " + std::to_string(j) + ")");
        }
    }

    // GA-10, GA-11: the user's rows repaired and checked, every row decoded
    // and its canonical genome written back
    void prepare()
    {
        invalid_.assign(u_.size(), 0);
        if (!user() && !o_.decode) return;
        for (std::size_t k = 0; k < u_.size(); ++k) {
            Vec& x = u_[k];
            if (user()) {
                if (o_.user_repair) {
                    guarded("repair", k, [&] { o_.user_repair(x); });
                } else if (o_.clip && x.size() == lo_.size()) {
                    for (std::size_t j = 0; j < x.size(); ++j)
                        if (std::isfinite(x[j])) x[j] = std::min(hi_[j], std::max(lo_[j], x[j]));
                }
                checked(x, k, "the user's operators");
            }
            if (!o_.decode) continue;
            Decoded d;
            guarded("decode", k, [&] { d = o_.decode(x); });
            if (!d.valid) {
                invalid_[k] = 1;
                continue;
            }
            checked(d.x_canon, k, "the decoder's canonical genome");
            x = std::move(d.x_canon);
        }
    }

    std::size_t tournament()                                             // GA-1
    {
        rnd::uniform_int_distribution<std::size_t> pick(0, o_.mu - 1);
        std::size_t w = pick(rng_);
        for (std::size_t t = 1; t < o_.tournament; ++t) {
            const std::size_t c = pick(rng_);
            if (!evo::better(f_[w], v_[w], f_[c], v_[c])) w = c;
        }
        return w;
    }

    Vec lo_, hi_;
    Options o_;
    std::mt19937& rng_;
    Bounds b_;
    Rows x_, u_;
    Vec f_, v_;
    std::vector<char> invalid_;
    long long gen_ = 0;
};

struct G3Options {
    std::size_t n = 100, mu = 3, lambda = 2;               // §3.2.1
    double sigma_zeta = 0.1, sigma_eta = 0.1;              // §3.2.1
    bool modified = false;                                 // §3.2.2
    bool best_centre = true;                               // §3.2.1, G3-1
    ops::BoundRepair repair = ops::BoundRepair::Reflect;   // GA-4
    Rows seeds;                         // the first population's first rows (task 24, §24.4.1; POP-7)
    std::vector<char> periodic;         // periodic variables (task 24, §24.4.2; periodic.hpp)
};

class G3 {
public:
    G3(Vec lo, Vec hi, const G3Options& o, std::mt19937& rng)
        : lo_(std::move(lo)), hi_(std::move(hi)), o_(o), rng_(rng)
    {
        if (lo_.empty() || lo_.size() != hi_.size()) throw std::invalid_argument("g3: lo, hi");
        if (o_.mu < 2 || o_.lambda < 1 || o_.n < std::max<std::size_t>(o_.mu, 2))
            throw std::invalid_argument("g3: mu >= 2, lambda >= 1, N >= mu");
        if (o_.repair == ops::BoundRepair::Resample) throw std::invalid_argument("g3: resample (GA-4)");
    }

    Rows ask()
    {
        if (x_.empty()) return u_ = evo::uniform(lo_, hi_, o_.n, rng_, o_.seeds);
        rnd::uniform_int_distribution<std::size_t> pick(0, o_.n - 1);
        std::vector<std::size_t> idx{evo::best(f_, v_)};                // (1), G3-2
        while (idx.size() < o_.mu) {
            const std::size_t c = pick(rng_);
            if (std::find(idx.begin(), idx.end(), c) == idx.end()) idx.push_back(c);
        }
        ops::ParentSet P;
        for (std::size_t i : idx) P.push_back(x_[i]);
        u_.clear();
        for (std::size_t k = 0; k < o_.lambda; ++k) {                    // (2), G3-1
            // P[0] is the best (step 1)
            auto [y, p] = ops::pcx_child(P, o_.sigma_zeta, o_.sigma_eta, rng_, o_.best_centre ? 0 : -1);
            {
                ops::RepairTally tally;                                  // the child (task 20, §20.2.2)
                for (std::size_t j = 0; j < y.size(); ++j)
                    if ((y[j] < lo_[j] || y[j] > hi_[j]) && !periodic::is(o_.periodic, j)) tally.out();
            }
            for (std::size_t j = 0; j < y.size(); ++j)                   // periodic.hpp PER-2
                if (y[j] < lo_[j] || y[j] > hi_[j]) evo::wrapped(y[j], lo_, hi_, o_.periodic, j);
            if (o_.repair != ops::BoundRepair::Native)
                for (std::size_t j = 0; j < y.size(); ++j)
                    if (y[j] < lo_[j] || y[j] > hi_[j]) y[j] = ops::repair_value(y[j], lo_[j], hi_[j], P[p][j], o_.repair, rng_);
            u_.push_back(std::move(y));
        }
        chosen_.clear();                                                 // (3)
        const std::size_t m = o_.modified ? 1 : 2;
        while (chosen_.size() < m) {
            const std::size_t c = pick(rng_);
            if (std::find(chosen_.begin(), chosen_.end(), c) == chosen_.end()) chosen_.push_back(c);
        }
        return u_;
    }

    void tell(const Vec& f, const Vec& v = {})
    {
        if (f.size() != u_.size() || (!v.empty() && v.size() != u_.size())) throw std::invalid_argument("g3: tell sizes");
        const Vec vv = v.empty() ? Vec(f.size(), 0.0) : v;
        if (x_.empty()) {
            x_ = u_;
            f_ = f;
            v_ = vv;
            return;
        }
        Rows fx;                                                         // (4), G3-3
        Vec ff, fv;
        for (std::size_t c : chosen_) {
            fx.push_back(x_[c]);
            ff.push_back(f_[c]);
            fv.push_back(v_[c]);
        }
        fx.insert(fx.end(), u_.begin(), u_.end());
        ff.insert(ff.end(), f.begin(), f.end());
        fv.insert(fv.end(), vv.begin(), vv.end());
        std::vector<std::size_t> order(fx.size());
        std::iota(order.begin(), order.end(), std::size_t{0});
        std::stable_sort(order.begin(), order.end(),
                         [&](std::size_t a, std::size_t b) { return evo::better(ff[a], fv[a], ff[b], fv[b]); });
        for (std::size_t k = 0; k < chosen_.size(); ++k) {
            x_[chosen_[k]] = fx[order[k]];
            f_[chosen_[k]] = ff[order[k]];
            v_[chosen_[k]] = fv[order[k]];
        }
    }

    const Rows& population() const { return x_; }
    const Vec& values() const { return f_; }
    const Vec& violations() const { return v_; }
    std::size_t best() const { return evo::best(f_, v_); }
    // a migrant (pap, task 19): the worst member's place where it is better
    void migrate(const Vec& x, double f, double v) { evo::replace_worst(x_, f_, v_, x, f, v); }

private:
    Vec lo_, hi_;
    G3Options o_;
    std::mt19937& rng_;
    Rows x_, u_;
    Vec f_, v_;
    std::vector<std::size_t> chosen_;
};

}   // namespace mootation::scalar::ga

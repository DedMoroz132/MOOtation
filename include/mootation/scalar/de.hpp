#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// Differential evolution, DE/rand/1/bin, as R. Storn, K. Price, "Differential
// Evolution - A Simple and Efficient Heuristic for Global Optimization over
// Continuous Spaces", J. Global Optim. 11:341-359, 1997, §2 (source
// de_storn1997): for every target x_i a mutant v = x_r1 + F (x_r2 - x_r3)
// (2), r1, r2, r3 mutually different and different from i; the trial takes
// v_j where randb(j) <= CR or j = rnbr(i), x_ij elsewhere (4); the trial
// replaces its target when it is not worse. All NP trials of a generation are
// built from generation G (Figure 3's two arrays), so a generation is one
// ask() of NP points and one tell().
//
// The caller asks for points, evaluates them and tells their values f and
// violations v. The first ask() gives the initial population, uniform in the
// box (§2: "chosen randomly and should cover the entire parameter space").
// Minimization; the random numbers are rnd:: of the caller's generator.
//
// Defaults are §4's first choices: F = 0.5, CR = 0.1; NP = 10 D, the top of
// its "between 5 D and 10 D"; NP >= 4 (§2), refused otherwise.
//
// Readings (OURS):
//   DE-1. The trial (4) is built coordinate by coordinate: rnbr(i) is drawn
//     first, then one U(0, 1) for every j (j = rnbr(i) included), and v_j is
//     taken where U < CR or j = rnbr(i). Figure 3 starts at a random j and
//     forces its last one — the same distribution; (4)'s "<=" is Figure 3's
//     "<" (they differ on a draw of exactly CR). The indices are drawn as in
//     Figure 3: a != i, b != i, a, c != i, a, b by redrawing, and the trial is
//     x_c + F (x_a - x_b) — (2) with r1 = c, r2 = a, r3 = b.
//   DE-2. Bounds. The paper's box is the initial parameter range (IPR) only;
//     its f9 and f28 have their minima outside it, and DE finds them. Here a
//     mutant coordinate outside the box is repaired before the crossover coin
//     by ops::repair_value (operators/bound_repair.hpp), `reflect` by
//     default as in gde3 (GDE3-3); `native` leaves it outside — the paper's
//     DE, for problems defined beyond the box; `resample` is refused.
//   DE-3. Selection: the trial replaces the target when it is not worse —
//     Figure 3's "score <= cost[i]"; §2's text says "smaller".
//   DE-4. Constraints g <= 0: NSGA-II's constrained-domination at M = 1
//     (evo.hpp); not in the paper.
//   DE-5. done() is true once all NP points are equal: every difference
//     vector is zero and no trial can differ from its target. The paper
//     stops at a value to reach or a number of generations, the caller's.
//   DE-6. sym (task 19, symmetry.hpp SYM-8; off by default): after the index
//     draws, x_r1, x_r2 and x_r3 aligned to the target (align), or the four
//     canonical (canon: the crossover and the bound repair then take
//     canon(x_i)); no draw changes, and sym = off is the trial as before.
//
// With F = CR = 0.2 and `reflect` this is gde3 at n_objs = 1 without
// constraints, draw for draw (GDE3Core::setup and step): the same initial
// draws, the same index draws, the same trial and the same "not worse" rule.
// ============================================================================

#include <cstddef>
#include <random>
#include <stdexcept>
#include <vector>

#include "../operators/bound_repair.hpp"
#include "../random.hpp"
#include "evo.hpp"
#include "symmetry.hpp"

namespace mootation::scalar::de {

using evo::Rows;
using evo::Vec;

struct Options {
    std::size_t np = 0;                                    // 0: 10 D
    double F = 0.5;                                        // §4
    double CR = 0.1;                                       // §4
    ops::BoundRepair repair = ops::BoundRepair::Reflect;   // DE-2
    symmetry::Sym sym;                                     // DE-6
    Rows seeds;                         // the first population's first rows (task 24, §24.4.1; POP-7)
    std::vector<char> periodic;         // periodic variables (task 24, §24.4.2; periodic.hpp)
};

// The trial of x_i (DE-1, (2), (4)); sym: DE-6.
inline Vec trial(const Rows& x, std::size_t i, const Options& o, const Vec& lo, const Vec& hi, std::mt19937& rng)
{
    const std::size_t n = x.size(), d = lo.size();
    const int last = static_cast<int>(n) - 1, ii = static_cast<int>(i);
    rnd::uniform_int_distribution<int> pick(0, last);
    rnd::uniform_int_distribution<int> dim(0, static_cast<int>(d) - 1);
    rnd::uniform_real_distribution<double> U01(0.0, 1.0);
    int a, b, c;                                                   // DE-1
    do a = pick(rng); while (a == ii);
    do b = pick(rng); while (b == ii || b == a);
    do c = pick(rng); while (c == ii || c == a || c == b);
    Vec ci, ca, cb, cc;                                            // DE-6's copies
    const Vec& xi = o.sym.mode == symmetry::Mode::canon ? o.sym.view(x[i], x[i], ci) : x[i];
    const Vec &xa = o.sym.view(x[static_cast<std::size_t>(a)], xi, ca),
              &xb = o.sym.view(x[static_cast<std::size_t>(b)], xi, cb),
              &xc = o.sym.view(x[static_cast<std::size_t>(c)], xi, cc);
    const std::size_t jr = static_cast<std::size_t>(dim(rng));
    Vec u(d);
    ops::RepairTally tally;                                        // the mutant (task 20, §20.2.2)
    for (std::size_t j = 0; j < d; ++j) {
        double v = xc[j] + o.F * (xa[j] - xb[j]);                  // (2)
        if ((v < lo[j] || v > hi[j]) && !evo::wrapped(v, lo, hi, o.periodic, j)) {   // PER-2
            tally.out();
            if (o.repair != ops::BoundRepair::Native)
                v = ops::repair_value(v, lo[j], hi[j], xi[j], o.repair, rng);   // DE-2
        }
        u[j] = U01(rng) < o.CR || j == jr ? v : xi[j];             // (4)
    }
    return u;
}

class DE {
public:
    DE(Vec lo, Vec hi, const Options& o, std::mt19937& rng)
        : lo_(std::move(lo)), hi_(std::move(hi)), o_(o), rng_(rng)
    {
        if (lo_.empty() || lo_.size() != hi_.size()) throw std::invalid_argument("de: lo, hi");
        if (o_.np == 0) o_.np = 10 * lo_.size();
        if (o_.np < 4) throw std::invalid_argument("de: NP >= 4 (§2)");
        if (o_.repair == ops::BoundRepair::Resample)
            throw std::invalid_argument("de: bound repair 'resample' is not available (DE-2)");
    }

    std::size_t np() const { return o_.np; }

    Rows ask()
    {
        if (x_.empty()) return trial_ = evo::uniform(lo_, hi_, o_.np, rng_, o_.seeds);
        trial_.assign(o_.np, Vec());
        for (std::size_t i = 0; i < o_.np; ++i) trial_[i] = trial(x_, i, o_, lo_, hi_, rng_);
        return trial_;
    }

    // f and v of the points of the last ask(); v empty: no constraints.
    void tell(const Vec& f, const Vec& v = {})
    {
        const std::size_t n = o_.np;
        if (f.size() != n || (!v.empty() && v.size() != n)) throw std::invalid_argument("de: tell sizes");
        if (x_.empty()) {
            x_ = trial_;
            f_ = f;
            v_ = v.empty() ? Vec(n, 0.0) : v;
            return;
        }
        for (std::size_t i = 0; i < n; ++i) {
            const double vi = v.empty() ? 0.0 : v[i];
            if (evo::not_worse(f[i], vi, f_[i], v_[i])) {          // DE-3, DE-4
                x_[i] = trial_[i];
                f_[i] = f[i];
                v_[i] = vi;
            }
        }
    }

    bool done() const                                              // DE-5
    {
        for (std::size_t i = 1; i < x_.size(); ++i)
            if (x_[i] != x_[0]) return false;
        return !x_.empty();
    }

    const Rows& population() const { return x_; }
    const Vec& values() const { return f_; }
    const Vec& violations() const { return v_; }
    std::size_t best() const { return evo::best(f_, v_); }
    // a migrant (pap, task 19): the worst member's place where it is better
    void migrate(const Vec& x, double f, double v) { evo::replace_worst(x_, f_, v_, x, f, v); }

private:
    Vec lo_, hi_;
    Options o_;
    std::mt19937& rng_;
    Rows x_, trial_;
    Vec f_, v_;
};

}   // namespace mootation::scalar::de

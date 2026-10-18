#pragma once
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "../constraint_mode.hpp"
#include "../data_vault.hpp"
#include "../detail/constrained.hpp"
#include "../operators/poly_mutation.hpp"

namespace mootation {

// ============================================================================
// PAES — the (1+1) Pareto Archived Evolution Strategy
// J.D. Knowles, D.W. Corne, "Approximating the Nondominated Front Using the
// Pareto Archived Evolution Strategy", Evol. Comput. 8(2):149-172, 2000,
// doi:10.1162/106365600568167, and its first paper, "The Pareto Archived
// Evolution Strategy: A New Baseline Algorithm for Pareto Multiobjective
// Optimisation", CEC 1999, pp. 98-105, doi:10.1109/CEC.1999.781913. Neither
// text was available: the algorithm as the materials of task 9 give the
// CEC'99 paper (§3, Figures 1-2; §4), the adaptive grid as jMetal's
// AdaptiveGrid and AdaptiveGridArchive (MIT; main 67de131d591a), which PAES
// there uses.
//
// One step, one evaluation: a mutant m of the current solution c,
//   c dominates m        → m is discarded;
//   m dominates c        → m becomes c and goes to the archive;
//   otherwise            → m is discarded if a member of the archive
//                          dominates it; else test(c, m, archive): m goes to
//                          the archive, and m becomes c if its grid cell is
//                          less crowded than c's.
// The archive keeps non-dominated points only; a new one removes those it
// dominates; when the archive is full, m enters only if its cell is not the
// most crowded, and then a member of the most crowded cell leaves.
// THE GRID divides the archive's objective range — its least and greatest
// value per objective, widened and rebuilt when a point falls outside —
// into 2^depth intervals per objective by recursive halving (depth bisections:
// 2^(depth·M) cells), CEC'99 §4: depth 5, archive 100. The answer is the
// archive, written into the active slots after every step.
//
// Parameters: paes_depth (5, CEC'99 §4; Platypus's "divisions" is 2^depth,
// not the depth); the archive's bound archive_size (pop_size, CEC'99's 100
// at pop_size 100); polynomial mutation, η_m = 20, p_m = 1/n (eta_m, pm), as
// jMetal's PAESRunner — the papers mutate bit strings.
//
// READINGS (declared):
//   PAES-1. c's density when c lies outside the current grid (it is not in
//     the archive): m becomes c; m outside: c stays (jMetal 4/5's test()).
//   PAES-2. The member that leaves a full archive: the first, in archive
//     order, of the most crowded cell (jMetal's prune()).
//   PAES-3. Equal objectives count as mutually non-dominated; the archive
//     takes no copy of a member's objectives.
//   PAES-4. Constraints: Deb's constrained domination wherever domination is
//     asked (constraint_mode not NONE), as jMetal's comparator.
//   PAES-5. The first c: a uniform random point (seeded: the first seed).
// Real variables only.
// ============================================================================

template <typename Ind_t>
class PAESCore {
public:
    ConstraintMode constraint_mode = ConstraintMode::NONE;

    void set_seed(unsigned s)       { rng_.seed(s); }
    void set_eta_mutation(double e) { eta_m_ = e; }
    void set_pm(double p)           { pm_ = p; }

    bool set_param(const std::string& k, double v)
    {
        if (k == "paes_depth") {
            if (!(v >= 1.0) || v != std::floor(v) || v > 12)
                throw std::invalid_argument("paes_depth must be a whole number in [1, 12]");
            depth_ = static_cast<int>(v);
            return true;
        }
        if (k == "archive_size") {
            if (!(v >= 1.0) || v != std::floor(v) || v > 1e7)
                throw std::invalid_argument("archive_size must be a whole number >= 1");
            cap_ = static_cast<int>(v);
            return true;
        }
        return false;
    }

    void setup(DataVault<Ind_t>& vault)
    {
        start(vault);
        const auto& b = vault.get_bounds();
        std::uniform_real_distribution<double> U(0.0, 1.0);
        std::vector<double> x(b.size());
        for (std::size_t j = 0; j < b.size(); ++j) x[j] = *b[j].first + U(rng_) * (*b[j].second - *b[j].first);
        vault.reduce(0);
        c_ = evaluate(vault, x);
        add(c_);
        publish(vault);
    }

    void setup_seeded(DataVault<Ind_t>& vault)
    {
        start(vault);
        std::vector<Point> seeds;
        for (std::size_t v = 0; v < vault.active_n(); ++v) seeds.push_back(read(vault, v));
        c_ = seeds.front();                                  // PAES-5
        for (const auto& p : seeds) add(p);
        publish(vault);
    }

    void step(DataVault<Ind_t>& vault)
    {
        const bool with_cv = constraint_mode != ConstraintMode::NONE;
        refresh_copy(vault, c_.pt, c_.f, c_.cv, with_cv);
        for (auto& a : arch_) refresh_copy(vault, a.pt, a.f, a.cv, with_cv);
        grid_rebuild();
        std::vector<double> x = c_.pt.variables;
        const double pm = pm_ > 0.0 ? pm_ : 1.0 / static_cast<double>(x.size());
        ops::polynomial_mutation(x, vault.get_bounds(), eta_m_, pm, rng_);
        Point m = evaluate(vault, x);
        if (dominates(c_, m)) {
            // discarded
        } else if (dominates(m, c_)) {
            c_ = m;
            add(m);
        } else {
            bool beaten = false;
            for (const auto& a : arch_) beaten = beaten || dominates(a, m);
            if (!beaten) {
                add(m);
                const int lc = location(c_.f), lm = location(m.f);
                if (lc < 0) c_ = m;                          // PAES-1
                else if (lm >= 0 && density(lm) < density(lc)) c_ = m;
            }
        }
        publish(vault);
    }

private:
    struct Point {
        std::vector<double> f;
        double cv = 0.0;
        VaultPoint pt{};
    };

    double eta_m_ = 20.0, pm_ = -1.0;
    int depth_ = 5, cap_ = 0;
    std::vector<double> lo_, hi_;
    std::vector<int> count_;                                  // members per cell
    std::vector<Point> arch_;
    Point c_;
    std::mt19937 rng_{0};

    void start(DataVault<Ind_t>& vault)
    {
        if (vault.bin_vars_n() > 0)
            throw std::invalid_argument("paes: continuous variables only");
        for (const auto& b : vault.get_bounds())
            if (!b.first || !b.second)
                throw std::invalid_argument("paes: every variable needs finite bounds");
        if (cap_ == 0) cap_ = vault.pop_size();
        arch_.clear();
    }

    bool dominates(const Point& a, const Point& b) const
    {
        return detail::dominates(constraint_mode, a.f, a.cv, b.f, b.cv);
    }

    Point read(DataVault<Ind_t>& vault, std::size_t v) const
    {
        Point p;
        p.f = vault.objectives_of(v);
        p.cv = constraint_mode != ConstraintMode::NONE ? vault.get_cv(v) : 0.0;
        p.pt = vault.point_of(v);
        return p;
    }

    Point evaluate(DataVault<Ind_t>& vault, const std::vector<double>& x)
    {
        const int keep = static_cast<int>(vault.active_n());
        const int at = vault.expand(1);
        vault.set_variables(static_cast<std::size_t>(at), x);
        vault.sync();
        Point p = read(vault, static_cast<std::size_t>(at));
        vault.reduce(keep);
        return p;
    }

    void publish(DataVault<Ind_t>& vault)
    {
        vault.reduce(0);
        vault.expand(static_cast<int>(arch_.size()));
        for (std::size_t i = 0; i < arch_.size(); ++i) vault.put_point(i, arch_[i].pt);
    }

    // ── the adaptive grid (jMetal's AdaptiveGrid) ───────────────────────────
    int cells_per_axis() const { return 1 << depth_; }

    // The cell of f, or −1 outside the grid's range.
    int location(const std::vector<double>& f) const
    {
        if (lo_.empty()) return -1;
        long long loc = 0, mult = 1;
        const int per = cells_per_axis();
        for (std::size_t j = 0; j < f.size(); ++j) {
            if (f[j] < lo_[j] || f[j] > hi_[j]) return -1;
            int pos = 0;
            if (f[j] == hi_[j]) pos = per - 1;
            else if (f[j] > lo_[j]) {
                double size = hi_[j] - lo_[j], account = lo_[j];
                int ranges = per;
                for (int b = 0; b < depth_; ++b) {
                    size /= 2.0;
                    ranges /= 2;
                    if (f[j] > account + size) { pos += ranges; account += size; }
                }
            }
            loc += pos * mult;
            mult *= per;
        }
        return static_cast<int>(loc);
    }

    int density(int cell) const
    {
        int n = 0;
        for (const auto& a : arch_) n += location(a.f) == cell;
        return n;
    }

    // The range of the archive's members, and with it every cell.
    void grid_rebuild()
    {
        if (arch_.empty()) { lo_.clear(); hi_.clear(); return; }
        const std::size_t m = arch_.front().f.size();
        lo_.assign(m, std::numeric_limits<double>::infinity());
        hi_.assign(m, -std::numeric_limits<double>::infinity());
        for (const auto& a : arch_)
            for (std::size_t j = 0; j < m; ++j) {
                lo_[j] = std::min(lo_[j], a.f[j]);
                hi_[j] = std::max(hi_[j], a.f[j]);
            }
    }

    int most_crowded() const
    {
        std::vector<int> cells;
        for (const auto& a : arch_) cells.push_back(location(a.f));
        int best = -1, most = 0;
        for (int c : cells) {
            const int n = static_cast<int>(std::count(cells.begin(), cells.end(), c));
            if (n > most) { most = n; best = c; }
        }
        return best;
    }

    // The archive's add (jMetal's AdaptiveGridArchive.add, PAES-2, -3).
    bool add(const Point& p)
    {
        for (const auto& a : arch_)
            if (dominates(a, p) || a.f == p.f) return false;
        arch_.erase(std::remove_if(arch_.begin(), arch_.end(),
                                   [&](const Point& a) { return dominates(p, a); }),
                    arch_.end());
        if (static_cast<int>(arch_.size()) < cap_) {
            arch_.push_back(p);
            grid_rebuild();
            return true;
        }
        // full: widen the grid to p if p lies outside it, then the cells decide
        auto with = arch_;
        with.push_back(p);
        std::swap(arch_, with);
        grid_rebuild();
        std::swap(arch_, with);
        const int lp = location(p.f);
        if (lp == most_crowded()) { grid_rebuild(); return false; }
        const int crowded = most_crowded();
        for (auto it = arch_.begin(); it != arch_.end(); ++it)
            if (location(it->f) == crowded) { arch_.erase(it); break; }
        arch_.push_back(p);
        grid_rebuild();
        return true;
    }
};

}   // namespace mootation

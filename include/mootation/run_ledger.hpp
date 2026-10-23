#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// A campaign run's accounting of its evaluations in C++ (task 17, stage 2b,
// step 2): python/mootation/run/campaign.py's account() — every row a core
// evaluated, in order: its count, whether it failed (a non-finite objective),
// whether it is feasible, the first feasible evaluation, and the row offered
// to the run archive, the second answer's archive, the noisy copy's oracle
// and everything evaluated (IncrementalFront), each at its own count.
//
// Reading at exactly a count. Where the run reads something after exactly b
// evaluations (the grid of everything evaluated, hv_archive, the hard
// budgets, the archive checkpoints; set_reads), a batch is taken in parts:
// next() offers the rows up to b and returns b, the caller reads, next()
// goes on; -1 when the batch is done. Everything evaluated is read here
// (all_at). The archives take every row as offered: campaign.py's pending
// lists and its "posthoc" mode only batch the same additions, one point after
// the other, so the points are the same.
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <map>
#include <set>
#include <utility>
#include <vector>

#include "grid_archive.hpp"
#include "indicators.hpp"
#include "set_archive.hpp"

namespace mootation {

class RunLedger {
public:
    // m objectives, d variables; constrained: feasibility counts (the rows
    // come with their constraint values). Any archive may be null.
    RunLedger(std::size_t m, std::size_t d, bool constrained, GridArchive* archive, GridArchive* oracle,
              indicators::IncrementalFront* all, SetArchive* second)
        : m_(m), d_(d), constrained_(constrained), arc_(archive), oracle_(oracle), inc_(all), sarc_(second)
    {
    }

    // The counts where the caller reads (sorted, distinct), and among them
    // those where everything evaluated is read (all_at).
    void set_reads(std::vector<long long> at, const std::vector<long long>& grid)
    {
        std::sort(at.begin(), at.end());
        at.erase(std::unique(at.begin(), at.end()), at.end());
        reads_ = std::move(at);
        grid_.clear();
        grid_.insert(grid.begin(), grid.end());
    }

    // A batch of n rows: the variables the archives keep X (n x d; a
    // categorical variable's index for its label), objectives F (n x m), the
    // constraint values G (n x c, c = 0 or null without), the noise-free
    // values T (n x m; null but on a noisy copy, where the oracle takes them).
    // The arrays must live until the batch is done.
    void begin(const double* X, const double* F, std::size_t n, const double* G, std::size_t c,
               const double* T)
    {
        X_ = X;
        F_ = F;
        T_ = T;
        n_ = n;
        lo_ = 0;
        start_ = fe_;
        finite_.assign(n, 1);
        ok_.assign(n, 1);
        bool first = true;
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < m_; ++j)
                if (!std::isfinite(F[i * m_ + j])) { finite_[i] = 0; break; }
            ok_[i] = finite_[i];
            if (G && c)
                for (std::size_t k = 0; k < c; ++k)
                    if (!(G[i * c + k] <= 0.0)) { ok_[i] = 0; break; }
            if (!finite_[i]) {
                ++n_failed_;
                failed_fe_.push_back(start_ + 1 + static_cast<long long>(i));
            }
            if (constrained_ && first_feasible_ < 0 && ok_[i] && first) {
                first_feasible_ = start_ + 1 + static_cast<long long>(i);
                first = false;
            }
        }
        next_read_ = static_cast<std::size_t>(std::upper_bound(reads_.begin(), reads_.end(), start_) - reads_.begin());
    }

    // Offer the rows up to the next count read in this batch and return it
    // (fe() is then that count; everything evaluated has been read there if
    // it is on its grid), or offer the rest and return -1.
    long long next()
    {
        if (next_read_ < reads_.size() && reads_[next_read_] <= start_ + static_cast<long long>(n_)) {
            const long long b = reads_[next_read_++];
            offer(lo_, static_cast<std::size_t>(b - start_));
            lo_ = static_cast<std::size_t>(b - start_);
            fe_ = b;
            if (inc_ && grid_.count(b)) all_at_[b] = {inc_->igd_plus(), inc_->eps_plus()};
            return b;
        }
        offer(lo_, n_);
        lo_ = n_;
        fe_ = start_ + static_cast<long long>(n_);
        return -1;
    }

    long long fe() const { return fe_; }
    long long n_failed() const { return n_failed_; }
    const std::vector<long long>& failed_fe() const { return failed_fe_; }
    long long first_feasible() const { return first_feasible_; }     // -1: none yet
    const std::map<long long, std::pair<double, double>>& all_at() const { return all_at_; }

private:
    void offer(std::size_t lo, std::size_t hi)
    {
        for (std::size_t i = lo; i < hi; ++i) {
            if (!finite_[i]) continue;
            const double* f = F_ + i * m_;
            const double* x = d_ ? X_ + i * d_ : nullptr;
            if (arc_ && ok_[i]) arc_->add(f, x);
            if (sarc_ && ok_[i]) sarc_->add(f, x);
            const double* t = nullptr;
            if (oracle_ && T_) {
                t = T_ + i * m_;
                oracle_->add(t, x);
            }
            if (inc_ && ok_[i]) {
                const long long at = start_ + 1 + static_cast<long long>(i);
                inc_->update(t ? t : f, 1, &at);
            }
        }
    }

    std::size_t m_, d_;
    bool constrained_;
    GridArchive* arc_;
    GridArchive* oracle_;
    indicators::IncrementalFront* inc_;
    SetArchive* sarc_;
    std::vector<long long> reads_;
    std::set<long long> grid_;
    long long fe_ = 0, n_failed_ = 0, first_feasible_ = -1;
    std::vector<long long> failed_fe_;
    std::map<long long, std::pair<double, double>> all_at_;
    // the batch in progress
    const double* X_ = nullptr;
    const double* F_ = nullptr;
    const double* T_ = nullptr;
    std::size_t n_ = 0, lo_ = 0, next_read_ = 0;
    long long start_ = 0;
    std::vector<char> finite_, ok_;
};

}   // namespace mootation

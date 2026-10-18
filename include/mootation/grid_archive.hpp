#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// The run archive in C++: python/mootation/run/archive.py's GridArchive, point
// for point. A campaign offers it every point a run evaluates (online, a batch
// at a time with add_many, or all of them after the run); the Python class
// stays as the reference the tests compare with, and describes the archive
// (what is kept, the grid, the extremes outside it, the frame).
//
// THE SAME ARCHIVE. Every decision is made from the values the Python class
// computes, by the same expressions: the cell, floor(((f - lo) / span) /
// delta) as an int64 (a value out of its range becomes the smallest int64, as
// NumPy's cast gives it on x86); the distance to the cell's corner, the sum of
// squares in NumPy's order (numpy_sum.hpp); the drift test of the archive's
// own frame, f < lo - span/4 or f > (lo + span) + span/4. The slots are
// numbered as the Python class numbers them (appended in order, compacted when
// the store of 256, 512, ... is full and more than half of it is dead), and the
// extremes a point dethrones rejoin the grid in the order Python's set of
// their slot numbers iterates in (cpython_set_order below): two of them can
// tie for a cell, and then the first placed keeps it.
//
// Dominance is checked against the stored points kept sorted by the first
// objective at two objectives: only those at or below f_1 can weakly dominate
// f, only those at or above it can be dominated by it; from three
// (kTreeFrom), through an ND-tree of the stored points (nd_tree.hpp; task 11
// of 2026-10-18, §2), which gives the same answers as the pass over every
// stored point (set_tree(false)) without comparing f with every one of them.
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <map>
#include <set>
#include <utility>
#include <vector>

#include "nd_tree.hpp"
#include "numpy_sum.hpp"

namespace mootation {

namespace detail {

// The order in which CPython (3.8 to 3.14) iterates a set built by adding the
// non-negative ints `keys` one after the other ({k for k in keys}): an int's
// hash is itself, the table starts at 8 entries and grows to the next power of
// two above four times its size when it is three fifths full; a key goes to
// slot hash & mask, else probes up to 9 slots further while they fit, else
// jumps to (5 i + 1 + perturb) & mask with perturb shifted right by 5 each
// time (Objects/setobject.c, set_add_entry and set_insert_clean).
inline std::vector<long long> cpython_set_order(const std::vector<long long>& keys)
{
    std::vector<long long> table(8, -1);           // -1: empty
    std::size_t fill = 0;
    auto insert_clean = [](std::vector<long long>& t, long long key) {
        const std::size_t mask = t.size() - 1;
        std::size_t perturb = static_cast<std::size_t>(key);
        std::size_t i = static_cast<std::size_t>(key) & mask;
        for (;;) {
            if (t[i] < 0) { t[i] = key; return; }
            if (i + 9 <= mask)
                for (std::size_t j = 1; j <= 9; ++j)
                    if (t[i + j] < 0) { t[i + j] = key; return; }
            perturb >>= 5;
            i = (i * 5 + 1 + perturb) & mask;
        }
    };
    for (long long key : keys) {
        const std::size_t mask = table.size() - 1;
        std::size_t perturb = static_cast<std::size_t>(key);
        std::size_t i = static_cast<std::size_t>(key) & mask;
        bool present = false, placed = false;
        while (!present && !placed) {
            const std::size_t probes = (i + 9 <= mask) ? 9 : 0;
            for (std::size_t j = 0; j <= probes; ++j) {
                if (table[i + j] < 0) { table[i + j] = key; placed = true; break; }
                if (table[i + j] == key) { present = true; break; }
            }
            if (present || placed) break;
            perturb >>= 5;
            i = (i * 5 + 1 + perturb) & mask;
        }
        if (present) continue;
        ++fill;
        if (fill * 5 < mask * 3) continue;
        std::size_t size = 8;
        while (size <= fill * 4) size <<= 1;       // used == fill: nothing is ever removed
        std::vector<long long> grown(size, -1);
        for (long long k : table)
            if (k >= 0) insert_clean(grown, k);
        table.swap(grown);
    }
    std::vector<long long> out;
    for (long long k : table)
        if (k >= 0) out.push_back(k);
    return out;
}

}   // namespace detail

class GridArchive {
public:
    // archive.default_delta: 1e-3, and 1e-2 from five objectives up
    static double default_delta(int n_obj) { return n_obj >= 5 ? 1e-2 : 1e-3; }

    // From how many objectives the dominance checks go through the ND-tree.
    static constexpr std::size_t kTreeFrom = 3;

    // ideal and nadir both given: the problem's frame; else the archive's own.
    GridArchive(int n_obj, int n_vars, const std::vector<double>& ideal,
                const std::vector<double>& nadir, double delta)
        : m_(static_cast<std::size_t>(n_obj)), d_(static_cast<std::size_t>(n_vars)),
          delta_(delta != 0.0 ? delta : default_delta(n_obj)),
          problem_frame_(!ideal.empty() && !nadir.empty()), ext_(m_, -1),
          use_tree_(m_ >= kTreeFrom), tree_(m_)
    {
        if (m_ != 2) cols_.resize(m_);
        if (problem_frame_) {
            lo_ = ideal;
            span_.resize(m_);
            for (std::size_t j = 0; j < m_; ++j) {
                const double s = nadir[j] - lo_[j];
                span_[j] = s > 0.0 ? s : 1.0;
            }
        }
    }

    std::size_t n_obj() const { return m_; }
    std::size_t n_vars() const { return d_; }
    double delta() const { return delta_; }
    bool problem_frame() const { return problem_frame_; }
    long long offered() const { return offered_; }
    long long accepted() const { return accepted_; }
    std::size_t size() const { return live_; }

    // Offer one evaluated point (x: n_vars values, or null for zeros); true if
    // it is stored. archive.GridArchive.add.
    bool add(const double* f, const double* x)
    {
        ++offered_;
        for (std::size_t j = 0; j < m_; ++j)
            if (!std::isfinite(f[j])) return false;
        if (!take(f)) return false;                  // dominated or a duplicate
        const std::size_t s = append(f, x);
        ++accepted_;
        if (!problem_frame_ && frame_drifted(f)) regrid(static_cast<long long>(s));
        std::vector<std::size_t> new_ext;
        for (std::size_t j = 0; j < m_; ++j)
            if (ext_[j] < 0 || f[j] < F_[static_cast<std::size_t>(ext_[j]) * m_ + j])
                new_ext.push_back(j);
        if (new_ext.empty()) return place(s);
        std::vector<long long> old;
        for (std::size_t j : new_ext)
            if (ext_[j] >= 0) old.push_back(ext_[j]);
        for (std::size_t j : new_ext) ext_[j] = static_cast<long long>(s);
        for (long long o : detail::cpython_set_order(old)) {
            const std::size_t so = static_cast<std::size_t>(o);
            if (alive_[so] && std::find(ext_.begin(), ext_.end(), o) == ext_.end()) place(so);
        }
        return true;
    }

    // An offer of the wrong shape: counted, not stored (add() would return false).
    void add_invalid() { ++offered_; }

    // The dominance checks through the ND-tree (true) or against every stored
    // point (false), from three objectives; before the first offer only. The
    // archive is the same either way (the tests compare the two).
    void set_tree(bool on)
    {
        if (offered_ == 0 && m_ > 2) use_tree_ = on;
    }
    bool tree() const { return use_tree_ && m_ > 2; }

    // The stored points in insertion order: objectives, variables, and whether
    // each is an objective's best (archive.GridArchive.points).
    void points(std::vector<double>& F, std::vector<double>& X, std::vector<char>& extreme) const
    {
        F.clear();
        X.clear();
        extreme.clear();
        for (std::size_t s = 0; s < n_; ++s) {
            if (!alive_[s]) continue;
            F.insert(F.end(), F_.begin() + static_cast<std::ptrdiff_t>(s * m_),
                     F_.begin() + static_cast<std::ptrdiff_t>((s + 1) * m_));
            X.insert(X.end(), X_.begin() + static_cast<std::ptrdiff_t>(s * d_),
                     X_.begin() + static_cast<std::ptrdiff_t>((s + 1) * d_));
            extreme.push_back(std::find(ext_.begin(), ext_.end(), static_cast<long long>(s)) !=
                              ext_.end());
        }
    }

    // The frame the grid uses: (lo, lo + span); empty before the archive's own
    // frame is first estimated (archive.GridArchive.frame takes the points'
    // range then).
    bool frame(std::vector<double>& lo, std::vector<double>& hi) const
    {
        if (lo_.empty()) return false;
        lo = lo_;
        hi.resize(m_);
        for (std::size_t j = 0; j < m_; ++j) hi[j] = lo_[j] + span_[j];
        return true;
    }

private:
    std::size_t m_, d_;
    double delta_;
    bool problem_frame_;
    std::vector<double> lo_, span_;                 // empty: no frame yet
    std::vector<double> F_, X_;                     // slot s at F_[s * m_], X_[s * d_]
    std::vector<char> alive_;
    std::size_t n_ = 0;                             // slots used
    std::size_t cap_ = 256;                         // the Python store's size
    std::size_t live_ = 0;
    std::map<std::vector<std::int64_t>, std::size_t> cell_;   // cell -> slot
    std::vector<std::vector<std::int64_t>> slot_cell_;        // slot -> its cell, if any
    std::vector<char> in_cell_;
    std::vector<long long> ext_;                    // slot of each objective's best, -1 none
    std::multimap<double, std::size_t> by_f1_;      // live slots by f_1 (two objectives)
    std::vector<std::size_t> beaten_;               // take()'s, kept for its storage
    // from three objectives: objective j of every slot, F_ by columns, and
    // take()'s max_j (p_j - f_j) and max_j (f_j - p_j) of every slot
    std::vector<std::vector<double>> cols_;
    std::vector<double> above_, below_;
    long long offered_ = 0, accepted_ = 0;
    // from kTreeFrom objectives: the live slots in an ND-tree (nd_tree.hpp)
    bool use_tree_;
    nd::NDTree tree_;

    const double* f_of(std::size_t s) const { return F_.data() + s * m_; }

    // Whether f is to be stored: no stored point weakly dominates it (p <= f
    // everywhere, a copy included); if so, the stored points it weakly
    // dominates are removed. The stored points are mutually nondominated, so
    // a dominated f weakly dominates none of them, and the order of the
    // removals does not matter (each touches its own slot). At two objectives
    // they form a staircase, f_2 falling as f_1 rises, and one look decides;
    // from three, one pass over the slots in memory.
    bool take(const double* f)
    {
        beaten_.clear();
        if (m_ == 2) {
            auto it = by_f1_.upper_bound(f[0]);         // the last p with p_1 <= f_1 ...
            if (it != by_f1_.begin() && f_of(std::prev(it)->second)[1] <= f[1])
                return false;                           // ... has the smallest p_2 of them
            for (it = by_f1_.lower_bound(f[0]); it != by_f1_.end(); ++it) {
                const double* p = f_of(it->second);
                if (p[1] < f[1]) break;                 // the staircase: none further on
                if (f[0] <= p[0] && f[1] <= p[1]) beaten_.push_back(it->second);
            }
        } else if (use_tree_) {
            // the ND-tree: the same answers, the beaten slots ascending as
            // the pass below finds them
            if (tree_.covered(f, &beaten_)) return false;
        } else {
            // one pass over each objective's column, every slot: p does not
            // weakly dominate f where max_j (p_j - f_j) > 0, f does not weakly
            // dominate p where max_j (f_j - p_j) > 0 — a positive difference
            // means the strict inequality. Where a maximum is not positive
            // the comparisons decide: a difference of two unequal doubles can
            // be flushed to 0 where the processor drops subnormals.
            above_.assign(n_, -std::numeric_limits<double>::infinity());
            below_.assign(n_, -std::numeric_limits<double>::infinity());
            double* up = above_.data();
            double* dn = below_.data();
            for (std::size_t j = 0; j < m_; ++j) {
                const double* c = cols_[j].data();
                const double fj = f[j];
                for (std::size_t s = 0; s < n_; ++s) {
                    const double d = c[s] - fj;
                    up[s] = up[s] < d ? d : up[s];
                    dn[s] = dn[s] < -d ? -d : dn[s];
                }
            }
            for (std::size_t s = 0; s < n_; ++s) {
                if (!alive_[s]) continue;
                bool above = up[s] > 0.0, below = dn[s] > 0.0;
                if (!above || !below) {
                    const double* p = f_of(s);
                    above = below = false;
                    for (std::size_t j = 0; j < m_; ++j) {
                        above = above || p[j] > f[j];
                        below = below || f[j] > p[j];
                    }
                }
                if (!above) return false;
                if (!below) beaten_.push_back(s);
            }
        }
        for (std::size_t s : beaten_) remove(s);
        return true;
    }

    std::size_t append(const double* f, const double* x)
    {
        if (n_ == cap_) {
            if (n_ > 2 * live_ + 64) compact();
            if (n_ == cap_) cap_ *= 2;
        }
        const std::size_t s = n_++;
        F_.insert(F_.end(), f, f + m_);
        if (x) X_.insert(X_.end(), x, x + d_);
        else   X_.insert(X_.end(), d_, 0.0);
        alive_.push_back(1);
        slot_cell_.emplace_back();
        in_cell_.push_back(0);
        if (m_ == 2) by_f1_.emplace(f[0], s);
        else if (use_tree_) tree_.insert(s, f);
        else
            for (std::size_t j = 0; j < m_; ++j) cols_[j].push_back(f[j]);
        ++live_;
        return s;
    }

    void forget_f1(std::size_t s)
    {
        if (m_ != 2) return;
        auto range = by_f1_.equal_range(F_[s * m_]);
        for (auto it = range.first; it != range.second; ++it)
            if (it->second == s) { by_f1_.erase(it); return; }
    }

    void remove(std::size_t s)
    {
        alive_[s] = 0;
        --live_;
        forget_f1(s);
        if (use_tree_ && m_ > 2) tree_.erase(s);
        if (in_cell_[s]) {
            auto it = cell_.find(slot_cell_[s]);
            if (it != cell_.end() && it->second == s) cell_.erase(it);
            in_cell_[s] = 0;
        }
        for (auto& e : ext_)
            if (e == static_cast<long long>(s)) e = -1;
    }

    void compact()
    {
        std::vector<std::size_t> remap(n_, n_);
        std::size_t k = 0;
        for (std::size_t s = 0; s < n_; ++s) {
            if (!alive_[s]) continue;
            remap[s] = k;
            std::copy(F_.begin() + static_cast<std::ptrdiff_t>(s * m_),
                      F_.begin() + static_cast<std::ptrdiff_t>((s + 1) * m_),
                      F_.begin() + static_cast<std::ptrdiff_t>(k * m_));
            std::copy(X_.begin() + static_cast<std::ptrdiff_t>(s * d_),
                      X_.begin() + static_cast<std::ptrdiff_t>((s + 1) * d_),
                      X_.begin() + static_cast<std::ptrdiff_t>(k * d_));
            slot_cell_[k] = slot_cell_[s];
            in_cell_[k] = in_cell_[s];
            ++k;
        }
        F_.resize(k * m_);
        X_.resize(k * d_);
        alive_.assign(k, char{1});
        slot_cell_.resize(k);
        in_cell_.resize(k);
        n_ = k;
        cell_.clear();
        for (std::size_t s = 0; s < k; ++s)
            if (in_cell_[s]) cell_[slot_cell_[s]] = s;
        for (auto& e : ext_)
            e = e >= 0 ? static_cast<long long>(remap[static_cast<std::size_t>(e)]) : -1;
        by_f1_.clear();
        if (m_ == 2)
            for (std::size_t s = 0; s < k; ++s) by_f1_.emplace(F_[s * m_], s);
        else if (use_tree_) {
            tree_.clear();                        // the slots are numbered anew
            for (std::size_t s = 0; s < k; ++s) tree_.insert(s, F_.data() + s * m_);
        } else
            for (std::size_t j = 0; j < m_; ++j) {
                cols_[j].resize(k);
                for (std::size_t s = 0; s < k; ++s) cols_[j][s] = F_[s * m_ + j];
            }
    }

    std::vector<std::int64_t> key(const double* f) const
    {
        std::vector<std::int64_t> c(m_);
        for (std::size_t j = 0; j < m_; ++j) {
            const double v = std::floor((f[j] - lo_[j]) / span_[j] / delta_);
            c[j] = (v >= -9223372036854775808.0 && v < 9223372036854775808.0)
                       ? static_cast<std::int64_t>(v)
                       : std::numeric_limits<std::int64_t>::min();
        }
        return c;
    }

    double corner(const double* f, const std::vector<std::int64_t>& c) const
    {
        double buf[64] = {};                 // gcc cannot see that t[0..m) is written below
        std::vector<double> big;
        double* t = buf;
        if (m_ > 64) { big.resize(m_); t = big.data(); }
        for (std::size_t j = 0; j < m_; ++j) {
            const double g = (f[j] - lo_[j]) / span_[j];
            const double d = g - static_cast<double>(c[j]) * delta_;
            t[j] = d * d;
        }
        return np::sum(t, m_);
    }

    // Put stored slot s into its cell; false (and s removed) if it loses.
    bool place(std::size_t s)
    {
        std::vector<std::int64_t> c = key(f_of(s));
        auto it = cell_.find(c);
        if (it == cell_.end()) {
            slot_cell_[s] = c;
            in_cell_[s] = 1;
            cell_.emplace(std::move(c), s);
            return true;
        }
        const std::size_t q = it->second;
        if (corner(f_of(s), c) < corner(f_of(q), c)) {
            remove(q);
            slot_cell_[s] = c;
            in_cell_[s] = 1;
            cell_[std::move(c)] = s;
            return true;
        }
        remove(s);
        return false;
    }

    // Re-estimate the frame from the stored points and rebuild every cell;
    // `exclude` is the slot the caller places itself.
    void regrid(long long exclude)
    {
        const double inf = std::numeric_limits<double>::infinity();
        std::vector<double> mn(m_, inf), mx(m_, -inf);
        for (std::size_t s = 0; s < n_; ++s) {
            if (!alive_[s]) continue;
            for (std::size_t j = 0; j < m_; ++j) {
                mn[j] = std::min(mn[j], F_[s * m_ + j]);
                mx[j] = std::max(mx[j], F_[s * m_ + j]);
            }
        }
        lo_ = mn;
        span_.resize(m_);
        for (std::size_t j = 0; j < m_; ++j) {
            const double sp = mx[j] - mn[j];
            span_[j] = sp > 0.0 ? sp : 1.0;
        }
        cell_.clear();
        std::fill(in_cell_.begin(), in_cell_.end(), char{0});
        std::vector<long long> ext = ext_;
        for (std::size_t s = 0; s < n_; ++s) {
            if (!alive_[s] || static_cast<long long>(s) == exclude) continue;
            if (std::find(ext.begin(), ext.end(), static_cast<long long>(s)) != ext.end()) continue;
            place(s);
        }
    }

    bool frame_drifted(const double* f) const
    {
        if (lo_.empty()) return true;
        for (std::size_t j = 0; j < m_; ++j) {
            const double tol = 0.25 * span_[j];
            if (f[j] < lo_[j] - tol) return true;
        }
        for (std::size_t j = 0; j < m_; ++j) {
            const double tol = 0.25 * span_[j];
            if (f[j] > lo_[j] + span_[j] + tol) return true;
        }
        return false;
    }
};

}   // namespace mootation

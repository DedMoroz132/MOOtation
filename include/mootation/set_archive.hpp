#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// The candidates of the second answer (python/mootation/run/archive.py,
// SetArchive; task 15, §6 and task 16, supplement 1, §3): every point offered
// that no point offered strictly dominates, points with equal objectives and
// other variables kept apart. The same points in the same order as the NumPy
// SetArchive, which compares each point with every stored one; here:
//
//   GROUPS. The stored points with one objective vector form a group; the
//   groups' vectors are mutually nondominated. A new point f with the vector
//   of a group joins it (nothing strictly dominates it: nothing dominates the
//   group). Otherwise f is refused if a group strictly dominates it, and else
//   stored as a new group, every group it strictly dominates removed.
//   M = 2: the groups sorted by f1 (then f2 falls strictly), found by binary
//   search: the group at or after f1 and the one before.
//   M >= 3: the groups in an ND-tree (nd_tree.hpp), which tells whether some
//   group weakly dominates f — the same vector, or one that dominates it — and
//   else which groups f weakly dominates.
//   A point with a NaN value is compared by nothing (the NumPy code's
//   comparisons are false): it is kept, and removes nothing.
//
//   ORDER. points() gives the stored points in the order they were offered,
//   as the NumPy code appends them.
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <map>
#include <vector>

#include "nd_tree.hpp"

namespace mootation {

class SetArchive {
public:
    SetArchive(std::size_t n_obj, std::size_t n_vars)
        : m_(n_obj), n_(n_vars), tree_(n_obj) {}

    std::size_t n_obj() const { return m_; }
    std::size_t n_vars() const { return n_; }
    std::size_t size() const { return alive_; }

    // Offer one point (n_vars values of x, or none); true if it is stored.
    bool add(const double* f, const double* x)
    {
        for (std::size_t k = 0; k < m_; ++k)
            if (std::isnan(f[k])) {
                store(f, x, kLoose);
                return true;
            }
        std::vector<double> v(f, f + m_);
        auto same = by_vector_.find(v);
        if (same != by_vector_.end()) {
            store(f, x, same->second);
            return true;
        }
        std::vector<std::size_t> beaten;
        if (m_ == 2) {
            if (!sorted_insert(f, beaten)) return false;
        } else {
            if (tree_.covered(f, &beaten)) return false;   // a group dominates f
        }
        for (std::size_t g : beaten) drop_group(g);
        const std::size_t g = groups_.size();
        groups_.push_back(Group{v, {}, true});
        by_vector_.emplace(std::move(v), g);
        if (m_ == 2) {
            auto at = std::lower_bound(order_.begin(), order_.end(), f[0],
                                       [&](std::size_t a, double f1) { return groups_[a].f[0] < f1; });
            order_.insert(at, g);
        } else {
            tree_.insert(g, f);
        }
        store(f, x, g);
        return true;
    }

    // The stored points, in the order they were offered.
    void points(std::vector<double>& F, std::vector<double>& X) const
    {
        F.clear();
        X.clear();
        for (const auto& p : points_) {
            if (!p.alive) continue;
            F.insert(F.end(), p.f.begin(), p.f.end());
            X.insert(X.end(), p.x.begin(), p.x.end());
        }
    }

private:
    static constexpr std::size_t kLoose = static_cast<std::size_t>(-1);

    struct Group {
        std::vector<double> f;
        std::vector<std::size_t> members;    // indices into points_
        bool live;
    };
    struct Point {
        std::vector<double> f, x;
        bool alive;
    };

    void store(const double* f, const double* x, std::size_t g)
    {
        if (g != kLoose) groups_[g].members.push_back(points_.size());
        points_.push_back(Point{std::vector<double>(f, f + m_),
                                x ? std::vector<double>(x, x + n_) : std::vector<double>(n_, 0.0),
                                true});
        ++alive_;
    }

    void drop_group(std::size_t g)
    {
        Group& gr = groups_[g];
        for (std::size_t i : gr.members) {
            points_[i].alive = false;
            --alive_;
        }
        gr.live = false;
        by_vector_.erase(gr.f);
        if (m_ == 2)
            order_.erase(std::find(order_.begin(), order_.end(), g));
        else
            tree_.erase(g);
    }

    // M = 2: false if a group strictly dominates f (no group has f's vector);
    // else the groups f strictly dominates, in beaten.
    bool sorted_insert(const double* f, std::vector<std::size_t>& beaten) const
    {
        beaten.clear();
        auto at = std::lower_bound(order_.begin(), order_.end(), f[0],
                                   [&](std::size_t a, double f1) { return groups_[a].f[0] < f1; });
        if (at != order_.end() && groups_[*at].f[0] == f[0] && groups_[*at].f[1] < f[1])
            return false;                                   // the same f1, a smaller f2
        if (at != order_.begin() && groups_[*(at - 1)].f[1] <= f[1])
            return false;                                   // a smaller f1, no larger f2
        for (auto it = at; it != order_.end() && groups_[*it].f[1] >= f[1]; ++it)
            beaten.push_back(*it);                          // f1 >= f's, f2 >= f's, not equal
        return true;
    }

    std::size_t m_, n_;
    std::size_t alive_ = 0;
    std::vector<Point> points_;
    std::vector<Group> groups_;
    std::map<std::vector<double>, std::size_t> by_vector_;   // a live group's vector -> group
    std::vector<std::size_t> order_;                          // M = 2: live groups by f1
    nd::NDTree tree_;                                         // M >= 3: live groups
};

}  // namespace mootation

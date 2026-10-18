#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// The nondominated rows of a point set (minimisation), for the Python run
// layer: python/mootation/run/metrics.py `nondominated` and `nondominated_mask`
// call these when the extension is built, and keep their NumPy loops as the
// reference the tests compare with.
//
// unique_rows(F): the distinct rows in lexicographic order, as np.unique(F,
// axis=0) gives them: rows compared value by value, a NaN after every number
// and equal to another NaN in the order; two rows are the same row when every
// value is == (so a row with a NaN is never the same as another, and 0.0 and
// -0.0 are the same: of two such rows the first in F is kept, where NumPy's
// sort keeps either).
//
// filter(F): the rows of unique_rows(F) that no other row dominates — the
// metrics.nondominated set, in the same order. In lexicographic order only an
// earlier row can dominate a later one, so one pass decides every row against
// the nondominated rows before it:
//   * two objectives: a later row is dominated exactly when its second value
//     is not below the smallest second value so far (O(n log n) with the sort);
//   * three objectives: against the staircase of the earlier rows' (f2, f3),
//     kept in a balanced tree (Kung, Luccio & Preparata, J. ACM 22(4), 1975;
//     O(n log n));
//   * more: against the nondominated rows so far in an ND-tree (nd_tree.hpp;
//     task 11 of 2026-10-18, §2), which answers whether one of them dominates
//     the row as comparing it with each of them would.
// A row with a NaN neither dominates nor is dominated (every comparison with
// NaN is false), as in the NumPy loop: it is kept, and left out of the pass.
//
// mask(F): for every row of F, whether no row dominates it — copies all kept,
// metrics.nondominated_mask.
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <map>
#include <vector>

#include "nd_tree.hpp"

namespace mootation::nd {

// NumPy's order of two values in a sort: a NaN after every number.
inline int compare_value(double a, double b)
{
    if (a < b) return -1;
    if (a > b) return 1;
    if (a == b) return 0;
    const bool na = std::isnan(a), nb = std::isnan(b);
    return na == nb ? 0 : (na ? 1 : -1);
}

// Indices of the distinct rows of the n x m row-major array F, in lexicographic
// order (the module comment).
inline std::vector<std::size_t> unique_rows(const double* F, std::size_t n, std::size_t m)
{
    std::vector<std::size_t> idx(n);
    for (std::size_t i = 0; i < n; ++i) idx[i] = i;
    std::stable_sort(idx.begin(), idx.end(), [&](std::size_t a, std::size_t b) {
        for (std::size_t j = 0; j < m; ++j) {
            const int c = compare_value(F[a * m + j], F[b * m + j]);
            if (c != 0) return c < 0;
        }
        return false;
    });
    std::vector<std::size_t> out;
    out.reserve(n);
    for (std::size_t t = 0; t < n; ++t) {
        if (t > 0) {
            const double* p = F + idx[t - 1] * m;
            const double* q = F + idx[t] * m;
            bool same = true;
            for (std::size_t j = 0; j < m && same; ++j) same = p[j] == q[j];
            if (same) continue;
        }
        out.push_back(idx[t]);
    }
    return out;
}

// Does row p weakly dominate row q (p <= q everywhere)? Between distinct rows
// without NaN that is dominance.
inline bool weakly_dominates(const double* p, const double* q, std::size_t m)
{
    for (std::size_t j = 0; j < m; ++j)
        if (!(p[j] <= q[j])) return false;
    return true;
}

inline bool has_nan(const double* f, std::size_t m)
{
    for (std::size_t j = 0; j < m; ++j)
        if (std::isnan(f[j])) return true;
    return false;
}

// Which of the rows `rows` (distinct, in lexicographic order) no other of them
// dominates: one flag per entry of rows.
inline std::vector<char> nondominated_sorted(const double* F, std::size_t m,
                                             const std::vector<std::size_t>& rows)
{
    const std::size_t n = rows.size();
    std::vector<char> keep(n, 1);
    if (m == 0) return keep;
    if (m == 1) {
        // the smallest value alone, every row with a NaN besides
        bool first = true;
        for (std::size_t t = 0; t < n; ++t) {
            const double* f = F + rows[t] * m;
            if (has_nan(f, m)) continue;
            keep[t] = first ? 1 : 0;
            first = false;
        }
        return keep;
    }
    if (m == 2) {
        bool any = false;
        double best = 0.0;                        // the smallest f2 so far
        for (std::size_t t = 0; t < n; ++t) {
            const double* f = F + rows[t] * m;
            if (has_nan(f, m)) continue;
            if (any && !(f[1] < best)) { keep[t] = 0; continue; }
            best = f[1];
            any = true;
        }
        return keep;
    }
    if (m == 3) {
        // f2 -> f3 of the earlier nondominated rows, f3 falling as f2 rises:
        // the smallest f3 among the rows with f2 <= q2 is that of the last
        // entry at or below q2.
        std::map<double, double> stair;
        for (std::size_t t = 0; t < n; ++t) {
            const double* f = F + rows[t] * m;
            if (has_nan(f, m)) continue;
            auto it = stair.upper_bound(f[1]);
            if (it != stair.begin() && std::prev(it)->second <= f[2]) { keep[t] = 0; continue; }
            // the entries q now covers in (f2, f3) leave the staircase
            auto from = stair.lower_bound(f[1]);
            auto to = from;
            while (to != stair.end() && to->second >= f[2]) ++to;
            stair.erase(from, to);
            stair.emplace(f[1], f[2]);
        }
        return keep;
    }
    NDTree front(m);                              // the nondominated rows so far
    for (std::size_t t = 0; t < n; ++t) {
        const double* f = F + rows[t] * m;
        if (has_nan(f, m)) continue;
        if (front.covered(f, nullptr)) { keep[t] = 0; continue; }
        front.insert(t, f);
    }
    return keep;
}

// Indices of the rows metrics.nondominated keeps, in its order (the module
// comment): one per distinct nondominated row.
inline std::vector<std::size_t> filter(const double* F, std::size_t n, std::size_t m)
{
    const std::vector<std::size_t> rows = unique_rows(F, n, m);
    const std::vector<char> keep = nondominated_sorted(F, m, rows);
    std::vector<std::size_t> out;
    for (std::size_t t = 0; t < rows.size(); ++t)
        if (keep[t]) out.push_back(rows[t]);
    return out;
}

// For every row of F: no row dominates it (copies of a nondominated row all
// kept), metrics.nondominated_mask.
inline std::vector<char> mask(const double* F, std::size_t n, std::size_t m)
{
    std::vector<std::size_t> idx(n);
    for (std::size_t i = 0; i < n; ++i) idx[i] = i;
    std::stable_sort(idx.begin(), idx.end(), [&](std::size_t a, std::size_t b) {
        for (std::size_t j = 0; j < m; ++j) {
            const int c = compare_value(F[a * m + j], F[b * m + j]);
            if (c != 0) return c < 0;
        }
        return false;
    });
    // the first row of every run of equal rows stands for the run
    std::vector<std::size_t> rows;
    std::vector<std::size_t> run_of(n);
    for (std::size_t t = 0; t < n; ++t) {
        bool same = t > 0;
        if (same) {
            const double* p = F + idx[t - 1] * m;
            const double* q = F + idx[t] * m;
            for (std::size_t j = 0; j < m && same; ++j) same = p[j] == q[j];
        }
        if (!same) rows.push_back(idx[t]);
        run_of[idx[t]] = rows.size() - 1;
    }
    const std::vector<char> keep = nondominated_sorted(F, m, rows);
    std::vector<char> out(n);
    for (std::size_t i = 0; i < n; ++i) out[i] = keep[run_of[i]];
    return out;
}

}   // namespace mootation::nd

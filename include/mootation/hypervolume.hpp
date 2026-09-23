#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// HYPERVOLUME — exact, by the WFG algorithm.
// L. While, L. Bradstreet, L. Barone, "A Fast Way of Calculating Exact
// Hypervolumes", IEEE TEVC 16(1):86-95, 2012.
// doi:10.1109/TEVC.2010.2077298          (source: while2012)
//
// The hypervolume of a set is a sum of exclusive hypervolumes (Eq. 3),
//   Hyp({p_1, ..., p_m}) = sum_i ExcHyp(p_i, {p_i+1, ..., p_m}),
// and an exclusive hypervolume is the inclusive one minus the hypervolume of
// the underlying set LIMITED by the contributing point — each objective
// replaced by the worse of the two values — with its dominated points
// discarded (§III, Eqs. 5-7; Fig. 5):
//   ExcHyp(p, S) = Hyp({p}) − Hyp(nds({limit(s, p) | s in S})).
// With both optimizations of §IV-B:
//   1) SORTING: the points are ordered so that they improve monotonically in
//      one objective (here the last), so that the worse contributing points,
//      which limit their underlying sets more, are processed against the
//      larger sets, and the better ones against the smaller;
//   2) SLICING in that objective, as in HSO: every later point is no worse
//      than p_i in the sorted objective, so the region p_i dominates alone is
//      a prism of height (r_n − p_i[n]) over its exclusive hypervolume in the
//      other n − 1 objectives, and the recursion runs in one objective fewer,
//      down to the two-objective base case — a sweep over the points sorted
//      in one objective (§IV-B.2).
// The alternative three-objective base case (Beume et al.) is not used: the
// paper measures it as almost no faster than slicing to two (Fig. 7).
//
// Minimisation, so "worse" is the larger value. A point that is not strictly
// better than the reference point in every objective dominates no volume and
// is dropped, as is a row with a NaN. Duplicates are dropped too: equal
// points do not dominate each other, and every copy would branch the
// recursion.
//
// API:
//   double v = hypervolume::wfg(F, n, m, ref);   // F: n rows of m, row-major
//   double v = hypervolume::wfg(rows, ref);      // vector<vector<double>>
//   size_t c = hypervolume::covered(F, n, m, S, k);
//     how many of the k rows of S the rows of F weakly dominate: the count
//     behind a Monte-Carlo estimate, box volume * c / k. The caller draws the
//     samples, so an estimate is the same whichever language counted it.
// ============================================================================

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace mootation {
namespace hypervolume {

namespace detail {

class Wfg {
public:
    Wfg(const double* ref, std::size_t m)
        : ref_(ref, ref + m), lim_(m + 1), tmp_(m + 1), idx_(m + 1), alive_(m + 1) {}

    // The rows of P (k of d coordinates) that no other row weakly dominates,
    // the first of equal rows kept; they are moved to the front of P and
    // their number is returned. A row is compared with the rows kept so far:
    // one of them no worse in every objective drops it, and it drops every
    // kept row it is no worse than.
    std::size_t nds(std::vector<double>& P, std::size_t k, std::size_t d) {
        std::vector<char>& alive = alive_[d];
        alive.assign(k, 0);
        double* base = P.data();
        for (std::size_t a = 0; a < k; ++a) {
            const double* x = base + a * d;
            bool dropped = false;
            for (std::size_t b = 0; b < a && !dropped; ++b) {
                if (!alive[b]) continue;
                const double* y = base + b * d;
                bool y_le = true, x_le = true;           // y <= x, x <= y in every objective
                for (std::size_t j = 0; j < d && (y_le || x_le); ++j) {
                    y_le = y_le && y[j] <= x[j];
                    x_le = x_le && x[j] <= y[j];
                }
                if (y_le) dropped = true;                // dominated, or equal to a kept row
                else if (x_le) alive[b] = 0;
            }
            alive[a] = !dropped;
        }
        std::size_t kept = 0;
        for (std::size_t a = 0; a < k; ++a) {
            if (!alive[a]) continue;
            if (kept != a) std::copy(base + a * d, base + a * d + d, base + kept * d);
            ++kept;
        }
        return kept;
    }

    // Hypervolume of the k rows of P (d coordinates each: nondominated,
    // distinct, strictly better than the reference point) against ref_[0..d).
    double hv(std::vector<double>& P, std::size_t k, std::size_t d) {
        if (k == 0) return 0.0;
        // one or two points: the inclusive volumes, less the overlap of two
        // (Eq. 3 written out; no recursion needed)
        if (k == 1) return incl(P.data(), d);
        if (k == 2) {
            const double* a = P.data();
            const double* b = a + d;
            double both = 1.0;
            for (std::size_t j = 0; j < d; ++j) both *= ref_[j] - std::max(a[j], b[j]);
            return incl(a, d) + incl(b, d) - both;
        }
        if (d == 2) return hv2(P, k);
        sort_worst_first(P, k, d);
        const std::size_t e = d - 1;                  // the sliced objective
        std::vector<double>& L = lim_[e];
        double total = 0.0;
        for (std::size_t i = 0; i < k; ++i) {
            const double* p = P.data() + i * d;
            double excl = incl(p, e);                 // inclusive, in e objectives
            if (i + 1 < k) {
                // limitset: the later rows, each objective the worse of the two
                L.resize((k - i - 1) * e);
                double* out = L.data();
                for (std::size_t s = i + 1; s < k; ++s) {
                    const double* q = P.data() + s * d;
                    for (std::size_t j = 0; j < e; ++j) *out++ = std::max(p[j], q[j]);
                }
                const std::size_t n = nds(L, k - i - 1, e);
                excl -= hv(L, n, e);
            }
            total += (ref_[e] - p[e]) * excl;
        }
        return total;
    }

private:
    std::vector<double> ref_;
    std::vector<std::vector<double>> lim_;             // limited sets, by dimension
    std::vector<std::vector<double>> tmp_;             // scratch, by dimension
    std::vector<std::vector<std::size_t>> idx_;        // sort orders, by dimension
    std::vector<std::vector<char>> alive_;             // nds marks, by dimension
    std::vector<std::pair<double, double>> pairs_;     // the base case

    // the inclusive hypervolume of p in its first d objectives
    double incl(const double* p, std::size_t d) const {
        double v = 1.0;
        for (std::size_t j = 0; j < d; ++j) v *= ref_[j] - p[j];
        return v;
    }

    // the rows of P, worst first in the last objective; ties (frequent in a
    // limited set, whose rows share the limiting point's values) worst first
    // in the objective before it, and so on
    void sort_worst_first(std::vector<double>& P, std::size_t k, std::size_t d) {
        std::vector<std::size_t>& idx = idx_[d];
        idx.resize(k);
        std::iota(idx.begin(), idx.end(), std::size_t{0});
        const double* base = P.data();
        std::sort(idx.begin(), idx.end(), [base, d](std::size_t a, std::size_t b) {
            const double* x = base + a * d;
            const double* y = base + b * d;
            for (std::size_t j = d; j-- > 0;)
                if (x[j] != y[j]) return x[j] > y[j];
            return false;
        });
        std::vector<double>& out = tmp_[d];
        out.resize(k * d);
        for (std::size_t r = 0; r < k; ++r)
            std::copy(base + idx[r] * d, base + idx[r] * d + d, out.data() + r * d);
        P.swap(out);
    }

    // two objectives: a sweep in increasing f_1, adding the strip each row
    // adds below the lowest f_2 so far
    double hv2(const std::vector<double>& P, std::size_t k) {
        pairs_.resize(k);
        for (std::size_t i = 0; i < k; ++i) pairs_[i] = {P[2 * i], P[2 * i + 1]};
        std::sort(pairs_.begin(), pairs_.end());
        double area = 0.0, low = ref_[1];
        for (const auto& q : pairs_) {
            if (q.second < low) {
                area += (ref_[0] - q.first) * (low - q.second);
                low = q.second;
            }
        }
        return area;
    }
};

}  // namespace detail

// Exact hypervolume (minimisation) of the n rows of F (m objectives each,
// row-major) against the reference point ref.
inline double wfg(const double* F, std::size_t n, std::size_t m, const double* ref) {
    if (m == 0) throw std::invalid_argument("hypervolume: no objectives");
    std::vector<double> P;
    P.reserve(n * m);
    std::size_t k = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const double* f = F + i * m;
        std::size_t j = 0;
        while (j < m && f[j] < ref[j]) ++j;          // false for NaN as well
        if (j == m) {
            P.insert(P.end(), f, f + m);
            ++k;
        }
    }
    if (k == 0) return 0.0;
    detail::Wfg w(ref, m);
    k = w.nds(P, k, m);
    return w.hv(P, k, m);
}

// How many of the k rows of S (m coordinates each, row-major) at least one of
// the n rows of F weakly dominates (no worse in every objective). The rows of
// F are visited in increasing first objective, so a sample stops at the first
// row that covers it or at the first whose f_1 already exceeds its own.
inline std::size_t covered(const double* F, std::size_t n, std::size_t m,
                           const double* S, std::size_t k) {
    if (m == 0) throw std::invalid_argument("hypervolume: no objectives");
    std::vector<std::size_t> order(n);
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::sort(order.begin(), order.end(),
              [F, m](std::size_t a, std::size_t b) { return F[a * m] < F[b * m]; });
    std::vector<double> P(n * m);
    for (std::size_t r = 0; r < n; ++r)
        std::copy(F + order[r] * m, F + order[r] * m + m, P.data() + r * m);
    std::size_t hits = 0;
    for (std::size_t i = 0; i < k; ++i) {
        const double* s = S + i * m;
        for (std::size_t r = 0; r < n; ++r) {
            const double* p = P.data() + r * m;
            if (!(p[0] <= s[0])) break;              // sorted: no later row covers s
            std::size_t j = 1;
            while (j < m && p[j] <= s[j]) ++j;
            if (j == m) {
                ++hits;
                break;
            }
        }
    }
    return hits;
}

inline double wfg(const std::vector<std::vector<double>>& rows, const std::vector<double>& ref) {
    const std::size_t m = ref.size();
    std::vector<double> F;
    F.reserve(rows.size() * m);
    for (const auto& r : rows) {
        if (r.size() != m) throw std::invalid_argument("hypervolume: row size != reference size");
        F.insert(F.end(), r.begin(), r.end());
    }
    return wfg(F.data(), rows.size(), m, ref.data());
}

}  // namespace hypervolume
}  // namespace mootation

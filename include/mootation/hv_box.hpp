#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// The hypervolume by box decomposition: R. Lacour, K. Klamroth & C.M.
// Fonseca, "A box decomposition algorithm to compute the hypervolume
// indicator", Computers & Operations Research 79:347-360, 2017,
// doi:10.1016/j.cor.2016.06.021 (source hv-box-decomposition_lacour2017_arxiv;
// task 11 of 2026-10-18, §6.1) — the hv_algo = "box" of the cores that
// select by hypervolume and of metrics.hypervolume; "wfg" (hypervolume.hpp)
// stays their default.
//
// THE DECOMPOSITION (§2.1, after Kaplan et al. 2008). The region the points
// dominate, below the reference point r, is the disjoint union of the boxes
//   B(u) = [z^1_1(u), r_1] x prod_{j>=2} [max_{k<j} z^k_j(u), u_j)
// over the local upper bounds u of the points (the maximal points of the
// search region below r, property P1-P2), z^j(u) the point that defines u's
// j-th value (z^j_j(u) = u_j), among the points and the dummies
// zhat^j = (r_j, lo_{-j}), lo below every point. The hypervolume is the sum
// of their volumes.
//
// THE LOCAL UPPER BOUNDS (§2.2.2, Algorithm 2, HBDA-NI): the points taken in
// increasing last objective, starting from {r}; a point zbar replaces every
// bound u it strictly dominates (A) by (zbar_p, u_-p), p the last objective,
// and by (zbar_j, u_-j) for every other j with zbar_j >= max_{k != j}
// z^k_j(u) (Theorem 2.1, Klamroth et al. 2015), zbar defining the replaced
// value. A bound whose last value is no longer r_p is never replaced again
// (later points are not below it there): its box is summed and it is
// dropped. Ties (§2.4): the order of the points breaks them, the current
// point the last, so the >= of the condition holds as written.
//
// THE CONTRIBUTIONS (hv_contribution.hpp, contributions_box): as there,
// Hyp({p}) less the hypervolume of p's limited set, this one by boxes. The
// sums are another order of the same volume: the values agree with WFG's to
// the last bits, not bit for bit, and two nearly equal contributions may
// compare the other way (a selection by them may then take another point).
// ============================================================================

#include <algorithm>
#include <cstddef>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace mootation::hypervolume {

namespace detail {

// The box decomposition of k mutually nondominated rows of P (m >= 2), each
// strictly below ref in every objective.
class Boxes {
public:
    Boxes(const double* ref, std::size_t m) : ref_(ref, ref + m), m_(m) {}

    double hv(const double* P, std::size_t k)
    {
        const std::size_t m = m_, last = m - 1;
        // the points in increasing last objective (stable: ties keep their order)
        order_.resize(k);
        std::iota(order_.begin(), order_.end(), std::size_t{0});
        std::stable_sort(order_.begin(), order_.end(), [P, m, last](std::size_t a, std::size_t b) {
            return P[a * m + last] < P[b * m + last];
        });
        // the points, then the m dummies: zhat^j = (r_j, lo_-j)
        pts_.assign((k + m) * m, 0.0);
        std::vector<double> lo(m, std::numeric_limits<double>::infinity());
        for (std::size_t t = 0; t < k; ++t)
            for (std::size_t j = 0; j < m; ++j) {
                const double v = P[order_[t] * m + j];
                pts_[t * m + j] = v;
                lo[j] = std::min(lo[j], v);
            }
        for (std::size_t d = 0; d < m; ++d)
            for (std::size_t j = 0; j < m; ++j) pts_[(k + d) * m + j] = d == j ? ref_[j] : lo[j];
        // the working bounds: u and its defining points, m of each per bound
        u_.assign(ref_.begin(), ref_.end());
        def_.resize(m);
        for (std::size_t d = 0; d < m; ++d) def_[d] = k + d;
        double total = 0.0;
        for (std::size_t t = 0; t < k; ++t) {
            const double* z = pts_.data() + t * m;
            nu_.clear();
            ndef_.clear();
            const std::size_t nb = u_.size() / m;
            for (std::size_t b = 0; b < nb; ++b) {
                const double* u = u_.data() + b * m;
                const std::size_t* dz = def_.data() + b * m;
                bool below = true;                                    // z < u: u is in A
                for (std::size_t j = 0; j < m && below; ++j) below = z[j] < u[j];
                if (!below) {                                         // in A-bar: kept
                    nu_.insert(nu_.end(), u, u + m);
                    ndef_.insert(ndef_.end(), dz, dz + m);
                    continue;
                }
                // (z_p, u_-p): its last value below r_p, never replaced again
                cu_.assign(u, u + m);
                cd_.assign(dz, dz + m);
                cu_[last] = z[last];
                cd_[last] = t;
                total += volume(cu_.data(), cd_.data());
                // (z_j, u_-j), j < p, where z_j >= max_{k != j} z^k_j(u)
                for (std::size_t j = 0; j < last; ++j) {
                    double top = -std::numeric_limits<double>::infinity();
                    for (std::size_t q = 0; q < m; ++q)
                        if (q != j) top = std::max(top, pts_[dz[q] * m + j]);
                    if (!(z[j] >= top)) continue;
                    const std::size_t at = nu_.size();
                    nu_.insert(nu_.end(), u, u + m);
                    ndef_.insert(ndef_.end(), dz, dz + m);
                    nu_[at + j] = z[j];
                    ndef_[at + j] = t;
                }
            }
            u_.swap(nu_);
            def_.swap(ndef_);
        }
        const std::size_t nb = u_.size() / m;
        for (std::size_t b = 0; b < nb; ++b) total += volume(u_.data() + b * m, def_.data() + b * m);
        return total;
    }

private:
    std::vector<double> ref_;
    std::size_t m_;
    std::vector<std::size_t> order_;
    std::vector<double> pts_, u_, nu_, cu_;
    std::vector<std::size_t> def_, ndef_, cd_;

    // the volume of B(u), Eq. (2)
    double volume(const double* u, const std::size_t* dz) const
    {
        const std::size_t m = m_;
        double v = ref_[0] - u[0];
        for (std::size_t j = 1; j < m && v > 0.0; ++j) {
            double lo = -std::numeric_limits<double>::infinity();
            for (std::size_t q = 0; q < j; ++q) lo = std::max(lo, pts_[dz[q] * m + j]);
            const double w = u[j] - lo;
            v = w > 0.0 ? v * w : 0.0;
        }
        return v > 0.0 ? v : 0.0;
    }
};

// The rows of P (k of them, m values each) that no other row dominates, one
// of each set of copies, in their order.
inline std::size_t stable_rows(std::vector<double>& P, std::size_t k, std::size_t m)
{
    std::vector<char> keep(k, 1);
    for (std::size_t a = 0; a < k; ++a) {
        if (!keep[a]) continue;
        const double* x = P.data() + a * m;
        for (std::size_t b = 0; b < k && keep[a]; ++b) {
            if (b == a || !keep[b]) continue;
            const double* y = P.data() + b * m;
            bool le = true, lt = false;
            for (std::size_t j = 0; j < m && le; ++j) {
                le = y[j] <= x[j];
                lt = lt || y[j] < x[j];
            }
            if (le && (lt || b < a)) keep[a] = 0;         // dominated, or a later copy
        }
    }
    std::size_t out = 0;
    for (std::size_t a = 0; a < k; ++a) {
        if (!keep[a]) continue;
        if (out != a) std::copy(P.begin() + static_cast<std::ptrdiff_t>(a * m),
                                P.begin() + static_cast<std::ptrdiff_t>((a + 1) * m),
                                P.begin() + static_cast<std::ptrdiff_t>(out * m));
        ++out;
    }
    P.resize(out * m);
    return out;
}

}   // namespace detail

// Exact hypervolume (minimisation) of the n rows of F (m objectives each,
// row-major) against ref, by box decomposition: wfg()'s value up to the
// order of the sums. The rows not strictly below ref in every objective count
// for nothing, as in wfg().
inline double box(const double* F, std::size_t n, std::size_t m, const double* ref)
{
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
    if (m == 1) {
        double best = P[0];
        for (std::size_t i = 1; i < k; ++i) best = std::min(best, P[i]);
        return ref[0] - best;
    }
    k = detail::stable_rows(P, k, m);
    detail::Boxes b(ref, m);
    return b.hv(P.data(), k);
}

inline double box(const std::vector<std::vector<double>>& rows, const std::vector<double>& ref)
{
    const std::size_t m = ref.size();
    std::vector<double> F;
    F.reserve(rows.size() * m);
    for (const auto& r : rows) {
        if (r.size() != m) throw std::invalid_argument("hypervolume: row size != reference size");
        F.insert(F.end(), r.begin(), r.end());
    }
    return box(F.data(), rows.size(), m, ref.data());
}

}   // namespace mootation::hypervolume

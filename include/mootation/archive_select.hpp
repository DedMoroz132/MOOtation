#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// The run layer's subset selection in C++: python/mootation/run/archive.py's
// almost_dominated, set_aside, dss_order, dplus_matrix, greedy_igdp and
// igdp_self_order, index for index. archive.py calls these when the extension
// is built and keeps its NumPy code as the reference the tests compare with;
// the procedures and their sources are described there.
//
// THE SAME NUMBERS. Every value is computed by the expression archive.py
// evaluates, element for element: the normalisation (f - lo) / span, a zero
// span counting as 1; the difference of two rows; max(·, 0) as np.maximum
// (a NaN passes); the square; the sum over the objectives and the mean over a
// reference in NumPy's pairwise order (numpy_sum.hpp); argmax and argmin as
// NumPy's (the first maximum or minimum, a NaN first of all); a set aside
// point as almost_dominated asks it. So the cover values, the gains and the
// indices are the same, bit for bit, ties included.
//
// This is NOT include/mootation/dss.hpp, the selection DMS and the inner
// archive make inside a core: that one sums the squares one after the other
// (the same values up to seven objectives) and knows only the set's own frame.
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <queue>
#include <utility>
#include <vector>

#include "numpy_sum.hpp"

namespace mootation::select {

inline constexpr double DRS_GAIN = 1e-3;   // archive.DRS_GAIN
inline constexpr double DRS_LOSS = 0.1;    // archive.DRS_LOSS

// A point set: n rows of m values, row-major.
struct Rows {
    std::vector<double> v;
    std::size_t n = 0, m = 0;
    const double* row(std::size_t i) const { return v.data() + i * m; }
    double* row(std::size_t i) { return v.data() + i * m; }
};

namespace detail {

// np.maximum(d, 0.0)
inline double pos(double d) { return std::isnan(d) ? d : (d >= 0.0 ? d : 0.0); }

// d+ = ||max(a - b, 0)||, the sum of squares in NumPy's order. `buf` holds m values.
inline double dplus(const double* a, const double* b, std::size_t m, double* buf)
{
    for (std::size_t j = 0; j < m; ++j) {
        const double t = pos(a[j] - b[j]);
        buf[j] = t * t;
    }
    return std::sqrt(np::sum(buf, m));
}

// np.minimum(a, b): a NaN on either side gives NaN
inline double minimum(double a, double b)
{
    if (std::isnan(a)) return a;
    if (std::isnan(b)) return b;
    return std::min(a, b);
}

// np.argmax of v (the first maximum; the first NaN if there is one)
inline std::size_t argmax(const std::vector<double>& v)
{
    std::size_t arg = 0;
    if (v.empty() || std::isnan(v[0])) return 0;
    double best = v[0];
    for (std::size_t i = 1; i < v.size(); ++i) {
        if (std::isnan(v[i])) return i;
        if (v[i] > best) { best = v[i]; arg = i; }
    }
    return arg;
}

// np.argmin of v (the first minimum; the first NaN if there is one)
inline std::size_t argmin(const std::vector<double>& v)
{
    std::size_t arg = 0;
    if (v.empty() || std::isnan(v[0])) return 0;
    double best = v[0];
    for (std::size_t i = 1; i < v.size(); ++i) {
        if (std::isnan(v[i])) return i;
        if (v[i] < best) { best = v[i]; arg = i; }
    }
    return arg;
}

// NumPy's sort order of two values: a NaN after every number
inline bool before(double a, double b)
{
    if (std::isnan(a)) return false;
    return std::isnan(b) || a < b;
}

}   // namespace detail

// archive.normalized: (F - lo) / span, span = hi - lo where positive, else 1.
inline Rows normalized(const Rows& F, const std::vector<double>& lo, const std::vector<double>& hi)
{
    Rows G = F;
    for (std::size_t j = 0; j < F.m; ++j) {
        const double span = hi[j] - lo[j];
        const double s = span > 0.0 ? span : 1.0;
        for (std::size_t i = 0; i < F.n; ++i) G.v[i * F.m + j] = (F.v[i * F.m + j] - lo[j]) / s;
    }
    return G;
}

// Is row p of G almost dominated by another row (archive.almost_dominated)?
// Some q != p with G_p - G_q >= -DRS_GAIN everywhere and max(G_p - G_q) >
// DRS_LOSS. Only the rows with flag `among` != 0 are asked as q (all if empty).
inline bool almost_dominated(const Rows& G, std::size_t p, const std::vector<char>& among = {})
{
    const double* gp = G.row(p);
    for (std::size_t q = 0; q < G.n; ++q) {
        if (q == p || (!among.empty() && !among[q])) continue;
        const double* gq = G.row(q);
        bool near = true;
        double loss = -std::numeric_limits<double>::infinity();
        for (std::size_t j = 0; j < G.m; ++j) {
            const double d = gp[j] - gq[j];
            if (!(d >= -DRS_GAIN)) { near = false; break; }
            loss = std::max(loss, d);
        }
        if (near && loss > DRS_LOSS) return true;
    }
    return false;
}

// archive.set_aside: G normalised and the kept flags. A missing lo or hi
// (empty vector) is the minimum or maximum of the points kept, taken again
// after every round that sets points aside, until a round sets none aside; a
// round never sets aside every point left.
inline std::pair<Rows, std::vector<char>> set_aside(const Rows& F, const std::vector<double>& ideal,
                                                    const std::vector<double>& nadir)
{
    std::vector<char> kept(F.n, 1);
    if (F.n == 0) return {F, kept};
    const double inf = std::numeric_limits<double>::infinity();
    for (;;) {
        std::vector<double> lo = ideal, hi = nadir;
        if (lo.empty() || hi.empty()) {
            // F[kept].min(0) / .max(0): NumPy's min and max pass a NaN on
            std::vector<double> mn(F.m, inf), mx(F.m, -inf);
            std::vector<char> nan_in(F.m, 0);
            for (std::size_t i = 0; i < F.n; ++i) {
                if (!kept[i]) continue;
                for (std::size_t j = 0; j < F.m; ++j) {
                    const double x = F.v[i * F.m + j];
                    if (std::isnan(x)) nan_in[j] = 1;
                    mn[j] = std::min(mn[j], x);
                    mx[j] = std::max(mx[j], x);
                }
            }
            for (std::size_t j = 0; j < F.m; ++j)
                if (nan_in[j]) mn[j] = mx[j] = std::numeric_limits<double>::quiet_NaN();
            if (lo.empty()) lo = mn;
            if (hi.empty()) hi = mx;
        }
        Rows G = normalized(F, lo, hi);
        std::vector<char> drop(F.n, 0);
        std::size_t n_kept = 0, n_drop = 0;
        for (std::size_t p = 0; p < F.n; ++p) {
            if (!kept[p]) continue;
            ++n_kept;
            if (almost_dominated(G, p, kept)) { drop[p] = 1; ++n_drop; }
        }
        if (n_drop == 0 || n_drop == n_kept) return {G, kept};
        for (std::size_t i = 0; i < F.n; ++i)
            if (drop[i]) kept[i] = 0;
    }
}

// archive.dss_order: the first k indices of F in DSS order, in the frame
// [ideal, nadir] (an empty vector: the points' own, as set_aside takes it).
inline std::vector<std::size_t> dss_order(const Rows& F, std::size_t k,
                                          const std::vector<double>& ideal = {},
                                          const std::vector<double>& nadir = {})
{
    const std::size_t n = F.n, m = F.m;
    k = std::min(k, n);
    std::vector<std::size_t> order;
    if (k == 0) return order;
    Rows G;
    std::vector<signed char> state(n, 0);           // 1 kept, -1 set aside, 0 not known yet
    if (!ideal.empty() && !nadir.empty()) {
        G = normalized(F, ideal, nadir);            // whether a point is set aside is asked lazily
    } else {
        auto got = set_aside(F, ideal, nadir);
        G = std::move(got.first);
        for (std::size_t i = 0; i < n; ++i) state[i] = got.second[i] ? 1 : -1;
    }
    auto is_kept = [&](std::size_t i) {
        if (state[i] == 0) state[i] = almost_dominated(G, i) ? -1 : 1;
        return state[i] == 1;
    };
    std::vector<char> taken(n, 0);
    std::vector<double> cover(n, std::numeric_limits<double>::infinity());
    std::vector<double> buf(m);
    auto take = [&](std::size_t i) {
        taken[i] = 1;
        order.push_back(i);
        const double* gi = G.row(i);
        for (std::size_t c = 0; c < n; ++c)
            cover[c] = detail::minimum(cover[c], detail::dplus(gi, G.row(c), m, buf.data()));
    };
    // the best kept point of every objective: by value, NaN last, ties by index
    std::vector<std::size_t> by(n);
    for (std::size_t j = 0; j < m; ++j) {
        if (order.size() >= k) break;
        for (std::size_t i = 0; i < n; ++i) by[i] = i;
        std::stable_sort(by.begin(), by.end(), [&](std::size_t a, std::size_t b) {
            return detail::before(G.v[a * m + j], G.v[b * m + j]);
        });
        for (std::size_t i : by) {
            if (is_kept(i)) {
                if (!taken[i]) take(i);
                break;
            }
        }
    }
    std::vector<double> masked(n);
    while (order.size() < k) {
        bool any = false;                            // not taken, not known to be set aside
        for (std::size_t i = 0; i < n && !any; ++i) any = !taken[i] && state[i] >= 0;
        for (std::size_t i = 0; i < n; ++i)
            masked[i] = (!taken[i] && (!any || state[i] >= 0)) ? cover[i] : -1.0;
        const std::size_t i = detail::argmax(masked);
        if (taken[i]) break;
        if (state[i] >= 0 && !is_kept(i)) continue;  // set aside now: look again
        take(i);
    }
    return order;
}

// archive.dplus_matrix's row c: d+(z, c) = ||max(C_c - R_z, 0)|| for every
// reference point z.
inline void dplus_row(const Rows& C, std::size_t c, const Rows& R, double* out, double* buf)
{
    const double* a = C.row(c);
    for (std::size_t z = 0; z < R.n; ++z) out[z] = detail::dplus(a, R.row(z), C.m, buf);
}

// archive.greedy_igdp on D = dplus_matrix(C, R), its rows computed when needed
// (C x R doubles would not fit for a large archive): k row indices, the row of
// the smallest mean first, then lazily by the gain mean(max(cur - D_c, 0)),
// ties to the lowest index. Python's heap of (-gain, c) pairs and this one
// give the same sequence: every pair is distinct, so the smallest is unique.
inline std::vector<std::size_t> greedy_igdp(const Rows& C, const Rows& R, std::size_t k)
{
    const std::size_t n = C.n, r = R.n;
    k = std::min(k, n);
    std::vector<std::size_t> chosen;
    if (k == 0) return chosen;
    std::vector<double> row(r), tmp(r), buf(C.m);
    std::vector<double> means(n);
    for (std::size_t c = 0; c < n; ++c) {
        dplus_row(C, c, R, row.data(), buf.data());
        means[c] = np::mean(row.data(), r);
    }
    const std::size_t first = detail::argmin(means);
    std::vector<double> cur(r);
    dplus_row(C, first, R, cur.data(), buf.data());
    chosen.push_back(first);
    auto gain = [&](std::size_t c) {
        dplus_row(C, c, R, row.data(), buf.data());
        for (std::size_t z = 0; z < r; ++z) tmp[z] = detail::pos(cur[z] - row[z]);
        return -np::mean(tmp.data(), r);
    };
    using Entry = std::pair<double, std::size_t>;
    std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> heap;
    for (std::size_t c = 0; c < n; ++c)
        if (c != first) heap.emplace(gain(c), c);
    while (chosen.size() < k && !heap.empty()) {
        const std::size_t c = heap.top().second;
        heap.pop();
        const Entry fresh(gain(c), c);               // row now holds D_c
        if (heap.empty() || !(heap.top() < fresh)) {
            chosen.push_back(c);
            for (std::size_t z = 0; z < r; ++z) cur[z] = detail::minimum(cur[z], row[z]);
        } else {
            heap.push(fresh);
        }
    }
    return chosen;
}

inline constexpr std::size_t SELF_REFERENCE = 1000;   // archive.SELF_REFERENCE

// archive.igdp_self_order: k points of F chosen for IGD+ against F thinned by
// DSS to SELF_REFERENCE points, in the given frame or that of the points
// set_aside keeps (the frame then taken without ideal or nadir, as there).
inline std::vector<std::size_t> igdp_self_order(const Rows& F, std::size_t k,
                                                const std::vector<double>& ideal = {},
                                                const std::vector<double>& nadir = {})
{
    std::vector<std::size_t> all;
    if (F.n <= k) {
        for (std::size_t i = 0; i < F.n; ++i) all.push_back(i);
        return all;
    }
    Rows G = (!ideal.empty() && !nadir.empty()) ? normalized(F, ideal, nadir)
                                                : set_aside(F, {}, {}).first;
    const std::vector<std::size_t> ref = dss_order(F, SELF_REFERENCE, ideal, nadir);
    Rows R;
    R.m = G.m;
    R.n = ref.size();
    for (std::size_t i : ref) R.v.insert(R.v.end(), G.row(i), G.row(i) + G.m);
    return greedy_igdp(G, R, k);
}

}   // namespace mootation::select

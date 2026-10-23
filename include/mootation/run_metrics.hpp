#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// More of python/mootation/run/metrics.py in C++ (task 17, supplement 1,
// stage 2a): the indicators in the frame (F - ideal) / span — roi_dist,
// range_cover_each, nn_cv, the hypervolume of a set in that frame (exact or
// Monte-Carlo) — and the set's own shares, nd_share and dup_share, and
// NumPy's quantile and median of a vector, which tau90 and the medians read.
// Every value is what metrics.py's NumPy code gives, bit for bit: sums in
// NumPy's pairwise order (numpy_sum.hpp), products left to right (np.prod),
// a NaN passed on as np.minimum, np.maximum, min and max pass it. Two places
// take the library's own pow (detmath.hpp; the owner's decision of
// 2026-10-23): the hypervolume's ref_scale^M, where metrics.py's Python
// float ** is the C library's, and the Monte-Carlo points come from the
// library's generator (metrics._mc_draws repeats it). metrics.py keeps its
// NumPy code as the reference the tests compare with.
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "detmath.hpp"
#include "hv_box.hpp"
#include "hypervolume.hpp"
#include "nondominated.hpp"
#include "numpy_sum.hpp"
#include "random.hpp"

namespace mootation::run_metrics {

using Vec = std::vector<double>;

namespace detail {

inline double nan_min(double a, double b)
{
    if (std::isnan(a)) return a;
    if (std::isnan(b)) return b;
    return b < a ? b : a;
}

inline double nan_max(double a, double b)
{
    if (std::isnan(a)) return a;
    if (std::isnan(b)) return b;
    return b > a ? b : a;
}

}   // namespace detail

// The frame's divisor: nadir - ideal, 1 where that is not > 0.
inline Vec span_of(const Vec& ideal, const Vec& nadir)
{
    Vec s(ideal.size());
    for (std::size_t j = 0; j < s.size(); ++j) {
        const double v = nadir[j] - ideal[j];
        s[j] = v > 0.0 ? v : 1.0;
    }
    return s;
}

// (F - ideal) / span of the n x m rows F.
inline Vec normalised(const double* F, std::size_t n, std::size_t m, const Vec& ideal, const Vec& span)
{
    Vec G(n * m);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < m; ++j) G[i * m + j] = (F[i * m + j] - ideal[j]) / span[j];
    return G;
}

// metrics.roi_dist: min over the rows of sqrt(sum(max(G - 1, 0)^2)).
inline double roi_dist(const double* F, std::size_t n, std::size_t m, const Vec& ideal, const Vec& nadir)
{
    if (n == 0) return std::numeric_limits<double>::infinity();
    const Vec G = normalised(F, n, m, ideal, span_of(ideal, nadir));
    Vec buf(m);
    double best = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < m; ++j) {
            const double d = G[i * m + j] - 1.0;
            const double t = std::isnan(d) ? d : (d >= 0.0 ? d : 0.0);
            buf[j] = t * t;
        }
        const double v = std::sqrt(np::sum(buf.data(), m));
        best = i == 0 ? v : detail::nan_min(best, v);
    }
    return best;
}

// metrics.range_cover_each: per objective max - min of clip(G, 0, 1).
inline Vec range_cover_each(const double* F, std::size_t n, std::size_t m, const Vec& ideal, const Vec& nadir)
{
    const Vec G = normalised(F, n, m, ideal, span_of(ideal, nadir));
    Vec out(m);
    for (std::size_t j = 0; j < m; ++j) {
        double lo = 0.0, hi = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            double v = G[i * m + j];
            v = std::isnan(v) ? v : (v < 0.0 ? 0.0 : (v > 1.0 ? 1.0 : v));
            lo = i == 0 ? v : detail::nan_min(lo, v);
            hi = i == 0 ? v : detail::nan_max(hi, v);
        }
        out[j] = hi - lo;
    }
    return out;
}

// metrics.nd_share: 1 - the share of rows another row dominates.
inline double nd_share(const double* F, std::size_t n, std::size_t m)
{
    if (n == 0) return 0.0;
    const std::vector<char> keep = nd::mask(F, n, m);
    Vec dominated(n);
    for (std::size_t i = 0; i < n; ++i) dominated[i] = keep[i] ? 0.0 : 1.0;
    return 1.0 - np::mean(dominated.data(), n);
}

// metrics.dup_share: 1 - (distinct rows) / n, rows equal byte for byte as
// np.unique(F, axis=0) compares them.
inline double dup_share(const double* F, std::size_t n, std::size_t m)
{
    if (n == 0) return 0.0;
    std::set<std::string> seen;
    for (std::size_t i = 0; i < n; ++i)
        seen.emplace(reinterpret_cast<const char*>(F + i * m), m * sizeof(double));
    return 1.0 - static_cast<double>(seen.size()) / static_cast<double>(n);
}

// metrics.nondominated's rows (nd::filter's order) of the n x m rows G.
inline Vec nondominated_rows(const Vec& G, std::size_t n, std::size_t m)
{
    if (n <= 1) return G;
    const std::vector<std::size_t> keep = nd::filter(G.data(), n, m);
    Vec out;
    out.reserve(keep.size() * m);
    for (std::size_t i : keep) out.insert(out.end(), G.begin() + static_cast<std::ptrdiff_t>(i * m),
                                          G.begin() + static_cast<std::ptrdiff_t>((i + 1) * m));
    return out;
}

// metrics.nn_cv: the coefficient of variation (np.std / np.mean) of the
// distances from each distinct non-dominated normalised point to its
// nearest other one; NaN for "None" (fewer than two such points, a mean of 0).
inline double nn_cv(const double* F, std::size_t n, std::size_t m, const Vec& ideal, const Vec& nadir)
{
    const double none = std::numeric_limits<double>::quiet_NaN();
    const Vec G = nondominated_rows(normalised(F, n, m, ideal, span_of(ideal, nadir)), n, m);
    const std::size_t k = G.size() / (m ? m : 1);
    if (k < 2) return none;
    Vec nn(k), buf(m);
    for (std::size_t i = 0; i < k; ++i) {
        double best = 0.0;
        for (std::size_t j = 0; j < k; ++j) {
            double d = std::numeric_limits<double>::infinity();         // the diagonal
            if (j != i) {
                for (std::size_t c = 0; c < m; ++c) {
                    const double v = G[i * m + c] - G[j * m + c];
                    buf[c] = v * v;
                }
                d = std::sqrt(np::sum(buf.data(), m));
            }
            best = j == 0 ? d : detail::nan_min(best, d);
        }
        nn[i] = best;
    }
    const double mean = np::mean(nn.data(), k);
    if (!(mean > 0.0)) return none;
    Vec sq(k);
    for (std::size_t i = 0; i < k; ++i) {
        const double x = nn[i] - mean;
        sq[i] = x * x;
    }
    const double var = np::sum(sq.data(), k) / static_cast<double>(k);
    return std::sqrt(var) / mean;
}

// np.quantile(a, q) by the default "linear" method: NaN if a holds one.
inline double quantile(Vec a, double q)
{
    const std::size_t n = a.size();
    for (double v : a)
        if (std::isnan(v)) return v;
    std::sort(a.begin(), a.end());
    const double virt = static_cast<double>(n - 1) * q;
    double prev = std::floor(virt);
    std::size_t lo = static_cast<std::size_t>(prev), hi = lo + 1;
    if (virt >= static_cast<double>(n - 1)) {
        lo = hi = n - 1;
        prev = -1.0;                                    // np: the index -1, gamma from it
    } else if (virt < 0.0) {
        lo = hi = 0;
    }
    const double t = virt - prev, a0 = a[lo], b0 = a[hi];
    const double diff = b0 - a0;
    return t >= 0.5 ? b0 - diff * (1.0 - t) : a0 + diff * t;
}

// np.median(a): the middle value, or np.mean of the two middle ones; NaN if
// a holds one.
inline double median(Vec a)
{
    const std::size_t n = a.size();
    for (double v : a)
        if (std::isnan(v)) return v;
    std::sort(a.begin(), a.end());
    if (n % 2 == 1) return a[n / 2];
    const double two[2] = {a[n / 2 - 1], a[n / 2]};
    return np::mean(two, 2);
}

// metrics._mc_draws's stream: the seed XOR this seeds the generator
constexpr std::uint32_t kHvMcStream = 0x6A09E667u;

struct Hv {
    double value = 0.0;
    const char* method = "none";
    // the Monte-Carlo estimate's standard error, in the value's units (task
    // 17, supplement 2, §2): box sqrt(p (1 - p) / samples) / ref_scale^M, p
    // the share of the points covered; NaN for an exact value
    double se = std::numeric_limits<double>::quiet_NaN();
};

// metrics.hypervolume: the rows of F in the frame, the reference point
// ref_scale in every objective, the rows strictly inside it, their
// non-dominated distinct rows; exact up to exact_max_m objectives (algo
// "auto", "wfg" or "box"), else the Monte-Carlo share of `samples` points of
// [lo, ref], lo = min(min G, 0), drawn in chunks of 20 000 by rnd::uniform01
// of a std::mt19937 seeded with seed XOR kHvMcStream; divided by
// ref_scale^M (dm::pow).
inline Hv hypervolume(const double* F, std::size_t n, std::size_t m, const Vec& ideal, const Vec& nadir,
                      double ref_scale, std::size_t samples, std::uint64_t seed, int exact_max_m,
                      const std::string& algo)
{
    Hv out;
    if (n == 0 || m == 0) return out;
    const Vec all = normalised(F, n, m, ideal, span_of(ideal, nadir));
    Vec inside;
    std::size_t k = 0;
    for (std::size_t i = 0; i < n; ++i) {
        bool ok = true;
        for (std::size_t j = 0; j < m; ++j) ok = ok && all[i * m + j] < ref_scale;
        if (ok) {
            inside.insert(inside.end(), all.begin() + static_cast<std::ptrdiff_t>(i * m),
                          all.begin() + static_cast<std::ptrdiff_t>((i + 1) * m));
            ++k;
        }
    }
    if (k == 0) return out;
    const Vec G = nondominated_rows(inside, k, m);
    const std::size_t g = G.size() / m;
    const Vec ref(m, ref_scale);
    double v = 0.0;
    if (static_cast<int>(m) <= exact_max_m) {
        v = algo == "wfg" ? hypervolume::wfg(G.data(), g, m, ref.data())
          : algo == "box" ? hypervolume::box(G.data(), g, m, ref.data())
                          : hypervolume::automatic(G.data(), g, m, ref.data());
        out.method = "exact";
    } else {
        Vec lo(m);
        for (std::size_t j = 0; j < m; ++j) {
            double c = G[j];
            for (std::size_t i = 1; i < g; ++i) c = detail::nan_min(c, G[i * m + j]);
            lo[j] = detail::nan_min(c, 0.0);
        }
        double box = 1.0;
        for (std::size_t j = 0; j < m; ++j) box = j == 0 ? ref[j] - lo[j] : box * (ref[j] - lo[j]);
        if (box > 0.0) {
            std::mt19937 gen(static_cast<std::uint32_t>(seed) ^ kHvMcStream);
            std::size_t hit = 0, done = 0;
            Vec s;
            while (done < samples) {
                const std::size_t chunk = std::min<std::size_t>(20000, samples - done);
                s.resize(chunk * m);
                for (std::size_t i = 0; i < chunk * m; ++i) {
                    const std::size_t j = i % m;
                    s[i] = lo[j] + rnd::uniform01(gen) * (ref[j] - lo[j]);
                }
                hit += hypervolume::covered(G.data(), g, m, s.data(), chunk);
                done += chunk;
            }
            v = box * static_cast<double>(hit) / static_cast<double>(samples);
            const double share = static_cast<double>(hit) / static_cast<double>(samples);
            out.se = box * std::sqrt(share * (1.0 - share) / static_cast<double>(samples));
        } else {
            out.se = 0.0;
        }
        out.method = "mc";
    }
    const double scale_m = dm::pow(ref_scale, static_cast<double>(m));
    out.value = v / scale_m;
    if (!std::isnan(out.se)) out.se = out.se / scale_m;
    return out;
}

}   // namespace mootation::run_metrics

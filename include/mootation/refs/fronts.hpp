#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// The reference fronts of the registry's problems — task 17, §2.1. A problem
// names its sampler in the registry's export ("front": {"kind": ..., the
// sampler's numbers}); `sample` gives that sampler's rows for n, `front` the
// problem's reference front: exactly_n of the sample (sample.hpp), seeded
// with the CRC-32 of the name of the problem whose front it is ("seed"), or
// the sample as it is where the registry takes it whole ("whole").
//
// Each sampler is its Python function's (named beside it), with the
// library's functions (dm::) for sin, cos, exp and pow, the sums in NumPy's
// order (np::sum) and the library's generator where Python draws from
// default_rng (a new sample: reference version 5):
//
//   dtlz1, dtlz2         registry._pf_dtlz1 / _pf_dtlz2 (DTLZ2-4, wideDTLZ3)
//   dtlz5                registry._pf_dtlz5: the curve (DTLZ5, DTLZ6 to M = 3)
//   dtlz7                registry._pf_dtlz7: the grid of the left records of
//                        phi(x) = x (1 + sin 3 pi x), or random records with
//                        the corners, nondominated, checked (checked below)
//   idtlz1, idtlz2, minus_dtlz2, sdtlz1, sdtlz2
//                        dtlz_variants.pf_*: "k" the DTLZ2 k of minusDTLZ2,
//                        "scale" SDTLZ's factors
//   zdt1 .. zdt6         registry._pf_zdt* (zdt1 also ZDT4's, catZDT1's)
//   bt_concave, bt5, bt9 bt.pf_*
//   mop1 .. mop7         mop.pf_*
//   sch, fon, sph        classic.pareto_front ("M" SPH's)
//   binary               binary.front, minimized: "which" (LOTZ, OMM, OJZJ),
//                        "n", "k"
//   full                 fronts_full.front: the first n rows of a shipped,
//                        verified front ("file", under <data>/_fronts)
//
// checked (registry._checked): the rows of a front sampled on a grid that
// nothing outside the sample beats — against 20 000 fresh position vectors
// and every position moved alone, both ways, by 0.05 down to 0.001, and 32
// Gaussian moves of all at once (rnd::uniform01, rnd::normal01).
// ============================================================================

#include <cstdint>
#include <filesystem>
#include <functional>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "../io/npz.hpp"
#include "../json.hpp"
#include "../nondominated.hpp"
#include "../run_compute.hpp"
#include "../run_metrics.hpp"
#include "../suite/dascmop.hpp"
#include "../suite/dtlz.hpp"
#include "../suite/lircmop.hpp"
#include "../suite/mmo.hpp"
#include "../suite/polygon.hpp"
#include "fronts_mw.hpp"
#include "fronts_zcat.hpp"
#include "sample.hpp"

namespace mootation::refs {

// where the shipped fronts are (<data>/_fronts/<file>)
struct Data {
    std::filesystem::path dir;
};

namespace detail {

inline const json::Value& field(const json::Value& spec, const std::string& key)
{
    const json::Value* v = spec.find(key);
    if (!v) throw std::invalid_argument("a front's spec without \"" + key + "\"");
    return *v;
}

inline std::size_t count(const json::Value& spec, const std::string& key)
{
    return static_cast<std::size_t>(field(spec, key).i);
}

inline double number(const json::Value& v)
{
    return v.kind == json::Value::Kind::Int ? static_cast<double>(v.i) : v.f;
}

// a column of two: f1 and f2(f1)
inline Rows curve(const std::vector<double>& f1, const std::function<double(double)>& f2)
{
    Rows F = rows_of(f1.size(), 2);
    for (std::size_t i = 0; i < f1.size(); ++i) {
        F.row(i)[0] = f1[i];
        F.row(i)[1] = f2(f1[i]);
    }
    return F;
}

inline Rows nondominated(const Rows& F)
{
    const std::vector<char> keep = nd::mask(F.v.data(), F.n, F.m);
    std::vector<std::size_t> idx;
    for (std::size_t i = 0; i < F.n; ++i)
        if (keep[i]) idx.push_back(i);
    return take(F, idx);
}

// does some row of V strictly dominate c (all <=, one <)?
inline bool beaten(const double* c, const Rows& V)
{
    for (std::size_t i = 0; i < V.n; ++i) {
        const double* v = V.row(i);
        bool le = true, lt = false;
        for (std::size_t j = 0; j < V.m && le; ++j) {
            le = v[j] <= c[j];
            lt = lt || v[j] < c[j];
        }
        if (le && lt) return true;
    }
    return false;
}

using Images = std::function<Rows(const Rows&)>;

// registry._checked: which rows of the front F (images of the positions X) stay
inline std::vector<char> checked(const Rows& X, const Rows& F, const Images& images, std::uint32_t seed,
                                 std::size_t samples = 20000)
{
    std::mt19937 g(seed);
    const std::size_t d = X.m;
    Rows R = rows_of(samples, d);
    for (double& v : R.v) v = rnd::uniform01(g);
    const Rows fresh = images(R);
    std::vector<char> bad(F.n, 0);
    for (std::size_t i = 0; i < F.n; ++i) bad[i] = beaten(F.row(i), fresh);
    auto against = [&](const Rows& Q) {
        const Rows G = images(Q);
        for (std::size_t i = 0; i < F.n; ++i) {
            bool le = true, lt = false;
            for (std::size_t j = 0; j < F.m && le; ++j) {
                le = G.row(i)[j] <= F.row(i)[j];
                lt = lt || G.row(i)[j] < F.row(i)[j];
            }
            if (le && lt) bad[i] = 1;
        }
    };
    auto clip = [](double v) { return v < 0.0 ? 0.0 : v > 1.0 ? 1.0 : v; };
    for (double step : {0.05, 0.02, 0.01, 0.005, 0.002, 0.001})
        for (std::size_t j = 0; j < d; ++j)
            for (double sgn : {-1.0, 1.0}) {
                Rows Q = X;
                for (std::size_t i = 0; i < Q.n; ++i) Q.row(i)[j] = clip(Q.row(i)[j] + sgn * step);
                against(Q);
            }
    for (double s : {0.2, 0.1, 0.05, 0.02})
        for (int rep = 0; rep < 8; ++rep) {
            Rows Q = X;
            for (double& v : Q.v) v = clip(v + s * rnd::normal01(g));
            against(Q);
        }
    std::vector<char> keep(F.n);
    for (std::size_t i = 0; i < F.n; ++i) keep[i] = !bad[i];
    return keep;
}

inline Rows keep_rows(const Rows& F, const std::vector<char>& keep)
{
    std::vector<std::size_t> idx;
    for (std::size_t i = 0; i < F.n; ++i)
        if (keep[i]) idx.push_back(i);
    return take(F, idx);
}

// ---- DTLZ ----------------------------------------------------------------

inline Rows dtlz5(std::size_t M, std::size_t n)
{
    const std::vector<double> t = linspace(0.0, dm::kPi / 2.0, n);
    const double cd = dm::cos(dm::kPi / 4.0), sd = dm::sin(dm::kPi / 4.0);
    Rows F = rows_of(n, M);
    for (std::size_t r = 0; r < n; ++r) {
        const double c0 = dm::cos(t[r]), s0 = dm::sin(t[r]);
        double* f = F.row(r);
        f[0] = M > 1 ? c0 * dm::pow(cd, static_cast<double>(M - 2)) : 1.0;
        for (std::size_t i = 2; i < M; ++i) f[i - 1] = c0 * dm::pow(cd, static_cast<double>(M - i - 1)) * sd;
        if (M > 1) f[M - 1] = s0;
    }
    unit_rows(F);
    return F;
}

constexpr double kDtlz7Cell = 1.0 / 200000.0;

// registry._dtlz7_record_cells: the left ends of the cells of a 2e5 grid on
// [0, 1] where phi is a left record
inline const std::vector<double>& dtlz7_record_cells()
{
    static const std::vector<double> cells = [] {
        const std::vector<double> t = linspace(0.0, 1.0, 200001);
        std::vector<double> out;
        double best = -std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i < t.size(); ++i) {
            const double phi = t[i] * (1.0 + dm::sin(3.0 * dm::kPi * t[i]));
            best = phi > best ? phi : best;                  // np.maximum.accumulate
            if (phi >= best && i + 1 < t.size()) out.push_back(t[i]);
        }
        return out;
    }();
    return cells;
}

inline Rows dtlz7_image(std::size_t M, const Rows& X)
{
    Rows F = rows_of(X.n, M);
    std::vector<double> terms(M - 1);
    for (std::size_t i = 0; i < X.n; ++i) {
        for (std::size_t j = 0; j + 1 < M; ++j) {
            const double x = X.row(i)[j];
            F.row(i)[j] = x;
            terms[j] = x * (1.0 + dm::sin(3.0 * dm::kPi * x));
        }
        F.row(i)[M - 1] = 2.0 * static_cast<double>(M) - np::sum(terms.data(), M - 1);
    }
    return F;
}

inline Rows dtlz7(std::size_t M, std::size_t n)
{
    if (M == 1) {
        Rows F = rows_of(1, 1);
        F.v[0] = 2.0;
        return F;
    }
    const std::size_t d = M - 1;
    const long long p = std::max<long long>(
        2, round_half_even(dm::pow(static_cast<double>(n * 2), 1.0 / static_cast<double>(d))));
    const std::vector<double>& rec = dtlz7_record_cells();
    Rows pts = rows_of(0, d);
    std::vector<double> row(d);
    std::size_t cells = 1;                                // p^(M-1), up to 4n + 1
    for (std::size_t j = 0; j < d && cells <= 4 * n; ++j) cells *= static_cast<std::size_t>(p);
    if (p >= 3 && cells <= 4 * n) {
        std::vector<double> grid;
        for (double at : linspace(0.0, static_cast<double>(rec.size() - 1), static_cast<std::size_t>(p)))
            grid.push_back(rec[static_cast<std::size_t>(std::nearbyint(at))]);
        std::vector<std::size_t> digit(d, 0);               // itertools.product
        for (;;) {
            for (std::size_t j = 0; j < d; ++j) row[j] = grid[digit[j]];
            push_row(pts, row.data());
            std::size_t j = d;
            while (j > 0 && ++digit[j - 1] == grid.size()) digit[--j] = 0;
            if (j == 0) break;
        }
    } else {
        std::mt19937 g(static_cast<std::uint32_t>(20260709 + 1000 * M + n));
        const double ends[2] = {rec.front(), rec.back()};
        if (d <= 10) {
            for (std::size_t c = 0; c < (std::size_t{1} << d); ++c) {
                for (std::size_t j = 0; j < d; ++j) row[j] = ends[(c >> j) & 1];
                push_row(pts, row.data());
            }
        } else {
            for (int e = 0; e < 2; ++e) {
                std::fill(row.begin(), row.end(), ends[e]);
                push_row(pts, row.data());
            }
            for (std::size_t k = 0; k < d; ++k) {
                for (std::size_t j = 0; j < d; ++j) row[j] = ends[0] + (j == k ? 1.0 : 0.0) * (ends[1] - ends[0]);
                push_row(pts, row.data());
            }
        }
        // registry._dtlz7_records: a record cell each, then a place in it
        const std::size_t k = 2 * n > pts.n ? 2 * n - pts.n : 0;
        std::vector<std::size_t> cell(k * d);
        for (auto& c : cell) c = static_cast<std::size_t>(rnd::below(g, rec.size()));
        for (std::size_t i = 0; i < k; ++i) {
            for (std::size_t j = 0; j < d; ++j) row[j] = rec[cell[i * d + j]];
            for (std::size_t j = 0; j < d; ++j) row[j] = std::min(row[j] + rnd::uniform01(g) * kDtlz7Cell, 1.0);
            push_row(pts, row.data());
        }
    }
    Rows F = dtlz7_image(M, pts);
    const std::vector<char> nd_keep = nd::mask(F.v.data(), F.n, F.m);
    pts = keep_rows(pts, nd_keep);
    F = keep_rows(F, nd_keep);
    const auto keep = checked(pts, F, [M](const Rows& X) { return dtlz7_image(M, X); },
                              static_cast<std::uint32_t>(20260922 + M));
    return keep_rows(F, keep);
}

// ---- ZDT -----------------------------------------------------------------

inline Rows zdt3_image(const Rows& X)
{
    Rows F = rows_of(X.n, 2);
    for (std::size_t i = 0; i < X.n; ++i) {
        const double f1 = X.row(i)[0];
        F.row(i)[0] = f1;
        F.row(i)[1] = 1.0 - std::sqrt(f1) - f1 * dm::sin(10.0 * dm::kPi * f1);
    }
    return F;
}

// ZDT3's front (Zitzler, Deb & Thiele 2000, Eq. 9, at g = 1): the pieces of
// f2 = 1 - sqrt(f1) - f1 sin(10 pi f1) that no other point of the curve
// dominates (task 17, addendum 6, §4). The right end of each piece is a
// local minimum of f2 (where f2' = -1/(2 sqrt f1) - sin(10 pi f1) - 10 pi f1
// cos(10 pi f1) turns from negative to positive), found on a grid of 2e5 and
// bisected to the last bit; a minimum lower than every earlier one ends a
// piece, and the piece starts where f2, falling from the local maximum
// before it, comes down to the earlier minimum's level (bisected too; the
// first piece starts at 0). The n points lie on the pieces evenly by f1,
// each piece's share of n its share of the pieces' length (the largest
// remainders rounded up), both ends of every piece among them.
struct Zdt3Piece {
    double a = 0.0, b = 0.0;
};

inline double zdt3_f2(double x)
{
    return 1.0 - std::sqrt(x) - x * dm::sin(10.0 * dm::kPi * x);
}

inline double zdt3_slope(double x)
{
    return -0.5 / std::sqrt(x) - dm::sin(10.0 * dm::kPi * x) - 10.0 * dm::kPi * x * dm::cos(10.0 * dm::kPi * x);
}

// the root of fn between lo and hi (fn(lo) and fn(hi) of other signs), to the last bit
template <typename F>
inline double bisect(F&& fn, double lo, double hi)
{
    const bool lo_neg = fn(lo) < 0.0;
    for (;;) {
        const double mid = 0.5 * (lo + hi);
        if (mid <= lo || mid >= hi) return mid;
        if ((fn(mid) < 0.0) == lo_neg) lo = mid;
        else hi = mid;
    }
}

inline std::vector<Zdt3Piece> zdt3_pieces()
{
    constexpr std::size_t kGrid = 200000;
    std::vector<double> minima, maxima;
    double prev = zdt3_slope(1.0 / kGrid);
    for (std::size_t k = 2; k <= kGrid; ++k) {
        const double x = static_cast<double>(k) / kGrid, s = zdt3_slope(x);
        const double lo = static_cast<double>(k - 1) / kGrid;
        if (prev < 0.0 && s >= 0.0) minima.push_back(bisect(zdt3_slope, lo, x));
        if (prev >= 0.0 && s < 0.0) maxima.push_back(bisect(zdt3_slope, lo, x));
        prev = s;
    }
    std::vector<Zdt3Piece> out;
    double level = std::numeric_limits<double>::infinity();
    for (double b : minima) {
        const double fb = zdt3_f2(b);
        if (!(fb < level)) continue;
        Zdt3Piece p;
        p.b = b;
        if (!out.empty()) {
            double m = 0.0;                              // the local maximum before b
            for (double c : maxima)
                if (c < b) m = c;
            p.a = bisect([level](double x) { return zdt3_f2(x) - level; }, m, b);
        }
        out.push_back(p);
        level = fb;
    }
    return out;
}

inline Rows zdt3(std::size_t n)
{
    static const std::vector<Zdt3Piece> pieces = zdt3_pieces();
    double total = 0.0;
    for (const auto& p : pieces) total += p.b - p.a;
    // the largest-remainder split of n over the pieces, two points each at least
    const std::size_t k = pieces.size();
    std::vector<std::size_t> count(k);
    std::vector<std::pair<double, std::size_t>> rest;
    std::size_t used = 0;
    for (std::size_t i = 0; i < k; ++i) {
        const double share = static_cast<double>(n) * (pieces[i].b - pieces[i].a) / total;
        count[i] = std::max<std::size_t>(2, static_cast<std::size_t>(share));
        used += count[i];
        rest.emplace_back(-(share - std::floor(share)), i);
    }
    std::sort(rest.begin(), rest.end());
    for (std::size_t r = 0; used < n && r < rest.size(); ++r, ++used) ++count[rest[r].second];
    Rows X = rows_of(0, 1);
    for (std::size_t i = 0; i < k; ++i)
        for (double f1 : linspace(pieces[i].a, pieces[i].b, count[i])) push_row(X, &f1);
    return zdt3_image(X);
}

// ---- the grid of a meshgrid of two (indexing "xy": the second fastest) ---

inline Rows octant_grid(std::size_t p)
{
    const std::vector<double> lin = linspace(0.0, 1.0, p);
    Rows F = rows_of(p * p, 3);
    for (std::size_t i = 0; i < p; ++i)              // b = lin[i] down the rows
        for (std::size_t j = 0; j < p; ++j) {        // a = lin[j] along them
            const double a = lin[j], b = lin[i];
            const double c1 = dm::cos(0.5 * dm::kPi * a), c2 = dm::cos(0.5 * dm::kPi * b);
            const double s1 = dm::sin(0.5 * dm::kPi * a), s2 = dm::sin(0.5 * dm::kPi * b);
            double* f = F.row(i * p + j);
            f[0] = c1 * c2;
            f[1] = c1 * s2;
            f[2] = s1;
        }
    return F;
}

// mop._simplex_grid: the grid of the simplex without dtlz_variants' top-up
inline Rows simplex_grid(std::size_t M, std::size_t n)
{
    const long long p = std::max<long long>(
        1, round_half_even(dm::pow(static_cast<double>(n), 1.0 / static_cast<double>(M - 1))));
    const std::vector<double> grid = linspace(0.0, 1.0, static_cast<std::size_t>(p + 1));
    Rows W = rows_of(0, M);
    std::vector<std::size_t> digit(M - 1, 0);
    std::vector<double> row(M);
    for (;;) {
        double s = 0.0;
        for (std::size_t j = 0; j + 1 < M; ++j) s += grid[digit[j]];
        if (s <= 1.0 + 1e-9) {
            for (std::size_t j = 0; j + 1 < M; ++j) row[j] = grid[digit[j]];
            row[M - 1] = 1.0 - s;
            push_row(W, row.data());
        }
        std::size_t j = M - 1;
        while (j > 0 && ++digit[j - 1] == grid.size()) digit[--j] = 0;
        if (j == 0) break;
    }
    return W;
}

// ---- WFG -----------------------------------------------------------------

// f_m = 2m h_m: the rows of H scaled
inline void wfg_scale(Rows& H)
{
    for (std::size_t i = 0; i < H.n; ++i)
        for (std::size_t j = 0; j < H.m; ++j) H.row(i)[j] *= 2.0 * static_cast<double>(j + 1);
}

// registry._img_wfg_convex: WFG's convex h_1..h_M-1 of the positions X, and
// h_M = last(x_1)
inline Rows wfg_convex_image(std::size_t M, const Rows& X, const std::function<double(double)>& last)
{
    Rows H = rows_of(X.n, M);
    std::vector<double> s(M - 1), c(M - 1);
    for (std::size_t i = 0; i < X.n; ++i) {
        for (std::size_t j = 0; j + 1 < M; ++j) {
            s[j] = dm::sin(dm::kPi * X.row(i)[j] / 2.0);
            c[j] = dm::cos(dm::kPi * X.row(i)[j] / 2.0);
        }
        double* h = H.row(i);
        h[0] = 1.0;
        for (std::size_t j = 0; j + 1 < M; ++j) h[0] *= 1.0 - c[j];      // np.prod: one after another
        for (std::size_t m = 1; m + 1 < M; ++m) {
            double p = 1.0;
            for (std::size_t j = 0; j < M - 1 - m; ++j) p *= 1.0 - c[j];
            h[m] = p * (1.0 - s[M - 1 - m]);
        }
        h[M - 1] = last(X.row(i)[0]);
    }
    wfg_scale(H);
    return H;
}

inline Rows wfg1_image(std::size_t M, const Rows& X)
{
    return wfg_convex_image(M, X, [](double x1) {
        return 1.0 - x1 - dm::cos(10.0 * dm::kPi * x1 + dm::kPi / 2.0) / (10.0 * dm::kPi);
    });
}

inline Rows wfg2_image(std::size_t M, const Rows& X)
{
    return wfg_convex_image(M, X, [](double x1) {
        const double c = dm::cos(5.0 * dm::kPi * x1);
        return 1.0 - x1 * (c * c);
    });
}

// registry._wfg12_positions: the corners of [0, 1]^(M-1) (itertools.product's
// order), then 4n uniform draws
inline Rows wfg12_positions(std::size_t M, std::size_t n, std::uint32_t seed)
{
    const std::size_t d = M - 1;
    Rows X = rows_of(0, d);
    std::vector<double> row(d);
    for (std::size_t c = 0; c < (std::size_t{1} << d); ++c) {
        for (std::size_t j = 0; j < d; ++j) row[j] = static_cast<double>((c >> (d - 1 - j)) & 1);
        push_row(X, row.data());
    }
    std::mt19937 g(seed);
    for (std::size_t k = 0; k < 4 * n; ++k) {
        for (std::size_t j = 0; j < d; ++j) row[j] = rnd::uniform01(g);
        push_row(X, row.data());
    }
    return X;
}

// registry._pf_wfg1 / _pf_wfg2 (which 1 or 2)
inline Rows wfg12(std::size_t M, std::size_t n, int which)
{
    const std::uint32_t off = static_cast<std::uint32_t>(100 * which + M);
    Rows X = wfg12_positions(M, n, 20260929u + off);
    auto image = [M, which](const Rows& Q) { return which == 1 ? wfg1_image(M, Q) : wfg2_image(M, Q); };
    Rows F = image(X);
    const std::vector<char> nd_keep = nd::mask(F.v.data(), F.n, F.m);
    X = keep_rows(X, nd_keep);
    F = keep_rows(F, nd_keep);
    return keep_rows(F, checked(X, F, image, 20260922u + off));
}

// registry._pf_wfg3: the degenerate line, x_1 over [0, 1], the others 1/2
inline Rows wfg3(std::size_t M, std::size_t n)
{
    const std::vector<double> t = linspace(0.0, 1.0, n);
    Rows F = rows_of(n, M);
    std::vector<double> x(M - 1, 0.5);
    for (std::size_t r = 0; r < n; ++r) {
        x[0] = t[r];
        double* h = F.row(r);
        h[0] = 1.0;
        for (std::size_t j = 0; j + 1 < M; ++j) h[0] *= x[j];
        for (std::size_t m = 1; m + 1 < M; ++m) {
            double p = 1.0;
            for (std::size_t j = 0; j < M - 1 - m; ++j) p *= x[j];
            h[m] = p * (1.0 - x[M - 1 - m]);
        }
        if (M > 1) h[M - 1] = 1.0 - x[0];
    }
    wfg_scale(F);
    return nondominated(F);
}

// ---- MaF -----------------------------------------------------------------

// registry._dd_xset: the grid of p = round(n^(1/(M-1))) levels on [0, 1]^(M-1)
// (itertools.product), topped up to n with uniform draws
inline Rows dd_xset(std::size_t M, std::size_t n)
{
    const std::size_t d = M - 1;
    const long long p = std::max<long long>(
        2, round_half_even(dm::pow(static_cast<double>(n), 1.0 / static_cast<double>(d))));
    const std::vector<double> grid = linspace(0.0, 1.0, static_cast<std::size_t>(p));
    Rows X = rows_of(0, d);
    std::vector<std::size_t> digit(d, 0);
    std::vector<double> row(d);
    for (;;) {
        for (std::size_t j = 0; j < d; ++j) row[j] = grid[digit[j]];
        push_row(X, row.data());
        std::size_t j = d;
        while (j > 0 && ++digit[j - 1] == grid.size()) digit[--j] = 0;
        if (j == 0) break;
    }
    if (X.n < n) {
        std::mt19937 g(static_cast<std::uint32_t>(20260709 + 1000 * M + n));
        for (std::size_t k = X.n; k < n; ++k) {
            for (std::size_t jj = 0; jj < d; ++jj) row[jj] = rnd::uniform01(g);
            push_row(X, row.data());
        }
    }
    return X;
}

// registry._pf_maf2: the band of the sphere whose angles lie in [pi/8, 3 pi/8]
inline Rows maf2(std::size_t M, std::size_t n)
{
    const Rows X = dd_xset(M, n);
    Rows F = rows_of(X.n, M);
    std::vector<double> c(M - 1), s(M - 1);
    for (std::size_t r = 0; r < X.n; ++r) {
        for (std::size_t j = 0; j + 1 < M; ++j) {
            const double th = 0.5 * dm::kPi * (0.5 * X.row(r)[j] + 0.25);
            c[j] = dm::cos(th);
            s[j] = dm::sin(th);
        }
        double* f = F.row(r);
        f[0] = 1.0;
        for (std::size_t j = 0; j + 1 < M; ++j) f[0] *= c[j];
        for (std::size_t i = 2; i < M; ++i) {
            double p = 1.0;
            for (std::size_t j = 0; j < M - i; ++j) p *= c[j];
            f[i - 1] = p * s[M - i];
        }
        if (M > 1) f[M - 1] = s[0];
    }
    return F;
}

// the regular M-gon's vertices on the unit circle, vertex m at 2 pi m / M
inline std::vector<double> unit_polygon(std::size_t M)
{
    std::vector<double> V(2 * M);
    for (std::size_t m = 0; m < M; ++m) {
        const double ang = 2.0 * dm::kPi * static_cast<double>(m) / static_cast<double>(M);
        V[2 * m] = dm::cos(ang);
        V[2 * m + 1] = dm::sin(ang);
    }
    return V;
}

// registry._point_in_poly (the ray test; V: x0 y0 x1 y1 ...)
inline bool point_in_poly(double px, double py, const std::vector<double>& V)
{
    const std::size_t n = V.size() / 2;
    bool in = false;
    std::size_t j = n - 1;
    for (std::size_t i = 0; i < n; ++i) {
        const double xi = V[2 * i], yi = V[2 * i + 1], xj = V[2 * j], yj = V[2 * j + 1];
        if (((yi > py) != (yj > py)) && (px < (xj - xi) * (py - yi) / (yj - yi + 1e-30) + xi)) in = !in;
        j = i;
    }
    return in;
}

// n points of the polygon V by rejection: x then y of a candidate, lo + (hi -
// lo) U of its bounding box (registry._sample_poly_interior, _pf_polygon:
// default_rng(seed) before; the library's generator now)
inline Rows poly_interior(const std::vector<double>& V, std::size_t n, std::uint32_t seed)
{
    double lo[2] = {V[0], V[1]}, hi[2] = {V[0], V[1]};
    for (std::size_t k = 0; 2 * k < V.size(); ++k)
        for (int c = 0; c < 2; ++c) {
            lo[c] = std::min(lo[c], V[2 * k + c]);
            hi[c] = std::max(hi[c], V[2 * k + c]);
        }
    std::mt19937 g(seed);
    Rows P = rows_of(0, 2);
    while (P.n < n) {
        double c[2];
        for (int k = 0; k < 2; ++k) c[k] = lo[k] + (hi[k] - lo[k]) * rnd::uniform01(g);
        if (point_in_poly(c[0], c[1], V)) push_row(P, c);
    }
    return P;
}

// f_m = the distance of p to vertex m (registry._pf_polygon, _pf_maf8)
inline Rows vertex_distances(const Rows& P, const std::vector<double>& V)
{
    const std::size_t M = V.size() / 2;
    Rows F = rows_of(P.n, M);
    for (std::size_t i = 0; i < P.n; ++i)
        for (std::size_t m = 0; m < M; ++m) F.row(i)[m] = dm::hypot(P.row(i)[0] - V[2 * m], P.row(i)[1] - V[2 * m + 1]);
    return F;
}

// registry._pf_maf9: f_m = the distance of p to the line through edge m
inline Rows edge_line_distances(const Rows& P, const std::vector<double>& V)
{
    const std::size_t M = V.size() / 2;
    Rows F = rows_of(P.n, M);
    for (std::size_t m = 0; m < M; ++m) {
        const double ax = V[2 * m], ay = V[2 * m + 1];
        const double dx = V[2 * ((m + 1) % M)] - ax, dy = V[2 * ((m + 1) % M) + 1] - ay;
        const double nrm = dm::hypot(dx, dy);
        for (std::size_t i = 0; i < P.n; ++i)
            F.row(i)[m] = std::fabs(dy * (P.row(i)[0] - ax) - dx * (P.row(i)[1] - ay)) / nrm;
    }
    return F;
}

// polygon_ishibuchi: the four regular M-gons of radius 8 about (25, 25),
// (75, 25), (25, 75), (75, 75), vertex m at 2 pi m / M + pi / 2
inline std::vector<std::vector<double>> ipolygons(std::size_t M)
{
    static const double centres[4][2] = {{25.0, 25.0}, {75.0, 25.0}, {25.0, 75.0}, {75.0, 75.0}};
    std::vector<std::vector<double>> out(4, std::vector<double>(2 * M));
    for (std::size_t m = 0; m < M; ++m) {
        const double ang = 2.0 * dm::kPi * static_cast<double>(m) / static_cast<double>(M) + dm::kPi / 2.0;
        const double bx = 8.0 * dm::cos(ang), by = 8.0 * dm::sin(ang);
        for (int p = 0; p < 4; ++p) {
            out[static_cast<std::size_t>(p)][2 * m] = bx + centres[p][0];
            out[static_cast<std::size_t>(p)][2 * m + 1] = by + centres[p][1];
        }
    }
    return out;
}

// polygon_ishibuchi.make_pf: n points of the first polygon (by rejection, a
// std::mt19937 seeded 12345, x then y), f_i the distance to the nearest of the
// four vertices i
inline Rows ipolygon(std::size_t M, std::size_t n)
{
    const auto polys = ipolygons(M);
    const Rows P = poly_interior(polys[0], n, 12345u);
    Rows F = rows_of(P.n, M);
    for (std::size_t i = 0; i < P.n; ++i)
        for (std::size_t m = 0; m < M; ++m) {
            double best = std::numeric_limits<double>::infinity();
            for (const auto& V : polys) {
                const double dx = V[2 * m] - P.row(i)[0], dy = V[2 * m + 1] - P.row(i)[1];
                best = std::min(best, std::sqrt(dx * dx + dy * dy));
            }
            F.row(i)[m] = best;
        }
    return F;
}

// ---- C-DTLZ (c_dtlz.py) -------------------------------------------------

using suite::dtlz::Cons;

// c_dtlz.feasible: every constraint of the row f <= 0
inline bool cdtlz_feasible(Cons which, const double* f, std::size_t M)
{
    const int m = static_cast<int>(M);
    const double r = suite::dtlz::cdtlz_radius(which, m);
    std::vector<double> g(M, 0.0);
    suite::dtlz::constraints(which, f, m, r * r, 1.0 / std::sqrt(static_cast<double>(M)), g.data());
    const std::size_t k = which == Cons::C3_DTLZ1 || which == Cons::C3_DTLZ4 ? M : 1;
    double worst = g[0];
    for (std::size_t j = 1; j < k; ++j) worst = std::max(worst, g[j]);
    return worst <= 0.0;
}

inline Rows feasible_rows(Cons which, const Rows& F)
{
    std::vector<char> keep(F.n);
    for (std::size_t i = 0; i < F.n; ++i) keep[i] = cdtlz_feasible(which, F.row(i), F.m);
    return keep_rows(F, keep);
}

// does some row of R dominate p?
inline bool dominated_by(const double* p, const Rows& R)
{
    return beaten(p, R);
}

// c_dtlz.convex_c2_front: the feasible part of the convex DTLZ2 front at
// g = 0 (A), and for every direction in the cylinder's hole its first exit
// (B) that nothing in A or B dominates and no feasible point of g = 0 dominates
inline Rows convex_c2(std::size_t M, std::size_t n, std::uint32_t seed = 20261009u)
{
    std::mt19937 g(seed + static_cast<std::uint32_t>(M));
    const int m = static_cast<int>(M);
    Rows S = rows_of(0, M);
    std::vector<double> row(M), sq(M);
    for (std::size_t j = 0; j < M; ++j) {
        std::fill(row.begin(), row.end(), 0.0);
        row[j] = 1.0;
        push_row(S, row.data());
    }
    for (std::size_t k = 0; k < 20 * n; ++k) {
        for (std::size_t j = 0; j < M; ++j) row[j] = std::fabs(rnd::normal01(g));
        for (std::size_t j = 0; j < M; ++j) sq[j] = row[j] * row[j];
        const double nrm = std::sqrt(np::sum(sq.data(), M));
        for (std::size_t j = 0; j < M; ++j) row[j] /= nrm;
        push_row(S, row.data());
    }
    auto convex_at = [m, M](const double* s, double scale, double* f) {
        for (std::size_t j = 0; j < M; ++j) f[j] = scale * s[j];
        suite::dtlz::convex(f, m);
    };
    Rows A = rows_of(0, M), hole = rows_of(0, M);
    for (std::size_t i = 0; i < S.n; ++i) {
        convex_at(S.row(i), 1.0, row.data());
        if (cdtlz_feasible(Cons::C2_Convex, row.data(), M)) push_row(A, row.data());
        else push_row(hole, S.row(i));
    }
    // the first exit on np.arange(1.001, 3.5005, 0.001) (its values start + i delta,
    // delta the difference of its first two), then 40 bisections
    const double start = 1.001, delta = (1.001 + 0.001) - 1.001;
    const auto steps = static_cast<std::size_t>(std::ceil((3.5005 - 1.001) / 0.001));
    std::vector<double> rho(hole.n, 1.0);
    std::vector<char> found(hole.n, 0);
    std::size_t left = hole.n;
    for (std::size_t s = 0; s < steps && left > 0; ++s) {
        const double step = start + static_cast<double>(s) * delta;
        for (std::size_t i = 0; i < hole.n; ++i) {
            if (found[i]) continue;
            convex_at(hole.row(i), step, row.data());
            if (cdtlz_feasible(Cons::C2_Convex, row.data(), M)) {
                rho[i] = step;
                found[i] = 1;
                --left;
            }
        }
    }
    Rows B = rows_of(0, M);
    for (std::size_t i = 0; i < hole.n; ++i) {
        if (!found[i]) continue;
        double lo = rho[i] - 0.001, hi = rho[i];
        for (int k = 0; k < 40; ++k) {
            const double mid = 0.5 * (lo + hi);
            convex_at(hole.row(i), mid, row.data());
            if (cdtlz_feasible(Cons::C2_Convex, row.data(), M)) hi = mid;
            else lo = mid;
        }
        convex_at(hole.row(i), hi, row.data());
        push_row(B, row.data());
    }
    std::vector<char> keep(B.n);
    for (std::size_t i = 0; i < B.n; ++i) keep[i] = !dominated_by(B.row(i), A) && !dominated_by(B.row(i), B);
    B = keep_rows(B, keep);
    // a feasible point of g = 0 below a candidate: s_i <= P_i^(1/4), s_M <= P_M^(1/2)
    keep.assign(B.n, 1);
    std::vector<double> U(M - 1), s(M);
    for (std::size_t i = 0; i < B.n; ++i) {
        const double* P = B.row(i);
        const double cap_last = std::sqrt(P[M - 1]);
        bool any = false, beaten_here = false;
        for (int t = 0; t < 4000; ++t) {
            for (std::size_t j = 0; j + 1 < M; ++j) U[j] = rnd::uniform01(g) * dm::pow(P[j], 0.25);
            for (std::size_t j = 0; j + 1 < M; ++j) sq[j] = U[j] * U[j];
            const double last = 1.0 - np::sum(sq.data(), M - 1);
            if (!(last >= 0.0 && last <= cap_last * cap_last)) continue;
            any = true;
            std::copy(U.begin(), U.end(), s.begin());
            s[M - 1] = std::sqrt(last);
            convex_at(s.data(), 1.0, row.data());
            beaten_here = beaten_here || cdtlz_feasible(Cons::C2_Convex, row.data(), M);
        }
        if (any) keep[i] = !beaten_here;
    }
    B = keep_rows(B, keep);
    Rows out = A;
    out.v.insert(out.v.end(), B.v.begin(), B.v.end());
    out.n += B.n;
    return out;
}

// ---- the sweeps of re_fronts: the nondominated rows, sorted ---------------

// re_fronts._nd_2d: in order of f1, then f2 (a stable sort), a row whose f2
// is below every earlier one (copies: one kept)
inline std::vector<std::size_t> nd_2d(const Rows& F)
{
    std::vector<std::size_t> order(F.n);
    for (std::size_t i = 0; i < F.n; ++i) order[i] = i;
    std::stable_sort(order.begin(), order.end(), [&F](std::size_t a, std::size_t b) {
        if (F.row(a)[0] != F.row(b)[0]) return F.row(a)[0] < F.row(b)[0];
        return F.row(a)[1] < F.row(b)[1];
    });
    std::vector<std::size_t> keep;
    double best = std::numeric_limits<double>::infinity();
    for (auto k : order)
        if (F.row(k)[1] < best) {
            keep.push_back(k);
            best = F.row(k)[1];
        }
    return keep;
}

// re_fronts._nd_3d: a sweep in f1 over a staircase of (f2, f3)
inline std::vector<std::size_t> nd_3d(const Rows& F)
{
    std::vector<std::size_t> order(F.n);
    for (std::size_t i = 0; i < F.n; ++i) order[i] = i;
    std::stable_sort(order.begin(), order.end(), [&F](std::size_t a, std::size_t b) {
        for (int c = 0; c < 3; ++c)
            if (F.row(a)[c] != F.row(b)[c]) return F.row(a)[c] < F.row(b)[c];
        return false;
    });
    std::vector<double> s2, s3;
    std::vector<std::size_t> keep;
    for (auto i : order) {
        const double a = F.row(i)[1], b = F.row(i)[2];
        const std::size_t k = static_cast<std::size_t>(std::upper_bound(s2.begin(), s2.end(), a) - s2.begin());
        if (k > 0 && s3[k - 1] <= b) continue;
        keep.push_back(i);
        std::size_t j = k;
        while (j < s2.size() && s3[j] >= b) ++j;
        s2.erase(s2.begin() + static_cast<std::ptrdiff_t>(k), s2.begin() + static_cast<std::ptrdiff_t>(j));
        s3.erase(s3.begin() + static_cast<std::ptrdiff_t>(k), s3.begin() + static_cast<std::ptrdiff_t>(j));
        s2.insert(s2.begin() + static_cast<std::ptrdiff_t>(k), a);
        s3.insert(s3.begin() + static_cast<std::ptrdiff_t>(k), b);
    }
    return keep;
}

inline std::vector<std::size_t> nd_sweep(const Rows& F)
{
    return F.m == 2 ? nd_2d(F) : nd_3d(F);
}

// the first n rows of F in DSS order, in F's own frame
inline Rows dss_thinned(const Rows& F, std::size_t n)
{
    if (F.n == 0) return F;
    std::vector<double> lo(F.m), hi(F.m);
    for (std::size_t j = 0; j < F.m; ++j) {
        lo[j] = hi[j] = F.row(0)[j];
        for (std::size_t i = 1; i < F.n; ++i) {
            lo[j] = std::min(lo[j], F.row(i)[j]);
            hi[j] = std::max(hi[j], F.row(i)[j]);
        }
    }
    return take(F, select::dss_order(F, n, lo, hi));
}

// ---- DAS-CMOP (dascmop.py) ------------------------------------------------

struct DasParams {
    double b = 0.0, d = 0.0, e = 0.0, r = 0.0;
};

// dascmop.params: (b, d, e, r) of the triplet (eta, zeta, gamma), D = 0.5
inline DasParams das_params(double eta, double zeta, double gamma)
{
    constexpr double kD = 0.5;
    DasParams p;
    p.b = 2.0 * eta - 1.0;
    p.d = zeta == 0.0 ? 0.0 : kD;
    p.e = zeta == 0.0 ? std::numeric_limits<double>::infinity() : kD - dm::log(zeta);
    p.r = gamma / 2.0;
    return p;
}

// dascmop.constraints_c of the row f (x1, x2, g its point's): every c >= -1e-12
inline bool das_ok(int i, const double* f, double x1, double x2, double g, const DasParams& p)
{
    namespace dc = suite::dascmop;
    constexpr double tol = -1e-12;
    const double c_g = std::isinf(p.e) ? g - p.d : p.e == p.d ? dc::kEqEps - std::abs(g - p.d) : (p.e - g) * (g - p.d);
    if (dm::sin(dc::kA * dm::kPi * x1) - p.b < tol || c_g < tol) return false;
    if (dc::n_obj(i) == 2) {
        for (int k = 0; k < 9; ++k) {
            const double f1 = f[0] - dc::kP[k], f2 = f[1] - dc::kQ[k];
            const double u = f1 * dc::kCosT - f2 * dc::kSinT, v = f1 * dc::kSinT + f2 * dc::kCosT;
            if (u * u / dc::kAk2 + v * v / dc::kBk2 - p.r < tol) return false;
        }
        return true;
    }
    if (dm::cos(dc::kA * dm::kPi * x2) - p.b < tol) return false;
    const double sq[3] = {f[0] * f[0], f[1] * f[1], f[2] * f[2]};
    const double total = suite::row_sum(sq, 3);
    double centre[3];
    for (int k = 0; k < 3; ++k) {
        if (total - sq[k] + (f[k] - 1.0) * (f[k] - 1.0) - p.r * p.r < tol) return false;
        const double c = f[k] - dc::kInvSqrt3;
        centre[k] = c * c;
    }
    return suite::row_sum(centre, 3) - p.r * p.r >= tol;
}

// dascmop.front: the feasible points of a grid of (x1, g) — (x1, x2, g) at
// three objectives — nondominated, each at its least feasible g (40
// bisections down from its grid value), nondominated again, n by DSS
inline Rows dascmop(int i, double eta, double zeta, double gamma, std::size_t n)
{
    namespace dc = suite::dascmop;
    const DasParams p = das_params(eta, zeta, gamma);
    const double top = std::isinf(p.e) ? p.d + 1.0 : p.e;
    const std::size_t M = static_cast<std::size_t>(dc::n_obj(i));
    std::vector<double> X1, X2, G;
    const std::vector<double> gs = linspace(p.d, top, 401);
    if (M == 2) {
        for (double a : linspace(0.0, 1.0, 20001))
            for (double g : gs) {
                X1.push_back(a);
                X2.push_back(0.0);
                G.push_back(g);
            }
    } else {
        const std::vector<double> x = linspace(0.0, 1.0, 301), g3 = linspace(p.d, top, 41);
        for (double a : x)
            for (double b : x)
                for (double g : g3) {
                    X1.push_back(a);
                    X2.push_back(b);
                    G.push_back(g);
                }
    }
    auto point = [&](double a, double b, double g, double* f) {
        dc::shape(i, a, b, f);
        for (std::size_t k = 0; k < M; ++k) f[k] = f[k] + g;
    };
    Rows F = rows_of(0, M);
    std::vector<double> x1, x2, gg, f(M);
    for (std::size_t k = 0; k < G.size(); ++k) {
        point(X1[k], X2[k], G[k], f.data());
        if (!das_ok(i, f.data(), X1[k], X2[k], G[k], p)) continue;
        push_row(F, f.data());
        x1.push_back(X1[k]);
        x2.push_back(X2[k]);
        gg.push_back(G[k]);
    }
    if (F.n == 0) return F;
    const auto nd1 = nd_sweep(F);
    const double step = M == 2 ? gs[1] - gs[0] : (top - p.d) / 40.0;
    Rows out = rows_of(0, M);
    for (auto k : nd1) {
        double lo = p.e == p.d ? p.d - dc::kEqEps : std::max(gg[k] - step, p.d), hi = gg[k];
        for (int it = 0; it < 40; ++it) {
            const double mid = 0.5 * (lo + hi);
            point(x1[k], x2[k], mid, f.data());
            if (das_ok(i, f.data(), x1[k], x2[k], mid, p)) hi = mid;
            else lo = mid;
        }
        point(x1[k], x2[k], hi, f.data());
        push_row(out, f.data());
    }
    out = take(out, nd_sweep(out));
    return dss_thinned(out, n);
}

// ---- LIR-CMOP (lir_cmop.py) -----------------------------------------------

// lir_cmop.constraints_c of 5-12 at the objectives (f1, f2): every c >= 0
inline bool lir_ok(int i, double f1, double f2)
{
    namespace lc = suite::lircmop;
    auto ell = [&](double p, double q, double a, double b) { return lc::ellipse(f1, f2, p, q, a, b) >= 0.0; };
    switch (i) {
        case 5: return ell(1.6, 1.6, 2.0, 4.0) && ell(2.5, 2.5, 2.0, 8.0);
        case 6: return ell(1.8, 1.8, 2.0, 8.0) && ell(2.8, 2.8, 2.0, 8.0);
        case 7:
        case 8: return ell(1.2, 1.2, 2.0, 6.0) && ell(2.25, 2.25, 2.5, 12.0) && ell(3.5, 3.5, 2.5, 10.0);
        default: break;
    }
    static constexpr double kEll[4][4] = {{1.4, 1.4, 1.5, 6.0}, {1.1, 1.2, 2.0, 4.0},
                                          {1.2, 1.2, 1.5, 5.0}, {1.6, 1.6, 1.5, 6.0}};
    static constexpr double kSineE[4] = {2.0, 1.0, 2.1, 2.5};
    const double* q = kEll[i - 9];
    return ell(q[0], q[1], q[2], q[3]) &&
           f1 * lc::kSinA + f2 * lc::kCosA - dm::sin(4.0 * dm::kPi * (f1 * lc::kCosA - f2 * lc::kSinA)) - kSineE[i - 9] >= 0.0;
}

inline double lir_shape(int i, double u)
{
    return suite::lircmop::sqrt_shape(i) ? 1.0 - std::sqrt(u) : 1.0 - u * u;
}

// lir_cmop._least_feasible: (f1, the least feasible attainable f2) — a scan
// up from the floor by 1e-3 (4 above it at most), then 40 bisections below
// the first feasible step; an f1 with none left out
inline Rows lir_least_feasible(int i, const std::vector<double>& f1)
{
    constexpr double kShift = suite::lircmop::kShift;
    Rows out = rows_of(0, 2);
    for (double a : f1) {
        const double base = i <= 8 ? lir_shape(i, std::min(std::max(a - kShift, 0.0), 1.0)) + kShift
                                   : (1.0 + kShift) * lir_shape(i, std::min(std::max(a / (1.0 + kShift), 0.0), 1.0));
        std::size_t k = 0;
        for (; k <= 4000; ++k)
            if (lir_ok(i, a, base + 1e-3 * static_cast<double>(k))) break;
        if (k > 4000) continue;
        double hi = base + 1e-3 * static_cast<double>(k), lo = k > 0 ? base + 1e-3 * static_cast<double>(k - 1) : hi;
        for (int it = 0; it < 40; ++it) {
            const double mid = 0.5 * (lo + hi);
            if (lir_ok(i, a, mid)) hi = mid;
            else lo = mid;
        }
        const double r[2] = {a, hi};
        push_row(out, r);
    }
    return out;
}

// lir_cmop._scan_front: 6001 values of f1, and 40 times finer where the front
// is steep (neighbours one step of f1 apart and over 3 median steps apart)
inline Rows lir_scan(int i)
{
    const double lo1 = i <= 8 ? suite::lircmop::kShift : 0.0;
    const std::vector<double> f1 = linspace(lo1, lo1 + 3.0, 6001);
    Rows coarse = lir_least_feasible(i, f1);
    const Rows F = take(coarse, nd_2d(coarse));
    if (F.n < 2) return coarse;
    std::vector<double> along(F.n - 1);
    for (std::size_t k = 0; k + 1 < F.n; ++k) {
        const double dx = F.row(k + 1)[0] - F.row(k)[0], dy = F.row(k + 1)[1] - F.row(k)[1];
        along[k] = std::sqrt(dx * dx + dy * dy);
    }
    const double med = run_metrics::median(along), df = f1[1] - f1[0];
    std::vector<double> fine;
    for (std::size_t k = 0; k + 1 < F.n; ++k)
        if (F.row(k + 1)[0] - F.row(k)[0] < 1.5 * df && along[k] > 3.0 * med) {
            const std::vector<double> t = linspace(F.row(k)[0], F.row(k + 1)[0], 41);
            fine.insert(fine.end(), t.begin() + 1, t.end() - 1);
        }
    if (fine.empty()) return coarse;
    const Rows more = lir_least_feasible(i, fine);
    coarse.v.insert(coarse.v.end(), more.v.begin(), more.v.end());
    coarse.n += more.n;
    return coarse;
}

// lir_cmop.front
inline Rows lircmop(int i, std::size_t n)
{
    namespace lc = suite::lircmop;
    Rows F = rows_of(0, i >= 13 ? 3 : 2);
    if (i >= 13) {
        const double rad = i == 13 ? lc::kShift + 1.0 : std::sqrt(3.0625);
        const std::size_t k = static_cast<std::size_t>(std::ceil(std::sqrt(2.0 * static_cast<double>(n))));
        const std::vector<double> lin = linspace(0.0, 1.0, k);
        std::vector<std::vector<double>> rows;
        for (double a : lin)
            for (double b : lin) {
                const double ca = dm::cos(0.5 * dm::kPi * a);
                rows.push_back({round12(rad * (ca * dm::cos(0.5 * dm::kPi * b))),
                                round12(rad * (ca * dm::sin(0.5 * dm::kPi * b))), round12(rad * dm::sin(0.5 * dm::kPi * a))});
            }
        std::sort(rows.begin(), rows.end());                   // np.unique(np.round(F, 12), axis=0)
        rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
        for (const auto& r : rows) push_row(F, r.data());
    } else if (i <= 4) {
        for (double u : linspace(0.0, 1.0, 20001)) {
            if (i >= 3 && !(dm::sin(lc::kC * dm::kPi * u) - 0.5 >= 0.0)) continue;
            const double r[2] = {u + lc::kB, lir_shape(i, u) + lc::kB};
            push_row(F, r);
        }
        F = take(F, nd_2d(F));
    } else {
        F = lir_scan(i);
        F = take(F, nd_2d(F));
    }
    return F.n <= n ? F : dss_thinned(F, n);
}

// ---- DOC (doc.py) ---------------------------------------------------------

// doc._circle_ends: where f2 = 1 - sqrt(f1) leaves DOC2's circle about f1 = p
// (radius 0.15), or the box's end; bisection to the last bit
inline std::pair<double, double> doc_circle_ends(double p)
{
    auto inside = [p](double u) {
        const double a = u - p, b = std::sqrt(u) - std::sqrt(p);
        return a * a + b * b <= 0.0225;
    };
    double ends[2];
    const double far[2] = {0.0, 1.0};
    for (int k = 0; k < 2; ++k) {
        if (inside(far[k])) {
            ends[k] = far[k];
            continue;
        }
        double lo = p, hi = far[k];
        for (int it = 0; it < 200; ++it) {
            const double mid = 0.5 * (lo + hi);
            if (mid == lo || mid == hi) break;
            if (inside(mid)) lo = mid;
            else hi = mid;
        }
        ends[k] = lo;
    }
    return {ends[0], ends[1]};
}

// doc._line_points: (k / 20, 1 - k / 20)
inline void doc_line_points(Rows& F, int from, int to)
{
    for (int k = from; k <= to; ++k) {
        const double u = static_cast<double>(k) / 20.0, r[2] = {u, 1.0 - u};
        push_row(F, r);
    }
}

// doc.front
inline Rows doc(int i, std::size_t n)
{
    Rows F = rows_of(0, i >= 8 ? 3 : 2);
    if (i == 4) {
        doc_line_points(F, 0, 20);
        return F;
    }
    if (i == 5) {
        doc_line_points(F, 0, 8);
        doc_line_points(F, 16, 20);
        return F;
    }
    if (i == 1) {
        for (double u : linspace(0.0, 1.0, 20001)) {
            const double r[2] = {u, std::sqrt(1.0 - u * u)};
            push_row(F, r);
        }
    } else if (i == 2) {
        for (double p : {1.0 / 8.0, 0.5, 7.0 / 8.0}) {
            const auto [a, b] = doc_circle_ends(p);
            for (double u : linspace(a, b, 8001)) {
                const double r[2] = {u, 1.0 - std::sqrt(u)};
                push_row(F, r);
            }
        }
    } else if (i == 3) {
        std::vector<double> cuts = {0.0, 0.5 * dm::kPi};
        for (double c : {-0.5, 0.0, 0.5})
            for (double s : {-0.1, 0.1}) cuts.push_back(dm::acos((c + s) / std::sqrt(2.0)) - 0.25 * dm::kPi);
        std::sort(cuts.begin(), cuts.end());
        auto kept = [](double f1, double f2) {
            const double d = f1 - f2;
            return std::fabs(d - 0.5) >= 0.1 && std::fabs(d) >= 0.1 && std::fabs(d + 0.5) >= 0.1;
        };
        for (std::size_t k = 0; k + 1 < cuts.size(); ++k) {
            const double mid = 0.5 * (cuts[k] + cuts[k + 1]);
            if (!kept(dm::cos(mid), dm::sin(mid))) continue;
            for (double t : linspace(cuts[k], cuts[k + 1], 5001)) {
                const double r[2] = {dm::cos(t), dm::sin(t)};
                push_row(F, r);
            }
        }
    } else if (i == 6 || i == 7) {
        for (double u : linspace(0.0, i == 6 ? 0.5 : 0.45, 20001)) {
            const double r[2] = {u, 1.0 - u};
            push_row(F, r);
        }
        doc_line_points(F, 11, 20);
    } else if (i == 8) {
        constexpr int H = 60;                            // f3 = 0.4 and 0.6 on the grid
        for (int a = 0; a <= H; ++a)
            for (int b = 0; b <= H - a; ++b) {
                const double r[3] = {a / double(H), b / double(H), (H - a - b) / double(H)};
                if (r[2] <= 0.4 + 1e-12 || r[2] >= 0.6 - 1e-12) push_row(F, r);
            }
    } else if (i == 9) {
        for (double t : linspace(0.0, 0.5 * dm::kPi, 20001)) {
            const double r[3] = {dm::cos(t), dm::sin(t), 0.0};
            push_row(F, r);
        }
    } else {
        throw std::invalid_argument("DOC" + std::to_string(i) + ": 1 to 9");
    }
    return F.n <= n ? F : dss_thinned(F, n);
}

// ---- MMF, SYM-PART, Omni-test (mmo.py) ------------------------------------

// mmo.g_star: g's least value in the box, the front's
inline double mmo_g_star(suite::mmo::Which w)
{
    namespace mm = suite::mmo;
    using W = mm::Which;
    switch (w) {
        case W::MMF10: return mm::g10(mm::argmin(mm::g10, 0.1, 1.1));
        case W::MMF11: return mm::g11(mm::argmin(mm::g11, 0.1, 1.1));
        case W::MMF12: return mm::g11(mm::argmin(mm::g11, 0.0, 1.0));
        case W::MMF13: return mm::g11(mm::argmin(mm::g11, 0.1 + std::sqrt(0.1), 1.1 + std::sqrt(1.1)));
        case W::MMF15:
        case W::MMF15a: return mm::g15(mm::argmin(mm::g15, 0.0, 1.0));
        default: return 1.0;                              // MMF9, MMF14
    }
}

// mmo.front: about n points of the global front
inline Rows mmo(const std::string& name, std::size_t n)
{
    namespace mm = suite::mmo;
    using W = mm::Which;
    W w{};
    if (!mm::parse(name, w)) throw std::invalid_argument("not an MMF problem: " + name);
    const std::size_t k = std::max<std::size_t>(n, 2);
    Rows F = rows_of(0, mm::n_obj(w) == 3 ? 3 : 2);
    auto two = [&F](double a, double b) {
        const double r[2] = {a, b};
        push_row(F, r);
    };
    switch (w) {
        case W::SymSimple:
        case W::SymRotated:
            for (double v : linspace(0.0, 1.0, k)) two(4 * mm::kA * mm::kA * v * v, 4 * mm::kA * mm::kA * ((1 - v) * (1 - v)));
            return F;
        case W::Omni:
            for (double th : linspace(dm::kPi, 1.5 * dm::kPi, k)) two(3.0 * dm::sin(th), 3.0 * dm::cos(th));
            return F;
        case W::MMF4:
            for (double f1 : linspace(0.0, 1.0, k)) two(f1, 1.0 - f1 * f1);
            return F;
        case W::MMF8:
            for (double f1 : linspace(0.0, 1.0, k)) two(f1, std::sqrt(1.0 - f1 * f1));
            return F;
        case W::MMF9:
        case W::MMF10:
        case W::MMF11:
        case W::MMF13: {
            const double g = mmo_g_star(w);
            for (double f1 : linspace(0.1, 1.1, k)) two(f1, g / f1);
            return F;
        }
        case W::MMF12: {
            // mmo._mmf12_pieces: the f1 of the nondominated part of f2 = g h(f1, g)
            const double g = mmo_g_star(w);
            auto f2_of = [g](double f1) {
                const double t = f1 / g;
                return g * (1.0 - t * t - t * dm::sin(2.0 * dm::kPi * mm::kQ * f1));
            };
            std::vector<double> pieces;
            double best = std::numeric_limits<double>::infinity();
            for (double f1 : linspace(0.0, 1.0, 200001)) {
                const double f2 = f2_of(f1);
                if (pieces.empty() || f2 < best) pieces.push_back(f1);
                best = std::min(best, f2);
            }
            for (double at : linspace(0.0, static_cast<double>(pieces.size() - 1), k)) {
                const double f1 = pieces[static_cast<std::size_t>(at)];   // .astype(int): truncated
                two(f1, f2_of(f1));
            }
            return F;
        }
        case W::MMF14:
        case W::MMF14a:
        case W::MMF15:
        case W::MMF15a: {
            const std::size_t q = std::max<std::size_t>(static_cast<std::size_t>(std::sqrt(static_cast<double>(n))), 2);
            const std::vector<double> lin = linspace(0.0, 0.5 * dm::kPi, q);
            const double r = 1.0 + mmo_g_star(w);
            for (std::size_t i = 0; i < q; ++i)          // np.meshgrid, indexing "xy": a along a row
                for (std::size_t j = 0; j < q; ++j) {
                    const double a = lin[j], b = lin[i];
                    const double row[3] = {r * (dm::cos(a) * dm::cos(b)), r * (dm::cos(a) * dm::sin(b)), r * dm::sin(a)};
                    push_row(F, row);
                }
            return F;
        }
        default:                                          // MMF1, 1_z, 1_e, 2, 3, 5, 6, 7
            for (double f1 : linspace(0.0, 1.0, k)) two(f1, 1.0 - std::sqrt(f1));
            return F;
    }
}

}   // namespace detail

// The rows of the sampler a front's spec names, for n.
inline Rows sample(const json::Value& spec, std::size_t n, const Data& data = {})
{
    using namespace detail;
    const std::string kind = field(spec, "kind").s;
    auto M = [&] { return count(spec, "M"); };
    if (kind == "dtlz1") {
        Rows F = simplex(M(), n);
        for (double& v : F.v) v *= 0.5;
        return F;
    }
    if (kind == "dtlz2") return sphere_oct(M(), n);
    if (kind == "dtlz5") return dtlz5(M(), n);
    if (kind == "dtlz7") return dtlz7(M(), n);
    if (kind == "idtlz1") {
        Rows F = simplex(M(), n);
        for (double& v : F.v) v = 0.5 - 0.5 * v;
        return F;
    }
    if (kind == "idtlz2") {
        Rows F = sphere_oct(M(), n);
        for (double& v : F.v) v = 1.0 - v;
        return F;
    }
    if (kind == "minus_dtlz2") {
        const double k = number(field(spec, "k"));
        Rows F = sphere_oct(M(), n);
        for (double& v : F.v) v = -(1.0 + 0.25 * k) * v;
        return F;
    }
    if (kind == "sdtlz1" || kind == "sdtlz2") {
        const json::Value& scale = field(spec, "scale");
        Rows F = kind == "sdtlz1" ? simplex(M(), n) : sphere_oct(M(), n);
        if (kind == "sdtlz1")
            for (double& v : F.v) v *= 0.5;
        for (std::size_t i = 0; i < F.n; ++i)
            for (std::size_t j = 0; j < F.m; ++j) F.row(i)[j] *= number(scale.items[j]);
        return F;
    }
    if (kind == "zdt1" || kind == "bt_concave" || kind == "mop1")
        return curve(linspace(0.0, 1.0, n), [](double f1) { return 1.0 - std::sqrt(f1); });
    if (kind == "zdt2") return curve(linspace(0.0, 1.0, n), [](double f1) { return 1.0 - f1 * f1; });
    if (kind == "zdt3") return zdt3(n);
    if (kind == "zdt6") return curve(linspace(0.280775, 1.0, n), [](double f1) { return 1.0 - f1 * f1; });
    if (kind == "bt5")
        return nondominated(curve(linspace(0.0, 1.0, n), [](double u) {
            return (1.0 - u) * (1.0 - u * dm::sin(8.5 * dm::kPi * u));
        }));
    if (kind == "bt9") return octant_grid(std::max<std::size_t>(2, static_cast<std::size_t>(std::sqrt(static_cast<double>(n)))));
    if (kind == "mop2") return curve(linspace(0.0, 1.0, n), [](double f1) { return 1.0 - f1 * f1; });
    if (kind == "mop3") {
        Rows F = rows_of(0, 2);
        for (double th : linspace(0.0, 0.5 * dm::kPi, n)) {
            const double r[2] = {dm::cos(th), dm::sin(th)};
            push_row(F, r);
        }
        return F;
    }
    if (kind == "mop4")
        return nondominated(curve(linspace(0.0, 1.0, 20 * n), [](double u) {
            const double c = dm::cos(2.0 * dm::kPi * u);
            return 1.0 - std::sqrt(u) * (c * c);
        }));
    if (kind == "mop6") return simplex_grid(3, n);
    if (kind == "mop7")
        return octant_grid(std::max<std::size_t>(2, static_cast<std::size_t>(round_half_even(std::sqrt(static_cast<double>(n))))));
    if (kind == "sch") {
        Rows F = rows_of(0, 2);
        for (double t : linspace(0.0, 1.0, n)) {
            const double x = 2.0 * t, r[2] = {x * x, (x - 2.0) * (x - 2.0)};
            push_row(F, r);
        }
        return F;
    }
    if (kind == "fon") {
        const double c = 1.0 / std::sqrt(3.0);
        Rows F = rows_of(0, 2);
        for (double t : linspace(0.0, 1.0, n)) {
            const double x = -c + 2.0 * c * t;
            const double r[2] = {1.0 - dm::exp(-3.0 * ((x - c) * (x - c))), 1.0 - dm::exp(-3.0 * ((x + c) * (x + c)))};
            push_row(F, r);
        }
        return F;
    }
    if (kind == "sph") {
        const std::size_t m = M();
        const std::vector<double> W = run_compute::detail::das_dennis(m, run_compute::detail::lattice_h(m, n));
        const std::size_t rows = W.size() / m;
        Rows F = rows_of(rows, m);
        std::vector<double> sq(m);
        for (std::size_t i = 0; i < rows; ++i) {
            for (std::size_t j = 0; j < m; ++j) sq[j] = W[i * m + j] * W[i * m + j];
            const double s = np::sum(sq.data(), m);
            for (std::size_t j = 0; j < m; ++j) F.row(i)[j] = s - 2.0 * W[i * m + j] + 1.0;
        }
        return F;
    }
    if (kind == "binary") {
        const std::string which = field(spec, "which").s;
        const long long nb = static_cast<long long>(count(spec, "n")), k = static_cast<long long>(count(spec, "k"));
        Rows F = rows_of(0, 2);
        auto add = [&F](double a, double b) {
            const double r[2] = {-a, -b};
            push_row(F, r);
        };
        if (which == "OMM" || which == "LOTZ") {
            for (long long a = 0; a <= nb; ++a) add(static_cast<double>(a), static_cast<double>(nb - a));
        } else {
            std::vector<long long> a{k};
            for (long long v = 2 * k; v <= nb; ++v) a.push_back(v);
            a.push_back(nb + k);
            for (long long v : a) add(static_cast<double>(v), static_cast<double>(nb + 2 * k - v));
        }
        return F;
    }
    if (kind == "wfg_concave") {
        Rows F = sphere_oct(M(), n);
        wfg_scale(F);
        return F;
    }
    if (kind == "wfg1") return wfg12(M(), n, 1);
    if (kind == "wfg2") return wfg12(M(), n, 2);
    if (kind == "wfg3") return wfg3(M(), n);
    if (kind == "maf1" || kind == "lsmop14") {
        Rows F = simplex(M(), n);
        if (kind == "maf1")
            for (double& v : F.v) v = 1.0 - v;
        return F;
    }
    if (kind == "maf2") return maf2(M(), n);
    if (kind == "maf3" || kind == "maf4" || kind == "maf5" || kind == "lsmop15") {
        const std::size_t m = M();
        Rows F = sphere_oct(m, n);
        for (std::size_t i = 0; i < F.n; ++i)
            for (std::size_t j = 0; j < m; ++j) {
                double& v = F.row(i)[j];
                if (kind == "maf3") v = dm::pow(v, j + 1 == m ? 2.0 : 4.0);
                else if (kind == "maf4") v = dm::pow(2.0, static_cast<double>(j + 1)) * (1.0 - v);
                else if (kind == "maf5") v = dm::pow(2.0, static_cast<double>(m - j)) * dm::pow(v, 4.0);
                else v = 1.0 - v;
            }
        return F;
    }
    if (kind == "maf8" || kind == "maf9") {
        const std::vector<double> V = unit_polygon(M());
        const Rows P = poly_interior(V, n, 7u);
        return kind == "maf8" ? vertex_distances(P, V) : edge_line_distances(P, V);
    }
    if (kind == "maf13") {
        const std::size_t m = M();
        const Rows d = sphere_oct(3, n);
        Rows F = rows_of(d.n, m);
        for (std::size_t i = 0; i < d.n; ++i) {
            const double f1 = d.row(i)[0], f2 = d.row(i)[1], f3 = d.row(i)[2];
            const double tail = f1 * f1 + dm::pow(f2, 10.0) + dm::pow(f3, 10.0);
            double* f = F.row(i);
            f[0] = f1;
            f[1] = f2;
            f[2] = f3;
            for (std::size_t j = 3; j < m; ++j) f[j] = tail;
        }
        return F;
    }
    if (kind == "polygon") {
        const std::vector<double> V = unit_polygon(M());
        return vertex_distances(poly_interior(V, n, 2024u), V);
    }
    if (kind == "ipolygon") return ipolygon(M(), n);
    if (kind == "c1_dtlz1") {
        Rows F = simplex(M(), n);
        for (double& v : F.v) v *= 0.5;
        return F;
    }
    if (kind == "c1_dtlz3") return sphere_oct(M(), n);
    if (kind == "c3_dtlz1") {
        Rows W = simplex(M(), n);
        for (std::size_t i = 0; i < W.n; ++i) {
            double lo = W.row(i)[0];
            for (std::size_t j = 1; j < W.m; ++j) lo = std::min(lo, W.row(i)[j]);
            for (std::size_t j = 0; j < W.m; ++j) W.row(i)[j] /= 1.0 + lo;
        }
        return W;
    }
    if (kind == "c3_dtlz4") {
        Rows U = sphere_oct(M(), n);
        for (std::size_t i = 0; i < U.n; ++i) {
            double hi = U.row(i)[0] * U.row(i)[0];
            for (std::size_t j = 1; j < U.m; ++j) hi = std::max(hi, U.row(i)[j] * U.row(i)[j]);
            const double d = std::sqrt(1.0 - 0.75 * hi);
            for (std::size_t j = 0; j < U.m; ++j) U.row(i)[j] /= d;
        }
        return U;
    }
    if (kind == "c2_dtlz2") return feasible_rows(Cons::C2_DTLZ2, sphere_oct(M(), 20 * n));
    if (kind == "c2_convex") return convex_c2(M(), n);
    if (kind == "lircmop") return lircmop(static_cast<int>(count(spec, "i")), n);
    if (kind == "doc") return doc(static_cast<int>(count(spec, "i")), n);
    if (kind == "mw") return mw::front(static_cast<int>(count(spec, "i")), M(), n);
    if (kind == "zcat") return zcat::front(static_cast<int>(count(spec, "i")), M(), n);
    if (kind == "mmo") return mmo(field(spec, "name").s, n);
    if (kind == "dascmop") {
        const json::Value& t = field(spec, "triplet");
        return dascmop(static_cast<int>(count(spec, "i")), number(t.items.at(0)), number(t.items.at(1)),
                       number(t.items.at(2)), n);
    }
    if (kind == "full") {
        const auto path = data.dir / "_fronts" / field(spec, "file").s;
        const auto arrays = npz::load(path);
        const auto it = arrays.find("F");
        if (it == arrays.end()) throw std::runtime_error(path.string() + ": no array F");
        const npz::Array& A = it->second;
        const std::vector<double> all = A.doubles();
        const std::size_t m = A.shape.size() > 1 ? A.shape[1] : 1;
        const std::size_t rows = std::min<std::size_t>(std::max<std::size_t>(1, n), all.size() / m);
        Rows F = rows_of(rows, m);
        std::copy(all.begin(), all.begin() + static_cast<std::ptrdiff_t>(rows * m), F.v.begin());
        return F;
    }
    throw std::invalid_argument("a front of kind \"" + kind + "\" is not built in C++ yet");
}

// The problem's reference front of n rows (as its pareto_front(n) gives it),
// its last step the deterministic filter: the rows no other row dominates,
// in their order (task 17, addendum 6, §4: which dominated row goes does not
// depend on a draw). filter = false: without it (for the count of what it drops).
inline Rows front(const json::Value& spec, std::size_t n, const Data& data = {}, bool filter = true)
{
    const json::Value* whole = spec.find("whole");
    Rows F;
    if (whole && whole->b) {
        F = sample(spec, n, data);
    } else {
        const auto seed = static_cast<std::uint32_t>(detail::field(spec, "seed").i);
        F = exactly_n([&](std::size_t k) { return sample(spec, k, data); }, n, seed);
    }
    return filter ? detail::nondominated(F) : F;
}

}   // namespace mootation::refs

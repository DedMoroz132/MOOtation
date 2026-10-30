#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// ZCAT1-20's reference fronts and Pareto sets in C++ — task 17, §2.1:
// zcat.pareto_front and pareto_sets.zcat step for step, the shapes the
// suite's (suite/zcat.hpp's position_functions, the topologies g of Table 3),
// the draws the library's (std::mt19937, rnd::) where Python took
// default_rng's — reference version 5.
//
//   front  the position vectors: a grid in the degenerate problems (m = 1),
//          else the cube's corners (at most a quarter), a grid and uniform
//          draws, the distinct ones in lexicographic order; their images
//          (f_i = i^2 alpha_i), nondominated; n of them — every objective's
//          extremes, the others drawn; each checked against 60 000 fresh
//          positions, its own moves (every position alone, both ways, 0.05
//          down to 0.001) and 64 Gaussian moves, and a beaten one replaced by
//          a checked one of the rest (_dominated_elsewhere)
//   set    y_II = g(y_I | m) where the image is nondominated among the
//          candidates and an independent sample of 20 000, and no move beats
//          it; x_i = (y_i - 1/2) i
// ============================================================================

#include <algorithm>
#include <cstdint>
#include <random>
#include <vector>

#include "../nondominated.hpp"
#include "../suite/zcat.hpp"
#include "sample.hpp"

namespace mootation::refs::zcat {

namespace z = suite::zcat;

// n_vars of ZCAT at M objectives: 10 M (zcat.n_vars)
inline std::size_t n_vars(std::size_t M) { return 10 * M; }

// i^2 alpha_i(y) of the positions y (M - 1 of them)
inline void image(int which, const double* y, std::size_t M, double* f)
{
    z::position_functions(which, y, static_cast<int>(M), f);
    for (std::size_t j = 0; j < M; ++j) {
        const double s = static_cast<double>(j + 1);
        f[j] = s * s * f[j];
    }
}

inline Rows images(int which, const Rows& Y, std::size_t M)
{
    Rows F = rows_of(Y.n, M);
    for (std::size_t i = 0; i < Y.n; ++i) image(which, Y.row(i), M, F.row(i));
    return F;
}

// k of 0 .. N-1 without replacement, in the order drawn (a partial Fisher-Yates)
template <typename G>
inline std::vector<std::size_t> choose(G& g, std::size_t N, std::size_t k)
{
    std::vector<std::size_t> idx(N);
    for (std::size_t i = 0; i < N; ++i) idx[i] = i;
    for (std::size_t i = 0; i < k && i < N; ++i) std::swap(idx[i], idx[i + static_cast<std::size_t>(rnd::below(g, N - i))]);
    idx.resize(std::min(k, N));
    return idx;
}

// does some row of V strictly dominate c?
inline bool beaten_by(const double* c, const Rows& V)
{
    for (std::size_t i = 0; i < V.n; ++i) {
        bool le = true, lt = false;
        for (std::size_t j = 0; j < V.m && le; ++j) {
            le = V.row(i)[j] <= c[j];
            lt = lt || V.row(i)[j] < c[j];
        }
        if (le && lt) return true;
    }
    return false;
}

// the moves of a check: every position alone, both ways, by 0.05 down to
// 0.001, then `reps` Gaussian moves of all d at once at each scale
template <typename G, typename F>
inline void moves(const Rows& Y, std::size_t d, int reps, G& g, F&& each)
{
    auto clip = [](double v) { return v < 0.0 ? 0.0 : v > 1.0 ? 1.0 : v; };
    for (double step : {0.05, 0.02, 0.01, 0.005, 0.002, 0.001})
        for (std::size_t j = 0; j < d; ++j)
            for (double sgn : {-1.0, 1.0}) {
                Rows Q = Y;
                for (std::size_t i = 0; i < Q.n; ++i) Q.row(i)[j] = clip(Q.row(i)[j] + sgn * step);
                each(Q);
            }
    for (double s : {0.2, 0.1, 0.05, 0.02})
        for (int r = 0; r < reps; ++r) {
            Rows Q = Y;
            for (std::size_t i = 0; i < Q.n; ++i)
                for (std::size_t j = 0; j < d; ++j) Q.row(i)[j] = clip(Q.row(i)[j] + s * rnd::normal01(g));
            each(Q);
        }
}

// zcat._dominated_elsewhere: which rows (positions Y, images F) are beaten
inline std::vector<char> dominated_elsewhere(int which, std::size_t M, const Rows& Y, const Rows& F,
                                             std::size_t samples = 60000)
{
    std::mt19937 g(static_cast<std::uint32_t>(20260922 + 100 * M));
    const std::size_t d = z::degenerate(which) ? 1 : M - 1;
    Rows P = rows_of(samples, Y.m);
    for (std::size_t i = 0; i < samples; ++i)
        for (std::size_t j = 0; j < d; ++j) P.row(i)[j] = rnd::uniform01(g);
    const Rows fresh = images(which, P, M);
    std::vector<char> bad(F.n, 0);
    for (std::size_t i = 0; i < F.n; ++i) bad[i] = beaten_by(F.row(i), fresh);
    moves(Y, d, 16, g, [&](const Rows& Q) {
        const Rows G = images(which, Q, M);
        for (std::size_t i = 0; i < F.n; ++i) {
            bool le = true, lt = false;
            for (std::size_t j = 0; j < M && le; ++j) {
                le = G.row(i)[j] <= F.row(i)[j];
                lt = lt || G.row(i)[j] < F.row(i)[j];
            }
            if (le && lt) bad[i] = 1;
        }
    });
    return bad;
}

// rows of R in lexicographic order, the distinct ones (np.unique(R, axis=0))
inline Rows unique_sorted(const Rows& R)
{
    std::vector<std::vector<double>> rows(R.n);
    for (std::size_t i = 0; i < R.n; ++i) rows[i].assign(R.row(i), R.row(i) + R.m);
    std::sort(rows.begin(), rows.end());
    rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
    Rows out = rows_of(0, R.m);
    for (const auto& r : rows) push_row(out, r.data());
    return out;
}

// the cube's corners of [0, 1]^d (row c: bit j of c), at most `most` of them
// (drawn where there are more), none above 14 dimensions
template <typename G>
inline Rows corners(std::size_t d, std::size_t most, G& g)
{
    Rows C = rows_of(0, d);
    if (d > 14) return C;
    std::vector<double> row(d);
    for (std::size_t c = 0; c < (std::size_t{1} << d); ++c) {
        for (std::size_t j = 0; j < d; ++j) row[j] = static_cast<double>((c >> j) & 1);
        push_row(C, row.data());
    }
    return C.n > most ? take(C, choose(g, C.n, most)) : C;
}

// zcat.pareto_front
inline Rows front(int which, std::size_t M, std::size_t n)
{
    const std::size_t m = z::degenerate(which) ? 1 : M - 1;
    std::mt19937 g(static_cast<std::uint32_t>(20260916 + 100 * M));
    const std::size_t cand = std::max<std::size_t>(4000, 8 * n);
    Rows Y = rows_of(0, M - 1);
    if (m == 1) {
        std::vector<double> row(M - 1, 0.0);           // the other positions fixed at 0
        for (double t : linspace(0.0, 1.0, cand)) {
            row[0] = t;
            push_row(Y, row.data());
        }
    } else {
        Rows C = corners(m, cand / 4, g);
        const std::size_t room = cand - C.n;
        const long long p = std::max<long long>(
            2, round_half_even(dm::pow(static_cast<double>(room / 2), 1.0 / static_cast<double>(m))));
        const std::vector<double> grid = linspace(0.0, 1.0, static_cast<std::size_t>(p));
        Rows mesh = rows_of(0, m);                      // np.meshgrid(..., indexing="ij"): the last fastest
        std::vector<std::size_t> digit(m, 0);
        std::vector<double> row(m);
        for (;;) {
            for (std::size_t j = 0; j < m; ++j) row[j] = grid[digit[j]];
            push_row(mesh, row.data());
            std::size_t j = m;
            while (j > 0 && ++digit[j - 1] == grid.size()) digit[--j] = 0;
            if (j == 0) break;
        }
        if (mesh.n > room / 2) mesh = take(mesh, choose(g, mesh.n, room / 2));
        Rows all = C;
        all.v.insert(all.v.end(), mesh.v.begin(), mesh.v.end());
        all.n += mesh.n;
        for (std::size_t k = mesh.n; k < room; ++k) {
            for (std::size_t j = 0; j < m; ++j) row[j] = rnd::uniform01(g);
            push_row(all, row.data());
        }
        Y = unique_sorted(all);
    }
    Rows F = images(which, Y, M);
    const std::vector<char> keep = nd::mask(F.v.data(), F.n, F.m);
    std::vector<std::size_t> idx;
    for (std::size_t i = 0; i < F.n; ++i)
        if (keep[i]) idx.push_back(i);
    Y = take(Y, idx);
    F = take(F, idx);
    std::vector<std::size_t> sel(F.n);
    for (std::size_t i = 0; i < F.n; ++i) sel[i] = i;
    if (F.n > n) {
        std::vector<std::size_t> ext;                   // np.unique(argmin, argmax)
        for (std::size_t j = 0; j < M; ++j) {
            std::size_t lo = 0, hi = 0;
            for (std::size_t i = 1; i < F.n; ++i) {
                if (F.row(i)[j] < F.row(lo)[j]) lo = i;
                if (F.row(i)[j] > F.row(hi)[j]) hi = i;
            }
            ext.push_back(lo);
            ext.push_back(hi);
        }
        std::sort(ext.begin(), ext.end());
        ext.erase(std::unique(ext.begin(), ext.end()), ext.end());
        if (ext.size() >= n) {
            ext.resize(n);
            sel = ext;
        } else {
            std::vector<std::size_t> rest;
            for (std::size_t i = 0; i < F.n; ++i)
                if (!std::binary_search(ext.begin(), ext.end(), i)) rest.push_back(i);
            sel = ext;
            for (auto k : choose(g, rest.size(), n - ext.size())) sel.push_back(rest[k]);
        }
    }
    const std::vector<char> bad = dominated_elsewhere(which, M, take(Y, sel), take(F, sel));
    if (std::find(bad.begin(), bad.end(), 1) != bad.end()) {
        std::vector<char> in(F.n, 0);
        for (auto s : sel) in[s] = 1;
        std::vector<std::size_t> pool;
        for (std::size_t i = 0; i < F.n; ++i)
            if (!in[i]) pool.push_back(i);
        std::mt19937 gp(static_cast<std::uint32_t>(20260923 + 100 * M));
        rnd::shuffle(pool.begin(), pool.end(), gp);
        std::vector<std::size_t> good;
        std::size_t need = 0;
        for (std::size_t k = 0; k < sel.size(); ++k) {
            if (bad[k]) ++need;
            else good.push_back(sel[k]);
        }
        for (std::size_t start = 0; start < pool.size() && need > 0; start += 256) {
            const std::vector<std::size_t> chunk(pool.begin() + static_cast<std::ptrdiff_t>(start),
                                                 pool.begin() + static_cast<std::ptrdiff_t>(std::min(pool.size(), start + 256)));
            const std::vector<char> b = dominated_elsewhere(which, M, take(Y, chunk), take(F, chunk));
            for (std::size_t k = 0; k < chunk.size() && need > 0; ++k)
                if (!b[k]) {
                    good.push_back(chunk[k]);
                    --need;
                }
        }
        sel = good;
    }
    return take(F, sel);
}

// pareto_sets._zcat_points: the full normalised points (y_II = g(y_I | m), m
// per point) of the positions P, and their images
inline void set_points(int which, std::size_t M, const Rows& P, Rows& Y, Rows& F)
{
    const std::size_t nv = n_vars(M);
    Y = rows_of(P.n, nv);
    for (std::size_t r = 0; r < P.n; ++r) {
        double* y = Y.row(r);
        std::copy(P.row(r), P.row(r) + P.m, y);
        const int m = z::position_count(which, y, static_cast<int>(M));
        z::topology(z::kTopology[which], y, m, static_cast<int>(nv), y + m);
    }
    F = rows_of(P.n, M);
    for (std::size_t r = 0; r < P.n; ++r) image(which, Y.row(r), M, F.row(r));
}

// pareto_sets.zcat
inline Rows pareto_set(int which, std::size_t M, std::size_t n, std::size_t verify = 20000)
{
    const std::size_t nv = n_vars(M);
    std::mt19937 g(static_cast<std::uint32_t>(20260922 + 100 * M));
    const std::size_t cand = std::max<std::size_t>(4000, 4 * n);
    const std::size_t d = z::degenerate(which) ? 1 : M - 1;
    Rows P = rows_of(0, d);
    std::vector<double> row(d);
    if (d == 1) {
        for (double t : linspace(0.0, 1.0, cand)) push_row(P, &t);
        for (std::size_t k = 0; k < verify; ++k) {
            const double u = rnd::uniform01(g);
            push_row(P, &u);
        }
    } else {
        P = corners(d, cand / 4, g);
        const std::size_t more = cand - P.n + verify;
        for (std::size_t k = 0; k < more; ++k) {
            for (std::size_t j = 0; j < d; ++j) row[j] = rnd::uniform01(g);
            push_row(P, row.data());
        }
    }
    Rows Y, F;
    set_points(which, M, P, Y, F);
    const std::size_t k = P.n - verify;                 // the candidates come first
    std::vector<std::size_t> keep;
    for (std::size_t i = 0; i < k; ++i)
        if (!beaten_by(F.row(i), F)) keep.push_back(i);
    if (keep.size() > n + n / 4) {
        std::vector<std::size_t> pick;
        for (auto j : choose(g, keep.size(), n + n / 4)) pick.push_back(keep[j]);
        std::sort(pick.begin(), pick.end());
        keep = pick;
    }
    const Rows Pk = take(P, keep), Fk = take(F, keep);
    std::vector<char> bad(keep.size(), 0);
    moves(Pk, d, 8, g, [&](const Rows& Q) {
        Rows YQ, FQ;
        set_points(which, M, Q, YQ, FQ);
        for (std::size_t i = 0; i < Fk.n; ++i) {
            bool le = true, lt = false;
            for (std::size_t j = 0; j < M && le; ++j) {
                le = FQ.row(i)[j] <= Fk.row(i)[j];
                lt = lt || FQ.row(i)[j] < Fk.row(i)[j];
            }
            if (le && lt) bad[i] = 1;
        }
    });
    std::vector<std::size_t> good;
    for (std::size_t i = 0; i < keep.size(); ++i)
        if (!bad[i]) good.push_back(keep[i]);
    Rows X = take(Y, good);
    if (X.n > n) {
        std::vector<std::size_t> pick = choose(g, X.n, n);
        std::sort(pick.begin(), pick.end());
        X = take(X, pick);
    }
    for (std::size_t r = 0; r < X.n; ++r)
        for (std::size_t j = 0; j < nv; ++j) X.row(r)[j] = (X.row(r)[j] - 0.5) * static_cast<double>(j + 1);
    return X;
}

}   // namespace mootation::refs::zcat

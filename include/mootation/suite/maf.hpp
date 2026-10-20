#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// MaF1-9 and MaF13-15 (Cheng et al. 2017) as the library evaluates them:
// MaF1, 3, 4 and 5 as benchmarks/maf.py, MaF2, 6, 7, 8, 9 and 13 as
// registry.py, _maf_register, and MaF14 and MaF15 as maf.py, lsmop_rows
// (task 12 of 2026-10-19). MaF8's distances are hypot (dm::hypot; np.hypot
// in the NumPy code; version 1 took CPython's math.hypot, an algorithm of its
// own, version 2 the C library's). MaF10-12 are WFG's.
// The order of every operation is that of the NumPy rows (common.hpp).
// ============================================================================

#include <cstdlib>
#include <vector>

#include "../detmath.hpp"
#include "common.hpp"

namespace mootation::suite::maf {

using Vec = std::vector<double>;

// sum (x - 1/2)^2 over the distance block (maf._g_quad_rows)
inline double g_quad(const double* xm, int k)
{
    Vec t(static_cast<std::size_t>(k));
    for (int j = 0; j < k; ++j) {
        const double d = xm[j] - 0.5;
        t[static_cast<std::size_t>(j)] = d * d;
    }
    return row_sum(t);
}

// 100 (k + sum((x - 1/2)^2 - cos(20 pi (x - 1/2)))) (maf._g_rastrigin_rows)
inline double g_rastrigin(const double* xm, int k)
{
    Vec t(static_cast<std::size_t>(k));
    for (int j = 0; j < k; ++j) {
        const double d = xm[j] - 0.5;
        t[static_cast<std::size_t>(j)] = d * d - dm::cos(20.0 * kPi * d);
    }
    return 100.0 * (k + row_sum(t));
}

// The DTLZ1 products of positions P, summing to 1 (maf._dtlz1_products_rows):
// M values.
inline void products(const double* P, int M, double* pr)
{
    for (int i = 0; i < M; ++i) {
        if (i == 0) {
            pr[i] = row_prod(P, static_cast<std::size_t>(M - 1));
        } else {
            const double pre = M - 1 - i > 0 ? row_prod(P, static_cast<std::size_t>(M - 1 - i)) : 1.0;
            pr[i] = pre * (1.0 - P[M - 1 - i]);
        }
    }
}

// The DTLZ2 directions of positions P (maf._dtlz2_dirs_rows): M values.
inline void directions(const double* P, int M, double* d)
{
    Vec c(static_cast<std::size_t>(M - 1)), s(static_cast<std::size_t>(M - 1));
    for (int i = 0; i < M - 1; ++i) {
        c[static_cast<std::size_t>(i)] = dm::cos(0.5 * kPi * P[i]);
        s[static_cast<std::size_t>(i)] = dm::sin(0.5 * kPi * P[i]);
    }
    d[0] = row_prod(c.data(), static_cast<std::size_t>(M - 1));
    for (int i = 1; i < M; ++i) {
        const double pre = M - 1 - i > 0 ? row_prod(c.data(), static_cast<std::size_t>(M - 1 - i)) : 1.0;
        d[i] = pre * s[static_cast<std::size_t>(M - 1 - i)];
    }
}

// The angle shape of MaF6 (registry.maf6_rows) and its base 1 + 100 g.
inline void maf6(const double* y, int n, int M, double* f)
{
    const double g = g_quad(y + (M - 1), n - M + 1);
    Vec theta(static_cast<std::size_t>(M - 1)), c(theta.size()), s(theta.size());
    theta[0] = y[0] * kPi / 2;
    if (M > 2) {
        const double denom = 4.0 * (1 + g);
        for (int i = 1; i < M - 1; ++i)
            theta[static_cast<std::size_t>(i)] = (kPi / denom) * (1 + 2 * g * y[i]);
    }
    for (std::size_t i = 0; i < theta.size(); ++i) {
        c[i] = dm::cos(theta[i]);
        s[i] = dm::sin(theta[i]);
    }
    const double base = 1.0 + 100.0 * g;
    f[0] = base * row_prod(c.data(), static_cast<std::size_t>(M - 1));
    for (int i = 2; i < M; ++i)
        f[i - 1] = base * row_prod(c.data(), static_cast<std::size_t>(M - i)) * s[static_cast<std::size_t>(M - i)];
    f[M - 1] = base * s[0];
}

// MaF2 (registry.maf2_rows): one g per objective over its chunk of the
// distance block.
inline void maf2(const double* y, int n, int M, double* f)
{
    const int c = (n - M + 1) / M;
    Vec cs(static_cast<std::size_t>(M - 1)), sn(cs.size()), g(static_cast<std::size_t>(M), 0.0);
    for (int i = 0; i < M - 1; ++i) {
        const double th = 0.5 * kPi * (0.5 * y[i] + 0.25);
        cs[static_cast<std::size_t>(i)] = dm::cos(th);
        sn[static_cast<std::size_t>(i)] = dm::sin(th);
    }
    auto chunk = [&](int lo, int hi) {
        Vec t(static_cast<std::size_t>(hi - lo));
        for (int j = lo; j < hi; ++j) {
            const double d = 0.5 * y[j] + 0.25 - 0.5;
            t[static_cast<std::size_t>(j - lo)] = d * d;
        }
        return row_sum(t);
    };
    for (int i = 0; i < M - 1; ++i) {
        const int lo = (M - 1) + i * c, hi = lo + c;
        if (hi > lo) g[static_cast<std::size_t>(i)] = chunk(lo, hi);
    }
    g[static_cast<std::size_t>(M - 1)] = chunk((M - 1) + (M - 1) * c, n);
    f[0] = row_prod(cs.data(), static_cast<std::size_t>(M - 1)) * (1.0 + g[0]);
    for (int i = 2; i < M; ++i)
        f[i - 1] = row_prod(cs.data(), static_cast<std::size_t>(M - i)) * sn[static_cast<std::size_t>(M - i)] *
                   (1.0 + g[static_cast<std::size_t>(i - 1)]);
    f[M - 1] = sn[0] * (1.0 + g[static_cast<std::size_t>(M - 1)]);
}

// MaF1, 3, 4, 5 (maf.py), 2, 6, 7 (registry.py) of y, n variables, M objectives.
inline void evaluate(int which, const double* y, int n, int M, double* f)
{
    const double* xm = y + (M - 1);
    const int k = n - M + 1;
    Vec d(static_cast<std::size_t>(M));
    switch (which) {
        case 1: {
            const double g = g_quad(xm, k);
            products(y, M, d.data());
            for (int i = 0; i < M; ++i) f[i] = (1.0 - d[static_cast<std::size_t>(i)]) * (1.0 + g);
            return;
        }
        case 2: maf2(y, n, M, f); return;
        case 3: {
            const double g = g_rastrigin(xm, k);
            directions(y, M, d.data());
            for (int i = 0; i < M; ++i)
                f[i] = dm::pow(d[static_cast<std::size_t>(i)] * (1.0 + g), i < M - 1 ? 4.0 : 2.0);
            return;
        }
        case 4: {
            const double g = g_rastrigin(xm, k);
            directions(y, M, d.data());
            for (int i = 0; i < M; ++i)
                f[i] = std::ldexp(1.0, i + 1) * (1.0 - d[static_cast<std::size_t>(i)]) * (1.0 + g);
            return;
        }
        case 5: {
            const double g = g_quad(xm, k);
            Vec P(static_cast<std::size_t>(M - 1));
            for (int i = 0; i < M - 1; ++i) P[static_cast<std::size_t>(i)] = dm::pow(y[i], 100.0);
            directions(P.data(), M, d.data());
            for (int i = 0; i < M; ++i)
                f[i] = std::ldexp(1.0, M - i) * dm::pow(d[static_cast<std::size_t>(i)] * (1.0 + g), 4.0);
            return;
        }
        case 6: maf6(y, n, M, f); return;
        case 7: {
            const double g = 1.0 + 9.0 / (n - M + 1) * row_sum(xm, static_cast<std::size_t>(k));
            Vec t(static_cast<std::size_t>(M - 1));
            for (int i = 0; i < M - 1; ++i) {
                f[i] = y[i];
                t[static_cast<std::size_t>(i)] = (y[i] / (1 + g)) * (1 + dm::sin(3 * kPi * y[i]));
            }
            f[M - 1] = (1 + g) * (static_cast<double>(M) - row_sum(t));
            return;
        }
        default:
            return;
    }
}

// MaF13 (registry.maf13_rows): five variables, M objectives.
inline void maf13(const double* x, int M, double* f)
{
    constexpr int D = 5;
    double Y[D];
    for (int i = 0; i < D; ++i)
        Y[i] = x[i] - 2.0 * x[1] * dm::sin(2 * kPi * x[0] + (i + 1) * kPi / D);
    // J1 = {3}, J2 = {4}, J3 = {2}, J4 = {3, 4}: (j + 1) % 3 of j = 2..4
    const double s1 = 0.0 + Y[3] * Y[3];
    const double s2 = 0.0 + Y[4] * Y[4];
    const double s3 = 0.0 + Y[2] * Y[2];
    const double s4 = 0.0 + Y[3] * Y[3] + Y[4] * Y[4];
    const double f1 = dm::sin(kPi / 2 * x[0]) + 2.0 / 1 * s1;
    const double f2 = dm::cos(kPi / 2 * x[0]) * dm::sin(kPi / 2 * x[1]) + 2.0 / 1 * s2;
    const double f3 = dm::cos(kPi / 2 * x[0]) * dm::cos(kPi / 2 * x[1]) + 2.0 / 1 * s3;
    f[0] = f1;
    f[1] = f2;
    f[2] = f3;
    const double tail = dm::pow(f1, 2.0) + dm::pow(f2, 10.0) + dm::pow(f3, 10.0) + 2.0 / 2 * s4;
    for (int i = 3; i < M; ++i) f[i] = tail;
}

// MaF9 (registry.maf9_rows): the distance from x to the line of each edge of
// the regular M-gon; lines holds (ax, ay, dx, dy, |d|) per edge, computed in
// Python.
inline void maf9(const double* x, int M, const Vec& lines, double* f)
{
    for (int m = 0; m < M; ++m) {
        const double* l = lines.data() + 5 * static_cast<std::size_t>(m);
        f[m] = std::abs(l[3] * (x[0] - l[0]) - l[2] * (x[1] - l[1])) / l[4];
    }
}

// MaF8 (registry.maf8_rows): the distance from x to each vertex of the
// regular M-gon; vertices holds (x, y) per vertex, computed in Python.
inline void maf8(const double* x, int M, const Vec& vertices, double* f)
{
    for (int m = 0; m < M; ++m) {
        const double* v = vertices.data() + 2 * static_cast<std::size_t>(m);
        f[m] = dm::hypot(x[0] - v[0], x[1] - v[1]);
    }
}

// ---- MaF14 and MaF15 (maf.py, lsmop_rows) -----------------------------------
// The eta of a subcomponent z of k values (maf._eta_*).
inline double rastrigin(const double* z, std::size_t k)
{
    Vec t(k);
    for (std::size_t i = 0; i < k; ++i) t[i] = z[i] * z[i] - 10.0 * dm::cos(2 * kPi * z[i]) + 10.0;
    return row_sum(t);
}

inline double rosenbrock(const double* z, std::size_t k)
{
    if (k < 2) return 0.0;                          // np.sum of no terms
    Vec t(k - 1);
    for (std::size_t i = 0; i + 1 < k; ++i) {
        const double a = z[i] * z[i] - z[i + 1], b = z[i] - 1.0;
        t[i] = 100.0 * (a * a) + b * b;
    }
    return row_sum(t);
}

inline double griewank(const double* z, std::size_t k)
{
    Vec t(k), c(k);
    for (std::size_t i = 0; i < k; ++i) {
        t[i] = z[i] * z[i] / 4000.0;
        c[i] = dm::cos(z[i] / std::sqrt(static_cast<double>(i + 1)));
    }
    return row_sum(t) - row_prod(c.data(), k) + 1.0;
}

inline double sphere(const double* z, std::size_t k)
{
    Vec t(k);
    for (std::size_t i = 0; i < k; ++i) t[i] = z[i] * z[i];
    return row_sum(t);
}

// MaF14 (which = 14: Rastrigin and Rosenbrock, c = I, the DTLZ1 products) or
// MaF15 (15: Griewank and the sphere, objective j reading groups j and j + 1,
// 1 minus the DTLZ2 directions) of x, n variables, M objectives. link holds
// (a_i, l_i, u_i - l_i) per variable of x^s, parts the sizes of the
// subcomponents group by group, N_k per group: both computed in Python.
inline void lsmop(int which, const double* x, int n, int M, const Vec& link, const Vec& parts, double* f)
{
    const std::size_t ns = static_cast<std::size_t>(n - M + 1), nk = parts.size() / static_cast<std::size_t>(M);
    Vec z(ns);
    for (std::size_t i = 0; i < ns; ++i) {
        const double* l = link.data() + 3 * i;
        z[i] = l[0] * (x[M - 1 + static_cast<int>(i)] - l[1]) - x[0] * l[2];
    }
    std::vector<std::size_t> first(static_cast<std::size_t>(M) + 1, 0);   // each group's first column
    for (std::size_t k = 0; k < static_cast<std::size_t>(M); ++k) {
        first[k + 1] = first[k];
        for (std::size_t s = 0; s < nk; ++s) first[k + 1] += static_cast<std::size_t>(parts[k * nk + s]);
    }
    // gbar of objective j (eta_1 for j = 0, 2, ...) over group k
    auto gbar = [&](int j, int k) {
        double acc = 0.0;
        std::size_t at = first[static_cast<std::size_t>(k)];
        for (std::size_t s = 0; s < nk; ++s) {
            const std::size_t size = static_cast<std::size_t>(parts[static_cast<std::size_t>(k) * nk + s]);
            const double* zs = z.data() + at;
            const double e = which == 14 ? (j % 2 == 0 ? rastrigin(zs, size) : rosenbrock(zs, size))
                                         : (j % 2 == 0 ? griewank(zs, size) : sphere(zs, size));
            acc = acc + e / static_cast<double>(size);
            at += size;
        }
        return acc / static_cast<double>(nk);
    };
    Vec h(static_cast<std::size_t>(M));
    if (which == 14) {
        products(x, M, h.data());
    } else {
        Vec d(static_cast<std::size_t>(M));
        directions(x, M, d.data());
        for (int j = 0; j < M; ++j) h[static_cast<std::size_t>(j)] = 1.0 - d[static_cast<std::size_t>(j)];
    }
    for (int j = 0; j < M; ++j) {
        const double g = which == 14 || j == M - 1 ? gbar(j, j) : gbar(j, j) + gbar(j, j + 1);
        f[j] = h[static_cast<std::size_t>(j)] * (1.0 + g);
    }
}

// The distance from x to the nearest edge of polygon P (n vertices, x0 y0
// x1 y1 ...): registry._poly_edge_distance.
inline double poly_edge_distance(const double* x, const double* P, std::size_t n)
{
    double best = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < n; ++i) {
        const double a0 = P[2 * i], a1 = P[2 * i + 1];
        const double b0 = P[2 * ((i + 1) % n)], b1 = P[2 * ((i + 1) % n) + 1];
        const double ab0 = b0 - a0, ab1 = b1 - a1;
        const double L2 = ab0 * ab0 + ab1 * ab1;
        const double t = L2 == 0.0 ? 0.0 : clip(((x[0] - a0) * ab0 + (x[1] - a1) * ab1) / L2, 0.0, 1.0);
        const double e0 = x[0] - (a0 + t * ab0), e1 = x[1] - (a1 + t * ab1);
        const double d = std::sqrt(e0 * e0 + e1 * e1);
        if (d < best) best = d;
    }
    return best;
}

// Whether x lies in polygon P: the ray test of registry._InsidePolygons,
// edge i from vertex i to vertex i - 1, crossed an odd number of times.
inline bool inside(const double* x, const double* P, std::size_t n)
{
    const double px = x[0], py = x[1];
    int count = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t j = (i + n - 1) % n;
        const double xi = P[2 * i], yi = P[2 * i + 1], xj = P[2 * j], yj = P[2 * j + 1];
        if (((yi > py) != (yj > py)) && (px < (xj - xi) * (py - yi) / (yj - yi + 1e-30) + xi)) ++count;
    }
    return count % 2 == 1;
}

// MaF9's constraint: the distance to the nearest edge of every forbidden
// polygon x lies in, added in the polygons' order (registry.maf9_cons_rows).
inline double maf9_constraint(const double* x, const std::vector<Vec>& polygons)
{
    double cv = 0.0;
    for (const Vec& P : polygons) {
        const std::size_t n = P.size() / 2;
        if (inside(x, P.data(), n)) cv += poly_edge_distance(x, P.data(), n);
    }
    return cv;
}

}   // namespace mootation::suite::maf

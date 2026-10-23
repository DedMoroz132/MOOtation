#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// The polygon problems (task 17, supplement 3, stage 3a), as benchmarks/
// evaluates them; the vertices are handed over from Python, C++ does the
// arithmetic only:
//   Polygon   polygon.py, polygon_eval: one regular M-gon on the unit circle
//             (vertex m at angle 2 pi m / M), f_m = |p - v_m| + g,
//             g = sum x_i^2 over x_3..x_n (NumPy's pairwise sum); rcPolygon
//             and c0Polygon are it at the map "rotate" (suite.hpp)
//   IPolygon  polygon_ishibuchi.py, _eval_M: Ishibuchi, Akedo & Nojima,
//             GECCO 2011, four regular M-gons in [0, 100]^2, f_i the distance
//             to the nearest of the four vertices i (np.linalg.norm: the
//             square root of dx^2 + dy^2)
// Their Pareto sets (pareto_sets.polygon; polygon_ishibuchi._sample_interior):
//   Polygon   a square grid clipped to the polygon, after its M vertices, the
//             distance variables 0
//   IPolygon  in each of the four polygons the same number of uniform points
//             by rejection, from a std::mt19937 seeded 12345 for each
//             (rnd::uniform01; the library's generator, task 17, supplement 3,
//             stage 3a: NumPy's default_rng before)
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "../detmath.hpp"
#include "../random.hpp"
#include "common.hpp"

namespace mootation::suite::polygon {

// vertices: (x, y) per vertex, M of them
inline void polygon(const double* x, int n, int M, const std::vector<double>& vertices, double* f)
{
    double g = 0.0;
    if (n > 2) {
        std::vector<double> sq(static_cast<std::size_t>(n - 2));
        for (int i = 2; i < n; ++i) sq[static_cast<std::size_t>(i - 2)] = x[i] * x[i];
        g = row_sum(sq);
    }
    for (int m = 0; m < M; ++m) {
        const double dx = x[0] - vertices[static_cast<std::size_t>(2 * m)];
        const double dy = x[1] - vertices[static_cast<std::size_t>(2 * m + 1)];
        f[m] = std::sqrt(dm::pow(dx, 2.0) + dm::pow(dy, 2.0)) + g;
    }
}

// polygons: each of the m polygons' M vertices, x0 y0 x1 y1 ...
inline void ipolygon(const double* x, int M, const std::vector<std::vector<double>>& polygons, double* f)
{
    for (int i = 0; i < M; ++i) {
        double best = 0.0;
        for (std::size_t p = 0; p < polygons.size(); ++p) {
            const double dx = polygons[p][static_cast<std::size_t>(2 * i)] - x[0];
            const double dy = polygons[p][static_cast<std::size_t>(2 * i + 1)] - x[1];
            const double d = std::sqrt(dx * dx + dy * dy);
            best = p == 0 ? d : nan_min(best, d);
        }
        f[i] = best;
    }
}

// The Polygon's Pareto set, about n rows of n_var values (pareto_sets.polygon)
inline std::vector<double> polygon_set(int M, int n_var, int n, const std::vector<double>& V)
{
    const double area = 0.5 * M * dm::sin(2.0 * kPi / M);
    const double h = std::sqrt(area / (n - M > 1 ? n - M : 1));
    const std::vector<double> g = arange(-1.0, 1.0 + h, h);
    std::vector<double> out;
    auto row = [&](double a, double b) {
        out.push_back(a);
        out.push_back(b);
        for (int i = 2; i < n_var; ++i) out.push_back(0.0);
    };
    for (int m = 0; m < M; ++m) row(V[static_cast<std::size_t>(2 * m)], V[static_cast<std::size_t>(2 * m + 1)]);
    for (double p0 : g)
        for (double p1 : g) {
            bool ok = true;                              // inside or on the convex polygon
            for (int i = 0; i < M; ++i) {
                const int j = (i + 1) % M;
                const double a0 = V[static_cast<std::size_t>(2 * i)], a1 = V[static_cast<std::size_t>(2 * i + 1)];
                const double b0 = V[static_cast<std::size_t>(2 * j)], b1 = V[static_cast<std::size_t>(2 * j + 1)];
                ok = ok && (b0 - a0) * (p1 - a1) - (b1 - a1) * (p0 - a0) >= -1e-12;
            }
            if (ok) row(p0, p1);
        }
    return out;
}

// whether (px, py) is inside the polygon P (x0 y0 x1 y1 ...): ray casting
// (polygon_ishibuchi._inside)
inline bool inside(double px, double py, const std::vector<double>& P)
{
    const int n = static_cast<int>(P.size() / 2);
    bool in = false;
    for (int i = 0, j = n - 1; i < n; j = i++) {
        const double xi = P[static_cast<std::size_t>(2 * i)], yi = P[static_cast<std::size_t>(2 * i + 1)];
        const double xj = P[static_cast<std::size_t>(2 * j)], yj = P[static_cast<std::size_t>(2 * j + 1)];
        if (((yi > py) != (yj > py)) && (px < (xj - xi) * (py - yi) / (yj - yi + 1e-30) + xi)) in = !in;
    }
    return in;
}

constexpr std::uint32_t kIPolygonSeed = 12345;

// IPolygon's Pareto set: max(1, n / 4) points of each polygon, x then y of a
// candidate from lo + (hi - lo) U of its bounding box, kept inside
inline std::vector<double> ipolygon_set(int n, const std::vector<std::vector<double>>& polygons)
{
    const int per = n / 4 > 1 ? n / 4 : 1;
    std::vector<double> out;
    for (const auto& P : polygons) {
        double lo[2] = {P[0], P[1]}, hi[2] = {P[0], P[1]};
        for (std::size_t i = 2; i < P.size(); ++i) {
            lo[i % 2] = std::min(lo[i % 2], P[i]);
            hi[i % 2] = std::max(hi[i % 2], P[i]);
        }
        std::mt19937 gen(kIPolygonSeed);
        for (int k = 0; k < per;) {
            const double x = lo[0] + (hi[0] - lo[0]) * rnd::uniform01(gen);
            const double y = lo[1] + (hi[1] - lo[1]) * rnd::uniform01(gen);
            if (inside(x, y, P)) {
                out.push_back(x);
                out.push_back(y);
                ++k;
            }
        }
    }
    return out;
}

}   // namespace mootation::suite::polygon

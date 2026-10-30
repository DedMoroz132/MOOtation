#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// The numbers a problem of the suite is built from that Python computed and
// handed over in the spec until task 17, §2.2 — now computed here, so that a
// problem is built from its name, sizes and parameters alone (the registry's
// export, the C++ engine and `mootation refs build` without Python). The
// formulas are the Python modules' (named beside each), which stay as the
// cross-check:
//
//   shift_centres(k)     dtlz_variants.shift_centres: c_i = 1/2 + d_i for odd
//                        i, 1/2 - d_i for even, d_i = 0.15 + 0.2 frac(i phi),
//                        phi = (sqrt 5 - 1) / 2 (shiftDTLZ, shiftZDT, the
//                        rotated problems' optimum)
//   b_param_root, wfg8_optimum, wfg9_optimum
//                        rotated.py: WFG8's and WFG9's optimal distance
//                        variables (b_param of Huband et al. 2006, Table XIV)
//   lsmop_parts(M)       maf.lsmop_parts: the sizes of the two subcomponents
//                        of each of MaF14's and MaF15's M groups
//   das_params(eta, zeta, gamma)
//                        dascmop.params: (b, d, e, r) of a difficulty triplet
//   unit_polygon(M)      polygon.polygon_vertices: the regular M-gon on the
//                        unit circle, vertex m at 2 pi m / M (Polygon, MaF8)
//   ipolygons(M)         polygon_ishibuchi._vertices: four regular M-gons of
//                        radius 8 about (25, 25), (75, 25), (25, 75), (75, 75)
//   sdtlz_scale(f, M)    dtlz_variants._scale: p^(i - 1), p from Table VIII
//                        (SDTLZ1 and SDTLZ2; an M outside it takes the
//                        nearest tabulated M below, 10 below them all)
//
// The sums are left to right as the Python loops add them; the elementary
// functions are dm:: (the Python modules' are the C library's: the last bits
// may differ, the problem versions say where they do).
// ============================================================================

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <string>
#include <vector>

#include "../detmath.hpp"

namespace mootation::suite::numbers {

inline std::vector<double> shift_centres(std::size_t k)
{
    const double phi = (std::sqrt(5.0) - 1.0) / 2.0;
    std::vector<double> c(k);
    for (std::size_t i = 1; i <= k; ++i) {
        const double v = static_cast<double>(i) * phi;
        const double d = 0.15 + 0.2 * (v - std::floor(v));
        c[i - 1] = 0.5 + (i % 2 == 1 ? d : -d);
    }
    return c;
}

inline double b_param_root(double u)
{
    constexpr double kA = 0.98 / 49.98, kB = 0.02, kC = 50.0;
    const double v = kA - (1.0 - 2.0 * u) * std::abs(std::floor(0.5 - u) + kA);
    return dm::pow(0.35, 1.0 / (kB + (kC - kB) * v));
}

// the l optimal distance variables after the normalised positions y
inline std::vector<double> wfg8_optimum(std::vector<double> y, std::size_t l)
{
    const std::size_t positions = y.size();
    for (std::size_t j = 0; j < l; ++j) {
        double s = 0.0;
        for (double v : y) s = s + v;
        y.push_back(b_param_root(s / static_cast<double>(y.size())));
    }
    return std::vector<double>(y.begin() + static_cast<std::ptrdiff_t>(positions), y.end());
}

// the l optimal distance variables, the last first
inline std::vector<double> wfg9_optimum(std::size_t l)
{
    std::vector<double> y{0.35};
    for (std::size_t j = 1; j < l; ++j) {
        double s = 0.0;
        for (double v : y) s = s + v;
        y.insert(y.begin(), b_param_root(s / static_cast<double>(y.size())));
    }
    return y;
}

// the sizes of the N_k = 2 subcomponents of each of the M groups of x^s
// (|x^s| = 19 M + 1), group by group
inline std::vector<std::vector<std::size_t>> lsmop_parts(std::size_t M)
{
    constexpr std::size_t kNk = 2;
    const std::size_t ns = 20 * M - M + 1;
    std::vector<double> c{3.8 * 0.1 * (1.0 - 0.1)};
    while (c.size() < M) c.push_back(3.8 * c.back() * (1.0 - c.back()));
    double total = 0.0;
    for (double v : c) total += v;
    std::vector<double> share(M);
    std::vector<std::size_t> size(M);
    std::size_t used = 0;
    for (std::size_t k = 0; k < M; ++k) {
        share[k] = c[k] / total * static_cast<double>(ns);
        size[k] = static_cast<std::size_t>(std::floor(share[k]));
        used += size[k];
    }
    std::vector<std::size_t> order(M);
    for (std::size_t k = 0; k < M; ++k) order[k] = k;
    std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        return static_cast<double>(size[a]) - share[a] < static_cast<double>(size[b]) - share[b];
    });
    for (std::size_t r = 0; r < ns - used; ++r) ++size[order[r]];
    std::vector<std::vector<std::size_t>> parts(M);
    for (std::size_t k = 0; k < M; ++k)
        for (std::size_t j = 0; j < kNk; ++j) parts[k].push_back(size[k] / kNk + (j < size[k] % kNk ? 1 : 0));
    return parts;
}

// (b, d, e, r) of DAS-CMOP's triplet (eta, zeta, gamma), D = 0.5; e = +inf
// where zeta = 0
inline std::vector<double> das_params(double eta, double zeta, double gamma)
{
    constexpr double kD = 0.5;
    return {2.0 * eta - 1.0, zeta == 0.0 ? 0.0 : kD,
            zeta == 0.0 ? std::numeric_limits<double>::infinity() : kD - dm::log(zeta), gamma / 2.0};
}

// x0 y0 x1 y1 ...: the regular M-gon on the unit circle
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

// IPolygon's four M-gons, each x0 y0 x1 y1 ...
inline std::vector<std::vector<double>> ipolygons(std::size_t M)
{
    static const double centres[4][2] = {{25.0, 25.0}, {75.0, 25.0}, {25.0, 75.0}, {75.0, 75.0}};
    std::vector<std::vector<double>> out(4, std::vector<double>(2 * M));
    for (std::size_t m = 0; m < M; ++m) {
        const double ang = 2.0 * dm::kPi * static_cast<double>(m) / static_cast<double>(M) + dm::kPi / 2.0;
        const double bx = 8.0 * dm::cos(ang), by = 8.0 * dm::sin(ang);
        for (std::size_t p = 0; p < 4; ++p) {
            out[p][2 * m] = bx + centres[p][0];
            out[p][2 * m + 1] = by + centres[p][1];
        }
    }
    return out;
}

// SDTLZ's factors p^(i - 1), i = 1..M (Cheng et al. 2016, Table VIII)
inline std::vector<double> sdtlz_scale(const std::string& family, std::size_t M)
{
    static const std::map<std::string, std::map<std::size_t, double>> table = {
        {"SDTLZ1", {{3, 10.0}, {5, 10.0}, {8, 3.0}, {10, 2.0}, {15, 1.2}}},
        {"SDTLZ2", {{3, 10.0}, {5, 10.0}, {8, 3.0}, {10, 3.0}, {15, 2.0}}}};
    const auto& tab = table.at(family);
    double p = 10.0;
    const auto it = tab.find(M);
    if (it != tab.end()) {
        p = it->second;
    } else {
        for (const auto& [m, v] : tab)
            if (m < M) p = v;
    }
    std::vector<double> s(M);
    for (std::size_t i = 0; i < M; ++i) s[i] = dm::pow(p, static_cast<double>(i));
    return s;
}

}   // namespace mootation::suite::numbers

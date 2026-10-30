#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// Samples of the Pareto sets of the registry's problems — task 17, §2.1
// (benchmarks/pareto_sets.py and the registry's _ps samplers; Python's stay
// as the cross-check). A problem names its sampler in the registry's export
// ("set": {"kind": ..., its numbers}); pareto_set gives its n rows.
//
//   dtlz        pareto_sets.dtlz: the positions over [0, 1]^(M-1) (the cube's
//               corners, at most a quarter of n, then Halton), the distance
//               variables at "rest" (one value, or one per variable)
//   dtlz7       the positions of DTLZ7's front sample (fronts.hpp's dtlz7,
//               not thinned), the rest at "rest" (pareto_sets.fill)
//   minus_dtlz  pareto_sets.minus_dtlz: dtlz's positions, every distance
//               variable 0 or 1 at random (the library's draw)
//   zdt         registry._ps_zdt: x_1 over [0, 1] — ZDT3's the f_1 of its
//               front sample — the others at "rest"
//   sch, fon, sph
//               classic.pareto_set
//
// halton(n, d): the radical inverses of 1, 2, ... in the first d primes,
// the digits added as pareto_sets.halton adds them.
// ============================================================================

#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "../suite/maf.hpp"
#include "../suite/spec_numbers.hpp"
#include "fronts.hpp"

namespace mootation::refs {

namespace detail {

constexpr unsigned kPrimes[16] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53};

inline Rows halton(std::size_t n, std::size_t d, std::size_t skip = 1)
{
    if (d > 16) throw std::invalid_argument("halton: at most 16 dimensions");
    Rows H = rows_of(n, d);
    for (std::size_t j = 0; j < d; ++j) {
        const double b = kPrimes[j];
        for (std::size_t k = 0; k < n; ++k) {
            std::size_t i = skip + k;
            double f = 1.0, r = 0.0;
            while (i > 0) {
                f /= b;
                r += f * static_cast<double>(i % kPrimes[j]);
                i /= kPrimes[j];
            }
            H.row(k)[j] = r;
        }
    }
    return H;
}

// pareto_sets._cube_positions
inline Rows cube_positions(std::size_t d, std::size_t n)
{
    if (d == 0) return rows_of(n, 0);
    if (d == 1) {
        Rows P = rows_of(n, 1);
        P.v = linspace(0.0, 1.0, n);
        return P;
    }
    Rows P = rows_of(0, d);
    std::vector<double> row(d);
    if (d <= 12)
        for (std::size_t c = 0; c < (std::size_t{1} << d) && P.n < n / 4; ++c) {
            for (std::size_t j = 0; j < d; ++j) row[j] = static_cast<double>((c >> j) & 1);
            push_row(P, row.data());
        }
    const Rows H = halton(n - P.n, d);
    P.v.insert(P.v.end(), H.v.begin(), H.v.end());
    P.n += H.n;
    return P;
}

// "rest" of a spec: one value for every variable, one each, or "shift": the
// shifted problems' optimum
inline std::vector<double> rest_of(const json::Value& spec, std::size_t k, bool zdt4 = false)
{
    const json::Value& r = field(spec, "rest");
    std::vector<double> out(k);
    if (r.kind == json::Value::Kind::String && r.s == "shift") {
        // the shifted problems' optimum (dtlz_variants.shift_centres; shiftZDT4's
        // scaled to [-5, 5], shifted.zdt_centres)
        out = suite::numbers::shift_centres(k);
        if (zdt4)
            for (double& v : out) v = 10.0 * (v - 0.5);
    } else if (r.kind == json::Value::Kind::Array) {
        if (r.items.size() != k) throw std::invalid_argument("a set's rest of another length than its variables");
        for (std::size_t j = 0; j < k; ++j) out[j] = number(r.items[j]);
    } else {
        std::fill(out.begin(), out.end(), number(r));
    }
    return out;
}

// pareto_sets.fill: the columns of P, then `rest`
inline Rows fill(const Rows& P, std::size_t n_vars, const std::vector<double>& rest)
{
    Rows X = rows_of(P.n, n_vars);
    for (std::size_t i = 0; i < P.n; ++i) {
        std::copy(P.row(i), P.row(i) + P.m, X.row(i));
        std::copy(rest.begin(), rest.end(), X.row(i) + P.m);
    }
    return X;
}

inline Rows columns(const Rows& F, std::size_t k)
{
    Rows P = rows_of(F.n, k);
    for (std::size_t i = 0; i < F.n; ++i) std::copy(F.row(i), F.row(i) + k, P.row(i));
    return P;
}

using suite::numbers::shift_centres;
using suite::numbers::wfg8_optimum;
using suite::numbers::wfg9_optimum;

// registry._register_rotated's _ps: the base's Pareto set with the distance
// block at the rotated problem's optimum (rotated.Rotated.optimum)
inline Rows rotated_set(const std::string& family, std::size_t M, std::size_t nv, std::size_t n)
{
    auto starts = [&family](const char* k) { return family.rfind(k, 0) == 0; };
    if (starts("WFG")) {
        const std::size_t first = nv - 20, k = nv - first;
        Rows P = cube_positions(first, n);
        for (std::size_t i = 0; i < P.n; ++i)
            for (std::size_t j = 0; j < first; ++j) P.row(i)[j] = P.row(i)[j] * 2.0 * static_cast<double>(j + 1);
        if (family == "WFG8") {
            Rows X = fill(P, nv, std::vector<double>(k, 0.0));
            std::vector<double> pos(first);
            for (std::size_t i = 0; i < X.n; ++i) {
                for (std::size_t j = 0; j < first; ++j) pos[j] = X.row(i)[j] / (2.0 * static_cast<double>(j + 1));
                const std::vector<double> c = wfg8_optimum(pos, k);
                for (std::size_t j = 0; j < k; ++j) X.row(i)[first + j] = 2.0 * static_cast<double>(first + j + 1) * c[j];
            }
            return X;
        }
        const std::vector<double> u = family == "WFG9" ? wfg9_optimum(k) : std::vector<double>(k, 0.35);
        std::vector<double> opt(k);
        for (std::size_t j = 0; j < k; ++j) opt[j] = 2.0 * static_cast<double>(first + j + 1) * u[j];
        return fill(P, nv, opt);
    }
    if (starts("ZDT")) {
        std::vector<double> c = shift_centres(nv - 1);
        if (family == "ZDT4")
            for (double& v : c) v = 10.0 * (v - 0.5);
        Rows x1 = rows_of(0, 1);
        if (family == "ZDT3") x1 = columns(zdt3(n), 1);
        else
            for (double v : linspace(0.0, 1.0, n)) push_row(x1, &v);
        return fill(x1, nv, c);
    }
    if (family == "Polygon") {
        const std::vector<double> v = suite::polygon::polygon_set(static_cast<int>(M), static_cast<int>(nv),
                                                                  static_cast<int>(n), unit_polygon(M));
        Rows X = rows_of(v.size() / nv, nv);
        X.v = v;
        const std::vector<double> c = shift_centres(nv - 2);
        for (std::size_t i = 0; i < X.n; ++i)
            for (std::size_t j = 2; j < nv; ++j) X.row(i)[j] = 2.0 * c[j - 2] - 1.0;
        return X;
    }
    return fill(cube_positions(M - 1, n), nv, shift_centres(nv - M + 1));   // DTLZ, MaF
}

using suite::numbers::lsmop_parts;

// maf.ps_lsmop: the positions over [0, 1]^(M-1) (MaF14's x_1 scaled to its
// largest value at which every Rosenbrock subcomponent reaches z = 1), every
// subcomponent at its optimum z (1 for Rosenbrock, MaF14's even groups; else
// 0), x_i = l_i + (z + x_1 (u_i - l_i)) / a_i, kept in the box
inline Rows lsmop_set(std::size_t M, int which, std::size_t n)
{
    const std::vector<double> link = suite::maf::lsmop_link(static_cast<int>(M), which);
    const std::size_t ns = link.size() / 3, D = 20 * M;
    std::vector<double> z(ns, 0.0);
    double x1_max = 1.0;
    std::size_t at = 0;
    const auto parts = lsmop_parts(M);
    for (std::size_t k = 0; k < parts.size(); ++k) {
        const std::size_t start = at;
        for (std::size_t s : parts[k]) {
            if (which == 14 && k % 2 == 1 && s >= 2)
                for (std::size_t i = at; i < at + s; ++i) x1_max = std::min(x1_max, link[3 * i] - 1.0 / link[3 * i + 2]);
            at += s;
        }
        if (which == 14 && k % 2 == 1)
            for (std::size_t i = start; i < at; ++i) z[i] = 1.0;
    }
    Rows P = cube_positions(M - 1, n);
    if (which == 14)
        for (std::size_t i = 0; i < P.n; ++i) P.row(i)[0] *= x1_max;
    Rows X = rows_of(P.n, D);
    for (std::size_t r = 0; r < P.n; ++r) {
        std::copy(P.row(r), P.row(r) + P.m, X.row(r));
        for (std::size_t i = 0; i < ns; ++i) {
            const double a = link[3 * i], l = link[3 * i + 1], w = link[3 * i + 2];
            X.row(r)[M - 1 + i] = std::min(l + (z[i] + P.row(r)[0] * w) / a, l + w);
        }
    }
    return X;
}

}   // namespace detail

inline Rows pareto_set(const json::Value& spec, std::size_t n)
{
    using namespace detail;
    const std::string kind = field(spec, "kind").s;
    if (kind == "dtlz" || kind == "minus_dtlz") {
        const std::size_t M = count(spec, "M"), nv = count(spec, "n_vars");
        const std::vector<double> rest = kind == "dtlz" ? rest_of(spec, nv - M + 1) : std::vector<double>(nv - M + 1, 0.0);
        Rows X = fill(cube_positions(M - 1, n), nv, rest);
        if (kind == "minus_dtlz") {
            std::mt19937 g(static_cast<std::uint32_t>(20261009 + M));
            for (std::size_t i = 0; i < X.n; ++i)
                for (std::size_t j = M - 1; j < nv; ++j) X.row(i)[j] = static_cast<double>(rnd::below(g, 2));
        }
        return X;
    }
    if (kind == "lsmop") return lsmop_set(count(spec, "M"), static_cast<int>(count(spec, "which")), n);
    if (kind == "rotated") return rotated_set(field(spec, "family").s, count(spec, "M"), count(spec, "n_vars"), n);
    if (kind == "mmo" || kind == "polygon" || kind == "ipolygon") {
        // the suite's own samplers (suite/mmo.hpp, suite/polygon.hpp)
        std::vector<double> v;
        std::size_t d = 2;
        if (kind == "mmo") {
            suite::mmo::Which w{};
            if (!suite::mmo::parse(field(spec, "name").s, w)) throw std::invalid_argument("not an MMF problem");
            d = count(spec, "n_vars");
            v = suite::mmo::pareto_set(w, static_cast<int>(n));
        } else if (kind == "polygon") {
            const std::size_t M = count(spec, "M");
            d = count(spec, "n_vars");
            v = suite::polygon::polygon_set(static_cast<int>(M), static_cast<int>(d), static_cast<int>(n), unit_polygon(M));
        } else {
            v = suite::polygon::ipolygon_set(static_cast<int>(n), ipolygons(count(spec, "M")));
        }
        Rows X = rows_of(v.size() / d, d);
        X.v = std::move(v);
        return X;
    }
    if (kind == "zcat") return zcat::pareto_set(static_cast<int>(count(spec, "i")), count(spec, "M"), n);
    if (kind == "mw") {
        // mw.pareto_set: the front's positions at g = 1, the distance
        // variables at g's optimum (mw.optimal_distance)
        const int i = static_cast<int>(count(spec, "i"));
        const std::size_t m = count(spec, "M"), d = m - 1;
        Rows P = rows_of(0, d);
        if (i == 1 || i == 2 || i == 6) {
            P = columns(mw::front(i, m, n), 1);
        } else if (i == 14) {
            P = columns(mw::mw14_front(m, n), d);
        } else {
            const Rows F = i == 4 ? simplex(m, n) : mw::front(i, m, n);
            P = rows_of(F.n, d);
            for (std::size_t r = 0; r < F.n; ++r) {
                double rest = 1.0;
                for (std::size_t j = 0; j < d; ++j) {
                    double p = 0.0;
                    if (rest > 0.0) {
                        const double v = F.row(r)[m - 1 - j] / rest;
                        p = i == 4 ? v : dm::asin(std::min(std::max(v, 0.0), 1.0)) * 2.0 / dm::kPi;
                    }
                    P.row(r)[j] = p;
                    rest = i == 4 ? rest * (1.0 - p) : rest * dm::cos(0.5 * dm::kPi * p);
                }
            }
        }
        const std::size_t nv = suite::mw::scalable(i) ? m + 12 : 15;
        const int kind_g = suite::mw::distance(i);
        Rows X = rows_of(P.n, nv);
        for (std::size_t r = 0; r < P.n; ++r) {
            double* x = X.row(r);
            for (std::size_t j = 0; j < d; ++j) x[j] = std::min(std::max(P.row(r)[j], 0.0), mw::upper(i));
            double last = x[d - 1];
            for (std::size_t k = m; k <= nv; ++k) {          // i = m .. n, x_i at index i - 1
                const double fi = static_cast<double>(k), fn = static_cast<double>(nv);
                double v;
                if (kind_g == 1) v = dm::pow(0.5 + (fi - 1.0) / (2.0 * fn), 1.0 / (fn - static_cast<double>(m)));
                else if (kind_g == 2) v = (fi - 1.0) / fn;
                else v = last = 1.0 - (last - 0.5) * (last - 0.5);
                x[k - 1] = v;
            }
        }
        return X;
    }
    if (kind == "c2_dtlz2") {
        // c_dtlz.pareto_set: the feasible ones of dtlz's 20 n (DTLZ2's
        // objectives), thinned evenly to n (np.linspace, rounded, unique)
        const std::size_t M = count(spec, "M"), nv = count(spec, "n_vars");
        const Rows X = fill(cube_positions(M - 1, 20 * n), nv, std::vector<double>(nv - M + 1, 0.5));
        std::vector<double> f(M);
        std::vector<std::size_t> ok;
        for (std::size_t i = 0; i < X.n; ++i) {
            suite::dtlz::evaluate(2, X.row(i), static_cast<int>(nv), static_cast<int>(M), f.data());
            if (cdtlz_feasible(Cons::C2_DTLZ2, f.data(), M)) ok.push_back(i);
        }
        std::vector<std::size_t> pick;
        for (double at : linspace(0.0, static_cast<double>(ok.size()) - 1.0, std::min(n, ok.size()))) {
            const auto k = static_cast<std::size_t>(std::nearbyint(at));
            if (pick.empty() || pick.back() != ok[k]) pick.push_back(ok[k]);
        }
        return take(X, pick);
    }
    if (kind == "dtlz7") {
        const std::size_t M = count(spec, "M"), nv = count(spec, "n_vars");
        return fill(columns(dtlz7(M, n), M - 1), nv, rest_of(spec, nv - M + 1));
    }
    if (kind == "zdt") {
        const std::string base = field(spec, "base").s;
        const std::size_t nv = base == "ZDT1" || base == "ZDT2" || base == "ZDT3" ? 30 : 10;
        Rows x1 = rows_of(0, 1);
        if (base == "ZDT3") x1 = columns(zdt3(n), 1);
        else
            for (double v : linspace(0.0, 1.0, n)) push_row(x1, &v);
        return fill(x1, nv, rest_of(spec, nv - 1, base == "ZDT4"));
    }
    if (kind == "sch") {
        Rows X = rows_of(0, 1);
        for (double t : linspace(0.0, 1.0, n)) {
            const double x = 2.0 * t;
            push_row(X, &x);
        }
        return X;
    }
    if (kind == "fon") {
        const double c = 1.0 / std::sqrt(3.0);
        Rows X = rows_of(0, 3);
        for (double t : linspace(0.0, 1.0, n)) {
            const double x = -c + 2.0 * c * t, r[3] = {x, x, x};
            push_row(X, r);
        }
        return X;
    }
    if (kind == "sph") {
        const std::size_t m = count(spec, "M");
        const std::vector<double> W = run_compute::detail::das_dennis(m, run_compute::detail::lattice_h(m, n));
        const std::size_t rows = W.size() / m;
        Rows X = rows_of(rows, 100);
        for (std::size_t i = 0; i < rows; ++i) std::copy(W.begin() + static_cast<std::ptrdiff_t>(i * m),
                                                         W.begin() + static_cast<std::ptrdiff_t>((i + 1) * m), X.row(i));
        return X;
    }
    throw std::invalid_argument("a Pareto set of kind \"" + kind + "\" is not built in C++ yet");
}

}   // namespace mootation::refs

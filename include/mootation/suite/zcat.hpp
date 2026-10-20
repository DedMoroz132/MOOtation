#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// ZCAT1-20 (Zapotecas-Martinez, Coello Coello, Aguirre & Tanaka 2023) at the
// suite's defaults as the registry has them (benchmarks/zcat.py): Level 1, a
// complicated Pareto set, no bias, no imbalance; the readings ZCAT-1..5 of
// zcat.py. And rZCAT's map (rotated.py, Rotated._zcat_v and map): the
// distance block turned about its optimum g(y_I), clipped to [0, 1].
// The order of every operation is that of the NumPy rows (common.hpp).
// ============================================================================

#include <vector>

#include "../detmath.hpp"
#include "common.hpp"

namespace mootation::suite::zcat {

using Vec = std::vector<double>;

constexpr int kK = 5;            // zcat.K
constexpr int kA = 5;            // zcat.A (ZCAT-2)

// Table 2: the topology g of each problem (zcat.G_OF), index 1..20
constexpr int kTopology[21] = {0, 4, 5, 2, 7, 9, 4, 5, 2, 7, 9, 3, 10, 1, 6, 8, 10, 1, 8, 6, 3};

inline bool degenerate(int which) { return which == 14 || which == 15 || which == 16; }

// ZCAT19 and ZCAT20: m = 1 inside these intervals of y_1 (zcat.REGION_M)
inline bool in_regions(int which, double y1)
{
    if (which == 19) return (0.0 <= y1 && y1 <= 0.2) || (0.4 <= y1 && y1 <= 0.6);
    return (0.1 <= y1 && y1 <= 0.4) || (0.6 <= y1 && y1 <= 0.9);
}

// m, the number of position variables (zcat.position_count)
inline int position_count(int which, const double* Y, int M)
{
    if (degenerate(which)) return 1;
    if (which == 19 || which == 20) return in_regions(which, Y[0]) ? 1 : M - 1;
    return M - 1;
}

// ---- F (Table 2), of the normalised point Y ------------------------------
inline void prods(const double* a, int M, const double* last, double* F)
{
    F[0] = M > 1 ? row_prod(a, static_cast<std::size_t>(M - 1)) : 1.0;
    for (int j = 2; j < M; ++j)
        F[j - 1] = row_prod(a, static_cast<std::size_t>(M - j)) * last[M - j];
}

inline double wavy(double y1)
{
    return (dm::cos((2 * kK - 1) * y1 * kPi) + 2 * y1 + 4 * kK * (1 - y1) - 1) / (4 * kK);
}

inline double quintic_sum(const double* Y, int M)
{
    Vec t(static_cast<std::size_t>(M - 1));
    for (int i = 0; i < M - 1; ++i) t[static_cast<std::size_t>(i)] = dm::pow(0.5 - Y[i], 5.0);
    return row_sum(t) / (2 * (M - 1) * 0x1p-5) + 0.5;                // 0.5^5
}

inline double quintic(double v)
{
    return (dm::pow(0.5 - v, 5.0) + 0x1p-5) / (2.0 * 0x1p-5);   // 0.5^5
}

// sin and cos of Y * pi/2 over the first M - 1 variables
inline void sin_cos(const double* Y, int M, Vec& s, Vec& c)
{
    s.resize(static_cast<std::size_t>(M - 1));
    c.resize(static_cast<std::size_t>(M - 1));
    for (int i = 0; i < M - 1; ++i) {
        s[static_cast<std::size_t>(i)] = dm::sin(Y[i] * kHalfPi);
        c[static_cast<std::size_t>(i)] = dm::cos(Y[i] * kHalfPi);
    }
}

inline void position_functions(int which, const double* Y, int M, double* F)
{
    const std::size_t p = static_cast<std::size_t>(M - 1);
    Vec s, c, a, b;
    switch (which) {
        case 1:
            sin_cos(Y, M, s, c);
            prods(s.data(), M, c.data(), F);
            F[M - 1] = 1.0 - s[0];
            return;
        case 2:
            sin_cos(Y, M, s, c);
            a.resize(p);
            b.resize(p);
            for (std::size_t i = 0; i < p; ++i) {
                a[i] = 1.0 - c[i];
                b[i] = 1.0 - s[i];
            }
            prods(a.data(), M, b.data(), F);
            F[M - 1] = 1.0 - dm::sin(Y[0] * kHalfPi);
            return;
        case 3:
        case 11:
            F[0] = row_sum(Y, p) / (M - 1);
            for (int j = 2; j < M; ++j)
                F[j - 1] = (row_sum(Y, static_cast<std::size_t>(M - j)) + (1.0 - Y[M - j])) / (M - j + 1);
            F[M - 1] = which == 3 ? 1.0 - Y[0] : wavy(Y[0]);
            return;
        case 4:
            for (int i = 0; i < M - 1; ++i) F[i] = Y[i];
            F[M - 1] = 1.0 - row_sum(Y, p) / (M - 1);
            return;
        case 5:
        case 17: {
            a.resize(p);
            for (std::size_t i = 0; i < p; ++i) a[i] = 1.0 - Y[i];
            const double mu = row_sum(a) / (M - 1);
            for (int i = 0; i < M - 1; ++i) F[i] = Y[i];
            F[M - 1] = (dm::exp(8.0 * mu) - 1.0) / (dm::kExp8 - 1.0);
            if (which == 17) {                               // ZCAT-4: the degenerate region
                bool flat = true;
                for (int i = 0; i < M - 1; ++i) flat = flat && Y[i] <= 0.5;
                if (flat) {
                    for (int i = 0; i < M - 1; ++i) F[i] = Y[0];
                    F[M - 1] = (dm::exp(8.0 * (1.0 - Y[0])) - 1.0) / (dm::kExp8 - 1.0);
                }
            }
            return;
        }
        case 6: {
            const double kappa = 40.0, rho = 0.05;
            const double mu = row_sum(Y, p) / (M - 1);
            const double num = dm::pow(1 + dm::exp(2 * kappa * mu - kappa), -1.0) - rho * mu
                             - dm::pow(1 + dm::kExp40, -1.0) + rho;
            const double den = dm::pow(1 + dm::kExpMinus40, -1.0) - dm::pow(1 + dm::kExp40, -1.0) + rho;
            for (int i = 0; i < M - 1; ++i) F[i] = Y[i];
            F[M - 1] = num / den;
            return;
        }
        case 7:
        case 18:
        case 20: {
            for (int i = 0; i < M - 1; ++i) F[i] = Y[i];
            F[M - 1] = quintic_sum(Y, M);
            bool one = false;
            if (which == 18) {
                bool low = true, high = true;
                for (int i = 0; i < M - 1; ++i) {
                    low = low && Y[i] <= 0.4;
                    high = high && Y[i] >= 0.6;
                }
                one = low || high;
            } else if (which == 20) {
                one = in_regions(20, Y[0]) || M == 2;
            }
            if (one) {
                for (int i = 0; i < M - 1; ++i) F[i] = Y[0];
                F[M - 1] = quintic(Y[0]);
            }
            return;
        }
        case 8:
            sin_cos(Y, M, s, c);
            a.resize(p);
            b.resize(p);
            for (std::size_t i = 0; i < p; ++i) {
                a[i] = 1.0 - s[i];
                b[i] = 1.0 - c[i];
            }
            prods(a.data(), M, b.data(), F);
            for (int i = 0; i < M - 1; ++i) F[i] = 1.0 - F[i];
            F[M - 1] = dm::cos(Y[0] * kHalfPi);
            return;
        case 9:
        case 13:
            sin_cos(Y, M, s, c);
            F[0] = row_sum(s.data(), p) / (M - 1);
            for (int j = 2; j < M; ++j)
                F[j - 1] = (row_sum(s.data(), static_cast<std::size_t>(M - j)) + c[static_cast<std::size_t>(M - j)]) /
                           (M - j + 1);
            F[M - 1] = dm::cos(Y[0] * kHalfPi);
            if (which == 13) {
                for (int i = 0; i < M - 1; ++i) F[i] = 1.0 - F[i];
                F[M - 1] = 1.0 - wavy(Y[0]);
            }
            return;
        case 10: {
            const double rho = 0.02;
            a.resize(p);
            for (std::size_t i = 0; i < p; ++i) a[i] = 1.0 - Y[i];
            const double mu = row_sum(a) / (M - 1);
            for (int i = 0; i < M - 1; ++i) F[i] = Y[i];
            F[M - 1] = (dm::pow(rho, -1.0) - dm::pow(mu + rho, -1.0)) /
                       (dm::pow(rho, -1.0) - dm::pow(1 + rho, -1.0));
            return;
        }
        case 12:
            a.resize(p);
            for (std::size_t i = 0; i < p; ++i) a[i] = 1.0 - Y[i];
            prods(a.data(), M, Y, F);
            for (int i = 0; i < M - 1; ++i) F[i] = 1.0 - F[i];
            F[M - 1] = wavy(Y[0]);
            return;
        case 14:
        case 16: {
            const double s1 = dm::sin(Y[0] * kHalfPi);
            F[0] = which == 14 ? dm::pow(s1, 2.0) : s1;
            for (int j = 2; j < M - 1; ++j)
                F[j - 1] = dm::pow(s1, (which == 14 ? 2.0 : 1.0) + static_cast<double>(j - 1) / (M - 2));
            if (M > 2)
                F[M - 2] = 0.5 * (1.0 + dm::sin((which == 14 ? 6.0 : 10.0) * Y[0] * kHalfPi - kHalfPi));
            F[M - 1] = which == 14 ? dm::cos(Y[0] * kHalfPi) : wavy(Y[0]);
            return;
        }
        case 15:
            for (int j = 1; j < M; ++j) F[j - 1] = dm::pow(Y[0], 1.0 + static_cast<double>(j - 1) / (4.0 * M));
            F[M - 1] = wavy(Y[0]);
            return;
        case 19: {
            const bool one = in_regions(19, Y[0]) || M == 2;
            for (int i = 0; i < M - 1; ++i) F[i] = one ? Y[0] : Y[i];
            const double ybar = one ? Y[0] : row_sum(Y, p) / (M - 1);
            F[M - 1] = 1.0 - ybar - dm::cos(2 * kA * kPi * ybar + kHalfPi) / (2 * kA * kPi);
            return;
        }
        default:
            return;
    }
}

// ---- the Pareto set topologies g1..g10 (Table 3, zcat.topology_batch) ----
// g_j for j = m+1..n of the positions YI (m values), into out (n - m values)
inline void topology(int which, const double* YI, int m, int n, double* out)
{
    const int k = n - m;
    Vec terms(static_cast<std::size_t>(m));
    auto theta = [&](int p) { return 2.0 * kPi * static_cast<double>(p) / n; };
    auto sum_terms = [&](auto term, int p) {
        const double t = theta(p);
        for (int q = 0; q < m; ++q) terms[static_cast<std::size_t>(q)] = term(YI[q], t);
        return row_sum(terms);
    };
    switch (which) {
        case 1:
            for (int p = 0; p < k; ++p)
                out[p] = sum_terms([](double y, double t) { return dm::sin(1.5 * kPi * y + t); }, p) / (2 * m) + 0.5;
            return;
        case 2:
            for (int p = 0; p < k; ++p)
                out[p] = sum_terms([](double y, double t) { return y * y * dm::sin(4.5 * kPi * y + t); }, p) /
                             (2 * m) + 0.5;
            return;
        case 3:
            for (int p = 0; p < k; ++p)
                out[p] = sum_terms([](double y, double t) {
                             const double c = dm::cos(kPi * y + t);
                             return c * c;
                         }, p) / m;
            return;
        case 4: {
            const double s = row_sum(YI, static_cast<std::size_t>(m));
            for (int p = 0; p < k; ++p)
                out[p] = sum_terms([&](double y, double t) { return y * dm::cos(4.0 * kPi * s / m + t); }, p) /
                             (2 * m) + 0.5;
            return;
        }
        case 5:
            for (int p = 0; p < k; ++p)
                out[p] = sum_terms([](double y, double t) {
                             return dm::pow(dm::sin(2.0 * kPi * y - 1.0 + t), 3.0);
                         }, p) / (2 * m) + 0.5;
            return;
        case 6: {
            Vec sq(static_cast<std::size_t>(m));
            for (int q = 0; q < m; ++q) sq[static_cast<std::size_t>(q)] = YI[q] * YI[q];
            const double rms = std::sqrt(row_sum(sq) / m);
            const double num0 = -10.0 * dm::exp(-0.4 * rms) + 10.0 + kE;
            const double den = -10.0 * dm::kExpMinus04 - dm::kExpMinus1 + 10.0 + kE;
            for (int p = 0; p < k; ++p)
                out[p] = (num0 - dm::exp(sum_terms([](double y, double t) {
                                                return dm::pow(dm::cos(11.0 * kPi * y + t), 3.0);
                                            }, p) / m)) / den;
            return;
        }
        case 7: {
            const double mu = row_sum(YI, static_cast<std::size_t>(m)) / m;
            const double den = 1.0 + kE - dm::kExpMinus1;
            for (int p = 0; p < k; ++p)
                out[p] = (mu + dm::exp(dm::sin(7.0 * kPi * mu - kHalfPi + theta(p))) - dm::kExpMinus1) / den;
            return;
        }
        case 8:
            for (int p = 0; p < k; ++p)
                out[p] = sum_terms([](double y, double t) { return std::abs(dm::sin(2.5 * kPi * (y - 0.5) + t)); }, p) /
                         m;
            return;
        case 9: {
            const double s = row_sum(YI, static_cast<std::size_t>(m)) / (2 * m);
            for (int p = 0; p < k; ++p)
                out[p] = s - sum_terms([](double y, double t) { return std::abs(dm::sin(2.5 * kPi * y - kHalfPi + t)); }, p) /
                                 (2 * m) + 0.5;
            return;
        }
        case 10:
            for (int p = 0; p < k; ++p)
                out[p] = dm::pow(sum_terms([](double y, double t) { return dm::sin((4.0 * y - 2.0) * kPi + t); }, p), 3.0) /
                             (2.0 * (m * m * m)) + 0.5;
            return;
        default:
            for (int p = 0; p < k; ++p) out[p] = 0.2210;      // g0
            return;
    }
}

// ZCAT`which` at M objectives, n variables; with rotations (one k x k matrix
// for each distance-block size k the problem has; empty: ZCAT itself) the
// rZCAT of rotated.py.
struct Problem {
    int which = 1, M = 2, n = 20;
    std::vector<std::pair<int, Vec>> rotations;     // (k, row-major k x k)

    const Vec* rotation(int k) const
    {
        for (const auto& r : rotations)
            if (r.first == k) return &r.second;
        return nullptr;
    }

    // Y = clip(x / i + 1/2, 0, 1), i = 1..n: the normalised point
    void normalise(const double* x, double* Y) const
    {
        for (int i = 0; i < n; ++i) Y[i] = clip(x[i] / static_cast<double>(i + 1) + 0.5, 0.0, 1.0);
    }

    // rZCAT's map: the distance block of x turned about g(y_I) (Rotated.map)
    void rotate(const double* x, double* y) const
    {
        Vec u(static_cast<std::size_t>(n));
        normalise(x, u.data());
        const int m = position_count(which, u.data(), M);
        const int k = n - m;
        Vec g(static_cast<std::size_t>(k)), d(static_cast<std::size_t>(k)), v(static_cast<std::size_t>(k));
        topology(kTopology[which], u.data(), m, n, g.data());
        for (int p = 0; p < k; ++p) d[static_cast<std::size_t>(p)] = u[static_cast<std::size_t>(m + p)] - g[static_cast<std::size_t>(p)];
        matvec(*rotation(k), static_cast<std::size_t>(k), d.data(), v.data());
        for (int i = 0; i < m; ++i) y[i] = x[i];
        for (int p = 0; p < k; ++p)
            y[m + p] = (clip(g[static_cast<std::size_t>(p)] + v[static_cast<std::size_t>(p)], 0.0, 1.0) - 0.5) *
                       static_cast<double>(m + p + 1);
    }

    void evaluate(const double* x, double* f) const
    {
        Vec y;
        if (!rotations.empty()) {
            y.resize(static_cast<std::size_t>(n));
            rotate(x, y.data());
            x = y.data();
        }
        Vec Y(static_cast<std::size_t>(n));
        normalise(x, Y.data());
        const int m = position_count(which, Y.data(), M);
        const int k = n - m;
        Vec F(static_cast<std::size_t>(M)), g(static_cast<std::size_t>(k));
        position_functions(which, Y.data(), M, F.data());
        topology(kTopology[which], Y.data(), m, n, g.data());
        Vec w;
        for (int i = 1; i <= M; ++i) {
            // J_i: the distance variables p (0-based in y_II) with (p + 1 - i) mod M = 0
            w.clear();
            for (int q = i - 1; q < k; q += M) {
                const double z = Y[static_cast<std::size_t>(m + q)] - g[static_cast<std::size_t>(q)];
                w.push_back(z * z);
            }
            const double level = w.empty() ? 0.0 : 10.0 * (row_sum(w) / static_cast<double>(w.size()));
            const double alpha = static_cast<double>(i * i) * F[static_cast<std::size_t>(i - 1)];
            f[i - 1] = alpha + static_cast<double>(i * i) * level;
        }
    }
};

}   // namespace mootation::suite::zcat

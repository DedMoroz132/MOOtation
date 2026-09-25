#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// Multi-parent, rotation-invariant crossovers (task 2 of 2026-09-23, B2).
// Each makes children from μ parents x_1 … x_μ, centre g = their mean.
//
//   spx   SIMPLEX crossover (Tsutsui, Yamamura & Higuchi, GECCO 1999, §2;
//         source spx_tsutsui1999): a child uniform in the parents' simplex
//         expanded about its centre g. Tsutsui et al. write the expanded
//         vertices as g + (1 + ε)(x_j − g) and run ε = 1 (§4, "the α value
//         of 0.5 in BLX-α"). Higuchi, Tsutsui & Yamamura (PPSN VI, 2000,
//         §3.1 Eq. 5 and §4 Eq. 9; source higuchi2000) make the factor itself
//         the expansion rate, g + ε(x_j − g), and derive ε = √(m + 2) for
//         m + 1 parents, the value that keeps the population's covariance
//         (their Theorem 3). That is the setting here, e = √(n + 2) with
//         μ = n + 1 parents, as in Tanabe & Ishibuchi (GECCO 2019, Table 1
//         and §2.2; source bc_tanabe2019); it is ε = √(n + 2) − 1 in the 1999
//         notation. Given the parents, a uniform point of the expanded simplex
//         has covariance e²/(μ(μ + 1))·Σ (x_j − g)(x_j − g)ᵀ, the parents'
//         (1/μ)·Σ (x_j − g)(x_j − g)ᵀ exactly when e² = μ + 1.
//         The point is g + e·Σ w_j (x_j − g) with barycentric weights w from
//         a flat Dirichlet (normalised exponentials); the papers draw the same
//         uniform point by the recursion r_k = u^(1/(k+1)) (Higuchi et al.,
//         Eqs. 3, 6-7).
//   rex   REAL-CODED ENSEMBLE crossover (Akimoto, Sakuma, Ono & Kobayashi,
//         GECCO 2009 / JSAI 24(6), §2.2, Eq. 3; source rex_akimoto_arex2009):
//         x = g + α·Σ ξ_j (x_j − g), ξ_j ~ N(0, σ²) independent, α = 1,
//         σ² = 1/(μ − 1) — mean g and covariance (1/(μ − 1))·Σ (x_j − g)(x_j − g)ᵀ.
//   undx  UNIMODAL NORMAL DISTRIBUTION crossover (Ono & Kobayashi, ICGA-7,
//         1997, pp. 246-253 — not in the corpus; restated in Ono, Kita &
//         Kobayashi, GECCO 1999, §2.1, Eqs. 1-2, source
//         undx_ono1999_gecco_substitute, the same text as Ono_gecco99): from
//         x_1, x_2 and a third parent x_3, the pair
//         c = m ± (z_1 e_1 + Σ_{k≥2} z_k e_k), m = (x_1 + x_2)/2,
//         e_1 = (x_2 − x_1)/d_1, z_1 ~ N(0, (α d_1)²), z_k ~ N(0, (β d_2/√n)²),
//         d_1 = |x_2 − x_1|, d_2 the distance of x_3 from the line x_1x_2,
//         e_k an orthonormal basis of the complement of e_1 ("orthogonal" in
//         the paper; unit length is what makes σ_2 the spread); α = 0.5,
//         β = 0.35 ("recommended [Ono 97], [Kita 98]"). x_3 sets d_2 only.
//         The paper's MGG applies it many times to one pair (§2.2, step 3);
//         a host here, once per mating.
//   pcx   PARENT-CENTRIC crossover (Deb, Anand & Joshi, Evol. Comput. 10(4),
//         2002, §2.2, Eq. 2; source deb2002_ — and its precursor, KanGAL
//         2001003, source pcx_deb2002_kangal2001003_precursor): for each child
//         one parent x_p "chosen with equal probability", d = x_p − g, D̄ the
//         mean perpendicular distance of the other μ − 1 parents from the line
//         through g along d, and y = x_p + w_ζ·d + Σ w_η·D̄·e^(i),
//         w_ζ ~ N(0, σ_ζ²), w_η ~ N(0, σ_η²); σ_ζ = σ_η = 0.1 and μ = 3 (§4:
//         "In all PCX runs, we have used σ_η = σ_ζ = 0.1"). The journal's G3
//         runs take the BEST parent as x_p "for a faster convergence"; the
//         operator itself picks it at random, and so does this one — a host
//         here has no single fitness to call best.
//
// Declared readings:
//   MP-1 (UNDX, PCX). Σ z_k e_k over an orthonormal basis of the complement
//     of a direction is drawn as an isotropic normal vector with its
//     component along that direction removed: the same distribution, without
//     building the basis.
//   MP-2 (PCX). Both versions count "(μ − 1) orthonormal bases that span the
//     subspace perpendicular to d^(p)", but μ parents span only μ − 2
//     directions perpendicular to d, so the perpendicular part is taken over
//     the whole complement of d (MP-1), the reading under which every
//     perpendicular direction is reachable. (The report's "w_ζ |d^(p)|",
//     a scalar added to a vector, is w_ζ·d^(p) in the journal's Eq. 2, as
//     implemented.)
//   MP-3 (SPX). Tsutsui et al. split the coordinates into subspaces of
//     dimension μ − 1 when μ < n + 1; only μ = n + 1 — one simplex spanning
//     the space, the setting of the statistics-preserving analysis and of BC
//     — is implemented.
//   MP-4. A child variable outside the box is repaired by bound_repair
//     (reflect by default, the provisional default of the new operators); the
//     "parent" for midpoint is the centre the child was drawn about (g, m or
//     x_p). resample and native are refused: the child is one draw.
//   MP-5. A degenerate direction (x_p at g, or x_1 = x_2) leaves the normal
//     part isotropic in the whole space.
//   MP-6. Parents in a host with pairwise mating (real_crossover.hpp,
//     apply_any). The host's pair is x_1, x_2; the other parents come from
//     the host's own mating selection, and one equal to a parent already
//     taken is drawn again, at most 10·μ times per application. Tsutsui et
//     al. and Tanabe & Ishibuchi select μ DIFFERENT parents at random, and
//     Ono et al. (§2.2) the pair without replacement and x_3 at random; a
//     repeated one flattens the simplex (SPX, REX) or UNDX's perpendicular
//     part. A converged population may still repeat one once the redraws
//     run out. The host's pair itself is left as the host chose it.
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <optional>
#include <random>
#include <utility>
#include <vector>

#include "bound_repair.hpp"
#include "sbx.hpp"

namespace mootation::ops {

using Bounds = std::vector<std::pair<std::optional<double>, std::optional<double>>>;
using ParentSet = std::vector<std::vector<double>>;

namespace mp_detail {

inline std::vector<double> centre(const ParentSet& P)
{
    std::vector<double> g(P[0].size(), 0.0);
    for (const auto& x : P)
        for (std::size_t j = 0; j < g.size(); ++j) g[j] += x[j];
    for (double& v : g) v /= static_cast<double>(P.size());
    return g;
}

inline double norm(const std::vector<double>& v)
{
    double s = 0.0;
    for (double x : v) s += x * x;
    return std::sqrt(s);
}

// An isotropic N(0, s²) vector with its component along the unit vector e
// removed (e empty: nothing removed) — MP-1.
template <typename RNG>
inline std::vector<double> perpendicular_normal(std::size_t n, double s,
                                                const std::vector<double>& e, RNG& rng)
{
    std::normal_distribution<double> N01(0.0, 1.0);
    std::vector<double> h(n);
    for (auto& v : h) v = s * N01(rng);
    if (!e.empty()) {
        double dot = 0.0;
        for (std::size_t j = 0; j < n; ++j) dot += h[j] * e[j];
        for (std::size_t j = 0; j < n; ++j) h[j] -= dot * e[j];
    }
    return h;
}

// MP-4: a variable outside the box back in, about the centre the child came from
template <typename RNG>
inline void repair(std::vector<double>& y, const std::vector<double>& about, const Bounds& bounds,
                   BoundRepair r, RNG& rng)
{
    RepairTally tally;
    for (std::size_t j = 0; j < y.size(); ++j) {
        const double lo = sbx_require_bound(bounds[j].first,  "lower", static_cast<int>(j));
        const double hi = sbx_require_bound(bounds[j].second, "upper", static_cast<int>(j));
        if (y[j] >= lo && y[j] <= hi) continue;
        tally.out();
        y[j] = repair_value(y[j], lo, hi, std::clamp(about[j], lo, hi), r, rng);
    }
}

}   // namespace mp_detail

template <typename RNG>
inline std::vector<double> spx_child(const ParentSet& P, double eps, RNG& rng)
{
    const auto g = mp_detail::centre(P);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    std::vector<double> w(P.size());
    double sum = 0.0;
    for (auto& v : w) { v = -std::log(1.0 - U(rng)); sum += v; }
    std::vector<double> y = g;
    for (std::size_t i = 0; i < P.size(); ++i)
        for (std::size_t j = 0; j < y.size(); ++j)
            y[j] += eps * (w[i] / sum) * (P[i][j] - g[j]);
    return y;
}

template <typename RNG>
inline std::vector<double> rex_child(const ParentSet& P, RNG& rng)
{
    const auto g = mp_detail::centre(P);
    std::normal_distribution<double> xi(0.0, std::sqrt(1.0 / static_cast<double>(P.size() - 1)));
    std::vector<double> y = g;
    for (const auto& x : P) {
        const double e = xi(rng);
        for (std::size_t j = 0; j < y.size(); ++j) y[j] += e * (x[j] - g[j]);
    }
    return y;
}

template <typename RNG>
inline std::pair<std::vector<double>, std::vector<double>>
undx_pair(const std::vector<double>& x1, const std::vector<double>& x2, const std::vector<double>& x3,
          double alpha, double beta, RNG& rng)
{
    const std::size_t n = x1.size();
    std::vector<double> m(n), d(n);
    for (std::size_t j = 0; j < n; ++j) { m[j] = 0.5 * (x1[j] + x2[j]); d[j] = x2[j] - x1[j]; }
    const double d1 = mp_detail::norm(d);
    std::vector<double> e1;
    if (d1 > 0.0) { e1 = d; for (double& v : e1) v /= d1; }
    std::vector<double> v(n);
    for (std::size_t j = 0; j < n; ++j) v[j] = x3[j] - x1[j];
    if (!e1.empty()) {
        double dot = 0.0;
        for (std::size_t j = 0; j < n; ++j) dot += v[j] * e1[j];
        for (std::size_t j = 0; j < n; ++j) v[j] -= dot * e1[j];
    }
    const double d2 = mp_detail::norm(v);
    std::normal_distribution<double> z1(0.0, 1.0);
    const double a = alpha * d1 * z1(rng);
    const auto h = mp_detail::perpendicular_normal(n, beta * d2 / std::sqrt(static_cast<double>(n)), e1, rng);
    std::vector<double> c1(n), c2(n);
    for (std::size_t j = 0; j < n; ++j) {
        const double s = (e1.empty() ? 0.0 : a * e1[j]) + h[j];
        c1[j] = m[j] + s;
        c2[j] = m[j] - s;
    }
    return {c1, c2};
}

template <typename RNG>
inline std::pair<std::vector<double>, std::size_t>
pcx_child(const ParentSet& P, double sigma_zeta, double sigma_eta, RNG& rng)
{
    const std::size_t n = P[0].size();
    std::uniform_int_distribution<std::size_t> pick(0, P.size() - 1);
    const std::size_t p = pick(rng);
    const auto g = mp_detail::centre(P);
    std::vector<double> d(n);
    for (std::size_t j = 0; j < n; ++j) d[j] = P[p][j] - g[j];
    const double dn = mp_detail::norm(d);
    std::vector<double> e;
    if (dn > 0.0) { e = d; for (double& v : e) v /= dn; }
    double Dbar = 0.0;
    for (std::size_t i = 0; i < P.size(); ++i) {
        if (i == p) continue;
        std::vector<double> r(n);
        for (std::size_t j = 0; j < n; ++j) r[j] = P[i][j] - g[j];
        if (!e.empty()) {
            double dot = 0.0;
            for (std::size_t j = 0; j < n; ++j) dot += r[j] * e[j];
            for (std::size_t j = 0; j < n; ++j) r[j] -= dot * e[j];
        }
        Dbar += mp_detail::norm(r);
    }
    Dbar /= static_cast<double>(P.size() - 1);
    std::normal_distribution<double> N01(0.0, 1.0);
    const double wz = sigma_zeta * N01(rng);
    const auto h = mp_detail::perpendicular_normal(n, sigma_eta * Dbar, e, rng);
    std::vector<double> y(n);
    for (std::size_t j = 0; j < n; ++j) y[j] = P[p][j] + wz * d[j] + h[j];
    return {y, p};
}

}   // namespace mootation::ops

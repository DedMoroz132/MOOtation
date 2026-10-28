#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// Designs of experiments on the library's own generators (task 17, stage 5):
// the points python/mootation/designs.py lays out — the first population of
// minimize(init=...) and the sampling baselines (run/baselines.py) — drawn
// from std::mt19937 and rnd:: (random.hpp), which give the same numbers on
// every platform, where designs.py drew from NumPy's and SciPy's generators.
// This is the reference now (DESIGN version 2); designs.py's NumPy and SciPy
// code stays as a check of the distributions. All points are in [0, 1)^d,
// row by row; the caller scales them to its box.
//
//   uniform     uniform points, mt19937(seed) word after word (rnd::uniform01)
//   sobol       the Sobol sequence (sobol.hpp, index 0 included) under
//               Matousek's linear matrix scrambling with a digital shift
//               (J. Matousek, "On the L2-discrepancy for anchored boxes",
//               J. Complexity 14, 1998): per coordinate a random lower
//               triangular binary matrix with a unit diagonal and a random
//               shift, from mt19937(seed); what SciPy's qmc.Sobol(scramble=
//               True) does with its own generator
//   halton_rand the randomized Halton sequence of designs.py (Owen 2017,
//               Eq. 5), the digit permutations of coordinate j from mt19937
//               seeded by std::seed_seq{seed, j}
//   sobol_owen  the Sobol sequence under Burley's hash-based Owen scrambling
//               (designs.py, SOBOL-OWEN): no generator, the key a splitmix64
//               hash of (seed, j) — the same points as designs.py's
//   lhs         the maximin Latin hypercube of designs.py (Morris & Mitchell
//               1995, its annealing and its two departures), its random
//               choices from mt19937(seed), then a point drawn in each cell
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "detmath.hpp"
#include "random.hpp"
#include "sobol.hpp"

namespace mootation::designs {

constexpr int kVersion = 2;                         // the library's generators (task 17, stage 5)

inline std::uint32_t seed32(std::int64_t seed) { return static_cast<std::uint32_t>(seed); }

// ── uniform ─────────────────────────────────────────────────────────────────

inline std::vector<double> uniform(std::size_t n, int d, std::int64_t seed)
{
    std::mt19937 g(seed32(seed));
    std::vector<double> U(n * static_cast<std::size_t>(d));
    for (double& u : U) u = rnd::uniform01(g);
    return U;
}

// ── Sobol, linear matrix scrambling with a digital shift ────────────────────

inline std::vector<double> sobol_lms(std::size_t n, int d, std::int64_t seed)
{
    std::mt19937 g(seed32(seed));
    // per coordinate: row k of L (k = 0 the most significant digit) holds its
    // own digit and random digits above it, then the shift
    std::vector<std::vector<std::uint32_t>> rows(static_cast<std::size_t>(d));
    std::vector<std::uint32_t> shift(static_cast<std::size_t>(d));
    for (int j = 0; j < d; ++j) {
        auto& L = rows[static_cast<std::size_t>(j)];
        for (int k = 0; k < 32; ++k) {
            const std::uint32_t own = 1u << (31 - k);
            const std::uint32_t above = k == 0 ? 0u : ~((1u << (32 - k)) - 1u);     // the k digits above
            L.push_back(own | (rnd::word32(g) & above));
        }
        shift[static_cast<std::size_t>(j)] = rnd::word32(g);
    }
    const auto X = sobol::integers(n, d);
    std::vector<double> U(X.size());
    for (std::size_t i = 0; i < n; ++i)
        for (int j = 0; j < d; ++j) {
            const std::uint32_t x = X[i * static_cast<std::size_t>(d) + static_cast<std::size_t>(j)];
            std::uint32_t y = 0;
            for (int k = 0; k < 32; ++k) {
                std::uint32_t v = x & rows[static_cast<std::size_t>(j)][static_cast<std::size_t>(k)];
                v ^= v >> 16;                       // the parity of the digits L's row picks
                v ^= v >> 8;
                v ^= v >> 4;
                v ^= v >> 2;
                v ^= v >> 1;
                y |= (v & 1u) << (31 - k);
            }
            y ^= shift[static_cast<std::size_t>(j)];
            U[i * static_cast<std::size_t>(d) + static_cast<std::size_t>(j)] = y * (1.0 / 4294967296.0);
        }
    return U;
}

// ── randomized Halton ───────────────────────────────────────────────────────

inline std::vector<int> primes(int d)
{
    std::vector<int> out;
    for (int k = 2; static_cast<int>(out.size()) < d; ++k) {
        bool prime = true;
        for (int p : out) {
            if (p * p > k) break;
            if (k % p == 0) prime = false;
        }
        if (prime) out.push_back(k);
    }
    return out;
}

// how many base-b digits a double resolves (Owen 2017, Algorithm 1's loop)
inline int halton_digits(int b)
{
    int k = 0;
    double b2r = 1.0 / b;
    while (1.0 - b2r < 1.0) {
        ++k;
        b2r /= b;
    }
    return k;
}

inline std::vector<double> halton(std::size_t n, int d, std::int64_t seed, std::uint64_t start = 0)
{
    std::vector<double> U(n * static_cast<std::size_t>(d));
    const auto ps = primes(d);
    for (int j = 0; j < d; ++j) {
        const int b = ps[static_cast<std::size_t>(j)];
        std::seed_seq seq{static_cast<std::uint32_t>(static_cast<std::uint64_t>(seed) & 0xFFFFFFFFu),
                          static_cast<std::uint32_t>(static_cast<std::uint64_t>(seed) >> 32),
                          static_cast<std::uint32_t>(j)};
        std::mt19937 g(seq);
        std::vector<std::vector<int>> perms;
        for (int k = halton_digits(b); k > 0; --k) {
            std::vector<int> p(static_cast<std::size_t>(b));
            std::iota(p.begin(), p.end(), 0);
            rnd::shuffle(p.begin(), p.end(), g);
            perms.push_back(std::move(p));
        }
        for (std::size_t i = 0; i < n; ++i) {
            std::uint64_t rest = start + i;
            double scale = 1.0, x = 0.0;
            for (const auto& p : perms) {
                scale /= b;
                x += p[static_cast<std::size_t>(rest % static_cast<std::uint64_t>(b))] * scale;
                rest /= static_cast<std::uint64_t>(b);
            }
            U[i * static_cast<std::size_t>(d) + static_cast<std::size_t>(j)] = x;
        }
    }
    return U;
}

// ── Sobol with Owen's nested uniform scrambling (Burley 2020) ───────────────

inline std::uint32_t reverse_bits(std::uint32_t x)
{
    x = ((x >> 1) & 0x55555555u) | ((x & 0x55555555u) << 1);
    x = ((x >> 2) & 0x33333333u) | ((x & 0x33333333u) << 2);
    x = ((x >> 4) & 0x0F0F0F0Fu) | ((x & 0x0F0F0F0Fu) << 4);
    x = ((x >> 8) & 0x00FF00FFu) | ((x & 0x00FF00FFu) << 8);
    return (x >> 16) | (x << 16);
}

inline std::uint32_t laine_karras(std::uint32_t x, std::uint32_t key)
{
    x += key;
    x ^= x * 0x6c50b47cu;
    x ^= x * 0xb82f1e52u;
    x ^= x * 0xc7afe638u;
    x ^= x * 0x8d22f6e6u;
    return x;
}

inline std::uint64_t splitmix64(std::uint64_t z)
{
    z += 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

inline std::vector<double> sobol_owen(std::size_t n, int d, std::int64_t seed, std::uint64_t start = 0)
{
    const auto X = sobol::integers(n, d, start);
    std::vector<double> U(X.size());
    for (int j = 0; j < d; ++j) {
        const auto key = static_cast<std::uint32_t>(
            splitmix64(static_cast<std::uint64_t>(seed) ^ splitmix64(static_cast<std::uint64_t>(j) + 1)) & 0xFFFFFFFFu);
        for (std::size_t i = 0; i < n; ++i) {
            const std::size_t at = i * static_cast<std::size_t>(d) + static_cast<std::size_t>(j);
            U[at] = reverse_bits(laine_karras(reverse_bits(X[at]), key)) * (1.0 / 4294967296.0);
        }
    }
    return U;
}

// ── maximin Latin hypercubes (Morris & Mitchell 1995; designs.py) ───────────

constexpr int kPValues[] = {1, 2, 5, 10, 20, 50, 100};   // Section 5
constexpr double kFacT = 0.95, kAccept = 0.99;
constexpr long long kIMaxCap = 2000, kBudget = 20000, kBudgetWork = 20000000;

// d^-p for an integer p >= 1 by repeated squaring and one division;
// infinity and overflow give 0
inline double neg_power(double d, int p)
{
    double out = 1.0, base = d;
    for (int e = p; e; e >>= 1) {
        if (e & 1) out *= base;
        if (e > 1) base *= base;
    }
    return 1.0 / out;
}

// the squared distances (in levels) between the rows of the level matrix L
// (n x k, row-major), infinite on the diagonal
inline std::vector<double> squared_distances(const std::vector<int>& L, std::size_t n, std::size_t k)
{
    std::vector<double> D2(n * n, std::numeric_limits<double>::infinity());
    for (std::size_t a = 0; a < n; ++a)
        for (std::size_t b = a + 1; b < n; ++b) {
            double s = 0.0;
            for (std::size_t c = 0; c < k; ++c) {
                const double t = static_cast<double>(L[a * k + c] - L[b * k + c]);
                s += t * t;
            }
            D2[a * n + b] = D2[b * n + a] = s;
        }
    return D2;
}

// the maximin order's key, best first: (-d1, J1, -d2, J2, ...) over the
// `depth` smallest distinct distances
inline std::vector<double> maximin_order(const std::vector<double>& D2, int depth = 8)
{
    std::vector<double> key;
    double floor = -std::numeric_limits<double>::infinity();
    for (int t = 0; t < depth; ++t) {
        double dmin = std::numeric_limits<double>::infinity();
        for (double v : D2)
            if (v > floor && v < dmin) dmin = v;
        if (dmin == std::numeric_limits<double>::infinity()) break;
        long long pairs = 0;
        for (double v : D2) pairs += v == dmin;
        key.push_back(-dmin);
        key.push_back(static_cast<double>(pairs / 2));
        floor = dmin;
    }
    return key;
}

// t0 (Section 4): on a hypothetical design whose C(n, 2) squared distances
// spread evenly over 50-150 % of the class's mean, one of the smallest made
// one level smaller is accepted with probability kAccept
inline double initial_temperature(std::size_t n, std::size_t k, int p, double scale)
{
    const double m = static_cast<double>(n * (n - 1) / 2);
    const double mean = static_cast<double>(k * n * (n + 1)) / 6.0;
    const double lo = 0.5 * mean / scale, hi = 1.5 * mean / scale;
    double total = 0.0;
    const std::size_t count = n * (n - 1) / 2;
    for (std::size_t i = 0; i < count; ++i) total += neg_power(lo + (hi - lo) * static_cast<double>(i) / std::max(m - 1.0, 1.0), p);
    const double closer = std::max(lo - 1.0 / scale, 1e-12);
    const double total_try = total - neg_power(lo, p) + neg_power(closer, p);
    const double d_phi = dm::pow(total_try, 1.0 / p) - dm::pow(total, 1.0 / p);
    return d_phi / -dm::log(kAccept);
}

// Fig. 1 on phi_p of the squared distances, from the Latin hypercube L (n x k,
// row-major, levels); returns the best design it met
template <class G>
inline std::vector<int> anneal(std::vector<int> L, std::size_t n, std::size_t k, int p, G& g)
{
    std::vector<double> D2 = squared_distances(L, n, k);
    double scale = std::numeric_limits<double>::infinity();
    for (double v : D2) scale = std::min(scale, v);
    for (double& v : D2) v /= scale;
    std::vector<double> T(n * n);
    double S = 0.0;
    for (std::size_t i = 0; i < n * n; ++i) {
        T[i] = neg_power(D2[i], p);
        S += T[i];
    }
    S /= 2.0;
    double phi = dm::pow(S, 1.0 / p);
    double t = initial_temperature(n, k, p, scale);
    const long long i_max = std::min<long long>(10LL * static_cast<long long>(k * n * (n - 1) / 2), kIMaxCap);
    const long long budget = std::min<long long>(kBudget, kBudgetWork / static_cast<long long>(n));
    std::vector<int> best = L;
    double best_S = S;
    bool at_best = true;
    long long spent = 0;
    std::vector<double> delta(n), new_a(n), new_b(n), t_a(n), t_b(n);
    while (spent < budget) {
        bool accepted = false;
        long long tries = 1;
        while (tries < i_max && spent < budget) {
            ++spent;
            const std::size_t c = static_cast<std::size_t>(rnd::below(g, k));
            const std::size_t a = static_cast<std::size_t>(rnd::below(g, n));
            std::size_t b = static_cast<std::size_t>(rnd::below(g, n - 1));
            b += b >= a;                                     // a second row, not a
            // a takes b's level and b a's: the other rows' distances to them change
            double s_try_delta = 0.0;
            for (std::size_t r = 0; r < n; ++r) {
                const double lr = L[r * k + c], la = L[a * k + c], lb = L[b * k + c];
                delta[r] = (r == a || r == b) ? 0.0 : ((lb - lr) * (lb - lr) - (la - lr) * (la - lr)) / scale;
                new_a[r] = D2[a * n + r] + delta[r];
                new_b[r] = D2[b * n + r] - delta[r];
                t_a[r] = neg_power(new_a[r], p);
                t_b[r] = neg_power(new_b[r], p);
                s_try_delta += t_a[r] - T[a * n + r] + t_b[r] - T[b * n + r];
            }
            const double S_try = S + s_try_delta;
            const double phi_try = S_try > 0 ? dm::pow(S_try, 1.0 / p) : std::numeric_limits<double>::infinity();
            if (S_try < S || rnd::uniform01(g) < dm::exp(-(phi_try - phi) / t)) {
                if (at_best && S_try > best_S) {
                    best = L;                               // leaving the best design
                    at_best = false;
                }
                std::swap(L[a * k + c], L[b * k + c]);
                for (std::size_t r = 0; r < n; ++r) {
                    D2[a * n + r] = D2[r * n + a] = new_a[r];
                    D2[b * n + r] = D2[r * n + b] = new_b[r];
                    T[a * n + r] = T[r * n + a] = t_a[r];
                    T[b * n + r] = T[r * n + b] = t_b[r];
                }
                S = S_try;
                phi = phi_try;
                accepted = true;
            }
            if (S_try < best_S) {
                best_S = S_try;
                at_best = true;
                tries = 1;
            } else {
                ++tries;
            }
        }
        S = 0.0;                                             // again in full, against the drift
        for (double v : T) S += v;
        S /= 2.0;
        phi = dm::pow(S, 1.0 / p);
        if (!accepted) break;
        t *= kFacT;
    }
    return at_best ? L : best;
}

// the level matrix (n x k, row-major) of a maximin Latin hypercube: the best
// of one annealing per p by the maximin order
template <class G>
inline std::vector<int> maximin_lhs(std::size_t n, std::size_t k, G& g)
{
    std::vector<int> best;
    std::vector<double> best_key;
    for (int p : kPValues) {
        std::vector<int> L(n * k);
        for (std::size_t c = 0; c < k; ++c) {
            std::vector<int> col(n);
            std::iota(col.begin(), col.end(), 0);
            rnd::shuffle(col.begin(), col.end(), g);
            for (std::size_t r = 0; r < n; ++r) L[r * k + c] = col[r];
        }
        if (n > 2) L = anneal(std::move(L), n, k, p, g);
        const auto key = maximin_order(squared_distances(L, n, k));
        if (best.empty() || std::lexicographical_compare(key.begin(), key.end(), best_key.begin(), best_key.end())) {
            best = L;
            best_key = key;
        }
    }
    return best;
}

inline std::vector<double> lhs(std::size_t n, int d, std::int64_t seed)
{
    std::mt19937 g(seed32(seed));
    const std::size_t k = static_cast<std::size_t>(d);
    const auto L = maximin_lhs(n, k, g);
    std::vector<double> U(n * k);
    for (std::size_t i = 0; i < n * k; ++i) U[i] = (L[i] + rnd::uniform01(g)) / static_cast<double>(n);   // a point in each cell
    return U;
}

// ── by name ─────────────────────────────────────────────────────────────────

// designs.KINDS: the designs of a first population
inline const std::vector<std::string>& kinds()
{
    static const std::vector<std::string> v{"sobol", "lhs", "halton_rand", "sobol_owen"};
    return v;
}

// n points of kind `kind` in [0, 1)^d (designs.KINDS, and "uniform")
inline std::vector<double> design(const std::string& kind, std::size_t n, int d, std::int64_t seed)
{
    if (kind == "uniform") return uniform(n, d, seed);
    if (kind == "sobol") return sobol_lms(n, d, seed);
    if (kind == "halton_rand") return halton(n, d, seed);
    if (kind == "sobol_owen") return sobol_owen(n, d, seed);
    if (kind == "lhs") return lhs(n, d, seed);
    throw std::invalid_argument("unknown design '" + kind + "'; known: uniform, sobol, lhs, halton_rand, sobol_owen");
}

}   // namespace mootation::designs

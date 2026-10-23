#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// The numbers the suite's instances draw (task 17, supplement 5, §3): the
// rotations of the rotated problems and the bbob instances, drawn here, by
// the library's generator, where NumPy's default_rng drew them before. A key
// of a few integers names an instance; its generator is a std::mt19937 seeded
// with them (seed below), and every draw is rnd::'s (random.hpp), the same on
// every platform. benchmarks/library_rng.py draws the same numbers in Python,
// the cross-check's.
//
// haar: a k x k rotation from the Haar distribution — the Q of the QR
// factorization of a matrix of standard normal entries, normalised so that
// R's diagonal is positive (Higham, Accuracy and Stability of Numerical
// Algorithms, 2nd ed., 2002, §28.3, after Stewart 1980); the factorization by
// Householder reflections (ibid., §19.1-19.2, the sign of (19.1)), Q formed
// from right to left.
// ============================================================================

#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <random>
#include <string>
#include <vector>

#include "../random.hpp"

namespace mootation::suite::instances {

// CRC-32 of a name (IEEE 802.3, the reflected polynomial 0xEDB88320: zlib's
// crc32, which the Python side takes)
inline std::uint32_t crc32(const std::string& s)
{
    std::uint32_t c = 0xFFFFFFFFu;
    for (unsigned char byte : s) {
        c ^= byte;
        for (int k = 0; k < 8; ++k) c = (c & 1u) ? (c >> 1) ^ 0xEDB88320u : c >> 1;
    }
    return c ^ 0xFFFFFFFFu;
}

inline std::uint64_t splitmix64(std::uint64_t z)
{
    z = z + 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

// The seed of an instance's generator: SplitMix64 chained over the key's
// words (h = splitmix64(h ^ w), from h = 0), its low 32 bits.
inline std::uint32_t seed(std::initializer_list<std::uint64_t> key)
{
    std::uint64_t h = 0;
    for (std::uint64_t w : key) h = splitmix64(h ^ w);
    return static_cast<std::uint32_t>(h & 0xFFFFFFFFu);
}

// A k x k rotation from the Haar distribution, row-major: the k^2 normal
// draws row after row, Householder QR, Q's columns times the signs of R's
// diagonal.
inline std::vector<double> haar(std::mt19937& g, int k)
{
    const std::size_t n = static_cast<std::size_t>(k);
    std::vector<double> A(n * n), Q(n * n, 0.0), beta(n, 0.0), diag(n, 0.0);
    for (double& a : A) a = rnd::normal01(g);
    auto at = [n](std::vector<double>& M, std::size_t i, std::size_t j) -> double& { return M[i * n + j]; };
    std::vector<std::vector<double>> V(n);
    for (std::size_t j = 0; j + 1 < n; ++j) {
        double ss = 0.0;
        for (std::size_t i = j; i < n; ++i) ss = ss + at(A, i, j) * at(A, i, j);
        const double sigma = at(A, j, j) > 0.0 ? -std::sqrt(ss) : std::sqrt(ss);   // (19.1)
        std::vector<double>& v = V[j];
        for (std::size_t i = j; i < n; ++i) v.push_back(at(A, i, j));
        v[0] = v[0] - sigma;
        double vv = 0.0;
        for (double e : v) vv = vv + e * e;
        diag[j] = sigma;
        if (vv == 0.0) continue;                        // x = 0: no reflection
        beta[j] = 2.0 / vv;
        for (std::size_t c = j; c < n; ++c) {           // A[j:, c] -= beta v (v . A[j:, c])
            double s = 0.0;
            for (std::size_t i = j; i < n; ++i) s = s + v[i - j] * at(A, i, c);
            const double t = beta[j] * s;
            for (std::size_t i = j; i < n; ++i) at(A, i, c) = at(A, i, c) - t * v[i - j];
        }
    }
    diag[n - 1] = at(A, n - 1, n - 1);
    for (std::size_t i = 0; i < n; ++i) at(Q, i, i) = 1.0;
    for (std::size_t jj = n - 1; jj-- > 0;) {           // Q = P_1 (P_2 (... P_{n-1}))
        if (beta[jj] == 0.0) continue;
        const std::vector<double>& v = V[jj];
        for (std::size_t c = 0; c < n; ++c) {
            double s = 0.0;
            for (std::size_t i = jj; i < n; ++i) s = s + v[i - jj] * at(Q, i, c);
            const double t = beta[jj] * s;
            for (std::size_t i = jj; i < n; ++i) at(Q, i, c) = at(Q, i, c) - t * v[i - jj];
        }
    }
    for (std::size_t c = 0; c < n; ++c)
        if (diag[c] < 0.0)
            for (std::size_t i = 0; i < n; ++i) at(Q, i, c) = -at(Q, i, c);
    return Q;
}

}   // namespace mootation::suite::instances

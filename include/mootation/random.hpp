#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// Random numbers that are the same on every platform (task 14 of 2026-10-20,
// §1). The generator stays std::mt19937: the standard defines its output bit
// for bit. What the standard does NOT define is how its distributions turn
// those words into numbers — MSVC, libstdc++ and libc++ each do it their own
// way, so one seed gave one run on Windows and another on Linux. Everything
// that makes numbers from the generator's words is therefore here, defined
// down to the bit, in the namespace rnd:
//
//   uniform01(g)        53 bits from two 32-bit words, [0, 1): genrand_res53
//                       of Matsumoto & Nishimura's reference code mt19937ar.c
//                       (2002) — not of the 1998 paper (Mersenne Twister, ACM
//                       TOMACS 8(1):3-30, 1998, source mersenne-twister_
//                       matsumoto1998), which defines the generator only
//   below(g, n), integer(g, lo, hi)
//                       uniform integers by Lemire's multiply-and-reject (Fast
//                       Random Integer Generation in an Interval, ACM TOMACS
//                       29(1), 2019): one 32-bit word a draw for ranges up to
//                       2^32, two words (a 64-bit draw) above; no bias
//   normal01(g)         Marsaglia's polar method (Marsaglia & Bray, A
//                       convenient method for generating normal variables,
//                       SIAM Review 6(3):260-264, 1964, §1: X = v1 (-2 ln S /
//                       S)^(1/2), v uniform on [-1, 1], S = v1^2 + v2^2 < 1;
//                       source normal-polar-method_marsaglia1964), one value a
//                       call (the second of the pair is dropped, so a value
//                       never depends on an earlier call or on the
//                       distribution object's life); S = 0 is drawn again too
//   exponential, cauchy, gamma (Marsaglia & Tsang, ACM TOMS 26(3), 2000),
//   bernoulli, discrete (cumulative weights, a binary search), shuffle
//                       (Fisher-Yates from the last element down, by below)
//
// The class templates rnd::uniform_int_distribution, uniform_real_distribution,
// normal_distribution, ... take the place of the std ones of the same names
// (constructed the same way, called with the generator): the library's code
// changed `std::` to `rnd::` and nothing else. The repository's CI refuses
// std:: distributions and std::shuffle anywhere in include/ but here.
//
// The elementary functions in normal01, exponential, cauchy and gamma (log,
// tan, pow) are the library's own (dm::, detmath.hpp, §2), sqrt is exact by
// IEEE: the values are the same on every platform, given the floating-point
// environment: no contraction into fused multiply-adds (§3), no
// flush-to-zero. A run is reproducible only with a seed given (set_seed);
// without one the cores seed from std::random_device.
// ============================================================================

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#include "detmath.hpp"
#include "fp_contract.hpp"

namespace mootation::rnd {

// One 32-bit word of a generator of 32 or 64 bits (std::mt19937, mt19937_64;
// the 64-bit one gives its high half).
template <typename G>
inline std::uint32_t word32(G& g)
{
    static_assert(G::min() == 0, "rnd: a generator of whole words");
    if constexpr (G::max() == 0xFFFFFFFFu) {
        return static_cast<std::uint32_t>(g());
    } else {
        static_assert(G::max() == std::numeric_limits<std::uint64_t>::max(),
                      "rnd: a generator of 32- or 64-bit words");
        return static_cast<std::uint32_t>(static_cast<std::uint64_t>(g()) >> 32);
    }
}

template <typename G>
inline std::uint64_t word64(G& g)
{
    const std::uint64_t hi = word32(g);
    return (hi << 32) | word32(g);
}

// [0, 1) with 53 random bits: genrand_res53.
template <typename G>
inline double uniform01(G& g)
{
    const std::uint32_t a = word32(g) >> 5, b = word32(g) >> 6;
    return (a * 67108864.0 + b) * (1.0 / 9007199254740992.0);
}

// The high and low 64 bits of a * b.
inline void mul64(std::uint64_t a, std::uint64_t b, std::uint64_t& hi, std::uint64_t& lo)
{
    const std::uint64_t a0 = a & 0xFFFFFFFFu, a1 = a >> 32, b0 = b & 0xFFFFFFFFu, b1 = b >> 32;
    const std::uint64_t p00 = a0 * b0, p01 = a0 * b1, p10 = a1 * b0, p11 = a1 * b1;
    const std::uint64_t mid = (p00 >> 32) + (p01 & 0xFFFFFFFFu) + (p10 & 0xFFFFFFFFu);
    lo = (mid << 32) | (p00 & 0xFFFFFFFFu);
    hi = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
}

// Uniform in [0, n), n >= 1: Lemire's method (Algorithm 5 of the paper).
template <typename G>
inline std::uint64_t below(G& g, std::uint64_t n)
{
    if (n <= 0xFFFFFFFFu) {                                  // one word a draw
        const std::uint32_t n32 = static_cast<std::uint32_t>(n);
        std::uint64_t m = static_cast<std::uint64_t>(word32(g)) * n32;
        std::uint32_t l = static_cast<std::uint32_t>(m);
        if (l < n32) {
            const std::uint32_t t = static_cast<std::uint32_t>(0u - n32) % n32;
            while (l < t) {
                m = static_cast<std::uint64_t>(word32(g)) * n32;
                l = static_cast<std::uint32_t>(m);
            }
        }
        return m >> 32;
    }
    std::uint64_t hi, lo;                                    // two words a draw
    mul64(word64(g), n, hi, lo);
    if (lo < n) {
        const std::uint64_t t = (0u - n) % n;
        while (lo < t) mul64(word64(g), n, hi, lo);
    }
    return hi;
}

// Uniform in [lo, hi] of an integer type (lo <= hi).
template <typename T, typename G>
inline T integer(G& g, T lo, T hi)
{
    static_assert(std::is_integral<T>::value, "rnd::integer: an integer type");
    const std::uint64_t span = static_cast<std::uint64_t>(hi) - static_cast<std::uint64_t>(lo);
    const std::uint64_t r = span == std::numeric_limits<std::uint64_t>::max() ? word64(g)
                                                                             : below(g, span + 1);
    return static_cast<T>(static_cast<std::uint64_t>(lo) + r);
}

// Standard normal: Marsaglia's polar method, one value a call.
template <typename G>
inline double normal01(G& g)
{
    double u, v, s;
    do {
        u = 2.0 * uniform01(g) - 1.0;
        v = 2.0 * uniform01(g) - 1.0;
        s = u * u + v * v;
    } while (s >= 1.0 || s == 0.0);
    return u * std::sqrt(-2.0 * dm::log(s) / s);
}

// Exponential of rate lambda: -log(1 - U) / lambda, 1 - U in (0, 1].
template <typename G>
inline double exponential(G& g, double lambda)
{
    return -dm::log(1.0 - uniform01(g)) / lambda;
}

// Cauchy of location a and scale b: a + b tan(pi (U - 1/2)).
template <typename G>
inline double cauchy(G& g, double a, double b)
{
    const double pi = 3.141592653589793238462643383279502884;
    return a + b * dm::tan(pi * (uniform01(g) - 0.5));
}

// Gamma of shape alpha and scale beta: Marsaglia & Tsang 2000 (§2; alpha < 1
// by their §6: Gamma(alpha + 1) U^(1/alpha), U uniform on (0, 1) — uniform01
// is on [0, 1), so U = 0, a chance of 2^-53, is drawn again: the paper's U,
// and the values of every other draw unchanged).
template <typename G>
inline double gamma(G& g, double alpha, double beta)
{
    if (alpha < 1.0) {
        const double x = gamma(g, alpha + 1.0, 1.0);
        double u = uniform01(g);
        while (u == 0.0) u = uniform01(g);
        return beta * x * dm::pow(u, 1.0 / alpha);
    }
    const double d = alpha - 1.0 / 3.0, c = 1.0 / std::sqrt(9.0 * d);
    for (;;) {
        double x, v;
        do {
            x = normal01(g);
            v = 1.0 + c * x;
        } while (v <= 0.0);
        v = v * v * v;
        const double u = uniform01(g);
        const double x2 = x * x;
        if (u < 1.0 - 0.0331 * x2 * x2) return beta * d * v;
        if (dm::log(u) < 0.5 * x2 + d * (1.0 - v + dm::log(v))) return beta * d * v;
    }
}

// Fisher-Yates, from the last element down.
template <typename It, typename G>
inline void shuffle(It first, It last, G& g)
{
    const auto n = static_cast<std::uint64_t>(std::distance(first, last));
    for (std::uint64_t i = n; i > 1; --i) {
        const std::uint64_t j = below(g, i);
        if (j != i - 1) std::swap(*(first + static_cast<std::ptrdiff_t>(i - 1)),
                                  *(first + static_cast<std::ptrdiff_t>(j)));
    }
}

// ── the std:: distributions' places ─────────────────────────────────────────

template <typename T = int>
class uniform_int_distribution {
public:
    explicit uniform_int_distribution(T a = 0, T b = std::numeric_limits<T>::max()) : a_(a), b_(b) {}
    template <typename G> T operator()(G& g) const { return integer<T>(g, a_, b_); }
    T a() const { return a_; }
    T b() const { return b_; }
private:
    T a_, b_;
};

template <typename T = double>
class uniform_real_distribution {
public:
    explicit uniform_real_distribution(T a = 0, T b = 1) : a_(a), b_(b) {}
    template <typename G> T operator()(G& g) const
    {
        return static_cast<T>(a_ + (b_ - a_) * uniform01(g));
    }
    T a() const { return a_; }
    T b() const { return b_; }
private:
    T a_, b_;
};

template <typename T = double>
class normal_distribution {
public:
    explicit normal_distribution(T mean = 0, T stddev = 1) : mean_(mean), sd_(stddev) {}
    template <typename G> T operator()(G& g) const { return static_cast<T>(mean_ + sd_ * normal01(g)); }
    T mean() const { return mean_; }
    T stddev() const { return sd_; }
private:
    T mean_, sd_;
};

template <typename T = double>
class exponential_distribution {
public:
    explicit exponential_distribution(T lambda = 1) : lambda_(lambda) {}
    template <typename G> T operator()(G& g) const { return static_cast<T>(exponential(g, lambda_)); }
private:
    T lambda_;
};

template <typename T = double>
class cauchy_distribution {
public:
    explicit cauchy_distribution(T a = 0, T b = 1) : a_(a), b_(b) {}
    template <typename G> T operator()(G& g) const { return static_cast<T>(cauchy(g, a_, b_)); }
private:
    T a_, b_;
};

template <typename T = double>
class gamma_distribution {
public:
    explicit gamma_distribution(T alpha = 1, T beta = 1) : alpha_(alpha), beta_(beta) {}
    template <typename G> T operator()(G& g) const { return static_cast<T>(gamma(g, alpha_, beta_)); }
private:
    T alpha_, beta_;
};

class bernoulli_distribution {
public:
    explicit bernoulli_distribution(double p = 0.5) : p_(p) {}
    template <typename G> bool operator()(G& g) const { return uniform01(g) < p_; }
private:
    double p_;
};

// Index i with probability w_i / sum(w): the cumulative weights, summed in
// order, and the first that exceeds U * sum. No weight, or all zero: index 0.
template <typename I = int>
class discrete_distribution {
public:
    template <typename It>
    discrete_distribution(It first, It last)
    {
        double s = 0.0;
        for (; first != last; ++first) {
            s += static_cast<double>(*first);
            cum_.push_back(s);
        }
    }
    template <typename G> I operator()(G& g) const
    {
        if (cum_.empty() || !(cum_.back() > 0.0)) return 0;
        const double r = uniform01(g) * cum_.back();
        std::size_t lo = 0, hi = cum_.size() - 1;              // the first cum_ > r
        while (lo < hi) {
            const std::size_t mid = lo + (hi - lo) / 2;
            if (cum_[mid] > r) hi = mid; else lo = mid + 1;
        }
        return static_cast<I>(lo);
    }
private:
    std::vector<double> cum_;
};

}   // namespace mootation::rnd

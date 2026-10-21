// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// random.hpp (task 14 of 2026-10-20, §1): the numbers the library makes from
// std::mt19937's words, the same on every platform.
//   * Bit for bit: the first 10 000 values of uniform01, below (n of 32 and 64
//     bits), integer, bernoulli, discrete and shuffle at a fixed seed against
//     tests/data/random_reference.txt — one file for every platform, so CI's
//     platforms check each other. `test_random --write <file>` writes it.
//     Since detmath (§2) normal01, exponential, cauchy and gamma too (their
//     log, tan and pow are dm::'s, the same everywhere): 10 000 values each,
//     gamma 5 000 at alpha = 2.5 and 5 000 at alpha = 0.4 (the boost of
//     Marsaglia & Tsang's note for alpha < 1).
//   * As distributions, 10^6 values each: the moments; Kolmogorov-Smirnov
//     for uniform01, normal01, exponential, cauchy and gamma(2.5); chi-square
//     for below(n) at n not dividing 2^32 (7, 1000003, 3·2^30) and above 2^32,
//     for discrete's weights, and for shuffle's 24 orders of four.
// ============================================================================

#include <algorithm>
#include <array>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <mootation/random.hpp>

namespace {

namespace rnd = mootation::rnd;

int checks = 0, failed = 0;

void check(bool ok, const std::string& what)
{
    ++checks;
    if (!ok) {
        ++failed;
        std::printf("  FAIL  %s\n", what.c_str());
    }
}

// The reference values, one "name value" a line (a double as its 64 bits in hex).
std::vector<std::string> reference_lines()
{
    std::vector<std::string> out;
    char buf[64];
    auto put = [&](const char* name, const std::string& v) { out.push_back(std::string(name) + " " + v); };
    auto hex = [&](double x) {                          // the bits, not %a: printf's %a differs
        std::uint64_t u;
        std::memcpy(&u, &x, sizeof u);
        std::snprintf(buf, sizeof buf, "%016" PRIx64, u);
        return std::string(buf);
    };
    auto dec = [&](std::uint64_t x) { std::snprintf(buf, sizeof buf, "%" PRIu64, x); return std::string(buf); };
    {
        std::mt19937 g(20261020u);
        for (int i = 0; i < 10000; ++i) put("uniform01", hex(rnd::uniform01(g)));
    }
    {
        std::mt19937 g(1u);
        const std::uint64_t ns[] = {1, 2, 7, 1000003, 3221225472ull, 4294967295ull, 4294967296ull,
                                    10000000019ull, 18446744073709551557ull};
        for (int i = 0; i < 10000; ++i) put("below", dec(rnd::below(g, ns[i % 9])));
    }
    {
        std::mt19937 g(2u);
        for (int i = 0; i < 10000; ++i) {
            put("integer", dec(static_cast<std::uint64_t>(static_cast<std::int64_t>(
                               rnd::integer<int>(g, -50, 49)) + 1000)));
        }
    }
    {
        std::mt19937 g(3u);
        rnd::bernoulli_distribution b(0.3);
        for (int i = 0; i < 10000; ++i) put("bernoulli", b(g) ? "1" : "0");
    }
    {
        std::mt19937 g(4u);
        const std::vector<double> w = {0.5, 0.0, 2.0, 1.25, 0.25};
        rnd::discrete_distribution<int> d(w.begin(), w.end());
        for (int i = 0; i < 10000; ++i) put("discrete", dec(static_cast<std::uint64_t>(d(g))));
    }
    {
        std::mt19937 g(5u);
        for (int i = 0; i < 1000; ++i) {
            std::vector<int> v(10);
            for (int k = 0; k < 10; ++k) v[static_cast<std::size_t>(k)] = k;
            rnd::shuffle(v.begin(), v.end(), g);
            std::string s;
            for (int x : v) s += static_cast<char>('0' + x);
            put("shuffle", s);
        }
    }
    {
        std::mt19937_64 g(6u);                          // a 64-bit generator: its high words
        for (int i = 0; i < 1000; ++i) put("uniform01_64", hex(rnd::uniform01(g)));
    }
    {
        std::mt19937 g(7u);
        for (int i = 0; i < 10000; ++i) put("normal01", hex(rnd::normal01(g)));
    }
    {
        std::mt19937 g(8u);
        for (int i = 0; i < 10000; ++i) put("exponential", hex(rnd::exponential(g, 1.5)));
    }
    {
        std::mt19937 g(9u);
        for (int i = 0; i < 10000; ++i) put("cauchy", hex(rnd::cauchy(g, 0.5, 2.0)));
    }
    {
        std::mt19937 g(10u);
        for (int i = 0; i < 5000; ++i) put("gamma_2.5", hex(rnd::gamma(g, 2.5, 1.0)));
    }
    {
        std::mt19937 g(11u);
        for (int i = 0; i < 5000; ++i) put("gamma_0.4", hex(rnd::gamma(g, 0.4, 2.0)));
    }
    return out;
}

// Kolmogorov-Smirnov statistic of x against the CDF F.
template <typename F>
double ks(std::vector<double> x, F cdf)
{
    std::sort(x.begin(), x.end());
    const double n = static_cast<double>(x.size());
    double d = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        const double f = cdf(x[i]);
        d = std::max({d, f - static_cast<double>(i) / n, static_cast<double>(i + 1) / n - f});
    }
    return d;
}

// Chi-square of counts against expected counts.
double chi2(const std::vector<double>& obs, const std::vector<double>& exp)
{
    double c = 0.0;
    for (std::size_t i = 0; i < obs.size(); ++i) c += (obs[i] - exp[i]) * (obs[i] - exp[i]) / exp[i];
    return c;
}

}   // namespace

int main(int argc, char** argv)
{
    const auto lines = reference_lines();
    if (argc == 3 && std::strcmp(argv[1], "--write") == 0) {
        std::ofstream f(argv[2], std::ios::binary);
        for (const auto& l : lines) f << l << "\n";
        std::printf("wrote %zu lines to %s\n", lines.size(), argv[2]);
        return 0;
    }

    // ── bit for bit against the reference file ──────────────────────────────
    {
        std::ifstream f(MOOTATION_TEST_DATA "/random_reference.txt", std::ios::binary);
        std::vector<std::string> want;
        std::string l;
        while (std::getline(f, l)) {
            if (!l.empty() && l.back() == '\r') l.pop_back();
            want.push_back(l);
        }
        check(want.size() == lines.size(), "the reference file has every line");
        std::size_t bad = 0, first = want.size();
        for (std::size_t i = 0; i < std::min(want.size(), lines.size()); ++i)
            if (want[i] != lines[i]) { ++bad; first = std::min(first, i); }
        if (bad) std::printf("  %zu lines differ, the first: want '%s', got '%s'\n", bad,
                             want[first].c_str(), lines[first].c_str());
        check(bad == 0, "the values bit for bit as tests/data/random_reference.txt");
    }

    const int N = 1000000;
    std::mt19937 g(777u);
    // ── uniform01: range, mean, variance, KS ───────────────────────────────
    {
        std::vector<double> x(N);
        double lo = 1.0, hi = 0.0, s = 0.0, s2 = 0.0;
        for (auto& v : x) { v = rnd::uniform01(g); lo = std::min(lo, v); hi = std::max(hi, v); s += v; s2 += v * v; }
        const double m = s / N, var = s2 / N - m * m;
        check(lo >= 0.0 && hi < 1.0, "uniform01 in [0, 1)");
        check(std::abs(m - 0.5) < 2e-3 && std::abs(var - 1.0 / 12) < 2e-3, "uniform01 mean, variance");
        check(ks(x, [](double t) { return t; }) < 1.95 / std::sqrt(double(N)), "uniform01 KS");
        bool fine = true;                                     // 53 bits: multiples of 2^-53
        for (int i = 0; i < 1000; ++i) fine = fine && std::ldexp(x[static_cast<std::size_t>(i)], 53) ==
                                                          std::floor(std::ldexp(x[static_cast<std::size_t>(i)], 53));
        check(fine, "uniform01 on the 2^-53 grid");
    }
    // ── normal01, exponential, cauchy, gamma: KS and moments ───────────────
    {
        std::vector<double> x(N);
        double s = 0.0, s2 = 0.0;
        for (auto& v : x) { v = rnd::normal01(g); s += v; s2 += v * v; }
        check(std::abs(s / N) < 5e-3 && std::abs(s2 / N - 1.0) < 1e-2, "normal01 mean, variance");
        check(ks(x, [](double t) { return 0.5 * std::erfc(-t / std::sqrt(2.0)); }) < 1.95 / std::sqrt(double(N)),
              "normal01 KS");
        for (auto& v : x) v = rnd::exponential(g, 2.0);
        check(ks(x, [](double t) { return 1.0 - std::exp(-2.0 * t); }) < 1.95 / std::sqrt(double(N)),
              "exponential(2) KS");
        for (auto& v : x) v = rnd::cauchy(g, 1.0, 0.5);
        check(ks(x, [](double t) { return 0.5 + std::atan((t - 1.0) / 0.5) / 3.141592653589793; }) <
                  1.95 / std::sqrt(double(N)), "cauchy(1, 0.5) KS");
        s = s2 = 0.0;
        for (auto& v : x) { v = rnd::gamma(g, 2.5, 2.0); s += v; s2 += v * v; }
        const double m = s / N, var = s2 / N - m * m;
        check(std::abs(m - 5.0) < 2e-2 && std::abs(var - 10.0) < 0.15, "gamma(2.5, 2) mean, variance");
        s = 0.0;
        for (int i = 0; i < N; ++i) s += rnd::gamma(g, 0.5, 1.0);
        check(std::abs(s / N - 0.5) < 5e-3, "gamma(0.5, 1) mean");
    }
    // ── below: chi-square on the residues ───────────────────────────────────
    for (std::uint64_t n : {std::uint64_t(7), std::uint64_t(1000003), std::uint64_t(3221225472ull),
                            std::uint64_t(10000000019ull)}) {
        const std::size_t K = 100;                             // buckets of the range
        std::vector<double> obs(K, 0.0), exp(K, 0.0);
        bool inside = true;
        for (int i = 0; i < N; ++i) {
            const std::uint64_t r = rnd::below(g, n);
            inside = inside && r < n;
            const auto bucket = static_cast<std::size_t>(n <= K ? r : (r / ((n + K - 1) / K)));
            obs[std::min(bucket, K - 1)] += 1.0;
        }
        const std::size_t used = static_cast<std::size_t>(n <= K ? n : K);
        for (std::size_t b = 0; b < used; ++b) {
            double width;
            if (n <= K) width = 1.0;
            else {
                const double w = static_cast<double>((n + K - 1) / K);
                width = b + 1 < used ? w : static_cast<double>(n) - w * static_cast<double>(used - 1);
            }
            exp[b] = N * width / static_cast<double>(n);
        }
        obs.resize(used);
        exp.resize(used);
        const double c = chi2(obs, exp), df = static_cast<double>(used - 1);
        check(inside && c < df + 5.0 * std::sqrt(2.0 * df), "below(" + std::to_string(n) + ") chi-square");
    }
    // ── discrete: chi-square on the weights ─────────────────────────────────
    {
        const std::vector<double> w = {0.5, 0.0, 2.0, 1.25, 0.25};
        rnd::discrete_distribution<int> d(w.begin(), w.end());
        std::vector<double> obs(5, 0.0);
        for (int i = 0; i < N; ++i) obs[static_cast<std::size_t>(d(g))] += 1.0;
        check(obs[1] == 0.0, "discrete: a zero weight never");
        const std::vector<double> o = {obs[0], obs[2], obs[3], obs[4]};
        const std::vector<double> e = {N * 0.125, N * 0.5, N * 0.3125, N * 0.0625};
        check(chi2(o, e) < 3.0 + 5.0 * std::sqrt(6.0), "discrete chi-square");
    }
    // ── shuffle: the 24 orders of four ───────────────────────────────────────
    {
        std::vector<double> obs(24, 0.0);
        for (int i = 0; i < N / 4; ++i) {
            std::array<int, 4> v = {0, 1, 2, 3};
            rnd::shuffle(v.begin(), v.end(), g);
            int code = 0;                                       // the order's index (Lehmer code)
            for (int a = 0; a < 4; ++a) {
                int smaller = 0;
                for (int b = a + 1; b < 4; ++b) smaller += v[static_cast<std::size_t>(b)] < v[static_cast<std::size_t>(a)];
                code = code * (4 - a) + smaller;
            }
            obs[static_cast<std::size_t>(code)] += 1.0;
        }
        const std::vector<double> exp(24, (N / 4) / 24.0);
        check(chi2(obs, exp) < 23.0 + 5.0 * std::sqrt(46.0), "shuffle: the 24 orders alike");
    }
    // ── the classes are the functions ───────────────────────────────────────
    {
        std::mt19937 a(9u), b(9u);
        rnd::uniform_int_distribution<int> ui(3, 17);
        rnd::uniform_real_distribution<double> ur(-2.0, 5.0);
        rnd::normal_distribution<double> nd(1.0, 3.0);
        bool same = true;
        for (int i = 0; i < 1000; ++i) {
            same = same && ui(a) == rnd::integer<int>(b, 3, 17);
            same = same && ur(a) == -2.0 + 7.0 * rnd::uniform01(b);
            same = same && nd(a) == 1.0 + 3.0 * rnd::normal01(b);
        }
        check(same, "the distribution classes call the functions");
    }

    std::printf("random: %d/%d checks passed%s\n", checks - failed, checks, failed ? ", FAILED" : "");
    return failed == 0 ? 0 : 1;
}

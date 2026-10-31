// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// Readings of a core's paper changed on 2026-11-01 (the owner's decisions on
// the second verification wave), each with its earlier reading kept where a
// knob keeps it:
//   - lis_lcs: de_base = rand1 and k_share = dp (LIS-1, LIS-2) give the runs
//     of the code before, bit for bit — the fingerprints of
//     tests/compat_dump.cpp that tests/golden_fingerprints.txt held for
//     lis_lcs until then; the defaults (current1, pp) give others; a word
//     the knob does not take is refused;
//   - hcca: setup() evaluates two independent random samples, 2N points
//     (HCCA-13).
// ============================================================================

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "harness.hpp"

namespace mootation::testing {

// tests/compat_dump.cpp's problem with a hole of NaN objectives
struct ZDT1NanSpec {
    static constexpr int NVARS = ZDT1Spec::NVARS;
    static constexpr int NOBJS = 2;
    static void eval(const std::vector<double>& x, std::vector<double>& f)
    {
        ZDT1Spec::eval(x, f);
        if (x[0] >= 0.30 && x[0] < 0.32) f[0] = f[1] = std::numeric_limits<double>::quiet_NaN();
    }
};

// ZDT1, counting its evaluations
inline long long& zdt1_calls() { static long long n = 0; return n; }
struct ZDT1CountSpec {
    static constexpr int NVARS = ZDT1Spec::NVARS;
    static constexpr int NOBJS = 2;
    static void eval(const std::vector<double>& x, std::vector<double>& f)
    {
        ++zdt1_calls();
        ZDT1Spec::eval(x, f);
    }
};

inline std::uint64_t fnv1a(std::uint64_t h, const void* data, std::size_t n)
{
    const unsigned char* p = static_cast<const unsigned char*>(data);
    for (std::size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

// tests/compat_dump.cpp's fingerprint: 30 generations from its seed, with
// the given words set before setup.
template <typename Ind, typename Core>
std::string fingerprint(int pop, const std::vector<std::pair<std::string, std::string>>& words)
{
    Problem<Ind>         prob;
    DataVault<Ind>       vault(pop, prob);
    Optimizer<Ind, Core> opt(std::move(vault), defer_setup);
    auto& alg = opt.get_algorithm();
    alg.set_seed(20260923u);
    for (const auto& w : words) alg.set_word(w.first, w.second);
    opt.setup();
    opt.optimize(30);
    auto& v = opt.get_vault();
    const std::size_t n = std::min<std::size_t>(v.active_n(), static_cast<std::size_t>(v.pop_size()));
    std::uint64_t h = 1469598103934665603ULL;
    for (std::size_t i = 0; i < n; ++i) {
        const auto& x = v.variables_of(i);
        const auto& f = v.objectives_of(i);
        h = fnv1a(h, x.data(), x.size() * sizeof(double));
        h = fnv1a(h, f.data(), f.size() * sizeof(double));
    }
    char buf[64];
    std::snprintf(buf, sizeof buf, "%zu %016llx", n, static_cast<unsigned long long>(h));
    return buf;
}

}   // namespace mootation::testing

MOOTATION_TEST_PROBLEM(LD, LISLCS_Individual, DTLZ2Spec)
MOOTATION_TEST_PROBLEM(LZ, LISLCS_Individual, ZDT1Spec)
MOOTATION_TEST_PROBLEM(LN, LISLCS_Individual, ZDT1NanSpec)
MOOTATION_TEST_PROBLEM(HC, NSGAII_Individual, ZDT1CountSpec)

using namespace mootation;
using namespace mootation::testing;

void test_lis_lcs_earlier_reading()
{
    std::cout << "-- lis_lcs: de_base = rand1, k_share = dp give the code before 2026-11-01 --\n";
    const std::vector<std::pair<std::string, std::string>> before = {{"de_base", "rand1"}, {"k_share", "dp"}};
    // tests/golden_fingerprints.txt for lis_lcs up to 2026-11-01
    check(fingerprint<LD, LISLCSCore<LD>>(91, before) == "91 332ca4b1bab3cc59", "dtlz2_3 as before");
    check(fingerprint<LZ, LISLCSCore<LZ>>(100, before) == "100 fd4af147181c2c4d", "zdt1 as before");
    check(fingerprint<LN, LISLCSCore<LN>>(100, before) == "100 d692df5db454806b", "zdt1_nan as before");
    // the defaults, each knob alone, and the defaults named: other runs or the same
    const std::string now = fingerprint<LZ, LISLCSCore<LZ>>(100, {});
    check(now != "100 fd4af147181c2c4d", "the defaults run otherwise");
    check(fingerprint<LZ, LISLCSCore<LZ>>(100, {{"de_base", "current1"}, {"k_share", "pp"}}) == now,
          "current1 and pp are the defaults");
    check(fingerprint<LZ, LISLCSCore<LZ>>(100, {{"de_base", "rand1"}}) != now, "de_base alone changes the run");
    check(fingerprint<LZ, LISLCSCore<LZ>>(100, {{"k_share", "dp"}}) != now, "k_share alone changes the run");
}

void test_lis_lcs_words_refused()
{
    std::cout << "-- lis_lcs: words its knobs do not take --\n";
    LISLCSCore<LZ> core;
    for (const auto& w : std::vector<std::pair<std::string, std::string>>{
             {"de_base", "current_to_pbest1"}, {"de_base", "best1"}, {"k_share", "both"}}) {
        bool threw = false;
        try { core.set_word(w.first, w.second); } catch (const std::invalid_argument&) { threw = true; }
        check(threw, w.first + " = " + w.second + " refused");
    }
    check(!core.set_word("theta_ties", "slot"), "a knob of another core is not taken");
}

void test_hcca_two_samples()
{
    std::cout << "-- hcca: DP and PP from two independent samples (HCCA-13) --\n";
    Problem<HC>         prob;
    DataVault<HC>       vault(100, prob);
    Optimizer<HC, HCCACore<HC>> opt(std::move(vault), defer_setup);
    opt.get_algorithm().set_seed(7u);
    zdt1_calls() = 0;
    opt.setup();
    check(zdt1_calls() == 200, "setup evaluates 2N = 200 points (" + std::to_string(zdt1_calls()) + ")");
}

int main()
{
    test_lis_lcs_earlier_reading();
    test_lis_lcs_words_refused();
    test_hcca_two_samples();
    return report("test_core_readings");
}

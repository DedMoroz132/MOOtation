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
// And the knobs of 2026-11-02, one reading each with the default unchanged
// (edv pm_literal, naemo flag1/flag2, maoea_iamd noise_frac, irea
// fitness_reading, moead_ds extreme_weights, aprd slot_exclusive): the
// default, named or not, gives the golden fingerprint, the other value
// another run, a value the knob does not take is refused.
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

// The same fingerprint as tests/compat_dump.cpp's, for any core: the
// generation budget told to a core that takes it, as compat_dump does, and
// `set` applied to the core before setup.
template <typename Ind, typename Core, typename Set>
std::string fingerprint_with(int pop, int gens, Set set)
{
    Problem<Ind>         prob;
    DataVault<Ind>       vault(pop, prob);
    Optimizer<Ind, Core> opt(std::move(vault), defer_setup);
    auto& alg = opt.get_algorithm();
    alg.set_seed(20260923u);
    if constexpr (has_set_t_max<Core>::value) alg.set_t_max(gens);
    set(alg);
    opt.setup();
    opt.optimize(gens);
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
MOOTATION_TEST_PROBLEM(ED, EDV_Individual,       DTLZ2Spec)
MOOTATION_TEST_PROBLEM(EZ, EDV_Individual,       ZDT1Spec)
MOOTATION_TEST_PROBLEM(NA, NSGAII_Individual,    DTLZ2Spec)
MOOTATION_TEST_PROBLEM(MI, MaOEAIAMD_Individual, DTLZ2Spec)
MOOTATION_TEST_PROBLEM(IR, IREA_Individual,      DTLZ2Spec)
MOOTATION_TEST_PROBLEM(DS, NSGAII_Individual,    DTLZ2Spec)
MOOTATION_TEST_PROBLEM(AP, NSGAII_Individual,    DTLZ2Spec)

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

// A word knob of 2026-11-02: the default, unnamed and named, gives the
// golden fingerprint (tests/golden_fingerprints.txt, dtlz2_3); the other word
// gives another run over `gens` generations; a word the knob does not take is
// refused, and another core's knob is not taken.
template <typename Ind, typename Core>
void word_knob(const std::string& key, const std::string& golden, const std::string& knob,
               const std::string& dflt, const std::string& other, int gens = 30)
{
    std::cout << "-- " << key << ": " << knob << " (" << dflt << " by default, " << other << ") --\n";
    auto none  = [](Core&) {};
    auto named = [&](Core& c) { c.set_word(knob, dflt); };
    auto alt   = [&](Core& c) { c.set_word(knob, other); };
    check(fingerprint_with<Ind, Core>(91, 30, none) == golden, key + " default as golden");
    check(fingerprint_with<Ind, Core>(91, 30, named) == golden, key + " " + knob + " = " + dflt + " as golden");
    check(fingerprint_with<Ind, Core>(91, gens, alt) != fingerprint_with<Ind, Core>(91, gens, none),
          key + " " + knob + " = " + other + " runs otherwise");
    Core core;
    bool threw = false;
    try { core.set_word(knob, "maybe"); } catch (const std::invalid_argument&) { threw = true; }
    check(threw, key + " " + knob + " = maybe refused");
    check(!core.set_word("k_share", "pp"), key + " does not take another core's knob");
}

void test_knobs_of_2026_11_02()
{
    word_knob<ED, EDVCore<ED>>("edv", "91 535f03f7cd2cb7ec", "pm_literal", "on", "off");
    word_knob<NA, NAEMOCore<NA>>("naemo", "91 6bbfd8997ef61273", "flag1", "off", "on");
    word_knob<NA, NAEMOCore<NA>>("naemo", "91 6bbfd8997ef61273", "flag2", "off", "on");
    word_knob<IR, IREACore<IR>>("irea", "91 fa799b2a4188d24b", "fitness_reading", "swap", "ibea");
    word_knob<DS, MOEADDSCore<DS>>("moead_ds", "91 8fd287030360a665", "extreme_weights", "nsga3", "literal");
    word_knob<AP, APRDCore<AP>>("aprd", "91 dee9a93dd30901ff", "slot_exclusive", "off", "on", 300);
    // 2026-11-05: maoea_iamd's kpcm_pm (IAMD-1)
    word_knob<MI, MaOEAIAMDCore<MI>>("maoea_iamd", "91 1e27b57e7dab0642", "kpcm_pm", "off", "on");
    // 2026-11-05: edv's edv_tournament (EDV-5) and normalize (EDV-6)
    word_knob<ED, EDVCore<ED>>("edv", "91 535f03f7cd2cb7ec", "edv_tournament", "fitness", "random");
    // 2026-11-06: maoea_iamd's cv_rule (IAMD-2) and cv_norm (IAMD-3). On DTLZ2
    // every rule agrees: a distance variable moves the point along a ray (a
    // line, small angle), a position variable along an arc (large angle), so
    // the runs are the default's and CV is the distance variables 2..11.
    {
        using MC = MaOEAIAMDCore<MI>;
        const std::vector<int> distance{2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
        auto cv_of = [](const std::string& knob, const std::string& word) {
            Problem<MI> prob;
            DataVault<MI> vault(91, prob);
            Optimizer<MI, MC> opt(std::move(vault), defer_setup);
            opt.get_algorithm().set_seed(20260923u);
            if (!knob.empty()) opt.get_algorithm().set_word(knob, word);
            opt.setup();
            return opt.get_algorithm().convergence_variables();
        };
        for (const auto& kw : std::vector<std::pair<std::string, std::string>>{
                 {"", ""}, {"cv_rule", "lmea"}, {"cv_rule", "platemo"}, {"cv_norm", "nondominated"}}) {
            std::cout << "-- maoea_iamd: " << (kw.first.empty() ? "default" : kw.first + " = " + kw.second)
                      << " on DTLZ2 --\n";
            check(cv_of(kw.first, kw.second) == distance, "CV = the distance variables");
            if (!kw.first.empty())
                check(fingerprint_with<MI, MC>(91, 30, [&](MC& c) { c.set_word(kw.first, kw.second); })
                          == "91 1e27b57e7dab0642", "the default's run");
        }
        MC core;
        for (const auto& k : {"cv_rule", "cv_norm"}) {
            bool threw = false;
            try { core.set_word(k, "maybe"); } catch (const std::invalid_argument&) { threw = true; }
            check(threw, std::string(k) + " = maybe refused");
        }
    }
    // 2026-11-06: moead_ds's nadir_source (DDS-9)
    word_knob<DS, MOEADDSCore<DS>>("moead_ds", "91 8fd287030360a665", "nadir_source", "intercepts", "archive");
    // 2026-11-06: moead_ds's drs_guard (DDS-10)
    word_knob<DS, MOEADDSCore<DS>>("moead_ds", "91 8fd287030360a665", "drs_guard", "off", "normalize");
    word_knob<DS, MOEADDSCore<DS>>("moead_ds", "91 8fd287030360a665", "drs_guard", "off", "extreme");
    std::cout << "-- edv: normalize (false by default, true) --\n";
    using EC = EDVCore<ED>;
    check(fingerprint_with<ED, EC>(91, 30, [](EC& c) { c.set_normalize(false); }) == "91 535f03f7cd2cb7ec",
          "edv normalize = false as golden");
    check(fingerprint_with<ED, EC>(91, 30, [](EC& c) { c.set_normalize(true); }) != "91 535f03f7cd2cb7ec",
          "edv normalize = true runs otherwise");

    std::cout << "-- maoea_iamd: noise_frac (0.01 by default, 0.316 the letter) --\n";
    using MC = MaOEAIAMDCore<MI>;
    auto none = [](MC&) {};
    check(fingerprint_with<MI, MC>(91, 30, none) == "91 1e27b57e7dab0642", "maoea_iamd default as golden");
    check(fingerprint_with<MI, MC>(91, 30, [](MC& c) { c.set_param("noise_frac", 0.01); })
              == "91 1e27b57e7dab0642", "noise_frac = 0.01 as golden");
    check(fingerprint_with<MI, MC>(91, 30, [](MC& c) { c.set_param("noise_frac", 0.316); })
              != "91 1e27b57e7dab0642", "noise_frac = 0.316 runs otherwise");
    MC core;
    bool threw = false;
    try { core.set_param("noise_frac", -1.0); } catch (const std::invalid_argument&) { threw = true; }
    check(threw, "noise_frac = -1 refused");
    check(!core.set_param("n_offspring", 10.0), "maoea_iamd does not take another core's param");
}

int main()
{
    test_lis_lcs_earlier_reading();
    test_lis_lcs_words_refused();
    test_hcca_two_samples();
    test_knobs_of_2026_11_02();
    return report("test_core_readings");
}

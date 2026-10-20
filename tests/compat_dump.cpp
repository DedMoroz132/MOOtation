// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// Default behaviour, bit for bit: a fingerprint of every algorithm's final
// population, to compare one tree against another.
//
// The library adds options — bound repair, operators, diagnostics — under
// the rule that every algorithm at its defaults keeps producing exactly the
// population it produced before. Since task 14 of 2026-10-20 a run is the
// same on every supported platform (random.hpp, detmath.hpp, no fused
// multiply-adds, sorts with one result), so the fingerprints are stored:
// tests/golden_fingerprints.txt, and `compat_dump --check <file>` (the ctest
// golden_fingerprints, on every CI platform) compares this build with it.
// tools/compat_check.py writes the list (--write-golden) and still compares
// two trees on one machine (--baseline).
//
// Output: one line per algorithm and problem,
//   <key> <problem> <n> <fnv1a-64 of the returned rows' variables and objectives>
// or "<key> <problem> refused: <reason>" for a documented precondition.
//
// The probes: DTLZ2 at M = 3 and ZDT1, 30 generations, a fixed seed. Both
// make DE mutants and every other unbounded operator leave the box, which is
// where the repair options live. A third, zdt1_nan: ZDT1 whose objectives are
// NaN in a band of x_1 — failed evaluations reach every core's comparisons and
// sorts, which must give one result everywhere (task 14, §4).
// ============================================================================

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <string>
#include <vector>

#include "harness.hpp"

namespace mootation::testing {

inline std::uint64_t fnv1a(std::uint64_t h, const void* data, std::size_t n)
{
    const unsigned char* p = static_cast<const unsigned char*>(data);
    for (std::size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

// ZDT1 with NaN objectives where 0.30 <= x_1 < 0.32: failed evaluations.
struct ZDT1NanSpec {
    static constexpr int NVARS = ZDT1Spec::NVARS;
    static constexpr int NOBJS = 2;
    static void eval(const std::vector<double>& x, std::vector<double>& f)
    {
        ZDT1Spec::eval(x, f);
        if (x[0] >= 0.30 && x[0] < 0.32) f[0] = f[1] = std::numeric_limits<double>::quiet_NaN();
    }
};

// The lines of this run, by "<key> <problem>" (main collects them).
inline std::map<std::string, std::string>& lines()
{
    static std::map<std::string, std::string> m;
    return m;
}

inline int pop_for(const char* name, int m)
{
    const std::string n(name);
    if (n == "moead_m2m" || n == "sms_m2m" || n == "moead_am2m") return 100;
    return m == 3 ? 91 : 100;
}

template <typename Ind, typename Core, typename Spec>
void dump(const char* key, const char* problem, int m)
{
    const int pop = pop_for(key, m);
    const int gens = 30;
    const unsigned seed = 20260923u;
    try {
        Problem<Ind>         prob;
        DataVault<Ind>       vault(pop, prob);
        Optimizer<Ind, Core> opt(std::move(vault), defer_setup);
        auto& alg = opt.get_algorithm();
        alg.set_seed(seed);
        if constexpr (has_set_t_max<Core>::value) alg.set_t_max(gens);
        opt.setup();
        opt.optimize(gens);
        auto& v = opt.get_vault();
        const std::size_t n = std::min<std::size_t>(v.active_n(),
                                                    static_cast<std::size_t>(v.pop_size()));
        std::uint64_t h = 1469598103934665603ULL;
        for (std::size_t i = 0; i < n; ++i) {
            const auto& x = v.variables_of(i);
            const auto& f = v.objectives_of(i);
            h = fnv1a(h, x.data(), x.size() * sizeof(double));
            h = fnv1a(h, f.data(), f.size() * sizeof(double));
        }
        char buf[256];
        std::snprintf(buf, sizeof buf, "%s %s %zu %016llx", key, problem, n,
                      static_cast<unsigned long long>(h));
        lines()[std::string(key) + " " + problem] = buf;
    } catch (const std::exception& e) {
        lines()[std::string(key) + " " + problem] =
            std::string(key) + " " + problem + " refused: " + e.what();
    }
}

}   // namespace mootation::testing

#define MOOTATION_ALG(key, IND, CORE)                                        \
    MOOTATION_TEST_PROBLEM(CD_##key, IND, DTLZ2Spec)                         \
    MOOTATION_TEST_PROBLEM(CZ_##key, IND, ZDT1Spec)                          \
    MOOTATION_TEST_PROBLEM(CN_##key, IND, ZDT1NanSpec)
#include "mootation/algorithms.def"
#undef MOOTATION_ALG

// compat_dump                 the fingerprints, one line a run
// compat_dump --check <file>  compared with a stored list (golden_fingerprints.txt):
//                             every line equal, or the differences and exit 1
int main(int argc, char** argv)
{
    using namespace mootation;
    using namespace mootation::testing;
#define MOOTATION_ALG(key, IND, CORE)                                        \
    dump<testing::CD_##key, CORE<testing::CD_##key>, DTLZ2Spec>(#key, "dtlz2_3", 3); \
    dump<testing::CZ_##key, CORE<testing::CZ_##key>, ZDT1Spec>(#key, "zdt1", 2); \
    dump<testing::CN_##key, CORE<testing::CN_##key>, ZDT1NanSpec>(#key, "zdt1_nan", 2);
#include "mootation/algorithms.def"
#undef MOOTATION_ALG
    if (argc == 3 && std::strcmp(argv[1], "--check") == 0) {
        std::ifstream f(argv[2], std::ios::binary);
        std::map<std::string, std::string> want;
        std::string l;
        while (std::getline(f, l)) {
            if (!l.empty() && l.back() == '\r') l.pop_back();
            if (l.empty() || l[0] == '#') continue;
            const auto a = l.find(' '), b = l.find(' ', a + 1);
            want[l.substr(0, b)] = l;
        }
        int bad = 0;
        for (const auto& [k, v] : lines()) {
            const auto it = want.find(k);
            if (it == want.end()) { std::printf("new      %s\n", v.c_str()); ++bad; }
            else if (it->second != v) {
                std::printf("CHANGED  %s  (golden: %s)\n", v.c_str(), it->second.c_str());
                ++bad;
            }
        }
        for (const auto& [k, v] : want)
            if (!lines().count(k)) { std::printf("missing  %s\n", v.c_str()); ++bad; }
        std::printf("%zu fingerprints against %s: %d differ\n", lines().size(), argv[2], bad);
        return bad == 0 ? 0 : 1;
    }
    for (const auto& [k, v] : lines()) std::printf("%s\n", v.c_str());
    return 0;
}

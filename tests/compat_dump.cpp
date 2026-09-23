// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// Default behaviour, bit for bit: a fingerprint of every algorithm's final
// population, to compare one tree against another.
//
// The library adds options — bound repair, operators, diagnostics — under
// the rule that every algorithm at its defaults keeps producing exactly the
// population it produced before. No tolerance can check that across
// platforms: the standard library's distributions and the compiler's
// floating-point code differ between MSVC, libstdc++ and libc++, so a stored
// "golden" population holds on the machine that wrote it only. This driver
// is therefore compiled TWICE on the same machine, against the headers of a
// baseline commit and against the headers under test, and the two outputs
// must be identical (tools/compat_check.py builds both and compares them;
// the "default behaviour" CI job runs it).
//
// Output: one line per algorithm and problem,
//   <key> <problem> <n> <fnv1a-64 of the returned rows' variables and objectives>
// or "<key> <problem> refused: <reason>" for a documented precondition.
//
// The probes: DTLZ2 at M = 3 and ZDT1, 30 generations, a fixed seed. Both
// make DE mutants and every other unbounded operator leave the box, which is
// where the repair options live.
// ============================================================================

#include <cstdint>
#include <cstdio>
#include <cstring>
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
        std::printf("%s %s %zu %016llx\n", key, problem, n,
                    static_cast<unsigned long long>(h));
    } catch (const std::exception& e) {
        std::printf("%s %s refused: %s\n", key, problem, e.what());
    }
}

}   // namespace mootation::testing

#define MOOTATION_ALG(key, IND, CORE)                                        \
    MOOTATION_TEST_PROBLEM(CD_##key, IND, DTLZ2Spec)                         \
    MOOTATION_TEST_PROBLEM(CZ_##key, IND, ZDT1Spec)
#include "mootation/algorithms.def"
#undef MOOTATION_ALG

int main()
{
    using namespace mootation;
    using namespace mootation::testing;
#define MOOTATION_ALG(key, IND, CORE)                                        \
    dump<testing::CD_##key, CORE<testing::CD_##key>, DTLZ2Spec>(#key, "dtlz2_3", 3); \
    dump<testing::CZ_##key, CORE<testing::CZ_##key>, ZDT1Spec>(#key, "zdt1", 2);
#include "mootation/algorithms.def"
#undef MOOTATION_ALG
    return 0;
}

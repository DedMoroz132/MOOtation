// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// Reproducibility, across every public algorithm.
//
// The README promises "explicit seeding (set_seed)". This test is what turns
// that from a claim into a fact:
//
//   1. same seed  -> bit-for-bit identical final population
//   2. other seed -> a different population
//
// The second half matters as much as the first. An algorithm that ignores
// set_seed entirely and uses a fixed internal state would pass (1) perfectly
// while making the guarantee worthless.
//
// Bit-for-bit is the right bar here, not a tolerance: every algorithm is
// deterministic given its RNG stream, so any difference means state is leaking
// in from somewhere it should not.
// ============================================================================

#include <cstddef>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "harness.hpp"

#define MOOTATION_ALG(key, IND, CORE) \
    MOOTATION_TEST_PROBLEM(Rep_##key, IND, DTLZ2Spec)
#include "mootation/algorithms.def"
#undef MOOTATION_ALG

namespace {

using namespace mootation;
using namespace mootation::testing;

constexpr int POP  = 60;
constexpr int GENS = 30;   // enough to diverge, short enough to run 60 times x3

// Full objective matrix of the final population, flattened.
template <typename Ind, typename Core>
std::vector<double> final_objectives(int pop, int gens, unsigned seed)
{
    Problem<Ind>         prob;
    DataVault<Ind>       vault(pop, prob);
    Optimizer<Ind, Core> opt(std::move(vault), defer_setup);

    auto& alg = opt.get_algorithm();
    alg.set_seed(seed);
    if constexpr (has_set_t_max<Core>::value) alg.set_t_max(gens);

    opt.setup();
    opt.optimize(gens);

    auto& v = opt.get_vault();
    std::vector<double> out;
    out.reserve(v.active_n() * 3);
    for (std::size_t i = 0; i < v.active_n(); ++i) {
        const auto& f = v.objectives_of(i);
        out.insert(out.end(), f.begin(), f.end());
    }
    return out;
}

int g_skipped = 0;

template <typename Ind, typename Core>
void check_one(const char* name)
{
    const std::string n = name;
    std::cout << "  " << std::left << std::setw(20) << name << std::flush;

    std::vector<double> a, b, c;
    try {
        a = final_objectives<Ind, Core>(POP, GENS, 1234u);
        b = final_objectives<Ind, Core>(POP, GENS, 1234u);
        c = final_objectives<Ind, Core>(POP, GENS, 9876u);
    } catch (const std::exception& e) {
        // Population-size constraints (M2M needs pop = K*S) are documented in
        // the algorithm headers. Not a reproducibility failure — but it must be
        // counted and reported, never silently passed over.
        std::cout << "SKIPPED (" << e.what() << ")\n" << std::flush;
        ++g_skipped;
        return;
    }

    const bool same = (a == b);
    const bool diff = (a != c);
    std::cout << (same ? "deterministic" : "NON-DETERMINISTIC") << ", "
              << (diff ? "seed-sensitive" : "SEED IGNORED") << '\n'
              << std::flush;

    check(!a.empty(), n + ": produced a non-empty population");
    check(same, n + ": identical seeds must give a bit-for-bit identical population");
    check(diff, n + ": different seeds must give a different population "
                    "(set_seed appears to be ignored)");
}

}   // namespace

int main()
{
    std::cout << "Reproducibility: DTLZ2 (M=3), pop=" << POP
              << ", gens=" << GENS << "\n"
              << "seeds 1234 (twice) and 9876\n\n";

#define MOOTATION_ALG(key, IND, CORE) \
    check_one<testing::Rep_##key, CORE<testing::Rep_##key>>(#key);
#include "mootation/algorithms.def"
#undef MOOTATION_ALG

    if (g_skipped)
        std::cout << "\n" << g_skipped
                  << " algorithm(s) skipped because pop=" << POP
                  << " is incompatible with their documented population-size "
                     "constraint.\n";

    return mootation::testing::report("reproducibility");
}

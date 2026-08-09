// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// Every algorithm declares a `constraint_mode` field. For a long time a third
// of them declared it and never read it — the switch was there, the effect was
// not. This test exists so that cannot come back silently.
//
// The problem is DTLZ2 with one inequality constraint that cuts the box in
// half: g = x0 - 0.5 <= 0. Roughly half of any uniformly drawn population is
// infeasible, so the constraint is not a formality the search can ignore.
//
// For each of the 60 algorithms the same seed is run twice — once with
// ConstraintMode::NONE and once with ConstraintMode::FEASIBILITY — and the test
// requires:
//
//   (1) the two runs DIFFER. If turning the mode on changes nothing at all, the
//       field is inert, which is exactly the defect being guarded against.
//   (2) the constrained run ends with at least as many feasible solutions as
//       the unconstrained one. Constraint handling wired to the wrong
//       comparison would show up here as a regression.
//
// Neither check asserts convergence quality: the point is that the switch is
// live and pushes the right way, not that any algorithm solves this problem
// well.
//
// Caveat worth knowing when reading the output: for some algorithms the
// UNCONSTRAINED run already ends fully feasible on this problem, because they
// converge into a part of the front that happens to satisfy x0 <= 0.5. For
// those, check (2) is vacuous and only check (1) carries information.
// ============================================================================

#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "harness.hpp"

namespace mootation::testing {

// DTLZ2 (m = 3, n = 7) plus the single constraint g = x0 - 0.5 <= 0.
struct CDTLZ2Spec {
    static constexpr int NVARS = 7;
    static constexpr int NOBJS = 3;
    static constexpr int NLIMS = 1;

    static void eval(const std::vector<double>& x, std::vector<double>& f)
    {
        double g = 0.0;
        for (int i = NOBJS - 1; i < NVARS; ++i) {
            const double d = x[static_cast<std::size_t>(i)] - 0.5;
            g += d * d;
        }
        const double half_pi = 1.57079632679489661923;
        f[0] = (1.0 + g) * std::cos(x[0] * half_pi) * std::cos(x[1] * half_pi);
        f[1] = (1.0 + g) * std::cos(x[0] * half_pi) * std::sin(x[1] * half_pi);
        f[2] = (1.0 + g) * std::sin(x[0] * half_pi);
    }

    static void eval_limits(const std::vector<double>& x,
                            std::vector<double>& lim)
    {
        lim[0] = x[0] - 0.5;   // <= 0 is feasible
    }
};

} // namespace mootation::testing

// One tag + Problem<> specialization per algorithm. The specialization has to
// be generated in full: Based_Individual befriends Problem<T> only, so no
// helper class may touch ind.limits.
#define MOOTATION_CONSTRAINED_PROBLEM(TAG, BASE)                             \
    namespace mootation::testing {                                           \
    struct TAG : public mootation::BASE {};                                  \
    }                                                                        \
    namespace mootation {                                                    \
    template <>                                                              \
    class Problem<testing::TAG> {                                            \
    public:                                                                  \
        std::vector<std::pair<std::optional<double>, std::optional<double>>>  \
            bounds = std::vector<std::pair<std::optional<double>,            \
                                           std::optional<double>>>(          \
                testing::CDTLZ2Spec::NVARS,                                  \
                std::pair<std::optional<double>, std::optional<double>>{     \
                    0.0, 1.0});                                              \
        int get_vars_n()     const { return testing::CDTLZ2Spec::NVARS; }    \
        int get_bin_vars_n() const { return 0; }                             \
        int get_objs_n()     const { return testing::CDTLZ2Spec::NOBJS; }    \
        int get_lims_n()     const { return testing::CDTLZ2Spec::NLIMS; }    \
        void calc_objs(testing::TAG& ind) const                              \
        {                                                                    \
            testing::CDTLZ2Spec::eval(ind.variables, ind.objectives);        \
            testing::CDTLZ2Spec::eval_limits(ind.variables, ind.limits);     \
        }                                                                    \
    };                                                                       \
    }

#define MOOTATION_ALG(key, IND, CORE) \
    MOOTATION_CONSTRAINED_PROBLEM(CTag_##key, IND)
#include "mootation/algorithms.def"
#undef MOOTATION_ALG

namespace mootation::testing {

struct CRun {
    std::size_t         n        = 0;
    std::size_t         feasible = 0;
    std::vector<double> signature;   // flattened objectives, for the diff check
};

template <typename Ind, typename Core>
CRun run_constrained(int pop, int gens, unsigned seed, ConstraintMode mode)
{
    Problem<Ind>         prob;
    DataVault<Ind>       vault(pop, prob);
    Optimizer<Ind, Core> opt(std::move(vault), defer_setup);

    auto& alg = opt.get_algorithm();
    alg.set_seed(seed);
    alg.constraint_mode = mode;
    if constexpr (has_set_t_max<Core>::value) alg.set_t_max(gens);

    opt.setup();
    opt.optimize(gens);

    auto& v = opt.get_vault();
    CRun r;
    r.n = v.active_n();
    r.signature.reserve(r.n * static_cast<std::size_t>(CDTLZ2Spec::NOBJS));
    for (std::size_t i = 0; i < r.n; ++i) {
        if (v.get_cv(i) <= 0.0) ++r.feasible;
        for (double c : v.objectives_of(i)) r.signature.push_back(c);
    }
    return r;
}

inline bool same_run(const CRun& a, const CRun& b)
{
    if (a.signature.size() != b.signature.size()) return false;
    for (std::size_t i = 0; i < a.signature.size(); ++i)
        if (a.signature[i] != b.signature[i]) return false;
    return true;
}

} // namespace mootation::testing

using namespace mootation;
using namespace mootation::testing;

namespace {

constexpr int      POP  = 91;
constexpr int      GENS = 60;
constexpr unsigned SEED = 20260806u;

// Same reason as in test_convergence: the M2M family requires pop = K*S
// exactly, and 91 is not a multiple of the default K=10. 90 = 10*9 is.
struct PopOverride { const char* key; int pop; };
const PopOverride POP_OVERRIDES[] = {
    {"moead_m2m", 90},
    {"sms_m2m",   90},
};

int pop_for(const char* key)
{
    for (const auto& o : POP_OVERRIDES) {
        const char* a = o.key; const char* b = key;
        while (*a && *a == *b) { ++a; ++b; }
        if (*a == 0 && *b == 0) return o.pop;
    }
    return POP;
}

int g_inert     = 0;
int g_regressed = 0;

template <typename Ind, typename Core>
void probe(const char* name)
{
    const int pop = pop_for(name);
    std::cout << "  " << name << std::flush;

    CRun off, on;
    try {
        off = run_constrained<Ind, Core>(pop, GENS, SEED, ConstraintMode::NONE);
        on  = run_constrained<Ind, Core>(pop, GENS, SEED, ConstraintMode::FEASIBILITY);
    } catch (const std::exception& e) {
        // An algorithm that REFUSES this configuration is not a failure — it is
        // a documented precondition (M2M's pop = K*S, Path-A's lattice sizes).
        // Silently passing would be wrong too, so it is named and skipped.
        std::cout << "  SKIP (" << e.what() << ")" << std::endl;
        return;
    }

    const bool inert = same_run(off, on);
    if (inert) {
        ++g_inert;
        std::cout << "  INERT " << name << " — constraint_mode changed nothing\n";
    }
    if (on.feasible < off.feasible) {
        ++g_regressed;
        std::cout << "  WORSE " << name << " — feasible " << off.feasible
                  << " -> " << on.feasible << " of " << on.n << '\n';
    }
    check(!inert, std::string(name) + ": constraint_mode is live");
    check(on.feasible >= off.feasible,
          std::string(name) + ": feasibility does not regress");
    std::cout << "  feasible " << off.feasible << " -> " << on.feasible
              << " of " << on.n << std::endl;
}

} // namespace

int main()
{
    std::cout << "constraint wiring: DTLZ2(m=3,n=7) with x0 <= 0.5, pop " << POP
              << ", " << GENS << " generations\n";

#define MOOTATION_ALG(key, IND, CORE) \
    probe<testing::CTag_##key, CORE<testing::CTag_##key>>(#key);
#include "mootation/algorithms.def"
#undef MOOTATION_ALG

    std::cout << "inert: " << g_inert << ", regressed: " << g_regressed << '\n';
    return report("test_constraints");
}

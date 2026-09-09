// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// Scale invariance: multiplying every objective by a constant must not change
// what an algorithm decides.
//
// Nothing in a multi-objective method should depend on the UNITS of the
// objectives. Pareto dominance does not, the reference-point and clustering
// constructions rescale themselves, and a user who reports mass in grams
// rather than kilograms is entitled to the same run. An algorithm that fails
// this is silently a different algorithm on their problem.
//
// The probe is DTLZ2 (M = 3) against the same problem with every objective
// multiplied by 2^10. The factor is a power of two, so the scaling is EXACT in
// IEEE-754: an algorithm whose decisions are scale-equivariant takes
// bit-identical decisions on both, and dividing the scaled objectives back by
// the factor has to reproduce the plain error bit for bit. The test therefore
// compares for exact equality, not within a tolerance — a tolerance would have
// to be invented, and every algorithm that passes does so exactly.
//
// A pass is not a proof of independence: it says that on THIS problem at THIS
// factor no absolute constant was crossed. A failure is a proof of dependence.
//
// Found by this test on 2026-09-09, all now fixed or declared:
//   two_arch2  I_eps+ on raw objectives with kappa = 0.05: exp(-20*I)
//              underflowed to zero for every pair and the convergence archive
//              stopped ordering at all (IGD 0.0667 -> 0.2712 at 2^10, 0.5130
//              at 2^20). The paper is silent on scaling and the IBEA fitness
//              it reuses does scale; the scaled reading is now the default.
//   ar_moea    R' was initialised from the unit-simplex reference set while it
//              is consumed in translated objective coordinates. Fixed.
// ============================================================================

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "harness.hpp"

namespace mootation::testing {

// The factor. A power of two so that f * KS is exact.
constexpr double KS = 1024.0;

// DTLZ2 with every objective multiplied by KS. front_error divides it back
// out, so the two specs' errors are directly comparable.
struct DTLZ2ScaledSpec {
    static constexpr int M     = DTLZ2Spec::M;
    static constexpr int K     = DTLZ2Spec::K;
    static constexpr int NVARS = DTLZ2Spec::NVARS;
    static constexpr int NOBJS = DTLZ2Spec::NOBJS;

    static void eval(const std::vector<double>& x, std::vector<double>& f)
    {
        DTLZ2Spec::eval(x, f);
        for (double& v : f) v *= KS;
    }

    static double front_error(const std::vector<double>& f)
    {
        std::vector<double> g(f.size());
        for (std::size_t i = 0; i < f.size(); ++i) g[i] = f[i] / KS;
        return DTLZ2Spec::front_error(g);
    }
};

// Algorithms whose published definition contains an ABSOLUTE constant in
// objective space. Their scale dependence is the paper's, not the port's, so
// changing it would make the implementation unfaithful. Each entry names the
// constant and the measured effect; an exemption without both is how a suite
// quietly stops testing anything.
struct ScaleExempt { const char* name; const char* why; };

const ScaleExempt SCALE_EXEMPT[] = {
    {"adaw",
     "footnote 2 fixes z* = best - 1e-4, an absolute offset in objective "
     "space (measured: 10.3 % of IGD at 2^10, the same at 2^20 and 2^30)"},
    {"moead_awa",
     "Step 1.2/3.4 fixes z* = min f - 1e-7 and Eq.6 divides by (f - z*) with "
     "an epsilon (measured: 2.1 % at 2^10, 5.4 % at 2^30)"},
    {"mombi2",
     "Alg. line 9 compares |z^max - z^min| against eps = 1e-3 from Sec 5.1, "
     "an absolute threshold on a raw range (measured: 2.3 % at 2^10)"},
    {"r2ibea",
     "Eq.4 is written on raw objectives and the fitness is exp(-I_R2/0.005); "
     "set_normalize(true) removes the dependence but costs a factor of 35 on "
     "ZDT1 at the native scale, so the letter is the default and the "
     "underflow is reported through set_warn_handler instead"},
};

const char* scale_exempt_reason(const char* key)
{
    for (const auto& e : SCALE_EXEMPT) {
        const char* a = e.name; const char* b = key;
        while (*a && *a == *b) { ++a; ++b; }
        if (*a == 0 && *b == 0) return e.why;
    }
    return nullptr;
}

int g_exempt = 0;

// M2M-family cores need pop = K * S; 91 is not divisible by their default
// K = 10, so they run at 100. Every other algorithm runs at a lattice size.
int pop_for(const char* name)
{
    const std::string n(name);
    if (n == "moead_m2m" || n == "sms_m2m" || n == "moead_am2m") return 100;
    return 91;
}

template <typename IndA, typename CoreA, typename IndB, typename CoreB>
void probe(const char* name)
{
    const int pop  = pop_for(name);
    const int gens = 25;                       // enough for a decision to differ
    const unsigned seed = 20260909u;

    RunResult plain, scaled;
    try {
        plain  = run_algorithm<IndA, CoreA, DTLZ2Spec>(pop, gens, seed);
        scaled = run_algorithm<IndB, CoreB, DTLZ2ScaledSpec>(pop, gens, seed);
    } catch (const std::exception& e) {
        // A refused configuration is a documented precondition, not a failure.
        std::cout << "  SKIP " << name << " (" << e.what() << ")\n";
        return;
    }

    const bool same = (plain.mean_error == scaled.mean_error) &&
                      (plain.best_error == scaled.best_error) &&
                      (plain.n == scaled.n);
    const char* why = scale_exempt_reason(name);

    if (!same && why) {
        ++g_exempt;
        std::printf("  EXEMPT %-12s %.6f vs %.6f — %s\n",
                    name, plain.mean_error, scaled.mean_error, why);
        return;
    }
    if (!same && !why) {
        std::printf("  %-12s plain %.9f  scaled %.9f  (n %zu vs %zu)\n",
                    name, plain.mean_error, scaled.mean_error, plain.n, scaled.n);
    }
    check(same, std::string(name) + ": objectives x 1024 changed the result");
}

}   // namespace mootation::testing

// One tag per algorithm per variant.
#define MOOTATION_ALG(key, IND, CORE)                                        \
    MOOTATION_TEST_PROBLEM(SP_##key, IND, DTLZ2Spec)                         \
    MOOTATION_TEST_PROBLEM(SS_##key, IND, DTLZ2ScaledSpec)
#include "mootation/algorithms.def"
#undef MOOTATION_ALG

int main()
{
    using namespace mootation;
    using namespace mootation::testing;

    std::cout << "scale invariance: DTLZ2 (M=3) against the same problem with "
                 "every objective x 1024\n";

#define MOOTATION_ALG(key, IND, CORE)                                        \
    probe<testing::SP_##key, CORE<testing::SP_##key>,                        \
          testing::SS_##key, CORE<testing::SS_##key>>(#key);
#include "mootation/algorithms.def"
#undef MOOTATION_ALG

    if (g_exempt)
        std::cout << '\n' << g_exempt
                  << " algorithm(s) exempt: the absolute constant is the "
                     "paper's own, see the list in this file\n";
    return report("scale invariance");
}

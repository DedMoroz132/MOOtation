// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// Convergence smoke test over every public algorithm.
//
// What this actually proves: each algorithm runs to completion, produces a
// finite, correctly sized population, and moves that population measurably
// toward the analytic Pareto front of DTLZ2 (the unit sphere, so the distance
// to the front is exact and needs no reference set).
//
// What it does NOT prove: that an implementation matches its paper. Fidelity
// is established by review against the primary source; this suite only catches
// the class of regression where an algorithm stops working at all.
//
// The threshold is deliberately loose. Algorithms here differ by design in how
// many function evaluations one generation costs — the steady-state MOEA/D
// variants (DRA, NIMMO) advance far less per generation than a generational
// (mu+lambda) scheme. Comparing them against each other at equal generations
// would be meaningless; the bar is only "clearly converging", not "converging
// as fast as the fastest".
// ============================================================================

#include <cstddef>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "harness.hpp"

// ── 1. Declare a tag + DTLZ2 problem specialization per algorithm ───────────
#define MOOTATION_ALG(key, IND, CORE) \
    MOOTATION_TEST_PROBLEM(Tag_##key, IND, DTLZ2Spec)
#include "mootation/algorithms.def"
#undef MOOTATION_ALG

namespace {

// DTLZ2 with M = 3. 91 = C(12+2, 2) is an exact Das-Dennis lattice count for
// H = 12, so algorithms that require the population to equal a lattice size
// are satisfied without a special case.
constexpr int POP   = 91;
constexpr int GENS  = 200;
constexpr unsigned SEED = 20260804u;

// A run must at least reach this mean distance to the front. Calibrated from
// the observed spread across all algorithms, then given room so that ordinary
// stochastic variation does not turn the suite red. 53 of the 60 algorithms listed at calibration time (2026-08) land
// below 0.015 here, so this is a very loose floor, not a performance bar.
constexpr double MEAN_ERROR_LIMIT = 0.35;

// The single best solution should be much closer than the population mean —
// if it is not, the algorithm is not finding the front at all.
constexpr double BEST_ERROR_LIMIT = 0.10;

// ── Per-algorithm exceptions ────────────────────────────────────────────────
// Every entry needs a stated reason. An exception without a justification is
// how a suite quietly stops testing anything.
//
//   pop  > 0        run this algorithm at a different population size
//   gens > 0        run it for a different number of generations — the
//                   paper's own stopping criterion, when the suite's default
//                   is far outside the regime the algorithm was designed for
//   mean / best     relaxed thresholds
//   known_issue     report the numbers, do not fail the suite
struct Override {
    const char* name;
    int         pop;          // 0 = use the default
    int         gens;         // 0 = use the default
    double      mean;         // < 0 = use the default
    double      best;         // < 0 = use the default
    bool        known_issue;
    const char* why;
};

const Override OVERRIDES[] = {
    // M2M decomposition needs pop = K * S exactly (see M2M-8 in the header).
    // 91 is not a multiple of the default K=10; 90 = 10 * 9 is. This is a
    // documented structural constraint, not a defect, so the suite adapts
    // instead of reporting a failure.
    {"sms_m2m",   90, 0, -1.0, -1.0, false,
     "M2M requires pop = K*S; 90 = 10*9"},

    // pop=90 satisfies the K*S constraint. At the suite's default 200
    // generations this port lands at mean 1.02 / best 0.010 — which for a
    // year read as a defect. The second primary-source pass (2026-09) showed
    // it is a budget mismatch: the paper stops after 3000 generations
    // (liu2014 §III-A(4)) and the annealed operator is calibrated to that.
    // At 3000 the same port reaches mean 0.116 / median 0.001 with no
    // relaxation of the thresholds. The mechanism (a boundary-stuck minority
    // carrying the mean) and the experiments that excluded the alternatives
    // are written up as M2M-D in the file header.
    {"moead_m2m", 90, 3000, -1.0, -1.0, false,
     "M2M requires pop=K*S (90=10*9); the paper's 3000 generations"},

    // Steady-state schemes: one generation advances only a fraction of the
    // population, so at equal generation counts they have spent far fewer
    // function evaluations than a generational (mu+lambda) algorithm. The
    // honest comparison is at equal FE. Thresholds are relaxed rather than
    // removed, so a real regression still shows up.
    {"nimmo",     0, 0, 0.90, 0.30, false,
     "steady-state, floor(N/5) subproblems per generation"},
    {"moead_dra", 0, 0, 0.40, -1.0, false,
     "steady-state with dynamic resource allocation"},

    // Same story as moead_m2m, same family, same operator: mean 1.01 at 200
    // generations, mean 0.028 / median 0.000 at 3000. The paper runs
    // 300,000 evaluations, i.e. about 3300 generations at this population
    // (liu2011 §IV-A). Resolved in the file header (the RESOLVED block).
    {"liu_gu2011", 0, 3000, -1.0, -1.0, false,
     "the paper's budget: ~3300 generations at 300k evaluations"},

    // DMS is not generational: one step() is one poll of one list point, at
    // most 2n = 24 evaluations here and fewer at the bounds (steps leaving the
    // box are not evaluated). 200 polls spend 2 292 evaluations and leave the
    // diagonal's extreme points (g = 2.5) in the answer, mean 2.08 / best
    // 0.021 (2026-09-23); 1000 polls spend 19 175, about the 18 200 the
    // generational algorithms spend in 200 generations, and reach mean 0.048 /
    // best 0.015 under the default thresholds.
    {"dms",        0, 1000, -1.0, -1.0, false,
     "one generation is one poll (<= 2n evaluations); ~equal evaluations"},

    // (μ+1) schemes: one step() is ONE offspring. 200 steps are 200
    // evaluations — SMS-EMOA mean 0.54 / best 0.18, MO-CMA-ES 0.97 / 0.19
    // (2026-09-23) — so both run 18 200 steps, the evaluations the
    // generational algorithms spend in 200 generations of 91: SMS-EMOA
    // 0.00008 / 0.00001, MO-CMA-ES 0.18 / 0.06, under the default thresholds.
    // MO-CMA-ES adapts a step size per individual from σ_0 = 0.6 of the
    // range and is the slower of the two on this budget.
    {"sms_emoa",   0, 18200, -1.0, -1.0, false,
     "steady state, one evaluation a step; ~equal evaluations"},
    {"mo_cma_es",  0, 18200, -1.0, -1.0, false,
     "steady state, one evaluation a step; ~equal evaluations"},
};

const Override* find_override(const std::string& name)
{
    for (const auto& o : OVERRIDES)
        if (name == o.name) return &o;
    return nullptr;
}

struct Row {
    std::string name;
    mootation::testing::RunResult r;
    bool        known_issue = false;
    bool        skipped     = false;
};

std::vector<Row> g_rows;
int g_known = 0;

template <typename Ind, typename Core>
void run_one(const char* name)
{
    using namespace mootation::testing;

    // Flush before running: if an algorithm crashes the process outright, the
    // last name printed identifies it. Buffered output would be lost.
    std::cout << "  running " << std::left << std::setw(20) << name
              << std::flush;

    const std::string n = name;

    const Override* ov       = find_override(n);
    const int       pop      = (ov && ov->pop > 0)     ? ov->pop  : POP;
    const int       gens     = (ov && ov->gens > 0)    ? ov->gens : GENS;
    const double    lim_mean = (ov && ov->mean >= 0.0) ? ov->mean : MEAN_ERROR_LIMIT;
    const double    lim_best = (ov && ov->best >= 0.0) ? ov->best : BEST_ERROR_LIMIT;

    RunResult r;
    try {
        r = run_algorithm<Ind, Core, DTLZ2Spec>(pop, gens, SEED);
    } catch (const std::exception& e) {
        // Reaching here means an algorithm refused this population size and no
        // override accounts for it — a genuine failure. Caught rather than
        // propagated so the remaining algorithms still get exercised.
        std::cout << " THREW: " << e.what() << '\n' << std::flush;
        check(false, n + ": threw std::exception: " + e.what());
        Row row{name, r, false, true};
        g_rows.push_back(row);
        return;
    }

    Row row{name, r, ov && ov->known_issue, false};
    g_rows.push_back(row);

    std::cout << std::fixed << std::setprecision(5) << " mean=" << r.mean_error
              << " best=" << r.best_error << " n=" << r.n;
    if (ov) std::cout << "   [" << ov->why << "]";
    std::cout << '\n' << std::flush;

    if (row.known_issue) {
        ++g_known;
        // Still assert the things that must hold even for a known issue: a
        // crash or NaN is never acceptable, whatever the open question is.
        check(r.finite, n + ": objectives are finite");
        check(r.n > 0,  n + ": population is not empty");
        return;
    }

    check(r.finite, n + ": objectives are finite");
    check(r.n > 0,  n + ": population is not empty");
    check(r.mean_error <= lim_mean,
          n + ": mean distance to front " + std::to_string(r.mean_error) +
              " exceeds " + std::to_string(lim_mean));
    check(r.best_error <= lim_best,
          n + ": best distance to front " + std::to_string(r.best_error) +
              " exceeds " + std::to_string(lim_best));
}

}   // namespace

int main()
{
    using namespace mootation;
    using namespace mootation::testing;

    std::cout << "DTLZ2 (M=3, n=12), pop=" << POP << ", gens=" << GENS
              << ", seed=" << SEED << "\n"
              << "Pareto front is the unit sphere; error = | ||f|| - 1 |\n\n";

#define MOOTATION_ALG(key, IND, CORE) \
    run_one<testing::Tag_##key, CORE<testing::Tag_##key>>(#key);
#include "mootation/algorithms.def"
#undef MOOTATION_ALG

    std::cout << std::left << std::setw(20) << "algorithm" << std::right
              << std::setw(8) << "n" << std::setw(14) << "mean err"
              << std::setw(14) << "best err" << '\n'
              << std::string(56, '-') << '\n'
              << std::fixed << std::setprecision(5);
    for (const auto& row : g_rows) {
        std::cout << std::left << std::setw(20) << row.name << std::right
                  << std::setw(8) << row.r.n
                  << std::setw(14) << row.r.mean_error
                  << std::setw(14) << row.r.best_error;
        if (row.skipped)          std::cout << "   NOT RUN";
        else if (row.known_issue) std::cout << "   KNOWN ISSUE";
        else if (!row.r.finite)   std::cout << "   NON-FINITE";
        std::cout << '\n';
    }

    std::cout << "\n" << g_rows.size() << " algorithms exercised";
    if (g_known)
        std::cout << ", " << g_known
                  << " carrying a known open issue (reported, not failed)";
    std::cout << ".\n";

    // Keep the exception list honest: if a known issue starts passing its
    // normal thresholds, say so, so the entry gets removed instead of rotting.
    for (const auto& row : g_rows) {
        if (!row.known_issue) continue;
        if (row.r.mean_error <= MEAN_ERROR_LIMIT &&
            row.r.best_error <= BEST_ERROR_LIMIT)
            std::cout << "NOTE: " << row.name
                      << " now meets the default thresholds — remove its "
                         "known_issue override.\n";
    }

    return report("convergence");
}

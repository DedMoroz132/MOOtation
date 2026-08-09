#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// Shared test harness. Deliberately dependency-free, like the library itself:
// a project whose selling point is "nothing beyond the standard library"
// should not need a test framework to prove it.
//
// Provides:
//   - check / check_close / report        minimal assertion + reporting
//   - DTLZ2Spec, ZDT1Spec                 benchmark problems with a closed-form
//                                         Pareto front, so the distance to the
//                                         front is exact and needs no reference
//                                         point set
//   - MOOTATION_TEST_PROBLEM(...)         declares a tag individual and its
//                                         Problem<> specialization
//   - run_algorithm<Ind, Core, Spec>()    seed, optional t_max, setup, run
//
// Note on the macro: Based_Individual keeps variables/objectives private and
// befriends `Problem<T>` only. A helper class is therefore *not* allowed to
// touch those fields — the evaluation has to happen inside the Problem<>
// specialization itself, which is why the macro generates the whole class and
// the *Spec types deal in plain vectors.
// ============================================================================

#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <mootation/mootation.hpp>

namespace mootation::testing {

// ── Minimal assertion harness ───────────────────────────────────────────────

inline int g_failures = 0;
inline int g_checks   = 0;

inline bool check(bool cond, const std::string& what)
{
    ++g_checks;
    if (!cond) {
        ++g_failures;
        std::cout << "  FAIL  " << what << '\n';
    }
    return cond;
}

inline bool check_close(double got, double want, double tol,
                        const std::string& what)
{
    ++g_checks;
    if (!(std::abs(got - want) <= tol)) {
        ++g_failures;
        std::cout << std::setprecision(10) << "  FAIL  " << what << ": got "
                  << got << ", want " << want << " (tol " << tol << ")\n";
        return false;
    }
    return true;
}

inline int report(const char* suite)
{
    std::cout << '\n' << suite << ": " << (g_checks - g_failures) << '/'
              << g_checks << " checks passed";
    if (g_failures) {
        std::cout << ", " << g_failures << " FAILED\n";
        return 1;
    }
    std::cout << "\n";
    return 0;
}

// ── DTLZ2 ───────────────────────────────────────────────────────────────────
// Deb, Thiele, Laumanns & Zitzler (2005), "Scalable Test Problems for
// Evolutionary Multiobjective Optimization".
//
// M objectives, n = M + K - 1 variables in [0,1]:
//   g      = sum_{i=M-1}^{n-1} (x_i - 0.5)^2
//   f_1    = (1+g) * prod_{i=1}^{M-1} cos(x_i * pi/2)
//   f_j    = (1+g) * sin(x_{M-j+1} * pi/2) * prod_{i=1}^{M-j} cos(x_i * pi/2)
//
// The Pareto front is the unit sphere in the positive orthant, ||f||_2 = 1.
// That closed form is what makes DTLZ2 the right convergence probe here: the
// distance of a point to the true front is |‖f‖ - 1| exactly.

struct DTLZ2Spec {
    static constexpr int M     = 3;
    static constexpr int K     = 10;
    static constexpr int NVARS = M + K - 1;   // 12
    static constexpr int NOBJS = M;

    static void eval(const std::vector<double>& x, std::vector<double>& f)
    {
        const double pi_2 = std::acos(-1.0) / 2.0;

        double g = 0.0;
        for (int i = M - 1; i < NVARS; ++i) {
            const double d = x[i] - 0.5;
            g += d * d;
        }

        for (int j = 0; j < M; ++j) {
            double v = 1.0 + g;
            for (int i = 0; i < M - 1 - j; ++i) v *= std::cos(x[i] * pi_2);
            if (j > 0) v *= std::sin(x[M - 1 - j] * pi_2);
            f[j] = v;
        }
    }

    static double front_error(const std::vector<double>& f)
    {
        double s = 0.0;
        for (double v : f) s += v * v;
        return std::abs(std::sqrt(s) - 1.0);
    }
};

// ── ZDT1 ────────────────────────────────────────────────────────────────────
// Zitzler, Deb & Thiele (2000). 30 variables in [0,1], 2 objectives.
//   f1 = x_0
//   g  = 1 + 9 * sum(x_1..x_{n-1}) / (n-1)
//   f2 = g * (1 - sqrt(f1/g))
// Pareto front: f2 = 1 - sqrt(f1), so the error is |f2 + sqrt(f1) - 1|.

struct ZDT1Spec {
    static constexpr int NVARS = 30;
    static constexpr int NOBJS = 2;

    static void eval(const std::vector<double>& x, std::vector<double>& f)
    {
        double sum = 0.0;
        for (int i = 1; i < NVARS; ++i) sum += x[i];
        const double g = 1.0 + 9.0 * sum / (NVARS - 1);
        f[0] = x[0];
        f[1] = g * (1.0 - std::sqrt(x[0] / g));
    }

    static double front_error(const std::vector<double>& f)
    {
        return std::abs(f[1] + std::sqrt(std::max(f[0], 0.0)) - 1.0);
    }
};

// ── Optional-setter detection ───────────────────────────────────────────────
// Some algorithms anneal a parameter over the run (RVEA's APD, MBRA, CLIA...)
// and must be told the generation budget. Detect the setter rather than keep a
// hand-written list that will drift as algorithms are added.

template <typename A, typename = void>
struct has_set_t_max : std::false_type {};
template <typename A>
struct has_set_t_max<A, std::void_t<decltype(std::declval<A&>().set_t_max(0))>>
    : std::true_type {};

// ── Runner ──────────────────────────────────────────────────────────────────

struct RunResult {
    double      mean_error = 0.0;   // mean distance of the population to the front
    double      best_error = 0.0;   // distance of the single closest solution
    std::size_t n          = 0;     // final population size
    bool        finite     = true;  // no NaN / inf in any objective
};

template <typename Ind, typename Core, typename Spec>
RunResult run_algorithm(int pop, int gens, unsigned seed)
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
    RunResult r;
    r.n = v.active_n();
    if (r.n == 0) { r.finite = false; return r; }

    double sum  = 0.0;
    double best = 1e300;
    for (std::size_t i = 0; i < r.n; ++i) {
        const auto& f = v.objectives_of(i);
        for (double c : f)
            if (!std::isfinite(c)) r.finite = false;
        const double e = Spec::front_error(f);
        sum += e;
        if (e < best) best = e;
    }
    r.mean_error = sum / static_cast<double>(r.n);
    r.best_error = best;
    return r;
}

}   // namespace mootation::testing

// Declares a tag individual and specializes Problem<> for it.
//   TAG   unique tag name
//   BASE  the individual type the algorithm requires
//   SPEC  mootation::testing::DTLZ2Spec / ZDT1Spec
#define MOOTATION_TEST_PROBLEM(TAG, BASE, SPEC)                              \
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
                testing::SPEC::NVARS,                                        \
                std::pair<std::optional<double>, std::optional<double>>{     \
                    0.0, 1.0});                                              \
        int get_vars_n() const { return testing::SPEC::NVARS; }              \
        int get_bin_vars_n() const { return 0; }                             \
        int get_objs_n() const { return testing::SPEC::NOBJS; }              \
        int get_lims_n() const { return 0; }                                 \
        void calc_objs(testing::TAG& ind) const                              \
        {                                                                    \
            testing::SPEC::eval(ind.variables, ind.objectives);              \
        }                                                                    \
    };                                                                       \
    }

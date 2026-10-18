// SPDX-License-Identifier: Apache-2.0
// Failed evaluations (data_vault.hpp, FAILED EVALUATIONS; task 7 of
// 2026-10-10, supplement 3, §2): a non-finite value marks the slot failed and
// gives it the penalty — above the worst finite value of every objective, a
// violation beyond the worst seen, the same for every failure, raised when a
// finite value moves the range; set_fail_raise(true) throws instead; every
// algorithm runs on a problem that fails on a third of its box, with no
// non-finite value anywhere in its population and its answer.

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "harness.hpp"

namespace mootation::testing {

// ZDT1 in 6 variables and one constraint (x1 - 0.95 <= 0); the evaluation
// fails where sin(40 x0) sin(40 x1) > 0.5 (a third of the box), and the
// constraint alone where x2 > 0.97.
struct FailSpec {
    static constexpr int NVARS = 6;
    static constexpr int NOBJS = 2;

    static bool fails(const std::vector<double>& x)
    {
        return std::sin(40.0 * x[0]) * std::sin(40.0 * x[1]) > 0.5;
    }
    static void eval(const std::vector<double>& x, std::vector<double>& f)
    {
        if (fails(x)) {
            f.assign(NOBJS, std::numeric_limits<double>::quiet_NaN());
            return;
        }
        double sum = 0.0;
        for (int i = 1; i < NVARS; ++i) sum += x[static_cast<std::size_t>(i)];
        const double g = 1.0 + 9.0 * sum / (NVARS - 1);
        f[0] = x[0];
        f[1] = g * (1.0 - std::sqrt(x[0] / g));
    }
    static void eval_limits(const std::vector<double>& x, std::vector<double>& lim)
    {
        lim[0] = x[2] > 0.97 ? std::numeric_limits<double>::quiet_NaN() : x[1] - 0.95;
    }
};

}   // namespace mootation::testing

// One tag and Problem<> per algorithm (Based_Individual befriends Problem<T>
// only), with or without the constraint.
#define MOOTATION_FAILING_PROBLEM(TAG, BASE, NLIM)                           \
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
                testing::FailSpec::NVARS,                                    \
                std::pair<std::optional<double>, std::optional<double>>{     \
                    0.0, 1.0});                                              \
        int get_vars_n()     const { return testing::FailSpec::NVARS; }      \
        int get_bin_vars_n() const { return 0; }                             \
        int get_objs_n()     const { return testing::FailSpec::NOBJS; }      \
        int get_lims_n()     const { return NLIM; }                          \
        void calc_objs(testing::TAG& ind) const                              \
        {                                                                    \
            testing::FailSpec::eval(ind.variables, ind.objectives);          \
            if constexpr (NLIM > 0)                                          \
                testing::FailSpec::eval_limits(ind.variables, ind.limits);   \
        }                                                                    \
    };                                                                       \
    }

MOOTATION_FAILING_PROBLEM(FailPlain, NSGAII_Individual, 0)
MOOTATION_FAILING_PROBLEM(FailCons, NSGAII_Individual, 1)

#define MOOTATION_ALG(key, IND, CORE) MOOTATION_FAILING_PROBLEM(FTag_##key, IND, 0)
#include "mootation/algorithms.def"
#undef MOOTATION_ALG

namespace {

using namespace mootation;
using namespace mootation::testing;

using X = std::vector<double>;

// A point that fails and one that does not, in x0, x1.
const X FAILING = {0.04, 0.04, 0.1, 0.1, 0.1, 0.1};   // sin(1.6)^2 > 0.5
const X GOOD_A  = {0.1, 0.5, 0.1, 0.1, 0.1, 0.1};     // sin(4) sin(20) < 0.5
const X GOOD_B  = {0.9, 0.5, 0.2, 0.2, 0.2, 0.2};

template <typename Tag>
void put(DataVault<Tag>& v, const std::vector<X>& rows)
{
    for (std::size_t i = 0; i < rows.size(); ++i) v.set_variables(i, rows[i]);
    v.sync();
}

void test_penalty()
{
    std::cout << "-- the penalty: above the worst, equal for every failure --\n";
    check(FailSpec::fails(FAILING) && !FailSpec::fails(GOOD_A) && !FailSpec::fails(GOOD_B),
          "the test points fail and do not as meant");
    Problem<FailPlain> prob;
    DataVault<FailPlain> v(4, prob);
    put(v, {GOOD_A, FAILING, GOOD_B, FAILING});
    check(!v.is_failed(0) && v.is_failed(1) && !v.is_failed(2) && v.is_failed(3)
          && v.failed_n() == 2, "two slots failed, counted");
    const X fa = v.objectives_of(0), fb = v.objectives_of(2), p = v.objectives_of(1);
    bool finite = true, above = true;
    for (std::size_t j = 0; j < p.size(); ++j) {
        finite = finite && std::isfinite(p[j]);
        above = above && p[j] > std::max(fa[j], fb[j]);
    }
    check(finite && above, "the penalty is finite and worse than both good points in every objective");
    check(v.objectives_of(3) == p, "every failure has the same penalty");
    const double span0 = std::max(fa[0], fb[0]) - std::min(fa[0], fb[0]);
    check(std::abs(p[0] - (std::max(fa[0], fb[0]) + span0)) < 1e-12,
          "the penalty is one span above the worst");
    // a worse finite point moves the penalty of every failure
    const int base = v.expand(1);
    v.set_variables(static_cast<std::size_t>(base), {0.95, 0.5, 1.0, 1.0, 1.0, 1.0});
    v.sync();
    check(!v.is_failed(static_cast<std::size_t>(base))
          && v.objectives_of(1)[1] > v.objectives_of(static_cast<std::size_t>(base))[1]
          && v.objectives_of(1) == v.objectives_of(3), "a new worst moves the penalty of both failures");
}

void test_constraints_and_raise()
{
    std::cout << "-- a failure is infeasible beyond any; raise --\n";
    Problem<FailCons> prob;
    DataVault<FailCons> v(4, prob);
    const X late = {0.1, 0.5, 0.99, 0.1, 0.1, 0.1};     // objectives fine, constraint NaN
    put(v, {GOOD_A, FAILING, {0.1, 0.99, 0.1, 0.1, 0.1, 0.1}, late});
    check(v.is_failed(1) && v.is_failed(3) && !v.is_failed(2), "a NaN constraint alone fails too");
    check(v.get_cv(1) > v.get_cv(2) && v.get_cv(2) > 0.0 && v.get_cv(0) == 0.0,
          "a failure's violation is beyond the infeasible point's");
    DataVault<FailPlain> w(2, Problem<FailPlain>{});
    w.set_fail_raise(true);
    bool threw = false;
    try { put(w, {GOOD_A, FAILING}); } catch (const std::runtime_error&) { threw = true; }
    check(threw, "set_fail_raise(true): a non-finite value throws");
}

// Every algorithm, 2 000 evaluations (about 60 generations of 30; 2 000
// steps for paes, whose step is one evaluation from one starting point), on
// the problem that fails on a third of its box: no exception, and no
// non-finite value in the population, its answer included.
template <typename Ind, typename Core>
bool runs_through(int pop, int steps)
{
    Problem<Ind> prob;
    DataVault<Ind> vault(pop, prob);
    Optimizer<Ind, Core> opt(std::move(vault), defer_setup);
    opt.get_algorithm().set_seed(7);
    if constexpr (has_set_t_max<Core>::value) opt.get_algorithm().set_t_max(steps);
    opt.setup();
    opt.optimize(steps);
    auto& v = opt.get_vault();
    const std::size_t n = std::min<std::size_t>(v.active_n(), static_cast<std::size_t>(v.pop_size()));
    for (std::size_t i = 0; i < n; ++i)
        for (double f : v.objectives_of(i))
            if (!std::isfinite(f)) return false;
    return v.failed_n() > 0;
}

void test_every_algorithm()
{
    std::cout << "-- every algorithm on a third of the box failing --\n";
    int ok = 0, all = 0;
    std::string bad;
#define MOOTATION_ALG(key, IND, CORE)                                         \
    {                                                                         \
        ++all;                                                                \
        const std::string k = #key;                                           \
        const int pop = (k == "moead_m2m" || k == "sms_m2m") ? 30 : 31;       \
        bool good = false;                                                    \
        const int steps = k == "paes" ? 2000 : 60;                            \
        try { good = runs_through<testing::FTag_##key, CORE<testing::FTag_##key>>(pop, steps); } \
        catch (const std::exception& e) { bad += k + " (" + e.what() + ") "; } \
        if (good) ++ok; else if (bad.find(k) == std::string::npos) bad += k + " "; \
    }
#include "mootation/algorithms.def"
#undef MOOTATION_ALG
    check(ok == all, std::to_string(ok) + " of " + std::to_string(all) +
                     " ran with failures and finite populations" + (bad.empty() ? "" : ": " + bad));
}

}   // namespace

int main()
{
    std::cout << "failed evaluations\n";
    test_penalty();
    test_constraints_and_raise();
    test_every_algorithm();
    return mootation::testing::report("failures");
}

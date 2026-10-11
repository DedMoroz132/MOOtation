// SPDX-License-Identifier: Apache-2.0
// Equality constraints (data_vault.hpp, EQUALITY CONSTRAINTS; task 7 of
// 2026-10-10, supplement 3, §3): an equality h = 0 holds within the tolerance
// delta, |h| <= delta, and adds max(|h| - delta, 0) to cv; limits_of gives
// |h| - delta, an inequality value; a new tolerance rewrites every evaluated
// slot from |h| exactly; a seed from outside carries raw values, a write-back
// the vault's; a failure stays a failure; without equalities nothing moves.

#include <cmath>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "harness.hpp"

namespace mootation::testing {

// Two variables, two objectives (x0, 1 - x0 + x1); the inequality x0 - 0.9
// <= 0 and the equality h = x1 - 0.5 = 0, last; h is NaN where x0 > 0.99.
struct EqTag : public mootation::NSGAII_Individual {};

}   // namespace mootation::testing

namespace mootation {
template <>
class Problem<testing::EqTag> {
public:
    std::vector<std::pair<std::optional<double>, std::optional<double>>> bounds =
        std::vector<std::pair<std::optional<double>, std::optional<double>>>(
            2, std::pair<std::optional<double>, std::optional<double>>{0.0, 1.0});
    int get_vars_n()     const { return 2; }
    int get_bin_vars_n() const { return 0; }
    int get_objs_n()     const { return 2; }
    int get_lims_n()     const { return 2; }
    void calc_objs(testing::EqTag& ind) const
    {
        const auto& x = ind.variables;
        ind.objectives[0] = x[0];
        ind.objectives[1] = 1.0 - x[0] + x[1];
        ind.limits[0] = x[0] - 0.9;
        ind.limits[1] = x[0] > 0.99 ? std::numeric_limits<double>::quiet_NaN() : x[1] - 0.5;
    }
};
}   // namespace mootation

namespace {

using namespace mootation;
using namespace mootation::testing;

using X = std::vector<double>;

void put(DataVault<EqTag>& v, const std::vector<X>& rows)
{
    for (std::size_t i = 0; i < rows.size(); ++i) v.set_variables(i, rows[i]);
    v.sync();
}

void test_tolerance()
{
    std::cout << "-- |h| <= delta holds, the rest is violation --\n";
    const double near = 0.5 + 5e-5, far = 0.5 + 3e-4;
    const double h_near = std::abs(near - 0.5), h_far = std::abs(far - 0.5);
    DataVault<EqTag> v(4, Problem<EqTag>{});
    v.set_equalities(1, 1e-4);
    put(v, {{0.2, near}, {0.2, far}, {0.2, 0.5 - 5e-5}, {0.95, 0.5}});
    check(v.get_cv(0) == 0.0 && v.get_cv(2) == 0.0, "|h| below delta, either side of 0: feasible");
    check(v.get_cv(1) == h_far - 1e-4, "cv = |h| - delta beyond it, exactly");
    check(v.limits_of(1)[1] == h_far - 1e-4 && v.limits_of(1)[0] == 0.2 - 0.9,
          "limits_of: |h| - delta for the equality, the inequality as it came");
    check(v.get_cv(3) == 0.95 - 0.9, "an inequality's violation alone, the equality held");
    // the tolerance exactly |h|: on the boundary, feasible
    DataVault<EqTag> w(1, Problem<EqTag>{});
    w.set_equalities(1, h_near);
    put(w, {{0.2, near}});
    check(w.get_cv(0) == 0.0, "|h| = delta: feasible");
    // a new tolerance rewrites every evaluated slot from |h|
    v.set_eq_tol(1e-5);
    check(v.get_cv(0) == h_near - 1e-5 && v.limits_of(0)[1] == h_near - 1e-5,
          "a tighter delta: the near point violates by |h| - delta, exactly");
    check(v.get_cv(1) == h_far - 1e-5, "and the far one by more");
    v.set_eq_tol(1e-4);
    check(v.get_cv(0) == 0.0 && v.get_cv(1) == h_far - 1e-4, "back to 1e-4: as at first");
    check(v.eq_n() == 1 && v.eq_tol() == 1e-4, "eq_n, eq_tol");
}

void test_seeds_and_failures()
{
    std::cout << "-- seeds, write-backs, failures --\n";
    DataVault<EqTag> v(3, Problem<EqTag>{});
    v.set_equalities(1, 1e-4);
    const double h = 3e-4;
    v.seed_individual(0, {0.2, 0.5 + h}, {0.2, 0.8 + h}, {}, {0.2 - 0.9, h}, /*raw=*/true);
    check(v.limits_of(0)[1] == h - 1e-4 && v.get_cv(0) == h - 1e-4,
          "a raw seed's h becomes |h| - delta");
    v.seed_individual(1, {0.2, 0.5 + h}, {0.2, 0.8 + h}, {}, {0.2 - 0.9, h - 1e-4});
    check(v.limits_of(1)[1] == h - 1e-4, "a write-back keeps the vault's value");
    v.set_eq_tol(2e-4);
    check(std::abs(v.limits_of(1)[1] - (h - 2e-4)) < 1e-18 && v.limits_of(0)[1] == h - 2e-4,
          "and both follow a new delta");
    v.seed_individual(2, {0.2, 0.5}, {0.2, 0.8});
    v.set_eq_tol(1e-6);
    check(v.limits_of(2)[1] == 0.0 && v.get_cv(2) == 0.0,
          "a write-back without values (all 0) stays 0");
    // a failure: NaN h
    DataVault<EqTag> f(2, Problem<EqTag>{});
    f.set_equalities(1, 1e-4);
    put(f, {{0.995, 0.5}, {0.2, 0.5 + h}});
    const double pen = f.get_cv(0);
    check(f.is_failed(0) && std::isfinite(pen) && pen > f.get_cv(1), "a NaN h fails the point");
    f.set_eq_tol(1e-8);
    check(f.get_cv(0) == pen && f.is_failed(0), "a new delta leaves the failure's penalty");
    bool threw = false;
    try { DataVault<EqTag>(1, Problem<EqTag>{}).set_equalities(3, 1e-4); }
    catch (const std::invalid_argument&) { threw = true; }
    check(threw, "more equalities than constraints: refused");
}

void test_without()
{
    std::cout << "-- without equalities nothing moves --\n";
    DataVault<EqTag> v(2, Problem<EqTag>{});
    put(v, {{0.2, 0.5 + 3e-4}, {0.95, 0.4}});
    check(v.limits_of(0)[1] == (0.5 + 3e-4) - 0.5 && v.get_cv(0) == (0.5 + 3e-4) - 0.5,
          "h is an inequality value as before");
    check(v.limits_of(1)[1] == 0.4 - 0.5 && v.get_cv(1) == 0.95 - 0.9, "a negative h satisfies it");
    v.set_eq_tol(0.5);
    check(v.get_cv(0) == (0.5 + 3e-4) - 0.5, "set_eq_tol without equalities changes nothing");
}

void test_optimizer_seed()
{
    std::cout << "-- the caller's seed population carries raw values --\n";
    DataVault<EqTag> vault(2, Problem<EqTag>{});
    vault.set_equalities(1, 1e-4);
    Optimizer<EqTag, NSGAIICore<EqTag>> opt(std::move(vault), defer_setup);
    opt.setup_with_seed({{0.2, 0.5 - 2e-4}, {0.3, 0.5}}, {{0.2, 0.6}, {0.3, 1.2}}, {},
                        {{0.2 - 0.9, -2e-4}, {0.3 - 0.9, 0.0}});
    auto& v = opt.get_vault();
    check(v.limits_of(0)[1] == 2e-4 - 1e-4 && v.get_cv(1) == 0.0,
          "setup_with_seed: h = -2e-4 is |h| - delta = 1e-4");
}

}   // namespace

int main()
{
    std::cout << "equality constraints\n";
    test_tolerance();
    test_seeds_and_failures();
    test_without();
    test_optimizer_seed();
    return mootation::testing::report("equalities");
}

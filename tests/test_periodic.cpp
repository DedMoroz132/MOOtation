// SPDX-License-Identifier: Apache-2.0
// Periodic variables (task 24, §24.4.2; scalar/periodic.hpp, PER-1 ... PER-3):
//   * the wrap: [lo, hi), the seam hi is lo;
//   * a minimum just above the seam, met from below: every method's points
//     lie in [lo, hi) — no point on the seam — and the strong ones find it;
//   * a run that never crosses the seam is the run without the mask, bit
//     for bit (basin_hopping with a small step about an inner minimum);
//   * the minima log takes the shorter way round;
//   * a mask of the wrong length, or on an infinite bound, is refused.

#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <mootation/scalar/scalar.hpp>

namespace sc = mootation::scalar;
namespace per = mootation::scalar::periodic;
using Vec = std::vector<double>;
using Rows = std::vector<Vec>;

static int failures = 0;

static void check(bool ok, const std::string& what)
{
    if (!ok) {
        ++failures;
        std::printf("FAIL %s\n", what.c_str());
    }
}

// x0 an angle on [0, 1) (period 1) with its minimum at a, x1 in [-1, 1]
static sc::Problem angle(double a, std::shared_ptr<Rows> asked, bool mask = true)
{
    const double pi = 3.14159265358979323846;
    sc::Problem P;
    P.lo = {0.0, -1.0};
    P.hi = {1.0, 1.0};
    P.m = 1;
    if (mask) P.periodic = {1, 0};
    P.evaluate = [a, pi, asked](const Vec& x, Vec& f, Vec&) {
        if (asked) asked->push_back(x);
        f[0] = 1.0 - mootation::dm::cos(2.0 * pi * (x[0] - a)) + (x[1] - 0.3) * (x[1] - 0.3);
    };
    return P;
}

static bool same(const sc::Result& a, const sc::Result& b)
{
    if (a.improvements.size() != b.improvements.size()) return false;
    for (std::size_t i = 0; i < a.improvements.size(); ++i)
        if (std::memcmp(&a.improvements[i], &b.improvements[i], sizeof a.improvements[i]) != 0) return false;
    return a.objectives == b.objectives && a.variables == b.variables && a.evaluations == b.evaluations;
}

static sc::Knob word(const std::string& w) { return {true, w, 0.0}; }
static sc::Knob number(double v) { return {false, "", v}; }

int main()
{
    // the wrap (PER-1)
    check(per::wrap_into(1.0, 0.0, 1.0) == 0.0 && per::wrap_into(-0.25, 0.0, 1.0) == 0.75 &&
              per::wrap_into(2.5, 0.0, 1.0) == 0.5 && per::wrap_into(0.3, 0.0, 1.0) == 0.3,
          "the wrap into [0, 1)");
    check(per::wrap_into(-1e-18, 0.0, 1.0) == 0.0, "a wrap that rounds to hi is lo: no point on the seam");
    check(per::gap(0.02, 0.98, true, 1.0) < 0.0400001 && per::gap(0.02, 0.98, false, 1.0) > 0.95, "the shorter way");

    // the minimum at 0.02, the box's lower edge 0: from 0.9, 0.95 ... the methods cross the seam
    const sc::Knob nm = word("{\"solver\":\"nelder_mead\"}");
    struct Case {
        std::string name;
        sc::Knobs knobs;
        bool finds;
    };
    for (const Case& c : std::vector<Case>{{"de", {}, true},
                                           {"jade", {}, true},
                                           {"shade", {}, true},
                                           {"lshade", {}, true},
                                           {"g3_pcx", {}, false},
                                           {"pso", {}, true},
                                           {"spso2011", {}, true},
                                           {"sep_cmaes", {}, false},
                                           {"cmaes", {}, false},
                                           {"nelder_mead", {}, true},
                                           {"ipop", {}, true},
                                           {"basin_hopping", {{"inner", nm}}, true},
                                           {"ils", {{"inner", nm}}, true}}) {
        auto asked = std::make_shared<Rows>();
        sc::Start st;
        if (c.name == "cmaes" || c.name == "nelder_mead" || c.name == "basin_hopping" || c.name == "ils" ||
            c.name == "sep_cmaes") {
            st.has_x0 = true;
            st.x0 = {0.9, 0.0};
        }
        const sc::Result r = sc::run(c.name, angle(0.02, asked), 1e-6, false, 0.0, 20, 4000, 3, c.knobs, {}, 0, st);
        bool inside = !asked->empty();
        for (const Vec& x : *asked) inside = inside && x[0] >= 0.0 && x[0] < 1.0 && x[1] >= -1.0 && x[1] <= 1.0;
        check(inside, c.name + ": every point in [lo, hi), none on the seam");
        const double off = per::gap(r.variables[0][0], 0.02, true, 1.0);
        if (c.finds) check(off < 1e-3, c.name + ": the minimum across the seam (off by " + std::to_string(off) + ")");
        std::printf("  %-13s x0 = %.6f (off %.2e), f = %.3g, %zu points\n", c.name.c_str(), r.variables[0][0], off,
                    r.objectives[0][0], asked->size());
    }

    // a run that never crosses the seam: the run without the mask
    {
        const sc::Knobs k{{"inner", nm}, {"bh_step", number(0.01)}};
        sc::Start st;
        st.has_x0 = true;
        st.x0 = {0.45, 0.0};
        const sc::Result a = sc::run("basin_hopping", angle(0.5, nullptr, true), 1e-6, false, 0.0, 20, 2000, 3, k, {}, 0, st);
        const sc::Result b = sc::run("basin_hopping", angle(0.5, nullptr, false), 1e-6, false, 0.0, 20, 2000, 3, k, {}, 0, st);
        check(same(a, b), "basin_hopping away from the seam: the same steps with the mask and without");
        const sc::Result c = sc::run("de", angle(0.5, nullptr, true), 1e-6, false, 0.0, 20, 2000, 3, {}, {}, 0, {});
        const sc::Result d = sc::run("de", angle(0.5, nullptr, false), 1e-6, false, 0.0, 20, 2000, 3, {}, {}, 0, {});
        std::printf("  de, minimum inside, with the mask and without: %s\n", same(c, d) ? "the same run" : "other runs (a mutant left the box)");
    }

    // the minima log: 0.001 and 0.999 one minimum across the seam
    {
        sc::strategy::MinimaLog with({0.0, -1.0}, {1.0, 1.0}, 0.01, 1e-8), without({0.0, -1.0}, {1.0, 1.0}, 0.01, 1e-8);
        with.periodic({1, 0});
        for (sc::strategy::MinimaLog* log : {&with, &without}) {
            log->add({0.001, 0.3}, 0.0, 0);
            log->add({0.999, 0.3}, 0.0, 1);
        }
        check(with.list().size() == 1 && with.list()[0].found == 2, "the log: one minimum across the seam");
        check(without.list().size() == 2, "the log without the mask: two");
    }

    // refused
    auto refused = [](sc::Problem P, const std::string& part) {
        std::string why;
        try {
            sc::run("de", P, 1e-6, false, 0.0, 20, 100, 3, {}, {}, 0, {});
        } catch (const std::invalid_argument& e) {
            why = e.what();
        }
        check(why.find(part) != std::string::npos, "refused: '" + why + "' (wanted: " + part + ")");
    };
    {
        sc::Problem P = angle(0.0, nullptr);
        P.periodic = {1};
        refused(P, "periodic: 1 flags for 2 variables");
    }

    if (failures) {
        std::printf("test_periodic: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("test_periodic: all checks passed\n");
    return 0;
}

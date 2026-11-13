// SPDX-License-Identifier: Apache-2.0
// The single-objective population methods' start with warm_start = on (task
// 24, §24.4.1; scalar.hpp POP-7, restarts.hpp RST-9):
//   * without the knob a start is ignored and reported, the run as without it
//     (what a strategy's starts rely on);
//   * seed points at the optimum: the best is the optimum at no evaluation,
//     its record at evaluation 0, for every population method;
//   * x0 and a design are the first population's first rows, evaluated
//     first and in order;
//   * the draw is made in full all the same: a start equal to the method's
//     own first rows gives the run without a start, bit for bit;
//   * ipop's and bipop's x0 is the first run's mean: x0 equal to the mean
//     they would draw gives the run without x0, bit for bit;
//   * a start outside the box is refused.

#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <mootation/scalar/scalar.hpp>

namespace sc = mootation::scalar;
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

// a shifted sphere in [-1, 1]^4 that writes down the points it is asked for
static sc::Problem sphere(std::shared_ptr<Rows> asked)
{
    sc::Problem P;
    P.lo.assign(4, -1.0);
    P.hi.assign(4, 1.0);
    P.m = 1;
    P.evaluate = [asked](const Vec& x, Vec& f, Vec&) {
        if (asked) asked->push_back(x);
        double s = 0.0;
        for (std::size_t i = 0; i < x.size(); ++i) s += (static_cast<double>(i) + 1.0) * (x[i] - 0.3) * (x[i] - 0.3);
        f[0] = s;
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

static const sc::Knobs kWarm{{"warm_start", {true, "on", 0.0}}};

int main()
{
    const std::vector<std::string> methods = {"de", "jade", "shade", "lshade", "ga_sbx", "ssga",
                                              "g3_pcx", "chc", "pso", "spso2011"};
    const Vec opt(4, 0.3);
    for (const std::string& m : methods) {
        // no start: the run as before (the method's own draws), and the rows it asks first
        auto first = std::make_shared<Rows>();
        const sc::Result plain = sc::run(m, sphere(first), 1e-6, false, 0.0, 20, 2000, 3, {}, {}, 0, {});
        {
            sc::Start any;
            any.has_x0 = true;
            any.x0 = Vec(4, 0.3);
            const sc::Result off = sc::run(m, sphere(nullptr), 1e-6, false, 0.0, 20, 2000, 3, {}, {}, 0, any);
            bool listed = false;
            for (const auto& w : off.ignored) listed = listed || w == "x0";
            check(same(plain, off) && listed, m + ": without warm_start, x0 ignored and reported, the same run");
        }

        // seed points at the optimum: f = 0 known, at no evaluation
        sc::Start st;
        st.has_seed_points = true;
        st.seed_points.push_back({opt, {0.0}, {}});
        const sc::Result seeded = sc::run(m, sphere(nullptr), 1e-6, false, 0.0, 20, 2000, 3, kWarm, {}, 0, st);
        check(!seeded.objectives.empty() && seeded.objectives[0][0] == 0.0 && seeded.variables[0] == opt,
              m + ": seeded at the optimum, the answer is the optimum");
        check(!seeded.improvements.empty() && seeded.improvements[0].f == 0.0 &&
                  seeded.improvements[0].evaluations == 0,
              m + ": its record at evaluation 0");
        bool listed = false;
        for (const auto& w : seeded.ignored) listed = listed || w == "seed_points";
        check(!listed, m + ": the seed points are not reported as ignored");

        // x0 and a design: the first rows, evaluated first and in order
        auto asked = std::make_shared<Rows>();
        sc::Start sd;
        sd.has_x0 = true;
        sd.x0 = {0.5, -0.5, 0.25, 0.0};
        sd.has_design = true;
        sd.design = {{0.1, 0.1, 0.1, 0.1}, {-0.2, 0.2, -0.2, 0.2}};
        sc::run(m, sphere(asked), 1e-6, false, 0.0, 20, 2000, 3, kWarm, {}, 0, sd);
        check(asked->size() >= 3 && (*asked)[0] == sd.x0 && (*asked)[1] == sd.design[0] && (*asked)[2] == sd.design[1],
              m + ": x0, then the design's rows, evaluated first");

        // the start equal to the method's own first rows: the run without a start
        sc::Start own;
        own.has_design = true;
        own.design.assign(first->begin(), first->begin() + 3);
        const sc::Result again = sc::run(m, sphere(nullptr), 1e-6, false, 0.0, 20, 2000, 3, kWarm, {}, 0, own);
        check(same(plain, again), m + ": the draw is made in full: its own first rows as a start, the same run");

        // outside the box: refused
        sc::Start out;
        out.has_x0 = true;
        out.x0 = {0.0, 0.0, 1.5, 0.0};
        std::string why;
        try {
            sc::run(m, sphere(nullptr), 1e-6, false, 0.0, 20, 2000, 3, kWarm, {}, 0, out);
        } catch (const std::invalid_argument& e) {
            why = e.what();
        }
        check(why.find("outside the box (variable 2)") != std::string::npos, m + ": a start outside the box: '" + why + "'");
    }

    // ipop and bipop: x0 the first run's mean (RST-9)
    for (const std::string& m : {std::string("ipop"), std::string("bipop")}) {
        const double share = m == "ipop" ? 0.0 : 0.1;
        std::mt19937 rng(sc::strategy::start_seed(3, 0));
        const Vec mean = sc::restarts::start_mean(Vec(4, -1.0), Vec(4, 1.0), share, rng);
        sc::Start st;
        st.has_x0 = true;
        st.x0 = mean;
        const sc::Result a = sc::run(m, sphere(nullptr), 1e-6, false, 0.0, 20, 4000, 3, {}, {}, 0, {});
        const sc::Result b = sc::run(m, sphere(nullptr), 1e-6, false, 0.0, 20, 4000, 3, kWarm, {}, 0, st);
        check(same(a, b), m + ": x0 = the mean it draws: the run without x0, bit for bit");
        st.x0 = opt;
        const sc::Result c = sc::run(m, sphere(nullptr), 1e-6, false, 0.0, 20, 4000, 3, kWarm, {}, 0, st);
        check(!same(a, c), m + ": another x0, another run");
        bool listed = false;
        for (const auto& w : c.ignored) listed = listed || w == "x0";
        check(!listed, m + ": x0 is not reported as ignored");
    }

    if (failures) {
        std::printf("test_warm_start_so: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("test_warm_start_so: all checks passed\n");
    return 0;
}

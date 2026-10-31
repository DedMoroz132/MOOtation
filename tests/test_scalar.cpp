// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// include/mootation/scalar/ (task 17, stage 6): the methods in C++.
//   eigen     A V = V diag(w), V orthogonal, w increasing, on random
//             symmetric matrices;
//   models    a quadratic is its own regression model; with fewer points than
//             coefficients, the minimum Frobenius norm model of a linear
//             function is linear;
//   methods   every name of scalar::names() but the single-objective solvers
//             on ZDT1 (4 variables) and on a
//             problem with an inequality and an equality: the budget never
//             exceeded, the answer feasible, non-dominated and pop_size at
//             most, the evaluations counted one by one; the same seed, the
//             same run;
//   single    slsqp, nelder_mead and cmaes at one objective (task 18, item
//             2.3): the minimum of a weighted sphere, with and without an
//             active inequality; a smaller budget the same run cut short
//             (SINGLE-1); two objectives and an infinite bound refused;
//   polish    the polished answer weakly dominates the algorithm's answer;
//   bits      a checksum of the runs' answers, the same on every platform of
//             the CI (no contraction into fused multiply-adds, dm:: for the
//             elementary functions, rnd:: for the random numbers).
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <mootation/scalar/polish.hpp>
#include <mootation/scalar/scalar.hpp>

namespace {

int g_failed = 0;

void check(bool ok, const std::string& what)
{
    std::printf("%s  %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++g_failed;
}

namespace sc = mootation::scalar;
using Vec = std::vector<double>;

void eigen_random()
{
    std::mt19937 gen(11);
    double worst = 0.0;
    bool ordered = true;
    for (int t = 0; t < 300; ++t) {
        const std::size_t n = 1 + gen() % 12;
        mootation::lsq::Matrix A(n, n);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j <= i; ++j)
                A(i, j) = A(j, i) = mootation::rnd::uniform01(gen) * 2.0 - 1.0;
        Vec w;
        mootation::lsq::Matrix V;
        mootation::eigen::symmetric(A, w, V);
        for (std::size_t k = 0; k + 1 < n; ++k) ordered = ordered && w[k] <= w[k + 1];
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t k = 0; k < n; ++k) {
                double av = 0.0, vv = 0.0;
                for (std::size_t j = 0; j < n; ++j) {
                    av += A(i, j) * V(j, k);
                    vv += V(j, i) * V(j, k);
                }
                worst = std::max(worst, std::fabs(av - w[k] * V(i, k)));
                worst = std::max(worst, std::fabs(vv - (i == k ? 1.0 : 0.0)));
            }
    }
    check(worst < 1e-12 && ordered, "eigen: A V = V diag(w), V^T V = I, w increasing (worst " +
                                        std::to_string(worst) + ")");
}

void models()
{
    std::mt19937 gen(3);
    auto u = [&]() { return mootation::rnd::uniform01(gen) * 2.0 - 1.0; };
    auto q = [](const Vec& s) {
        return 1.0 + s[0] - 2.0 * s[1] + 0.5 * s[2] + 0.5 * (2.0 * s[0] * s[0] + s[1] * s[1] + 4.0 * s[2] * s[2]);
    };
    std::vector<Vec> S, F;
    for (int i = 0; i < 40; ++i) {                      // regression: 40 > 10 coefficients
        Vec s{u(), u(), u()};
        S.push_back(s);
        F.push_back({q(s)});
    }
    sc::Model md = sc::models(S, F)[0];
    bool ok = std::fabs(md.c - 1.0) < 1e-10 && std::fabs(md.g[0] - 1) < 1e-10 &&
              std::fabs(md.g[1] + 2) < 1e-10 && std::fabs(md.g[2] - 0.5) < 1e-10 &&
              std::fabs(md.H(0, 0) - 2) < 1e-9 && std::fabs(md.H(1, 1) - 1) < 1e-9 &&
              std::fabs(md.H(2, 2) - 4) < 1e-9 && std::fabs(md.H(0, 1)) < 1e-9;
    check(ok, "models: a quadratic is its own regression model");
    S.clear();
    F.clear();
    for (int i = 0; i < 7; ++i) {                       // minimum Frobenius norm: 4 < 7 < 10
        Vec s{u(), u(), u()};
        S.push_back(s);
        F.push_back({1.0 + s[0] - 2.0 * s[1] + 0.5 * s[2]});
    }
    md = sc::models(S, F)[0];
    double hmax = 0.0;
    for (double v : md.H.a) hmax = std::max(hmax, std::fabs(v));
    check(hmax < 1e-9 && std::fabs(md.g[0] - 1) < 1e-9 && std::fabs(md.g[1] + 2) < 1e-9,
          "models: minimum Frobenius norm of a linear function is linear");
}

// ZDT1 on 4 variables, every call counted
struct Counted {
    sc::Problem P;
    std::shared_ptr<long long> n = std::make_shared<long long>(0);
    long long calls() const { return *n; }
};

Counted zdt1()
{
    Counted c;
    c.P.lo.assign(4, 0.0);
    c.P.hi.assign(4, 1.0);
    c.P.m = 2;
    c.P.evaluate = [n = c.n](const Vec& x, Vec& f, Vec&) {
        ++*n;
        double g = 0.0;
        for (std::size_t i = 1; i < x.size(); ++i) g += x[i];
        g = 1.0 + 9.0 * g / 3.0;
        f[0] = x[0];
        f[1] = g * (1.0 - std::sqrt(x[0] / g));
    };
    return c;
}

// min (x0, x1 + x2^2) s.t. x0 + x1 >= 0.5 (g = 0.5 - x0 - x1 <= 0), x2 = 0.25
Counted constrained()
{
    Counted c;
    c.P.lo.assign(3, 0.0);
    c.P.hi.assign(3, 1.0);
    c.P.m = 2;
    c.P.n_cons = 2;
    c.P.n_eq = 1;
    c.P.evaluate = [n = c.n](const Vec& x, Vec& f, Vec& g) {
        ++*n;
        f[0] = x[0];
        f[1] = x[1] + x[2] * x[2];
        g[0] = 0.5 - x[0] - x[1];
        g[1] = x[2] - 0.25;
    };
    return c;
}

bool nondominated(const sc::Rows& F)
{
    for (std::size_t a = 0; a < F.size(); ++a)
        for (std::size_t b = 0; b < F.size(); ++b) {
            if (a == b) continue;
            bool le = true, lt = false;
            for (std::size_t j = 0; j < F[a].size(); ++j) {
                le = le && F[b][j] <= F[a][j];
                lt = lt || F[b][j] < F[a][j];
            }
            if (le && lt) return false;
        }
    return true;
}

std::uint64_t g_hash = 1469598103934665603ULL;      // FNV-1a over the answers' bits

void hash(const sc::Rows& R)
{
    for (const Vec& r : R)
        for (double v : r) {
            std::uint64_t b;
            std::memcpy(&b, &v, sizeof b);
            for (int k = 0; k < 8; ++k) {
                g_hash ^= (b >> (8 * k)) & 0xFF;
                g_hash *= 1099511628211ULL;
            }
        }
}

void methods()
{
    const long long budget = 1500;
    for (const std::string& name : sc::names()) {
        if (std::find(sc::single_names().begin(), sc::single_names().end(), name) != sc::single_names().end())
            continue;                               // one objective only: single() below
        for (int which = 0; which < 2; ++which) {
            Counted c = which == 0 ? zdt1() : constrained();
            const sc::Result r = sc::run(name, c.P, 1e-3, false, 0.0, 20, budget, 7, {}, {}, 0, {});
            Counted c2 = which == 0 ? zdt1() : constrained();
            const sc::Result r2 = sc::run(name, c2.P, 1e-3, false, 0.0, 20, budget, 7, {}, {}, 0, {});
            bool feasible = true;
            for (double v : r.cv) feasible = feasible && v <= 0.0;
            const std::string what = name + (which == 0 ? " on ZDT1" : " with constraints");
            check(r.evaluations == c.calls() && c.calls() <= budget && !r.objectives.empty() &&
                      r.objectives.size() <= 20 && nondominated(r.objectives) && feasible,
                  what + ": " + std::to_string(r.evaluations) + " evaluations, " +
                      std::to_string(r.objectives.size()) + " points");
            check(r.objectives == r2.objectives && r.variables == r2.variables &&
                      r.evaluations == r2.evaluations,
                  what + ": the same seed, the same run");
            hash(r.objectives);
            hash(r.variables);
        }
    }
    // knobs: a word refused, unknown knobs ignored and reported, settings reported
    Counted c = zdt1();
    bool refused = false;
    try {
        sc::Knobs k{{"local", {true, "bfgs", 0.0}}};
        sc::run("scalar", c.P, 1e-6, false, 0.0, 10, 300, 1, k, {}, 0, {});
    } catch (const std::invalid_argument& e) {
        refused = std::string(e.what()) == "local = 'bfgs': sqp, nm, cmaes";
    }
    check(refused, "scalar: local = 'bfgs' refused with Python's message");
    sc::Knobs k{{"eta_c", {false, "", 15.0}}, {"sub_plan", {true, "gap", 0.0}}, {"scalarization", {true, "tch", 0.0}}};
    const sc::Result r = sc::run("scalar", c.P, 1e-6, false, 0.0, 10, 600, 1, k, {}, 0, {});
    bool gap = false, version = false;
    for (const auto& [key, v] : r.settings) {
        gap = gap || (key == "sub_plan" && v == "gap");
        version = version || (key == "method_version" && v == "4");
    }
    check(r.ignored == std::vector<std::string>{"eta_c"} && gap && version,
          "scalar: knobs taken, ignored and reported; method_version 4");
    // sqp_line_search (SQP-6 to SQP-9): powell by default, stop and accept
    // runs of their own, within the budget, reported
    for (const char* how : {"", "stop", "accept"}) {
        Counted cc = constrained();
        sc::Knobs ka;
        if (*how) ka = {{"sqp_line_search", {true, how, 0.0}}};
        const sc::Result ra = sc::run("eps_sqp", cc.P, 1e-3, false, 0.0, 20, 1500, 7, ka, {}, 0, {});
        const std::string want = *how ? how : "powell";
        bool reported = false;
        for (const auto& [key, v] : ra.settings) reported = reported || (key == "sqp_line_search" && v == want);
        bool feasible = !ra.objectives.empty();
        for (double v : ra.cv) feasible = feasible && v <= 0.0;
        check(reported && feasible && ra.evaluations == cc.calls() && ra.evaluations <= 1500,
              "eps_sqp with sqp_line_search = " + want + ": " + std::to_string(ra.evaluations) + " evaluations");
        hash(ra.objectives);
    }
}

// sum (j + 1) (x_j - 0.3)^2 on [-1, 1]^4, every point it is called at kept;
// constrained: g = 1 - x0 - x1 <= 0, active at the minimum: x0 - 0.3 = 2t,
// x1 - 0.3 = t (the weights 1 and 2), 0.6 + 3t = 1, f = 6 t^2 with t = 2/15
struct Single {
    sc::Problem P;
    std::shared_ptr<std::vector<Vec>> seen = std::make_shared<std::vector<Vec>>();
};

Single single_problem(bool constrained)
{
    Single c;
    c.P.lo.assign(4, -1.0);
    c.P.hi.assign(4, 1.0);
    c.P.m = 1;
    c.P.n_cons = constrained ? 1 : 0;
    c.P.evaluate = [seen = c.seen, constrained](const Vec& x, Vec& f, Vec& g) {
        seen->push_back(x);
        double s = 0.0;
        for (std::size_t j = 0; j < x.size(); ++j) s += static_cast<double>(j + 1) * (x[j] - 0.3) * (x[j] - 0.3);
        f[0] = s;
        if (constrained) g[0] = 1.0 - x[0] - x[1];
    };
    return c;
}

std::uint64_t g_single = 1469598103934665603ULL;    // FNV-1a of single()'s answers

void single()
{
    const double t = 2.0 / 15.0, f_con = 6.0 * t * t;
    for (const std::string& name : sc::single_names()) {
        for (int con = 0; con < 2; ++con) {
            Single a = single_problem(con == 1);
            const sc::Result r = sc::run(name, a.P, 1e-6, false, 0.0, 20, 4000, 3, {}, {}, 0, {});
            Single b = single_problem(con == 1);
            const sc::Result r2 = sc::run(name, b.P, 1e-6, false, 0.0, 20, 4000, 3, {}, {}, 0, {});
            const double want = con ? f_con : 0.0;
            bool at_min = !r.objectives.empty();
            double fbest = r.objectives.empty() ? 1e300 : r.objectives[0][0];
            for (std::size_t i = 0; i < r.objectives.size(); ++i)
                at_min = at_min && r.objectives[i][0] == fbest && r.cv[i] <= 0.0;
            const std::string what = name + (con ? " with g <= 0" : " on the weighted sphere");
            check(at_min && fbest - want < 1e-7 && fbest >= want - 1e-12 &&
                      r.evaluations == static_cast<long long>(a.seen->size()) && r.evaluations <= 4000,
                  what + ": f - f* = " + std::to_string(fbest - want) + " in " + std::to_string(r.evaluations) +
                      " evaluations, the answer the feasible points of least f");
            check(r.objectives == r2.objectives && r.variables == r2.variables && r.evaluations == r2.evaluations,
                  what + ": the same seed, the same run");
            // SINGLE-1: a smaller budget is the same run cut short
            Single c = single_problem(con == 1);
            const sc::Result r3 = sc::run(name, c.P, 1e-6, false, 0.0, 20, 150, 3, {}, {}, 0, {});
            const bool prefix = c.seen->size() <= a.seen->size() &&
                                std::equal(c.seen->begin(), c.seen->end(), a.seen->begin());
            check(prefix && r3.evaluations == std::min<long long>(150, r.evaluations),
                  what + ": at a budget of 150 the run cut short there");
            for (const sc::Rows* R : {&r.objectives, &r.variables, &r3.variables})
                for (const Vec& row : *R)
                    for (double v : row) {
                        std::uint64_t bits;
                        std::memcpy(&bits, &v, sizeof bits);
                        for (int k = 0; k < 8; ++k) {
                            g_single ^= (bits >> (8 * k)) & 0xFF;
                            g_single *= 1099511628211ULL;
                        }
                    }
        }
    }
    // refusals: two objectives, an infinite bound; knobs taken, others ignored and reported
    Counted z = zdt1();
    std::string why;
    try {
        sc::run("cmaes", z.P, 1e-6, false, 0.0, 10, 300, 1, {}, {}, 0, {});
    } catch (const std::invalid_argument& e) {
        why = e.what();
    }
    check(why == "cmaes is a single-objective algorithm: n_objs must be 1, got 2", "cmaes at two objectives: " + why);
    Single u = single_problem(false);
    u.P.hi[2] = std::numeric_limits<double>::infinity();
    why.clear();
    try {
        sc::run("nelder_mead", u.P, 1e-6, false, 0.0, 10, 300, 1, {}, {}, 0, {});
    } catch (const std::invalid_argument& e) {
        why = e.what();
    }
    check(why.find("every bound must be finite") != std::string::npos, "nelder_mead with an infinite bound: " + why);
    Single k = single_problem(false);
    const sc::Knobs kn{{"cma_sigma0", {false, "", 0.2}}, {"local", {true, "nm", 0.0}},
                       {"sqp_line_search", {true, "stop", 0.0}}};
    const sc::Result rk = sc::run("slsqp", k.P, 1e-6, false, 0.0, 10, 600, 1, kn, {}, 0, {});
    bool stop = false, sqp = false;
    for (const auto& [key, v] : rk.settings) {
        stop = stop || (key == "sqp_line_search" && v == "stop");
        sqp = sqp || (key == "local" && v == "sqp");
    }
    check(rk.ignored == std::vector<std::string>{"cma_sigma0", "local"} && stop && sqp,
          "slsqp: sqp_line_search taken, cma_sigma0 and local ignored and reported");
}

void polish()
{
    // an "algorithm": 60 random points of ZDT1, its answer the non-dominated ones
    Counted c = zdt1();
    std::mt19937 gen(5);
    std::vector<sc::Point> pts;
    for (int i = 0; i < 60; ++i) {
        sc::Point p;
        for (int j = 0; j < 4; ++j) p.x.push_back(mootation::rnd::uniform01(gen));
        p.f.assign(2, 0.0);
        c.P.evaluate(p.x, p.f, p.g);
        pts.push_back(p);
    }
    sc::Rows base, baseF;
    for (std::size_t a = 0; a < pts.size(); ++a) {
        bool dom = false;
        for (std::size_t b = 0; b < pts.size(); ++b)
            dom = dom || (pts[b].f[0] <= pts[a].f[0] && pts[b].f[1] <= pts[a].f[1] && pts[b].f != pts[a].f);
        if (!dom) {
            base.push_back(pts[a].x);
            baseF.push_back(pts[a].f);
        }
    }
    for (const std::string how : {"sqp", "nm"}) {
        *c.n = 0;
        const sc::PolishResult r = sc::polish(c.P, pts, base, 60, how, 5, 1e-6, 20, 400, {}, 0,
                                              1.4901161193847656e-08, 1e-14);
        bool covered = true;
        for (const Vec& b : baseF) {
            bool any = false;
            for (const Vec& a : r.objectives) any = any || (a[0] <= b[0] && a[1] <= b[1]);
            covered = covered && any;
        }
        check(covered && r.spent == c.calls() && r.spent <= 400 && r.chosen == 5,
              "polish " + how + ": the algorithm's answer weakly dominated, " + std::to_string(r.spent) +
                  " evaluations");
        hash(r.objectives);
    }
}

}   // namespace

int main()
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    eigen_random();
    models();
    methods();
    single();
    polish();
    std::printf("checksum %016llx\n", static_cast<unsigned long long>(g_hash));
    // the same on every platform of the CI (MSVC, g++, clang, macOS ARM)
    const std::uint64_t expected = 0x8f6c7458336771deULL;
    check(g_hash == expected, "bits: the checksum of every run's answer");
    std::printf("single checksum %016llx\n", static_cast<unsigned long long>(g_single));
    check(g_single == 0xad2a37d5e7d46f36ULL, "bits: the checksum of the single-objective runs");
    if (g_failed) {
        std::printf("%d check(s) failed\n", g_failed);
        return 1;
    }
    std::printf("all checks passed\n");
    return 0;
}

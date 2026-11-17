// SPDX-License-Identifier: Apache-2.0
// The move, the operator portfolio, the children's outcomes and the schedule
// of the strategies (task 31, layer 2; scalar/moves.hpp, scalar/portfolio.hpp,
// stop_rule.hpp STOP-7):
//   * step = user (the mutate hook) on basin_hopping, ils and pbh: a hook that
//     moves one coordinate moves only it — every point evaluated differs from
//     the start in that coordinate alone (the local search is one evaluation);
//   * a portfolio of one operator with op_select = fixed is the user move with
//     that operator, bit for bit (basin_hopping, ils, pbh; cluster_ga's
//     mutation; ga_sbx's and ssga's ga_mutation and ga_crossover); a user move
//     that repeats ils's box move is ils's default run, bit for bit;
//   * on_child (MOVE-4): every child's tag reaches its own call (the operators
//     write their numbers into the coordinates the f is read from, and the
//     local search keeps x), the calls add up to the children (cluster_ga,
//     pbh, basin_hopping, ga_sbx), the same calls in the same order on 1
//     thread and on 4, all of them on the run's thread; the taken ones are
//     cluster_ga's accepted;
//   * the 30.7 portfolio and the moves of before after the move to moves.hpp:
//     runs of 5169a71 (main before layer 2), bit for bit (kOld, written by the
//     report's driver layer2\old_runs.cpp on both);
//   * the schedule (STOP-7): bh_temperature_schedule = const, and geometric
//     with bh_temperature_end = bh_temperature, are the default run bit for
//     bit; geometric to 0.01 over max_steps = 20 ends at 0.01^(19/20);
//   * the refusals.

#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <mootation/scalar/scalar.hpp>

namespace {

namespace sc = mootation::scalar;
using Vec = std::vector<double>;

int g_failed = 0;

void check(bool ok, const std::string& what)
{
    std::printf("%s  %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++g_failed;
}

sc::Knob word(const std::string& w) { return {true, w, 0.0}; }
sc::Knob number(double v) { return {false, "", v}; }

std::string setting(const sc::Result& r, const std::string& key)
{
    for (const auto& kv : r.settings)
        if (kv.first == key) return kv.second;
    return "";
}

bool has_setting(const sc::Result& r, const std::string& key)
{
    for (const auto& kv : r.settings)
        if (kv.first == key) return true;
    return false;
}

bool same(const sc::Result& a, const sc::Result& b)
{
    return a.objectives == b.objectives && a.variables == b.variables && a.evaluations == b.evaluations &&
           a.improvements.size() == b.improvements.size();
}

unsigned long long hash_of(const Vec& x)
{
    unsigned long long h = 1469598103934665603ULL;
    for (double v : x) {
        unsigned char b[sizeof(double)];
        std::memcpy(b, &v, sizeof b);
        for (unsigned char c : b) h = (h ^ c) * 1099511628211ULL;
    }
    return h;
}

double u01(std::mt19937& g) { return mootation::rnd::uniform01(g); }

sc::Result run(const std::string& name, const sc::Problem& P, long long budget, const sc::Knobs& knobs,
               const sc::Start& start = sc::Start{}, unsigned seed = 7)
{
    return sc::run(name, P, 1e-6, false, 0.0, 20, budget, seed, knobs, {}, 0, start);
}

sc::Start at(const Vec& x)
{
    sc::Start s;
    s.has_x0 = true;
    s.x0 = x;
    return s;
}

double sphere_of(const Vec& x)
{
    double s = 0.0;
    for (double v : x) s += v * v;
    return s;
}

sc::Problem sphere(std::size_t n)
{
    sc::Problem P;
    P.lo.assign(n, -2.0);
    P.hi.assign(n, 2.0);
    P.m = 1;
    P.evaluate = [](const Vec& x, Vec& f, Vec&) { f.assign(1, sphere_of(x)); };
    return P;
}

// old_runs.cpp's problem: Rastrigin in 4 variables with the user's moves and operators
sc::Problem rastrigin4()
{
    sc::Problem P;
    P.lo.assign(4, -5.12);
    P.hi.assign(4, 5.12);
    P.m = 1;
    P.evaluate = [](const Vec& x, Vec& f, Vec&) {
        double s = 10.0 * static_cast<double>(x.size());
        for (double v : x) s += v * v - 10.0 * mootation::dm::cos(2.0 * 3.14159265358979323846 * v);
        f.assign(1, s);
    };
    P.mutate = [](Vec& x, std::mt19937& g) {
        const std::size_t j = (std::min)(x.size() - 1, static_cast<std::size_t>(u01(g) * static_cast<double>(x.size())));
        x[j] += 0.6 * (2.0 * u01(g) - 1.0);
    };
    P.perturb = [](Vec& x, double strength, std::mt19937& g) {
        for (double& v : x) v += strength * 10.24 * (2.0 * u01(g) - 1.0);
    };
    P.mutate_ops = {{"one", P.mutate}, {"all", [](Vec& x, std::mt19937& g) {
                                            for (double& v : x) v += 0.3 * (2.0 * u01(g) - 1.0);
                                        }}};
    auto uniform_x = [](const Vec& p1, const Vec& p2, Vec& c1, Vec& c2, std::mt19937& g) {
        c1 = p1;                                             // ga hands the hook empty vectors (GA-9)
        c2 = p2;
        for (std::size_t j = 0; j < p1.size(); ++j) {
            const bool first = u01(g) < 0.5;
            c1[j] = first ? p1[j] : p2[j];
            c2[j] = first ? p2[j] : p1[j];
        }
    };
    auto blend_x = [](const Vec& p1, const Vec& p2, Vec& c1, Vec& c2, std::mt19937& g) {
        c1 = p1;
        c2 = p2;
        const double t = u01(g);
        for (std::size_t j = 0; j < p1.size(); ++j) {
            c1[j] = t * p1[j] + (1.0 - t) * p2[j];
            c2[j] = (1.0 - t) * p1[j] + t * p2[j];
        }
    };
    P.crossover = uniform_x;
    P.crossover_ops = {{"uniform", uniform_x}, {"blend", blend_x}};
    return P;
}

const char* kNM = "{\"solver\":\"nelder_mead\",\"local_tol\":1e-8}";

// old_runs.cpp's runs on 5169a71 (main before layer 2): its case number,
// evaluations, the best f, hash_of(its x) (layer2\old_runs_5169a71.txt)
struct OldRun {
    int line;
    const char* name;
    sc::Knobs knobs;
    long long evaluations;
    double f;
    unsigned long long x_hash;
};

// THE TAGS: x[2] in [0, 1] is set to the crossover operator's number, x[3] in
// [0, 1] to the move's, and f = 1000 x[3] + 100 x[2] + x[0]^2 + x[1]^2 with
// x[0], x[1] in [-1, 1]: a child's f says which operators made it
sc::Problem tagged()
{
    sc::Problem P;
    P.lo = {-1.0, -1.0, 0.0, 0.0};
    P.hi = {1.0, 1.0, 1.0, 1.0};
    P.m = 1;
    P.evaluate = [](const Vec& x, Vec& f, Vec&) { f.assign(1, 1000.0 * x[3] + 100.0 * x[2] + x[0] * x[0] + x[1] * x[1]); };
    P.local_search = [](const Vec& x, sc::LocalContext&, const Vec&) {   // keeps x: the child is its start
        sc::LocalOutcome o;
        o.x = x;
        return o;
    };
    P.mutate_ops = {{"zero", [](Vec& x, std::mt19937& g) {
                         x[3] = 0.0;
                         x[0] += 0.2 * (2.0 * u01(g) - 1.0);
                     }},
                    {"one", [](Vec& x, std::mt19937& g) {
                         x[3] = 1.0;
                         x[1] += 0.2 * (2.0 * u01(g) - 1.0);
                     }}};
    P.crossover_ops = {{"x0", [](const Vec& p1, const Vec& p2, Vec& c1, Vec& c2, std::mt19937& g) {
                            c1 = p1;                                    // written whole (MOVE-6)
                            c2 = p2;
                            const double t = u01(g);
                            for (std::size_t j = 0; j < p1.size(); ++j) {
                                c1[j] = t * p1[j] + (1.0 - t) * p2[j];
                                c2[j] = (1.0 - t) * p1[j] + t * p2[j];
                            }
                            c1[2] = c2[2] = 0.0;
                        }},
                       {"x1", [](const Vec& p1, const Vec& p2, Vec& c1, Vec& c2, std::mt19937& g) {
                            c1 = p1;
                            c2 = p2;
                            for (std::size_t j = 0; j < p1.size(); ++j) {
                                const bool first = u01(g) < 0.5;
                                c1[j] = first ? p1[j] : p2[j];
                                c2[j] = first ? p2[j] : p1[j];
                            }
                            c1[2] = c2[2] = 1.0;
                        }}};
    return P;
}

struct Call {
    Vec parents_f;
    double child_f;
    long long crossover, mutation;
    bool taken;
    bool on_run_thread;

    bool operator==(const Call& o) const
    {
        return parents_f == o.parents_f && child_f == o.child_f && crossover == o.crossover && mutation == o.mutation &&
               taken == o.taken && on_run_thread == o.on_run_thread;
    }
};

// a run of the tagged problem with on_child recording its calls
struct Recorded {
    sc::Result r;
    std::vector<Call> calls;
};

Recorded recorded(const std::string& name, const sc::Knobs& knobs, long long budget)
{
    sc::Problem P = tagged();
    auto calls = std::make_shared<std::vector<Call>>();
    const std::thread::id me = std::this_thread::get_id();
    P.on_child = [calls, me](const Vec& parents_f, double child_f, const sc::ChildTag& t) {
        calls->push_back({parents_f, child_f, t.crossover, t.mutation, t.taken, std::this_thread::get_id() == me});
    };
    Recorded out;
    out.r = run(name, P, budget, knobs, at({0.5, -0.5, 0.5, 0.5}));
    out.calls = *calls;
    return out;
}

// each call's tags are its child's (TAGS), on the run's thread
bool tags_match(const std::vector<Call>& calls, bool crossed)
{
    for (const Call& c : calls) {
        const long long m = static_cast<long long>(std::floor(c.child_f / 1000.0));
        const long long x = static_cast<long long>(std::floor((c.child_f - 1000.0 * static_cast<double>(m)) / 100.0));
        if (m != c.mutation || (crossed && x != c.crossover) || !c.on_run_thread) return false;
    }
    return !calls.empty();
}

long long calls_of(const sc::Result& r, const char* kind)
{
    long long s = 0;
    for (const auto& row : mootation::json::parse(setting(r, "operators")).items)
        if (row.items[1].s == kind) s += row.items[2].i;
    return s;
}

long long taken_of(const std::vector<Call>& calls)
{
    long long s = 0;
    for (const Call& c : calls) s += c.taken ? 1 : 0;
    return s;
}

bool refused(const std::string& name, const sc::Problem& P, const sc::Knobs& knobs)
{
    try {
        run(name, P, 200, knobs, at(Vec(P.lo.size(), 0.5)));
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

}   // namespace

int main()
{
    // ── step = user: a hook that moves one coordinate moves only it ──────────
    {
        sc::Problem P = sphere(3);
        auto seen = std::make_shared<std::vector<Vec>>();
        P.evaluate = [seen](const Vec& x, Vec& f, Vec&) {
            seen->push_back(x);
            f.assign(1, sphere_of(x));
        };
        P.mutate = [](Vec& x, std::mt19937& g) { x[0] += 0.5 * (2.0 * u01(g) - 1.0); };
        const Vec x0{1.0, 1.5, -0.5};
        const sc::Knob nm = word("{\"solver\":\"nelder_mead\"}");
        for (const char* name : {"basin_hopping", "ils", "pbh"}) {
            seen->clear();
            sc::Knobs kn{{"inner", nm}, {"step", word("user")}, {"start_budget", number(1)}};
            if (std::string(name) == "pbh") kn.push_back({"pbh_K", number(1)});
            const sc::Result r = run(name, P, 60, kn, at(x0));
            bool only = seen->size() > 1, moved = false;
            for (const Vec& x : *seen) {
                only = only && x[1] == x0[1] && x[2] == x0[2];
                moved = moved || x[0] != x0[0];
            }
            check(only && moved && setting(r, "step") == "user" && !has_setting(r, "bh_step_end") &&
                      !has_setting(r, "bh_step") && !has_setting(r, "ils_step") && !has_setting(r, "pbh_step"),
                  std::string(name) + ", step = user: " + std::to_string(seen->size()) +
                      " points, every one moved in x[0] alone; no step knobs in the settings (MOVE-1)");
        }
    }

    const sc::Problem R = rastrigin4();
    const sc::Start from = at({1.5, -2.0, 0.5, 3.0});
    const sc::Knob nm = word(kNM);

    // ── one operator, op_select = fixed: the user move with it, bit for bit ──
    {
        sc::Problem U = R;
        U.mutate_ops = {{"one", R.mutate}};
        U.crossover_ops = {{"uniform", R.crossover}};
        struct Pair {
            const char* name;
            sc::Knobs user, ops;
        };
        const std::vector<Pair> pairs = {
            {"basin_hopping", {{"inner", nm}, {"step", word("user")}}, {{"inner", nm}, {"step", word("portfolio")}}},
            {"ils", {{"inner", nm}, {"step", word("user")}, {"ils_accept", word("lsmc")}},
             {{"inner", nm}, {"step", word("portfolio")}, {"ils_accept", word("lsmc")}}},
            {"pbh", {{"inner", nm}, {"pbh_K", number(3)}, {"step", word("user")}},
             {{"inner", nm}, {"pbh_K", number(3)}, {"step", word("portfolio")}, {"op_select", word("fixed")}}},
            {"cluster_ga", {{"inner", nm}, {"cga_pop", number(4)}, {"mutation", word("user")}, {"cga_mut_rate", number(0.5)}},
             {{"inner", nm}, {"cga_pop", number(4)}, {"mutation", word("ops")}, {"cga_mut_rate", number(0.5)}}},
            {"ga_sbx", {{"np", number(20)}, {"ga_mutation", word("user")}, {"ga_crossover", word("user")}},
             {{"np", number(20)}, {"ga_mutation", word("ops")}, {"ga_crossover", word("ops")}}},
            {"ssga", {{"np", number(20)}, {"ga_mutation", word("user")}}, {{"np", number(20)}, {"ga_mutation", word("ops")}}},
        };
        for (const Pair& p : pairs) {
            const sc::Result a = run(p.name, U, 3000, p.user, from), b = run(p.name, U, 3000, p.ops, from);
            check(same(a, b) && has_setting(b, "operators"),
                  std::string(p.name) + ": one operator, op_select = fixed = the user move with it, bit for bit");
        }
        // ils's box move by the user hook: the default run, bit for bit
        sc::Problem B = R;
        B.mutate = [](Vec& x, std::mt19937& g) {
            for (double& v : x) v += (2.0 * u01(g) - 1.0) * 0.1 * 10.24;
        };
        const sc::Result d = run("ils", B, 3000, {{"inner", nm}}, from);
        const sc::Result u = run("ils", B, 3000, {{"inner", nm}, {"step", word("user")}}, from);
        check(same(d, u), "ils: step = user with a hook that is the box move = step = box, bit for bit");
        const sc::Result bx = run("basin_hopping", R, 3000, {{"inner", nm}, {"step", word("box")}}, from);
        const sc::Result b0 = run("basin_hopping", R, 3000, {{"inner", nm}}, from);
        check(same(bx, b0) && setting(bx, "bh_step_end") == setting(b0, "bh_step_end"),
              "basin_hopping: step = box is the default run, bit for bit");
    }

    // ── on_child: tags, counts, 1 thread = 4 threads ────────────────────────
    {
        const sc::Knob ls = word("{\"solver\":\"local_search\"}");
        auto cga = [&](int threads) {
            return recorded("cluster_ga", {{"inner", ls}, {"cga_pop", number(6)}, {"crossover", word("ops")},
                                           {"mutation", word("ops")}, {"cga_mut_rate", number(1.0)},
                                           {"op_select", word("adaptive_pursuit")}, {"cga_maxgen", number(6)},
                                           {"threads", number(threads)}},
                            1000000);
        };
        const Recorded c1 = cga(1), c4 = cga(4);
        check(c1.calls == c4.calls && same(c1.r, c4.r), "cluster_ga: on_child, 1 thread = 4 threads (" +
                                                             std::to_string(c1.calls.size()) + " calls, in order)");
        check(tags_match(c4.calls, true), "cluster_ga, 4 threads: every child's tags reach its own on_child");
        check(c4.calls.size() == 36 && static_cast<long long>(c4.calls.size()) == calls_of(c4.r, "mutation") &&
                  static_cast<long long>(c4.calls.size()) == calls_of(c4.r, "crossover") &&
                  std::to_string(c4.calls.size()) == setting(c4.r, "crossings") &&
                  std::to_string(taken_of(c4.calls)) == setting(c4.r, "accepted"),
              "cluster_ga: the calls add up to the children (6 generations of 6), the crossings, the operators' "
              "calls; taken = accepted (" + setting(c4.r, "accepted") + ")");

        auto pbh = [&](int threads) {
            return recorded("pbh", {{"inner", ls}, {"pbh_K", number(4)}, {"step", word("portfolio")},
                                    {"op_select", word("probability_matching")}, {"pbh_maxstep", number(5)},
                                    {"threads", number(threads)}},
                            1000000);
        };
        const Recorded p1 = pbh(1), p4 = pbh(4);
        check(p1.calls == p4.calls && same(p1.r, p4.r) && tags_match(p4.calls, false) && p4.calls.size() == 20 &&
                  std::to_string(p4.calls.size()) == setting(p4.r, "steps") &&
                  static_cast<long long>(p4.calls.size()) == calls_of(p4.r, "mutation") &&
                  std::to_string(taken_of(p4.calls)) == setting(p4.r, "accepted"),
              "pbh: on_child on 1 thread = 4 threads, tags its children's, 20 calls = steps, taken = accepted");

        const Recorded bh = recorded("basin_hopping", {{"inner", ls}, {"step", word("portfolio")},
                                                       {"op_select", word("adaptive_pursuit")}, {"max_steps", number(30)}},
                                     1000000);
        check(tags_match(bh.calls, false) && bh.calls.size() == 30 && setting(bh.r, "steps") == "30" &&
                  std::to_string(taken_of(bh.calls)) == setting(bh.r, "accepted") &&
                  calls_of(bh.r, "mutation") == 30,
              "basin_hopping: on_child once a step (30), tags its children's, taken = accepted");

        const Recorded ga = recorded("ga_sbx", {{"np", number(10)}, {"ga_mutation", word("ops")},
                                                {"ga_crossover", word("ops")}, {"op_select", word("adaptive_pursuit")}},
                                     310);
        // 10 + 30 generations of 10 told; the 31st generation's 10 children made (MOVE-3), not evaluated
        check(tags_match(ga.calls, true) && ga.calls.size() == 300 && calls_of(ga.r, "mutation") == 310 &&
                  calls_of(ga.r, "crossover") == 310,
              "ga_sbx: on_child after tell, tags its children's, 300 calls; the operators made 310 (" +
                  std::to_string(ga.calls.size()) + ")");
    }

    // ── the moves of before and the 30.7 portfolio: 5169a71's runs ─────────
    {
        const std::vector<OldRun> old = {
            {0, "basin_hopping", {{"inner", nm}}, 4000, 0x1.3e62f6a286c86p+3, 0x36c708b215cb77f6ULL},
            {1, "basin_hopping", {{"inner", nm}, {"perturb", word("user")}}, 4000, 0x1.3e62f6a286c86p+3, 0x8df2fe6443712cd9ULL},
            {2, "basin_hopping", {{"inner", nm}, {"ladder_min", number(0.05)}, {"ladder_max", number(0.2)}, {"ladder_step", number(0.05)}, {"bh_accept", word("threshold")}}, 4000, 0x1.7e105f634bad6p+2, 0x429f8839232704c4ULL},
            {3, "basin_hopping", {{"inner", nm}, {"step_units", word("absolute")}, {"bh_step", number(0.5)}, {"bh_temperature", number(3.0)}}, 4000, 0x1.3e62f6a286da0p+3, 0x03d9a6cbde643021ULL},
            {4, "ils", {{"inner", nm}}, 4000, 0x1.fd6b474b7d9d8p+0, 0x2e44904759b7845dULL},
            {5, "ils", {{"inner", nm}, {"ils_accept", word("lsmc")}, {"ils_temperature", number(2.0)}}, 4000, 0x1.fd6b474b7d9d8p+0, 0x2e44904759b7845dULL},
            {6, "ils", {{"inner", nm}, {"ils_accept", word("restart")}, {"ils_restart", number(3)}}, 4000, 0x1.fd6b474b7d9d8p+0, 0x2e44904759b7845dULL},
            {7, "ils", {{"inner", nm}, {"perturb", word("user")}, {"ils_accept", word("rw")}}, 4000, 0x1.bdbd933860776p+3, 0x91c85ad7379f3bfaULL},
            {8, "pbh", {{"inner", nm}, {"pbh_K", number(4)}}, 4000, 0x1.9de6ded7fe1a9p+3, 0x977a6733b25e8fe5ULL},
            {9, "pbh", {{"inner", nm}, {"pbh_K", number(3)}, {"perturb", word("user")}, {"pbh_distance", word("euclid")}, {"pbh_dcut", number(0.5)}}, 4000, 0x1.3e62cc477f8dbp+3, 0xa7057b89b14f2327ULL},
            {10, "pbh", {{"inner", nm}, {"pbh_K", number(3)}, {"step_units", word("absolute")}, {"pbh_step", number(0.4)}, {"threads", number(3)}}, 4000, 0x1.1e8bed5afd337p+4, 0xdebb254182d32ec0ULL},
            {11, "cluster_ga", {{"inner", nm}, {"cga_pop", number(5)}}, 3996, 0x1.1e8c424224614p+3, 0x50ede38fa1cd8bfcULL},
            {12, "cluster_ga", {{"inner", nm}, {"cga_pop", number(5)}, {"mutation", word("user")}, {"cga_mut_rate", number(0.5)}}, 4000, 0x1.1e8c42422464ep+3, 0x0f2edcc660a8a554ULL},
            {13, "cluster_ga", {{"inner", nm}, {"cga_pop", number(5)}, {"crossover", word("ops")}, {"mutation", word("ops")}, {"cga_mut_rate", number(0.5)}, {"op_select", word("adaptive_pursuit")}}, 4000, 0x1.3e62f6a286f26p+2, 0x3ad4a7e3b3815dfbULL},
            {14, "cluster_ga", {{"inner", nm}, {"cga_pop", number(5)}, {"crossover", word("ops")}, {"mutation", word("ops")}, {"cga_mut_rate", number(0.7)}, {"op_select", word("probability_matching")}, {"op_pmin", number(0.1)}}, 4000, 0x1.bdbdc824107bcp+2, 0x11725eca5a315531ULL},
            {15, "cluster_ga", {{"inner", nm}, {"cga_pop", number(5)}, {"mutation", word("ops")}, {"cga_mut_rate", number(1.0)}, {"threads", number(3)}, {"op_select", word("adaptive_pursuit")}, {"op_alpha", number(0.5)}, {"op_beta", number(0.6)}}, 4000, 0x1.bdc280aad5e12p+2, 0x45b42648c5bf998cULL},
            {16, "cluster_ga", {{"inner", nm}, {"cga_pop", number(5)}, {"cga_replace", word("multi_dynamic")}, {"mutation", word("user")}}, 3999, 0x1.7e105f634bc46p+2, 0x9c06941d49f3a94aULL},
            {17, "cluster_ga", {{"inner", nm}, {"cga_pop", number(5)}, {"cga_replace", word("afpo")}}, 3999, 0x1.3e62f6a286ce2p+3, 0xee2760e770650dcbULL},
            {18, "cluster_ga", {{"inner", nm}, {"cga_pop", number(3)}, {"cga_islands", number(3)}, {"mutation", word("ops")}, {"cga_mut_rate", number(0.5)}, {"op_select", word("adaptive_pursuit")}}, 3999, 0x1.bdbd933860854p+3, 0x04ffd471968a3c35ULL},
            {19, "cluster_ga", {{"inner", nm}, {"cga_pop", number(4)}, {"cga_init", word("around_x0")}, {"cga_init_moves", number(2)}, {"mutation", word("ops")}}, 3999, 0x1.1e8c2d14a0b80p+4, 0x98e56dde01bb5112ULL},
            {20, "cluster_ga", {{"inner", nm}, {"cga_pop", number(1)}, {"cga_de", number(0.0)}}, 4000, 0x1.fd6b474b7d9d8p+0, 0x2e44904759b7845dULL},
            {21, "ga_sbx", {{"np", number(20)}}, 4000, 0x1.b3ef11696a000p-10, 0x7bf6649f2e57c5d2ULL},
            {22, "ga_sbx", {{"np", number(20)}, {"ga_mutation", word("user")}, {"ga_crossover", word("user")}}, 4000, 0x1.3f121f39e338cp+2, 0x530d1ded170ba161ULL},
            {23, "ssga", {{"np", number(20)}}, 4000, 0x1.0fd71a8800000p-13, 0xf6fa20516b435eb8ULL},
            {24, "ssga", {{"np", number(20)}, {"ga_mutation", word("user")}}, 4000, 0x1.3edb23a2c52b0p+2, 0x1d75cb9b0ed56b45ULL},
        };
        for (const OldRun& o : old) {
            const sc::Result r = run(o.name, R, 4000, o.knobs, from);
            const double f = r.objectives.empty() ? 0.0 : r.objectives[0][0];
            check(r.evaluations == o.evaluations && f == o.f && hash_of(r.variables[0]) == o.x_hash,
                  "old_runs case " + std::to_string(o.line) + " (" + o.name + "): 5169a71's run, bit for bit");
        }
    }

    // ── MOVE-6: the crossover hook gets c1, c2 of n values in every strategy ─
    {
        auto sizes_ok = std::make_shared<bool>(true);
        auto indexing = [sizes_ok](const Vec& p1, const Vec& p2, Vec& c1, Vec& c2, std::mt19937& g) {
            *sizes_ok = *sizes_ok && c1.size() == p1.size() && c2.size() == p1.size();
            if (c1.size() != p1.size() || c2.size() != p1.size()) throw std::runtime_error("c1, c2 not of n values");
            for (std::size_t j = 0; j < p1.size(); ++j) {          // indexes them, as cluster_ga's hooks may
                const bool first = u01(g) < 0.5;
                c1[j] = first ? p1[j] : p2[j];
                c2[j] = first ? p2[j] : p1[j];
            }
        };
        sc::Problem I = R;
        I.crossover = indexing;
        for (const char* name : {"ga_sbx", "ssga"}) {
            const sc::Knobs kn{{"np", number(20)}, {"ga_crossover", word("user")}};
            const sc::Result a = run(name, I, 3000, kn, from), b = run(name, R, 3000, kn, from);
            check(*sizes_ok && same(a, b),
                  std::string(name) + ": ga_crossover = user hands c1, c2 of n values; a hook that indexes them = "
                                      "one that assigns them whole, bit for bit");
        }
        const sc::Result c = run("cluster_ga", I, 3000, {{"inner", nm}, {"cga_pop", number(4)}, {"crossover", word("user")}}, from);
        check(*sizes_ok && c.evaluations > 0, "cluster_ga: the same hook gets c1, c2 of n values");
    }

    // ── the schedule of bh_temperature (STOP-7) ────────────────────────────
    {
        const sc::Knobs base{{"inner", nm}, {"bh_temperature", number(2.0)}};
        auto with = [&](sc::Knobs k) {
            sc::Knobs all = base;
            all.insert(all.end(), k.begin(), k.end());
            return run("basin_hopping", R, 3000, all, from);
        };
        const sc::Result d = with({});
        const sc::Result c = with({{"bh_temperature_schedule", word("const")}});
        const sc::Result g = with({{"bh_temperature_schedule", word("geometric")}, {"bh_temperature_end", number(2.0)}});
        check(same(d, c) && same(d, g) && setting(d, "bh_step_end") == setting(g, "bh_step_end"),
              "bh_temperature_schedule = const, and geometric to bh_temperature itself: the default run, bit for bit");
        const sc::Result a = run("basin_hopping", R, 1000000000LL,
                                 {{"inner", nm}, {"bh_temperature_schedule", word("geometric")},
                                  {"bh_temperature_end", number(0.01)}, {"max_steps", number(20)}},
                                 from);
        const double want = mootation::dm::pow(0.01, 19.0 / 20.0), got = std::stod(setting(a, "bh_temperature_reached"));
        check(setting(a, "steps") == "20" && std::fabs(got - want) <= 1e-5 * want,
              "geometric 1 -> 0.01 over max_steps = 20: the last step's T = 0.01^(19/20) = " +
                  setting(a, "bh_temperature_reached"));
    }

    // ── the refusals ────────────────────────────────────────────────────────
    {
        sc::Problem bare = sphere(3);
        check(refused("basin_hopping", bare, {{"inner", nm}, {"step", word("user")}}), "step = user without mutate refused");
        check(refused("pbh", bare, {{"inner", nm}, {"step", word("portfolio")}}),
              "step = portfolio without mutate_ops refused");
        check(refused("ils", R, {{"inner", nm}, {"step", word("user")}, {"perturb", word("user")}}),
              "perturb = user beside step = user refused (two moves)");
        check(refused("basin_hopping", R, {{"inner", nm}, {"step", word("user")}, {"ladder_max", number(0.2)}}),
              "the ladder with step = user refused (MOVE-1)");
        check(refused("ils", R, {{"inner", nm}, {"ils_temperature_schedule", word("geometric")}}),
              "ils_temperature_schedule with ils_accept = better refused (STOP-7)");
        check(refused("basin_hopping", R, {{"inner", nm}, {"step", word("portfolio")}, {"op_select", word("adaptive_pursuit")},
                                           {"op_pmin", number(0.5)}}),
              "op_pmin = 1/K refused (Thierens, footnote 2)");
        check(refused("ga_sbx", bare, {{"ga_mutation", word("ops")}}), "ga_mutation = ops without mutate_ops refused");
    }

    std::printf("%s\n", g_failed ? "FAILED" : "all passed");
    return g_failed ? 1 : 0;
}

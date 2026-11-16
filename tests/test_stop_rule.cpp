// SPDX-License-Identifier: Apache-2.0
// The ledger, the stop rule and the clocks of the strategies (task 31, layer
// 1; scalar/stop_rule.hpp, scalar/clocks.hpp):
//   * 30.10: cluster_ga whose inner solver is a local_search hook declaring
//     cost c and calling nothing ends after ceil(budget / c) searches — on 1
//     thread the hook is called exactly that often (STOP-5), on 4 threads
//     the same run is merged; settings: stopped_by = budget, spend;
//   * stall (STOP-3): a flat problem with an endless budget ends after stall
//     steps without a better best (basin_hopping, cluster_ga: cga_stall of
//     30.13 is stall a generation), max_steps after its steps;
//   * the knobs off: stall = 0, max_steps = 0, budget_units = budget are the
//     default run, bit for bit, and the default settings have no new rows;
//     without a gradient budget_units = evaluations is the same run;
//   * budget_units (STOP-2): budget, the default since method version 6,
//     keeps eleven strategies with lbfgsb inside within their budget in
//     used()'s units on a problem with a gradient; evaluations gives
//     cc2d22c's runs (method version 5) of the same, bit for bit (kOld);
//   * STOP-6: coevolution on the user's restrict_to counts the subproblem's
//     evaluations (outside) and goes on to its budget; derived() keeps the
//     earlier deadline and the clock;
//   * the clocks (30.12): with time_clock = cpu a problem that sleeps is not
//     cut by max_seconds (the wall clock cuts it), a busy one ends near
//     max_seconds of processor time; the settings have time_clock,
//     wall_seconds, cpu_seconds; a run within a run with a deadline keeps
//     its clock.
// With the argument "cpu30": time_clock = cpu, max_seconds = 30 on one
// thread, a problem half asleep — the run ends at ~30 s of processor time,
// ~60 s of the wall clock (the assignment's check; minutes, not in ctest).

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <mootation/scalar/scalar.hpp>

#if defined(_WIN32)                   // the declarations of clocks.hpp beside windows.h's (CLOCK-3)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

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

sc::Problem flat(std::size_t n)
{
    sc::Problem P = sphere(n);
    P.evaluate = [](const Vec&, Vec& f, Vec&) { f.assign(1, 1.0); };
    return P;
}

// Rosenbrock with its gradient
sc::Problem rosenbrock(std::size_t n)
{
    sc::Problem P;
    P.lo.assign(n, -2.0);
    P.hi.assign(n, 2.0);
    P.m = 1;
    P.evaluate = [](const Vec& x, Vec& f, Vec&) {
        double s = 0.0;
        for (std::size_t j = 0; j + 1 < x.size(); ++j)
            s += 100.0 * (x[j + 1] - x[j] * x[j]) * (x[j + 1] - x[j] * x[j]) + (1.0 - x[j]) * (1.0 - x[j]);
        f.assign(1, s);
    };
    P.gradient = [](const Vec& x, Vec& gf, sc::Rows&) {
        gf.assign(x.size(), 0.0);
        for (std::size_t j = 0; j + 1 < x.size(); ++j) {
            const double d = x[j + 1] - x[j] * x[j];
            gf[j] += -400.0 * x[j] * d - 2.0 * (1.0 - x[j]);
            gf[j + 1] += 200.0 * d;
        }
    };
    return P;
}

// The strategies of several rounds with a gradient inside, on rosenbrock(4)
// with its gradient, budget kCaseBudget, seed 7 (STOP-2's change)
struct Case {
    const char* name;
    sc::Knobs knobs;
};

constexpr long long kCaseBudget = 3000;

std::vector<Case> cases()
{
    const sc::Knob lb = word("{\"solver\":\"lbfgsb\"}");
    return {{"basin_hopping", {{"inner", lb}}},
            {"ils", {{"inner", lb}}},
            {"mlsl", {{"inner", lb}, {"mlsl_n", number(20)}}},
            {"cluster_ga", {{"inner", lb}, {"cga_pop", number(4)}}},
            {"pbh", {{"inner", lb}, {"pbh_K", number(3)}}},
            {"halving", {{"inner", lb}, {"n_starts", number(8)}}},
            {"memetic", {{"inner", word("{\"solver\":\"de\"}")}, {"local_solver", lb}, {"memetic_every", number(500)}}},
            {"tiktak", {{"inner", lb}}},
            {"coevolution", {{"inner", lb}, {"cc_window", number(2)}}},
            {"lns", {{"inner", lb}, {"lns_size", number(2)}}},
            {"multistart", {{"inner", lb}, {"n_starts", number(4)}}}};
}

// FNV-1a of the bytes of x
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

// cases()' runs on cc2d22c (method version 5), the same knobs: evaluations,
// gradient calls, the best f, hash_of(its x) (written by the driver of the
// task's report, layer1\old_runs.cpp, built on cc2d22c)
struct OldRun {
    long long evaluations, grad_calls;
    double f;
    unsigned long long x_hash;
};

const OldRun kOld[] = {
    {3000, 2977, 0x1.57d93f1e2db00p-60, 0x583cb6861fb57308ULL},   // basin_hopping
    {3000, 2977, 0x1.68e4a69e40c00p-62, 0x1e1aada6943a5cabULL},   // ils
    {2073, 996, 0x1.06060b133a05bp-50, 0x0d1697d30946b208ULL},    // mlsl
    {2997, 2960, 0x1.f259dab9b6200p-62, 0xb026e5deecb1bb27ULL},   // cluster_ga
    {3000, 2967, 0x1.00e5202b43000p-64, 0xd8392be02036e6ecULL},   // pbh
    {351, 332, 0x1.18fd76747a561p-52, 0xd3044cf3def3830fULL},     // halving
    {3000, 58, 0x1.87b1ba38d15b6p-50, 0x1f2994f6d17534e2ULL},     // memetic
    {393, 294, 0x1.718feba288a00p-61, 0x2dc15374aa4e8edaULL},     // tiktak
    {2992, 2990, 0x1.d9d37479289e5p+1, 0xd33bdd948a14191dULL},    // coevolution
    {3000, 2938, 0x1.86bb87d4b6e64p-47, 0x43302588f09c974cULL},   // lns
    {166, 147, 0x1.392688d7eaf85p-47, 0xe4820e6704b3d832ULL},     // multistart
};

// processor time spent on purpose: about `us` microseconds of arithmetic
// (spun on the steady clock: Windows' processor time moves in ticks of ~15.6 ms)
double busy(double us)
{
    const double t0 = sc::clock_seconds();
    double s = 0.0;
    while ((sc::clock_seconds() - t0) * 1e6 < us)
        for (int k = 0; k < 200; ++k) s += std::sqrt(static_cast<double>(k) + s * 1e-9);
    return s;
}

bool same(const sc::Result& a, const sc::Result& b)
{
    return a.objectives == b.objectives && a.variables == b.variables && a.evaluations == b.evaluations &&
           a.grad_calls == b.grad_calls && a.improvements.size() == b.improvements.size();
}

sc::Result run(const std::string& name, const sc::Problem& P, long long budget, const sc::Knobs& knobs,
               unsigned seed = 7, const sc::Start& start = sc::Start{})
{
    return sc::run(name, P, 1e-6, false, 0.0, 20, budget, seed, knobs, {}, 0, start);
}

// a start at x: the first local search starts there, no plan evaluated before it
sc::Start at(const Vec& x)
{
    sc::Start s;
    s.has_x0 = true;
    s.x0 = x;
    return s;
}

const std::string kNM = "{\"solver\":\"nelder_mead\"}";
const std::string kULS = "{\"solver\":\"local_search\"}";

void cpu30()
{
    sc::Problem P = sphere(4);
    P.evaluate = [](const Vec& x, Vec& f, Vec&) {          // half busy, half asleep
        busy(5000.0);
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        f.assign(1, sphere_of(x));
    };
    const sc::Result r = run("basin_hopping", P, 1000000000LL,
                             {{"inner", word(kNM)}, {"time_clock", word("cpu")}, {"max_seconds", number(30.0)}});
    std::printf("cpu30: wall_seconds %s, cpu_seconds %s, evaluations %lld, stopped_by %s, time_up %s\n",
                setting(r, "wall_seconds").c_str(), setting(r, "cpu_seconds").c_str(), r.evaluations,
                r.stopped_by.c_str(), setting(r, "time_up").c_str());
    check(r.stopped_by == "deadline" && r.cpu_seconds >= 30.0 && r.cpu_seconds < 31.0,
          "cpu30: ends at ~30 s of processor time");
    check(r.seconds > 1.5 * r.cpu_seconds, "cpu30: the wall clock went on past it (the sleep)");
}

}   // namespace

int main(int argc, char** argv)
{
    if (argc > 1 && std::strcmp(argv[1], "cpu30") == 0) {
        cpu30();
        std::printf("%s\n", g_failed ? "FAILED" : "all passed");
        return g_failed ? 1 : 0;
    }

    // 30.10: the local_search hook's declared cost ends cluster_ga
    {
        const double c = 7.0;
        const long long B = 100, want = 15;                            // ceil(100 / 7)
        auto calls = std::make_shared<std::atomic<long long>>(0);
        sc::Problem P = sphere(3);
        P.local_search = [calls, c](const Vec& x, sc::LocalContext&, const Vec&) {
            ++*calls;
            sc::LocalOutcome o;
            o.x = x;
            o.has_f = true;
            o.f = sphere_of(x);
            o.cost = c;
            return o;
        };
        const sc::Start x0 = at({0.5, 0.5, 0.5});                    // no plan: the hook's cost alone
        const sc::Knobs k1{{"inner", word(kULS)}, {"cga_pop", number(4)}};
        const sc::Result r1 = run("cluster_ga", P, B, k1, 7, x0);
        check(r1.hook_calls == want && calls->load() == want,
              "cluster_ga, local_search of cost 7, budget 100: 15 searches, 15 calls on 1 thread (got " +
                  std::to_string(r1.hook_calls) + ", " + std::to_string(calls->load()) + ")");
        check(r1.evaluations == 0 && r1.hook_spent == 105.0 && r1.used == 105.0, "... no evaluation, 105 spent");
        check(r1.stopped_by == "budget" && setting(r1, "stopped_by") == "budget", "... stopped_by budget");
        check(setting(r1, "spend").find("\"hook_cost\":105") != std::string::npos,
              "... the spend row: " + setting(r1, "spend"));
        calls->store(0);
        sc::Knobs k4 = k1;
        k4.push_back({"threads", number(4)});
        const sc::Result r4 = run("cluster_ga", P, B, k4, 7, x0);
        check(same(r1, r4) && r4.hook_calls == want && r4.hook_spent == 105.0,
              "... 4 threads: the same run, 15 searches merged (" + std::to_string(calls->load()) + " made)");
        for (const long long b : {1LL, 6LL, 7LL, 8LL, 50LL}) {
            const sc::Result rb = run("cluster_ga", P, b, k1, 7, x0);
            const long long w = (b + 6) / 7;
            check(rb.hook_calls == w, "... budget " + std::to_string(b) + ": " + std::to_string(w) + " searches");
        }
        const sc::Result rbh = run("basin_hopping", P, B, {{"inner", word(kULS)}}, 7, x0);
        check(rbh.hook_calls == want && rbh.stopped_by == "budget", "basin_hopping on the same hook: 15 searches");
        const sc::Result rpb = run("pbh", P, B, {{"inner", word(kULS)}, {"pbh_K", number(3)}}, 7, x0);
        check(rpb.hook_calls == want, "pbh on the same hook: 15 searches (got " + std::to_string(rpb.hook_calls) + ")");
    }

    // stall and max_steps (STOP-3): a flat problem, an endless budget
    {
        const long long endless = 1000000000000LL;
        const sc::Result a = run("basin_hopping", flat(2), endless, {{"inner", word(kNM)}, {"stall", number(5)}});
        check(a.stopped_by == "stall" && setting(a, "steps") == "5",
              "basin_hopping, stall = 5 on a flat f: 5 steps, stopped_by stall (" + setting(a, "steps") + ", " +
                  a.stopped_by + ")");
        const sc::Result g = run("cluster_ga", flat(2), endless,
                                 {{"inner", word(kNM)}, {"cga_pop", number(4)}, {"stall", number(3)}});
        check(g.stopped_by == "stall" && setting(g, "generations") == "3",
              "cluster_ga, stall = 3 (cga_stall): 3 generations (" + setting(g, "generations") + ")");
        const sc::Result m = run("mlsl", flat(2), endless, {{"inner", word(kNM)}, {"stall", number(2)}});
        check(m.stopped_by == "stall" && setting(m, "iterations") == "3",
              "mlsl, stall = 2: the first iteration's record, then 2 (" + setting(m, "iterations") + ")");
        const sc::Result s = run("ils", sphere(2), endless, {{"inner", word(kNM)}, {"max_steps", number(4)}});
        check(s.stopped_by == "steps" && setting(s, "steps") == "4", "ils, max_steps = 4: 4 steps");
        const sc::Result p = run("pbh", flat(2), endless, {{"inner", word(kNM)}, {"stall", number(4)}});
        check(p.stopped_by == "stall" && setting(p, "iterations") == "4", "pbh, stall = 4: 4 generations");
    }

    // the knobs off: the default run, bit for bit; no new settings rows by default
    {
        const sc::Knobs base{{"inner", word(kNM)}};
        for (const char* name : {"basin_hopping", "ils", "mlsl", "cluster_ga", "pbh", "multistart", "memetic"}) {
            sc::Knobs kb = base;
            if (std::string(name) == "memetic") kb.push_back({"local_solver", word(kNM)});
            const sc::Result d = run(name, sphere(3), 3000, kb);
            sc::Knobs off = kb;
            off.push_back({"stall", number(0)});
            off.push_back({"max_steps", number(0)});
            off.push_back({"budget_units", word("budget")});
            const sc::Result o = run(name, sphere(3), 3000, off);
            check(same(d, o), std::string(name) + ": stall = 0, max_steps = 0, budget_units = budget: bit for bit");
            sc::Knobs ev_count = kb;
            ev_count.push_back({"budget_units", word("evaluations")});
            check(same(d, run(name, sphere(3), 3000, ev_count)),
                  std::string(name) + ": without a gradient budget_units = evaluations is the same run");
            check(!has_setting(d, "stopped_by") && !has_setting(d, "spend") && has_setting(o, "spend"),
                  std::string(name) + ": the default settings have no stopped_by, spend; asked, they do");
        }
    }

    // budget_units (STOP-2): budget (the default, method version 6) keeps a run with a gradient within its
    // budget in used()'s units; evaluations is cc2d22c's run (method version 5), bit for bit (kOld)
    {
        for (const Case& c : cases()) {
            const sc::Result b = run(c.name, rosenbrock(4), kCaseBudget, c.knobs);
            check(b.used <= static_cast<double>(kCaseBudget) && b.grad_calls > 0,
                  std::string(c.name) + ", budget_units = budget (default): used " + std::to_string(b.used) +
                      " within " + std::to_string(kCaseBudget));
        }
        const std::vector<Case> cs = cases();
        check(cs.size() == sizeof(kOld) / sizeof(kOld[0]), "one stored line of cc2d22c per case");
        for (std::size_t q = 0; q < cs.size() && q < sizeof(kOld) / sizeof(kOld[0]); ++q) {
            sc::Knobs kv = cs[q].knobs;
            kv.push_back({"budget_units", word("evaluations")});
            const sc::Result v = run(cs[q].name, rosenbrock(4), kCaseBudget, kv);
            const OldRun& o = kOld[q];
            const bool ok = v.evaluations == o.evaluations && v.grad_calls == o.grad_calls && !v.objectives.empty() &&
                            v.objectives[0][0] == o.f && hash_of(v.variables[0]) == o.x_hash;
            check(ok, std::string(cs[q].name) + ", budget_units = evaluations: cc2d22c's run bit for bit (" +
                          std::to_string(v.evaluations) + " evaluations, " + std::to_string(v.grad_calls) +
                          " gradients)");
        }
    }

    // STOP-6: the user's subproblem counted; derived() keeps the deadline and the clock
    {
        auto direct = std::make_shared<std::atomic<long long>>(0);
        sc::Problem P = sphere(6);
        P.restrict_to = [direct](const Vec& xf, const std::vector<std::size_t>& idx) {
            sc::Problem Q;
            for (std::size_t j = 0; j < idx.size(); ++j) {
                Q.lo.push_back(-2.0);
                Q.hi.push_back(2.0);
            }
            Q.m = 1;
            Q.evaluate = [direct, xf, idx](const Vec& z, Vec& f, Vec&) {
                ++*direct;
                Vec x = xf;
                for (std::size_t k = 0; k < idx.size(); ++k) x[idx[k]] = z[k];
                f.assign(1, sphere_of(x));
            };
            return Q;
        };
        const sc::Result r = run("coevolution", P, 3000, {{"inner", word(kNM)}, {"cc_window", number(2)}});
        check(r.outside_evals == direct->load() && r.outside_evals > 0,
              "coevolution on the user's restrict_to: its evaluations counted outside (" +
                  std::to_string(r.outside_evals) + ")");
        check(r.evaluations + r.outside_evals <= 3000 + 3000 / 3 && r.used >= 3000.0 * 0.5,
              "... and the run goes on to its budget (used " + std::to_string(r.used) + ")");
        sc::Problem from = sphere(2), to = sphere(2);
        from.deadline = 5.0;
        from.time_clock = sc::Clock::cpu;
        to.deadline = 9.0;
        const sc::Problem d = sc::derived(from, to);
        check(d.deadline == 5.0 && d.time_clock == sc::Clock::cpu, "derived: the earlier deadline, the source's clock");
    }

    // the clocks (30.12)
    {
        const double c0 = sc::cpu_seconds(), u0 = sc::unbiased_seconds();
        busy(40000.0);                                  // Windows' processor time moves by ticks of ~15.6 ms
        check(sc::cpu_seconds() - c0 >= 0.015 && sc::unbiased_seconds() - u0 >= 0.03,
              "cpu and unbiased clocks go on with work");
        check(sc::clock_of("cpu") == sc::Clock::cpu && std::string(sc::clock_name(sc::Clock::unbiased)) == "unbiased",
              "time_clock's words");
        bool refused = false;
        try {
            (void)sc::clock_of("tsc");
        } catch (const std::invalid_argument&) {
            refused = true;
        }
        check(refused, "time_clock = tsc refused");

        sc::Problem sleepy = sphere(3);
        sleepy.evaluate = [](const Vec& x, Vec& f, Vec&) {
            std::this_thread::sleep_for(std::chrono::milliseconds(4));
            f.assign(1, sphere_of(x));
        };
        const sc::Result w = run("de", sleepy, 60, {{"max_seconds", number(0.15)}});
        const sc::Result c = run("de", sleepy, 60, {{"max_seconds", number(0.15)}, {"time_clock", word("cpu")}});
        check(w.time_up && w.evaluations < 60, "a sleeping problem: the wall clock cuts it at 0.15 s");
        check(!c.time_up && c.evaluations == 60 && setting(c, "time_clock") == "cpu",
              "... time_clock = cpu does not (the sleep is not processor time)");
        check(c.seconds > c.cpu_seconds && setting(c, "wall_seconds") != "" && setting(c, "cpu_seconds") != "",
              "... wall_seconds " + setting(c, "wall_seconds") + " > cpu_seconds " + setting(c, "cpu_seconds"));
        const sc::Result u = run("de", sleepy, 60, {{"max_seconds", number(0.15)}, {"time_clock", word("unbiased")}});
        check(u.time_up && u.evaluations < 60, "... time_clock = unbiased cuts it as the wall clock (awake)");

        sc::Problem hot = sphere(3);
        hot.evaluate = [](const Vec& x, Vec& f, Vec&) {
            busy(300.0);
            f.assign(1, sphere_of(x));
        };
        const sc::Result h = run("basin_hopping", hot, 1000000000LL,
                                 {{"inner", word(kNM)}, {"time_clock", word("cpu")}, {"max_seconds", number(0.3)}});
        check(h.stopped_by == "deadline" && h.cpu_seconds >= 0.3 && h.cpu_seconds < 0.6,
              "a busy problem, time_clock = cpu, max_seconds = 0.3: ends at ~0.3 s of processor time (" +
                  setting(h, "cpu_seconds") + ", " + h.stopped_by + ")");

        bool kept = false;
        try {
            (void)run("multistart", sphere(2), 400,
                      {{"inner", word("{\"solver\":\"nelder_mead\",\"time_clock\":\"cpu\"}")},
                       {"max_seconds", number(10.0)}});
        } catch (const std::invalid_argument&) {
            kept = true;
        }
        check(kept, "a run within a run with a deadline keeps its clock (another time_clock refused)");
    }

    std::printf("%s\n", g_failed ? "FAILED" : "all passed");
    return g_failed ? 1 : 0;
}

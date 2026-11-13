// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// scalar/structure_fit.hpp (task 27, item 27.1, the link with 24.1.1 and
// 24.1.4) on a typed grammar of expressions with parameters:
//   - the solver local_search with set_active_fit from one tree genome: the
//     structure's genes stay, the parameters move, f does not grow, the
//     evaluations within the budget, the hook called once;
//   - FIT-2: an invalid GE genome and a tree without parameters come back
//     (canonical) with nothing evaluated;
//   - memetic with the GA on the structure hooks (ga_crossover, ga_mutation,
//     ga_init = user) and local_solver = local_search, lamarck = on: every
//     local search keeps its start's structure and does not worsen f; the
//     runs with lamarck on and off differ (the answers are written back);
//     the same seed, the same run;
//   - the other way: the parameters fitted inside evaluate (one evaluation a
//     structure) with cache = canon: every structure evaluated once, the
//     repeats counted in cache_hits; the same seed, the same run.
// ============================================================================

#include <cmath>
#include <cstdio>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <vector>

#include <mootation/scalar/scalar.hpp>
#include <mootation/scalar/structure_fit.hpp>
#include <mootation/scalar/structure_hooks.hpp>

namespace sc = mootation::scalar;
using namespace mootation::scalar::structure;
using Vec = std::vector<double>;

static int g_failed = 0;

static void check(bool ok, const std::string& what)
{
    if (!ok) {
        ++g_failed;
        std::printf("FAIL: %s\n", what.c_str());
    }
}

static Production rule(const std::string& label, std::vector<int> children = {})
{
    Production p;
    p.label = label;
    p.children = std::move(children);
    return p;
}

// E := (+ E E) | (* E T) | x[a] ;  T := c[k] | t ;  x[a] = a t, c[k] = k
enum { E = 0, T = 1 };

static Grammar grammar()
{
    Grammar g;
    g.symbols = {"E", "T"};
    Production plus = rule("+", {E, E});
    plus.commutative = true;
    Production x = rule("x");
    x.param_lo = {-2.0};
    x.param_hi = {2.0};
    Production c = rule("c");
    c.param_lo = {-2.0};
    c.param_hi = {2.0};
    g.rules = {{plus, rule("*", {E, T}), x}, {c, rule("t")}};
    g.start = E;
    return g;
}

static double value(const Grammar& g, const Tree& tr, std::size_t i, std::size_t* end, double t)
{
    const Node& n = tr.nodes[i];
    const Production& p = production(g, n);
    std::size_t at = i + 1;
    std::vector<double> a;
    for (std::size_t c = 0; c < p.children.size(); ++c) a.push_back(value(g, tr, at, &at, t));
    *end = at;
    if (p.label == "+") return a[0] + a[1];
    if (p.label == "*") return a[0] * a[1];
    if (p.label == "x") return n.params[0] * t;
    if (p.label == "c") return n.params[0];
    return t;
}

// the squared error against 0.5 t + 1.5 t^2 at t = 0, 0.25, ..., 1
static double error(const Grammar& g, const Tree& tr)
{
    double s = 0.0;
    for (int k = 0; k <= 4; ++k) {
        const double t = 0.25 * k;
        std::size_t end = 0;
        const double d = value(g, tr, 0, &end, t) - (0.5 * t + 1.5 * t * t);
        s += d * d;
    }
    return s;
}

static sc::Knob word(const std::string& w) { return {true, w, 0.0}; }
static sc::Knob number(double v) { return {false, "", v}; }

static TreeHookOptions tree_options()
{
    TreeHookOptions o;
    o.tree.max_depth = 5;
    o.tree.init_depth = 4;
    o.tree.max_size = 30;
    return o;
}

static void one_fit()
{
    sc::Problem P;
    P.m = 1;
    const auto codec = make_tree_hooks(P, grammar(), tree_options());
    P.evaluate = [codec](const Vec& x, Vec& f, Vec&) { f[0] = error(codec->grammar, codec->unpack(x)); };
    ActiveFit a;
    a.fit_budget = 200;
    set_active_fit(P, a);

    std::mt19937 rng(3);
    int tried = 0;
    for (int k = 0; k < 40; ++k) {
        Vec x(codec->n());
        P.sample(x, rng);
        const sc::Decoded d = P.decode(x);
        if (params_of(codec->unpack(d.x_canon)).empty()) continue;
        ++tried;
        sc::Start from;
        from.has_x0 = true;
        from.x0 = d.x_canon;
        const sc::Result r = sc::run("local_search", P, 1e-6, false, 0.0, 20, 300, 11, {}, {}, 0, from);
        const Vec& y = r.variables[0];
        bool same_structure = true;
        for (std::size_t i = 0; i <= codec->M; ++i) same_structure = same_structure && y[i] == d.x_canon[i];
        check(same_structure, "the fit keeps the structure's genes");
        Vec f0(1), g;
        P.evaluate(d.x_canon, f0, g);
        check(r.objectives[0][0] <= f0[0], "the fit does not worsen f");
        check(r.evaluations <= 300 && r.hook_calls == 1, "within the budget, one call");
    }
    check(tried > 10, "enough trees with parameters");
}

static void nothing_to_fit()
{
    // a GE genome that does not decode: back as it was, nothing evaluated
    sc::Problem Q;
    Q.m = 1;
    GEHookOptions go;
    go.ge.max_len = 10;
    go.max_params = 8;
    go.ge.max_wraps = 0;
    const auto gc = make_ge_hooks(Q, grammar(), go);
    int calls = 0;
    Q.evaluate = [&calls](const Vec&, Vec& f, Vec&) {
        ++calls;
        f[0] = 0.0;
    };
    set_active_fit(Q, ActiveFit{});
    Genome endless;
    endless.codons = {0, 3, 6};                    // E -> (+ E E) for ever
    const Vec bad = gc->pack(endless);
    check(!Q.decode(bad).valid, "the endless genome does not decode");
    std::mt19937 rng(1);
    sc::LocalContext ctx{Q, 100, rng, {}};
    const sc::LocalOutcome out = Q.local_search(bad, ctx, {});
    check(out.x == bad && calls == 0, "FIT-2: an invalid genome comes back, nothing evaluated");

    // a tree without parameters: E := (+ E E) | t
    Grammar g;
    g.symbols = {"E"};
    Production plus = rule("+", {0, 0});
    plus.commutative = true;
    g.rules = {{plus, rule("t")}};
    sc::Problem P;
    P.m = 1;
    TreeHookOptions o = tree_options();
    const auto tc = make_tree_hooks(P, g, o);
    P.evaluate = Q.evaluate;
    set_active_fit(P, ActiveFit{});
    Vec x(tc->n());
    P.sample(x, rng);
    sc::LocalContext ctx2{P, 100, rng, {}};
    const sc::LocalOutcome out2 = P.local_search(x, ctx2, {});
    check(out2.x == P.decode(x).x_canon && calls == 0, "FIT-2: no parameters, the canonical genome, nothing evaluated");
}

static sc::Result memetic(const std::string& lamarck, std::shared_ptr<int> worse)
{
    sc::Problem P;
    P.m = 1;
    const auto codec = make_tree_hooks(P, grammar(), tree_options());
    P.evaluate = [codec](const Vec& x, Vec& f, Vec&) { f[0] = error(codec->grammar, codec->unpack(x)); };
    ActiveFit a;
    a.fit_budget = 60;
    set_active_fit(P, a);
    if (worse) {   // every local search: its start's structure kept, f not worse
        const auto fit = P.local_search;
        P.local_search = [fit, codec, worse](const Vec& x, sc::LocalContext& ctx, const Vec& s) {
            sc::LocalOutcome out = fit(x, ctx, s);
            const Vec xc = codec->decode(x).x_canon;
            for (std::size_t i = 0; i <= codec->M; ++i)
                if (out.x[i] != xc[i]) ++*worse;
            if (error(codec->grammar, codec->unpack(out.x)) > error(codec->grammar, codec->unpack(xc))) ++*worse;
            return out;
        };
    }
    const std::string ga = "{\"solver\":\"ga_sbx\",\"ga_crossover\":\"user\",\"ga_mutation\":\"user\","
                           "\"ga_init\":\"user\"}";
    sc::Knobs k{{"inner", word(ga)}, {"local_solver", word("{\"solver\":\"local_search\"}")},
                {"memetic_every", number(300)}};
    if (!lamarck.empty()) k.push_back({"lamarck", word(lamarck)});
    return sc::run("memetic", P, 1e-6, false, 0.0, 20, 2400, 5, k, {}, 0, {});
}

static void through_memetic()
{
    auto worse = std::make_shared<int>(0);
    const sc::Result on = memetic("on", worse), on2 = memetic("on", nullptr), off = memetic("off", nullptr);
    check(*worse == 0, "memetic: every local search keeps its structure and does not worsen f");
    check(on.hook_calls > 0, "memetic: the local searches ran (" + std::to_string(on.hook_calls) + ")");
    check(on.objectives == on2.objectives && on.variables == on2.variables && on.evaluations == on2.evaluations,
          "memetic: the same seed, the same run");
    check(on.objectives != off.objectives || on.evaluations != off.evaluations || on.variables != off.variables,
          "memetic: lamarck = on writes the fits back (the run differs from off)");
    std::printf("  memetic + fit: f %.4g (lamarck on), %.4g (off); %lld local searches\n", on.objectives[0][0],
                off.objectives[0][0], on.hook_calls);
}

// the fit inside evaluate: a compass search of the parameters, deterministic
static double fitted_error(const Grammar& g, Tree t)
{
    std::vector<double> p = params_of(t);
    double best = error(g, t), step = 0.5;
    while (step > 1e-4) {
        bool moved = false;
        for (std::size_t k = 0; k < p.size(); ++k)
            for (double s : {step, -step}) {
                std::vector<double> q = p;
                q[k] += s;
                set_params(t, q);
                const double e = error(g, t);
                if (e < best) {
                    best = e;
                    p = q;
                    moved = true;
                }
            }
        if (!moved) step *= 0.5;
    }
    return best;
}

static sc::Result cached_run(std::shared_ptr<std::vector<std::string>> keys)
{
    sc::Problem P;
    P.m = 1;
    const auto codec = make_tree_hooks(P, grammar(), tree_options());
    P.evaluate = [codec, keys](const Vec& x, Vec& f, Vec&) {
        const Tree t = codec->unpack(x);
        if (keys) keys->push_back(canon_key(codec->grammar, t));
        f[0] = fitted_error(codec->grammar, t);
    };
    P.cache = sc::Problem::Cache::canon;
    const sc::Knobs user{{"ga_crossover", word("user")}, {"ga_mutation", word("user")}, {"ga_init", word("user")}};
    return sc::run("ga_sbx", P, 1e-6, false, 0.0, 20, 1500, 7, user, {}, 0, {});
}

static void fit_inside_evaluate()
{
    auto keys = std::make_shared<std::vector<std::string>>();
    const sc::Result a = cached_run(keys), b = cached_run(nullptr);
    const std::set<std::string> distinct(keys->begin(), keys->end());
    check(!keys->empty() && distinct.size() == keys->size(), "cache = canon: every structure evaluated once");
    check(a.cache_hits > 0, "cache = canon: repeats of a structure counted (" + std::to_string(a.cache_hits) + ")");
    check(a.objectives == b.objectives && a.variables == b.variables && a.cache_hits == b.cache_hits,
          "cache = canon: the same seed, the same run");
    std::printf("  ga_sbx, fit inside evaluate, cache = canon: %zu structures evaluated, %lld repeats, f %.4g\n",
                keys->size(), a.cache_hits, a.objectives[0][0]);
}

int main()
{
    one_fit();
    nothing_to_fit();
    through_memetic();
    fit_inside_evaluate();
    if (g_failed) {
        std::printf("%d check(s) failed\n", g_failed);
        return 1;
    }
    std::printf("all checks passed\n");
    return 0;
}

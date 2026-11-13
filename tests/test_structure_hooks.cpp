// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// scalar/structure_hooks.hpp (task 27 on task 24's genome contract):
//   - pack / unpack: a round trip for random trees and GE genomes; the
//     packed vector lies in the bounds make_*_hooks set (every coordinate in
//     [lo, hi], length n);
//   - the hooks as a GA calls them (sample, crossover, mutate, decode on
//     every child, x_canon written back), 2 * 10^4 children: every vector
//     unpacks to a tree that obeys the grammar within max_depth and max_size,
//     stays in the bounds; decode is idempotent (decode(x_canon).x_canon =
//     x_canon) and its key is the tree's; the active coordinates are exactly
//     the used parameter slots;
//   - the parameters a fit changes in the active slots come back through
//     decode (Lamarckian), the structure's coordinates untouched;
//   - GE: an invalid genome decodes to valid = false with x as it was; a
//     valid one's x_canon holds the derivation's parameters; the hooks keep
//     codons in [0, 255] and lengths in [1, max_len];
//   - malformed vectors refused (HOOK-1): a production out of range, a count
//     the derivation does not end at, a non-integer, the wrong length;
//   - canon_key (HOOK-6): the structure's key, unchanged by decode's
//     write-back; "(invalid)" for a GE genome that does not decode;
//   - the same seed, the same vectors;
//   - through scalar::run, ga_sbx and ssga with ga_crossover, ga_mutation,
//     ga_init = user: trees — evaluate gets only canonical well-formed
//     genomes, no decode fails; GE — no invalid genome reaches evaluate,
//     the invalid ones are counted apart (decode_invalid); the same seed
//     gives the same run.
// ============================================================================

#include <cstdio>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <mootation/scalar/scalar.hpp>
#include <mootation/scalar/structure_hooks.hpp>

using namespace mootation::scalar;
using namespace mootation::scalar::structure;
namespace sc = mootation::scalar;

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

// a small typed grammar with parameters: E := (+ E E) | (* E T) | x[a] ; T := c[k] | t
enum { E = 0, T = 1 };

static Grammar grammar()
{
    Grammar g;
    g.symbols = {"E", "T"};
    Production plus = rule("+", {E, E});
    plus.commutative = true;
    Production x = rule("x");
    x.param_lo = {-1.0};
    x.param_hi = {1.0};
    Production c = rule("c");
    c.param_lo = {0.5};
    c.param_hi = {2.0};
    g.rules = {{plus, rule("*", {E, T}), x}, {c, rule("t")}};
    g.start = E;
    return g;
}

static bool in_bounds(const Problem& P, const std::vector<double>& x)
{
    if (x.size() != P.lo.size()) return false;
    for (std::size_t k = 0; k < x.size(); ++k)
        if (!(x[k] >= P.lo[k] && x[k] <= P.hi[k])) return false;
    return true;
}

static void tree_hooks()
{
    Problem P;
    TreeHookOptions o;
    o.tree.max_depth = 6;
    o.tree.init_depth = 4;
    o.tree.max_size = 40;
    const auto codec = make_tree_hooks(P, grammar(), o);
    check(P.lo.size() == codec->n() && P.hi.size() == codec->n() && codec->n() == 1 + 40 + 40,
          "n = 1 + M + M K");
    check(codec->param_lo == -1.0 && codec->param_hi == 2.0, "HOOK-2: the union of the ranges");
    check(!P.repair, "repair left off");

    std::mt19937 rng(17);
    std::vector<std::vector<double>> pop(30, std::vector<double>(codec->n()));
    for (auto& x : pop) {
        P.sample(x, rng);
        check(in_bounds(P, x), "sample in the bounds");
        const Tree t = codec->unpack(x);
        check(obeys(codec->grammar, t) && codec->pack(t) == x, "pack(unpack(x)) = x");
    }
    int bad = 0, active_wrong = 0, not_idem = 0, key_wrong = 0;
    for (int it = 0; it < 10000; ++it) {
        const auto i = static_cast<std::size_t>(mootation::rnd::below(rng, pop.size()));
        const auto j = static_cast<std::size_t>(mootation::rnd::below(rng, pop.size()));
        std::vector<double> c1, c2;
        P.crossover(pop[i], pop[j], c1, c2, rng);
        for (std::vector<double>* c : {&c1, &c2}) {
            P.mutate(*c, rng);
            const std::string key_before = P.canon_key(*c);
            const Decoded d = P.decode(*c);
            // HOOK-6: the key is the structure's, the same before and after the write-back
            if (P.canon_key(d.x_canon) != key_before ||
                key_before != canon_key(codec->grammar, codec->unpack(*c)))
                ++key_wrong;
            *c = d.x_canon;
            if (!d.valid || !in_bounds(P, *c)) ++bad;
            const Tree t = codec->unpack(*c);
            if (!obeys(codec->grammar, t) || depth(codec->grammar, t) > o.tree.max_depth ||
                t.nodes.size() > static_cast<std::size_t>(o.tree.max_size))
                ++bad;
            const Decoded again = P.decode(*c);
            if (again.x_canon != *c) ++not_idem;
            const std::size_t used = params_of(t).size();
            for (std::size_t k = 0; k < codec->n(); ++k) {
                const bool want = k >= 1 + codec->M && k < 1 + codec->M + used;
                if ((d.active[k] != 0) != want) ++active_wrong;
            }
        }
        pop[i] = c1;
        pop[j] = c2;
    }
    check(bad == 0, "2 * 10^4 children: valid, in the bounds and the limits (" + std::to_string(bad) + ")");
    check(not_idem == 0, "decode is idempotent on its x_canon");
    check(active_wrong == 0, "active = the used parameter slots");
    check(key_wrong == 0, "HOOK-6: canon_key is the structure's, unchanged by decode's write-back");

    // a fit moves the active slots; decode writes them back with the structure untouched
    std::vector<double> x = pop[0];
    const Decoded d0 = P.decode(x);
    x = d0.x_canon;
    std::vector<double> fitted = x;
    for (std::size_t k = 0; k < fitted.size(); ++k)
        if (d0.active[k]) fitted[k] = 0.75;
    const Decoded d1 = P.decode(fitted);
    bool same_structure = true;
    for (std::size_t k = 0; k <= codec->M; ++k) same_structure = same_structure && d1.x_canon[k] == x[k];
    check(same_structure, "a parameter fit leaves the structure's coordinates");
    for (std::size_t k = 0; k < fitted.size(); ++k)
        if (d0.active[k]) check(d1.x_canon[k] == 0.75, "the fitted parameters are written back");

    // HOOK-1: malformed vectors refused
    const auto refused = [&](std::vector<double> v, const std::string& what) {
        bool r = false;
        try {
            P.decode(v);
        } catch (const std::invalid_argument&) {
            r = true;
        }
        check(r, "HOOK-1: " + what + " refused");
    };
    std::vector<double> v = x;
    v[1] = 7.0;
    refused(v, "a production out of range");
    v = x;
    v[0] = x[0] + 1.0;
    refused(v, "a count past the derivation's end");
    v = x;
    v[1] = 0.5;
    refused(v, "a non-integer production");
    refused(std::vector<double>(x.begin(), x.end() - 1), "the wrong length");

    // the same seed, the same vectors
    std::mt19937 r1(5), r2(5);
    for (int k = 0; k < 100; ++k) {
        std::vector<double> a(codec->n()), b(codec->n());
        P.sample(a, r1);
        P.sample(b, r2);
        P.mutate(a, r1);
        P.mutate(b, r2);
        check(a == b, "same seed, same vectors");
    }
}

static void ge_hooks()
{
    Problem P;
    GEHookOptions o;
    o.ge.max_len = 30;
    o.max_params = 12;
    o.p_prune = 0.1;
    const auto codec = make_ge_hooks(P, grammar(), o);
    check(codec->n() == 2 + 30 + 12 && P.lo.size() == codec->n(), "n = 2 + L + P");

    std::mt19937 rng(23);
    std::vector<std::vector<double>> pop(30, std::vector<double>(codec->n()));
    for (auto& x : pop) {
        P.sample(x, rng);
        check(in_bounds(P, x) && codec->pack(codec->unpack(x)) == x, "GE: sample in the bounds, round trip");
    }
    int bad = 0, invalid = 0, params_wrong = 0;
    for (int it = 0; it < 10000; ++it) {
        const auto i = static_cast<std::size_t>(mootation::rnd::below(rng, pop.size()));
        const auto j = static_cast<std::size_t>(mootation::rnd::below(rng, pop.size()));
        std::vector<double> c1, c2;
        P.crossover(pop[i], pop[j], c1, c2, rng);
        for (std::vector<double>* c : {&c1, &c2}) {
            P.mutate(*c, rng);
            const std::vector<double> before = *c;
            const Decoded d = P.decode(*c);
            if (!d.valid) {
                ++invalid;
                if (d.x_canon != before || P.canon_key(before) != "(invalid)") ++bad;
                check(!codec->derivation(before).valid, "GE: derivation agrees on invalid");
            } else {
                const Mapping m = codec->derivation(d.x_canon);
                std::vector<double> want = params_of(m.tree);
                if (want.size() > codec->P) want.resize(codec->P);
                const Genome g = codec->unpack(d.x_canon);
                if (g.params != want) ++params_wrong;
            }
            *c = d.x_canon;
            if (!in_bounds(P, *c)) ++bad;
        }
        pop[i] = c1;
        pop[j] = c2;
    }
    check(bad == 0, "GE hooks: in the bounds; invalid ones keep x (" + std::to_string(bad) + ")");
    check(invalid > 0, "GE: invalid genomes do occur and are flagged");
    check(params_wrong == 0, "GE: x_canon holds the derivation's parameters");
}

// ── through the GA (ga_sbx, ssga with ga_crossover, ga_mutation, ga_init = user) ─
// the expression's value at t: + and * of their arguments, x[a] = a t, c[k] = k, t = t
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

static void through_the_ga()
{
    const sc::Knobs user{{"ga_crossover", word("user")}, {"ga_mutation", word("user")}, {"ga_init", word("user")}};
    for (const std::string m : {"ga_sbx", "ssga"}) {
        // trees
        Problem P;
        P.m = 1;
        TreeHookOptions o;
        o.tree.max_depth = 6;
        o.tree.init_depth = 4;
        o.tree.max_size = 40;
        const auto tc = make_tree_hooks(P, grammar(), o);
        int foreign = 0;
        P.evaluate = [tc, &foreign](const std::vector<double>& x, std::vector<double>& f, std::vector<double>&) {
            const Tree t = tc->unpack(x);
            if (tc->pack(t) != x) ++foreign;
            Tree c = t;
            canonicalize(tc->grammar, c);
            if (tc->pack(c) != x) ++foreign;   // evaluate sees the canonical genome (written back)
            f[0] = error(tc->grammar, t);
        };
        const sc::Result a = sc::run(m, P, 1e-6, false, 0.0, 20, 2000, 7, user, {}, 0, {});
        const sc::Result b = sc::run(m, P, 1e-6, false, 0.0, 20, 2000, 7, user, {}, 0, {});
        check(foreign == 0, m + ", trees: evaluate gets canonical, well-formed genomes");
        check(!a.objectives.empty() && a.evaluations > 0 && a.evaluations <= 2000, m + ", trees: the run runs");
        check(a.objectives == b.objectives && a.variables == b.variables && a.evaluations == b.evaluations,
              m + ", trees: the same seed, the same run");
        check(a.decode_invalid == 0, m + ", trees: every tree decodes");

        // grammatical evolution
        Problem Q;
        Q.m = 1;
        GEHookOptions go;
        go.ge.max_len = 40;
        go.max_params = 20;
        const auto gc = make_ge_hooks(Q, grammar(), go);
        int invalid_seen = 0;
        Q.evaluate = [gc, &invalid_seen](const std::vector<double>& x, std::vector<double>& f,
                                         std::vector<double>&) {
            const Mapping d = gc->derivation(x);
            if (!d.valid) {
                ++invalid_seen;
                f[0] = 1e300;
                return;
            }
            f[0] = error(gc->grammar, d.tree);
        };
        const sc::Result c = sc::run(m, Q, 1e-6, false, 0.0, 20, 2000, 7, user, {}, 0, {});
        const sc::Result e = sc::run(m, Q, 1e-6, false, 0.0, 20, 2000, 7, user, {}, 0, {});
        check(invalid_seen == 0, m + ", GE: no invalid genome reaches evaluate");
        check(c.decode_invalid > 0, m + ", GE: invalid genomes counted apart (" + std::to_string(c.decode_invalid) + ")");
        check(c.objectives == e.objectives && c.variables == e.variables && c.decode_invalid == e.decode_invalid,
              m + ", GE: the same seed, the same run");
    }
}

int main()
{
    tree_hooks();
    ge_hooks();
    through_the_ga();
    if (g_failed) {
        std::printf("%d check(s) failed\n", g_failed);
        return 1;
    }
    std::printf("all checks passed\n");
    return 0;
}

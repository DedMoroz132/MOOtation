// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// scalar/typed_tree.hpp and scalar/grammatical_evolution.hpp (task 27):
//   - Ratle & Sebag 2001 §6.2 on their Fig. 7 grammar: the indices (V 1, E 2;
//     <op><V><V> 2, the other three 3), D_max = 2 leaves only <op><V><V>
//     (TREE-1), Fig. 8: at budget 2 only <op><V><V>; with D_max = 4 every
//     tree is within depth 4 and every production of E occurs at the root;
//   - Montana 1995 Examples 1-3: the types possibilities table is the index
//     (VECTOR-2, VECTOR-3 at depth 1, FLOAT from 2), and at point 5 of
//     Example 2 (budget 2) SCALAR-VEC-MULT-3 is masked;
//   - the operators keep the types: 10^5 crossovers, subtree and point
//     mutations on a population, every tree obeys the grammar, within
//     max_depth and max_size; a crossover with no node of the symbol in the
//     second parent changes nothing (TREE-3);
//   - correct by construction (item 27.1): on a series-parallel grammar with
//     a "determined" attribute (leaves A, B determined, C not; a parallel
//     node determined iff all its children are, a series node iff one is;
//     C only in a series node; node kinds alternate; at most one leaf of a
//     kind in a node; two children at least), 10^4 random trees and the
//     valid ones of 10^4 random GE genomes, canonicalized: none undetermined
//     and none that the reduction (one-child nodes removed, nested nodes of
//     one kind merged, equal leaves of a node merged, children sorted)
//     changes;
//   - the canonical key: the same for every permutation of the children of
//     a commutative node, canonicalize idempotent and key-preserving;
//   - the parameters: params_of / set_params round trip, write_params and
//     decode give them back (GE-7);
//   - grammatical evolution: O'Neill & Ryan 2001 §IV-B's worked example
//     (the first nine codons give 1.0 - Sin(X); GE-9); decoding is a function of
//     the genome; a genome that keeps choosing the recursive production is
//     invalid after max_wraps wraps and valid = false is all it gives;
//     duplication and pruning of a valid unwrapped genome leave its
//     derivation as it was; the GE operators keep codons in range;
//   - the same seed gives the same trees and genomes;
//   - refusals: a symbol with no finite tree, a codon out of range,
//     set_params of the wrong length.
// ============================================================================

#include <algorithm>
#include <cstdio>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include <mootation/scalar/grammatical_evolution.hpp>
#include <mootation/scalar/typed_tree.hpp>

using namespace mootation::scalar::structure;

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

// ── Ratle & Sebag 2001, Fig. 7 (start symbol E: TREE-1) ─────────────────────
enum { E = 0, V = 1, OP = 2 };

static Grammar fig7()
{
    Grammar g;
    g.symbols = {"E", "V", "op"};
    g.rules = {{rule("", {OP, E, E}), rule("", {OP, E, V}), rule("", {OP, V, E}), rule("", {OP, V, V})},
               {rule("a"), rule("b"), rule("c")},
               {rule("+"), rule("-"), rule("*"), rule("/")}};
    g.start = E;
    return g;
}

static void ratle_sebag()
{
    const Grammar g = fig7();
    const DepthIndex ix = depth_index(g);
    check(ix.of[V] == 1 && ix.of[OP] == 1 && ix.of[E] == 2, "Fig. 7 indices of V, op, E");
    check(ix.rule[E][0] == 3 && ix.rule[E][1] == 3 && ix.rule[E][2] == 3 && ix.rule[E][3] == 2,
          "Fig. 7 indices of E's derivations");
    std::mt19937 rng(7);
    TreeOptions o;
    o.max_depth = 2;
    o.init_depth = 0;
    for (int k = 0; k < 1000; ++k) {
        const Tree t = random_tree(g, ix, o, rng);
        check(t.nodes[0].rule == 3 && depth(g, t) == 2 && obeys(g, t), "D_max = 2: only <op><V><V>");
    }
    // Fig. 8: an E with budget 2 (level 3 under D_max = 4) is <op><V><V>
    for (int k = 0; k < 1000; ++k) {
        std::vector<Node> sub;
        grow(g, ix, E, 2, sub, rng);
        check(sub[0].rule == 3, "Fig. 8: budget 2 masks all but <op><V><V>");
    }
    o.max_depth = 4;
    std::set<int> roots;
    for (int k = 0; k < 10000; ++k) {
        const Tree t = random_tree(g, ix, o, rng);
        roots.insert(t.nodes[0].rule);
        check(obeys(g, t) && depth(g, t) <= 4, "D_max = 4: within depth");
    }
    check(roots.size() == 4, "D_max = 4: every derivation of E at the root");
}

// ── Montana 1995, Examples 1-3 ─────────────────────────────────────────────
static void montana()
{
    enum { V2 = 0, V3 = 1, FL = 2 };
    Grammar g;
    g.symbols = {"VECTOR-2", "VECTOR-3", "FLOAT"};
    g.rules = {{rule("VECTOR-ADD-2", {V2, V2}), rule("SCALAR-VEC-MULT-2", {FL, V2}), rule("V3")},
               {rule("VECTOR-ADD-3", {V3, V3}), rule("SCALAR-VEC-MULT-3", {FL, V3}), rule("V1"), rule("V2")},
               {rule("DOT-PRODUCT-2", {V2, V2}), rule("DOT-PRODUCT-3", {V3, V3})}};
    g.start = V3;
    const DepthIndex ix = depth_index(g);
    check(ix.of[V2] == 1 && ix.of[V3] == 1 && ix.of[FL] == 2, "Example 3: the types possibilities table");
    check(ix.rule[V3][1] == 3 && ix.rule[V3][0] == 2, "Example 2: SCALAR-VEC-MULT-3 needs depth 3");
    std::mt19937 rng(11);
    for (int k = 0; k < 1000; ++k) {
        std::vector<Node> sub;
        grow(g, ix, V3, 2, sub, rng);
        check(sub[0].rule != 1, "Example 2, point 5: SCALAR-VEC-MULT-3 masked at budget 2");
    }
}

// ── the operators keep the types ────────────────────────────────────────────
static void operators_keep_types()
{
    const Grammar g = fig7();
    const DepthIndex ix = depth_index(g);
    TreeOptions o;
    o.max_depth = 6;
    o.init_depth = 4;
    o.max_size = 60;
    std::mt19937 rng(2026);
    std::vector<Tree> pop;
    for (int k = 0; k < 50; ++k) pop.push_back(random_tree(g, ix, o, rng));
    int bad = 0, crossed = 0, mutated = 0;
    for (int it = 0; it < 100000; ++it) {
        const std::size_t i = static_cast<std::size_t>(mootation::rnd::below(rng, pop.size()));
        const std::size_t j = static_cast<std::size_t>(mootation::rnd::below(rng, pop.size()));
        switch (it % 3) {
        case 0:
            if (i != j) crossed += crossover(g, o, pop[i], pop[j], rng) ? 1 : 0;
            break;
        case 1: mutated += subtree_mutation(g, ix, o, pop[i], rng) ? 1 : 0; break;
        default: mutated += point_mutation(g, pop[i], rng) ? 1 : 0; break;
        }
        for (std::size_t k : {i, j})
            if (!obeys(g, pop[k]) || depth(g, pop[k]) > o.max_depth ||
                pop[k].nodes.size() > static_cast<std::size_t>(o.max_size))
                ++bad;
    }
    check(bad == 0, "10^5 operations: no tree breaks the types or the limits (" + std::to_string(bad) + ")");
    check(crossed > 10000 && mutated > 20000, "the operators do change trees");

    // TREE-3: no node of the symbol in the second parent
    enum { X = 0, Y = 1 };
    Grammar h;
    h.symbols = {"X", "Y"};
    h.rules = {{rule("x")}, {rule("y")}};
    Tree a{{Node{X, 0, {}}}}, b{{Node{Y, 0, {}}}};
    TreeOptions ho;
    h.start = X;
    check(!crossover(h, ho, a, b, rng) && a.nodes[0].symbol == X && b.nodes[0].symbol == Y,
          "TREE-3: no partner, parents unchanged");
}

// ── the series-parallel grammar with the "determined" attribute ─────────────
enum { ROOT = 0, SN = 1, PN = 2, LA = 3, LB = 4, LC = 5 };

static Grammar series_parallel(int max_arity)
{
    Grammar g;
    g.symbols = {"root", "series", "parallel", "A", "B", "C"};
    g.rules.resize(6);
    for (int leaf : {LA, LB, LC}) {
        Production p = rule(g.symbols[static_cast<std::size_t>(leaf)]);
        p.param_lo = {0.1};
        p.param_hi = {10.0};
        g.rules[static_cast<std::size_t>(leaf)].push_back(p);
    }
    const auto add = [&](int sym, const std::string& label, const std::vector<int>& leaves, int sub) {
        for (int mask = 0; mask < (1 << leaves.size()); ++mask) {
            std::vector<int> kids;
            for (std::size_t b = 0; b < leaves.size(); ++b)
                if (mask & (1 << b)) kids.push_back(leaves[b]);
            for (int k = 0; static_cast<int>(kids.size()) + k <= max_arity; ++k) {
                if (static_cast<int>(kids.size()) + k < 2) continue;
                std::vector<int> all = kids;
                all.insert(all.end(), static_cast<std::size_t>(k), sub);
                Production p = rule(label, all);
                p.commutative = true;
                g.rules[static_cast<std::size_t>(sym)].push_back(p);
                g.rules[ROOT].push_back(p);
            }
        }
    };
    add(SN, "S", {LA, LB, LC}, PN);   // C only in a series node
    add(PN, "P", {LA, LB}, SN);
    g.start = ROOT;
    return g;
}

// the phenotype as gen_audit.py sees it: a leaf's kind or (S|P, children)
struct Net {
    std::string op;   // "A", "B", "C" for a leaf
    std::vector<Net> kids;
};

static Net net_of(const Grammar& g, const Tree& t, std::size_t i, std::size_t* end)
{
    Net n;
    const Production& p = production(g, t.nodes[i]);
    n.op = p.label;
    std::size_t at = i + 1;
    for (std::size_t c = 0; c < p.children.size(); ++c) n.kids.push_back(net_of(g, t, at, &at));
    *end = at;
    return n;
}

static bool determined(const Net& n)
{
    if (n.kids.empty()) return n.op != "C";
    bool all = true, any = false;
    for (const Net& k : n.kids) {
        const bool d = determined(k);
        all = all && d;
        any = any || d;
    }
    return n.op == "P" ? all : any;
}

static std::string key(const Net& n)
{
    if (n.kids.empty()) return n.op;
    std::string s = n.op + "(";
    for (std::size_t k = 0; k < n.kids.size(); ++k) s += (k ? "," : "") + key(n.kids[k]);
    return s + ")";
}

// gen_audit.py's reduce: nested nodes of one kind merged, equal leaves of a
// node merged, leaves sorted then subtrees by key, one child: the child
static Net reduce(const Net& n)
{
    if (n.kids.empty()) return n;
    std::vector<Net> flat;
    for (const Net& k : n.kids) {
        Net r = reduce(k);
        if (!r.kids.empty() && r.op == n.op)
            flat.insert(flat.end(), r.kids.begin(), r.kids.end());
        else
            flat.push_back(r);
    }
    std::set<std::string> leaves;
    std::vector<Net> subs;
    for (const Net& k : flat) {
        if (k.kids.empty())
            leaves.insert(k.op);
        else
            subs.push_back(k);
    }
    std::stable_sort(subs.begin(), subs.end(), [](const Net& a, const Net& b) { return key(a) < key(b); });
    Net out;
    out.op = n.op;
    for (const std::string& l : leaves) out.kids.push_back(Net{l, {}});
    out.kids.insert(out.kids.end(), subs.begin(), subs.end());
    if (out.kids.size() == 1) return out.kids[0];
    return out;
}

static void correct_by_construction()
{
    const Grammar g = series_parallel(3);
    const DepthIndex ix = depth_index(g);
    TreeOptions o;
    o.max_depth = 4;     // gen_audit.py: nodes at depths 0-2, leaves down to 3
    o.init_depth = 0;
    std::mt19937 rng(1);
    int wrong = 0, unreduced = 0;
    for (int k = 0; k < 10000; ++k) {
        Tree t = random_tree(g, ix, o, rng);
        canonicalize(g, t);
        std::size_t end = 0;
        const Net n = net_of(g, t, 0, &end);
        wrong += determined(n) ? 0 : 1;
        unreduced += key(reduce(n)) == key(n) ? 0 : 1;
        check(key(n) == canon_key(g, t), "the canonical key is gen_audit's key of the reduced tree");
    }
    check(wrong == 0, "typed trees: none undetermined (" + std::to_string(wrong) + ")");
    check(unreduced == 0, "typed trees: none unreduced (" + std::to_string(unreduced) + ")");

    GEOptions ge;
    int valid = 0;
    wrong = unreduced = 0;
    for (int k = 0; k < 10000; ++k) {
        const Genome x = random_genome(ge, rng);
        Decoded d = decode(g, ge, x);
        if (!d.valid) continue;
        ++valid;
        check(obeys(g, d.tree), "GE: a valid derivation obeys the grammar");
        canonicalize(g, d.tree);
        std::size_t end = 0;
        const Net n = net_of(g, d.tree, 0, &end);
        wrong += determined(n) ? 0 : 1;
        unreduced += key(reduce(n)) == key(n) ? 0 : 1;
    }
    check(valid > 3000, "GE: enough random genomes decode (" + std::to_string(valid) + ")");
    check(wrong == 0, "GE: none undetermined (" + std::to_string(wrong) + ")");
    check(unreduced == 0, "GE: none unreduced (" + std::to_string(unreduced) + ")");
}

// ── the canonical key ───────────────────────────────────────────────────────
static void canonical_key()
{
    const Grammar g = series_parallel(3);
    const DepthIndex ix = depth_index(g);
    TreeOptions o;
    o.max_depth = 5;
    o.init_depth = 0;
    std::mt19937 rng(5);
    for (int k = 0; k < 2000; ++k) {
        Tree t = random_tree(g, ix, o, rng);
        const std::string key0 = canon_key(g, t);
        // permute the children of one symbol of a random commutative node
        const std::size_t i = static_cast<std::size_t>(mootation::rnd::below(rng, t.nodes.size()));
        const Production& p = production(g, t.nodes[i]);
        std::vector<std::vector<Node>> kids;
        std::size_t at = i + 1;
        for (std::size_t c = 0; c < p.children.size(); ++c) {
            const std::size_t e = subtree_end(g, t, at);
            kids.emplace_back(t.nodes.begin() + static_cast<std::ptrdiff_t>(at),
                              t.nodes.begin() + static_cast<std::ptrdiff_t>(e));
            at = e;
        }
        std::vector<std::size_t> same;
        for (std::size_t c = 0; c < p.children.size(); ++c)
            if (p.children[c] == p.children.back()) same.push_back(c);
        if (same.size() >= 2) std::swap(kids[same.front()], kids[same.back()]);
        Tree u;
        u.nodes.assign(t.nodes.begin(), t.nodes.begin() + static_cast<std::ptrdiff_t>(i + 1));
        for (const auto& kd : kids) u.nodes.insert(u.nodes.end(), kd.begin(), kd.end());
        u.nodes.insert(u.nodes.end(), t.nodes.begin() + static_cast<std::ptrdiff_t>(at), t.nodes.end());
        check(obeys(g, u), "a permutation of one symbol's children obeys the grammar");
        check(canon_key(g, u) == key0, "the key ignores the order of a commutative node's children");
        Tree cu = u, ct = t;
        canonicalize(g, cu);
        canonicalize(g, ct);
        check(canon_key(g, cu) == key0, "canonicalize keeps the key");
        check(params_of(cu) == params_of(ct) && cu.nodes.size() == ct.nodes.size(),
              "a tree and its permutation have one canonical form");
        Tree cc = cu;
        canonicalize(g, cc);
        check(params_of(cc) == params_of(cu), "canonicalize is idempotent");
    }
}

// ── parameters ──────────────────────────────────────────────────────────────
static void parameters()
{
    const Grammar g = series_parallel(3);
    const DepthIndex ix = depth_index(g);
    TreeOptions o;
    o.max_depth = 4;
    o.init_depth = 0;
    std::mt19937 rng(9);
    Tree t = random_tree(g, ix, o, rng);
    std::vector<double> v = params_of(t);
    check(!v.empty(), "a tree with leaves has parameters");
    for (double& x : v) {
        check(x >= 0.1 && x <= 10.0, "a new parameter is in its range");
        x *= 2.0;
    }
    set_params(t, v);
    check(params_of(t) == v, "set_params writes them back");
    bool refused = false;
    try {
        set_params(t, std::vector<double>(v.size() + 1, 1.0));
    } catch (const std::invalid_argument&) {
        refused = true;
    }
    check(refused, "set_params of the wrong length refused");

    GEOptions ge;
    Genome x;
    for (int k = 0; k < 100 && x.codons.empty(); ++k) {
        Genome y = random_genome(ge, rng);
        if (decode(g, ge, y).valid) x = y;
    }
    Decoded d = decode(g, ge, x);
    for (const Node& n : d.tree.nodes)
        for (double p : n.params) check(p == 0.5 * (0.1 + 10.0), "GE-7: a missing parameter is the middle");
    std::vector<double> fitted = params_of(d.tree);
    for (std::size_t k = 0; k < fitted.size(); ++k) fitted[k] = 1.0 + static_cast<double>(k);
    set_params(d.tree, fitted);
    write_params(x, d.tree);
    check(params_of(decode(g, ge, x).tree) == fitted, "GE-7: written back, decoded again");
}

// ── grammatical evolution ───────────────────────────────────────────────────
enum { EXPR = 0, GOP = 1, PRE = 2, VAR = 3 };

static Grammar oneill_ryan()
{
    Grammar g;
    g.symbols = {"expr", "op", "pre_op", "var"};
    g.rules = {{rule("", {EXPR, GOP, EXPR}), rule("()", {EXPR, GOP, EXPR}), rule("", {PRE, EXPR}),
                rule("", {VAR})},
               {rule("+"), rule("-"), rule("/"), rule("*")},
               {rule("Sin")},
               {rule("X"), rule("1.0")}};
    g.start = EXPR;
    return g;
}

static std::string text(const Grammar& g, const Tree& t, std::size_t i, std::size_t* end)
{
    const Node& n = t.nodes[i];
    const Production& p = production(g, n);
    std::string s;
    std::size_t at = i + 1;
    std::vector<std::string> kids;
    for (std::size_t c = 0; c < p.children.size(); ++c) kids.push_back(text(g, t, at, &at));
    *end = at;
    if (n.symbol == EXPR && n.rule == 1) return "(" + kids[0] + kids[1] + kids[2] + ")";
    if (n.symbol == EXPR && n.rule == 2) return kids[0] + "(" + kids[1] + ")";
    if (p.children.empty()) return p.label;
    for (const std::string& k : kids) s += k;
    return s;
}

static void grammatical_evolution()
{
    const Grammar g = oneill_ryan();
    depth_index(g);
    GEOptions o;
    // §IV-B's steps: Fig. 2's first nine codons 220 240 220 203 101 53 202 203
    // 102 give 1.0 - Sin(X) with <op><expr><op><expr> still open (GE-9: the
    // later codons of Fig. 2 do not give the paper's final expression), here
    // closed by 55 (*), 203 102 (X), 53 (-), 203 101 (1.0)
    const Genome steps{{220, 240, 220, 203, 101, 53, 202, 203, 102, 55, 203, 102, 53, 203, 101}, {}};
    const Decoded d = decode(g, o, steps);
    std::size_t end = 0;
    const std::string s = d.valid ? text(g, d.tree, 0, &end) : std::string();
    check(d.valid && d.wraps == 0 && d.codons_read == 15, "§IV-B: the steps decode without wrapping");
    check(s == "1.0-Sin(X)*X-1.0", "§IV-B: 1.0 - Sin(X) then the closing codons (got " + s + ")");
    const std::vector<std::pair<int, int>> choices{{EXPR, 0}, {EXPR, 0}, {EXPR, 0}, {EXPR, 3}, {VAR, 1},
                                                   {GOP, 1},  {EXPR, 2}, {PRE, 0},  {EXPR, 3}, {VAR, 0}};
    bool same = d.tree.nodes.size() >= choices.size();
    for (std::size_t k = 0; same && k < choices.size(); ++k)
        same = d.tree.nodes[k].symbol == choices[k].first && d.tree.nodes[k].rule == choices[k].second;
    check(same, "§IV-B: the paper's choices, <pre-op> reading no codon");
    const Genome fig2{{220, 240, 220, 203, 101, 53, 202, 203, 102, 55, 220, 202,
                       241, 130, 37, 202, 203, 140, 39, 202, 203, 102},
                      {}};
    const Decoded d2 = decode(g, o, fig2);
    const Decoded d3 = decode(g, o, fig2);
    check(d2.valid == d3.valid && d2.codons_read == d3.codons_read && d2.wraps == d3.wraps,
          "decoding is a function of the genome");
    check(!d2.valid, "GE-9: Fig. 2 as printed does not complete under Table I's grammar within 10 wraps");

    // a genome choosing <expr><op><expr> for ever: invalid after max_wraps wraps
    for (int wraps : {0, 1, 10}) {
        GEOptions w = o;
        w.max_wraps = wraps;
        const Decoded bad = decode(g, w, Genome{{0, 4, 8}, {}});
        check(!bad.valid && bad.tree.nodes.empty() && bad.wraps == wraps,
              "an endless derivation is invalid after max_wraps = " + std::to_string(wraps));
    }
    // wrapping that completes: one codon 3 -> <var>, read again: 3 mod 2 = 1, 1.0
    {
        GEOptions w = o;
        w.max_wraps = 1;
        const Decoded one = decode(g, w, Genome{{3}, {}});
        check(one.valid && one.wraps == 1 && one.codons_read == 2, "one codon read twice by wrapping");
        w.max_wraps = 0;
        check(!decode(g, w, Genome{{3}, {}}).valid, "the same with no wrapping is invalid");
    }

    // duplication and pruning leave a valid unwrapped derivation as it was
    std::mt19937 rng(3);
    int checked = 0;
    for (int k = 0; k < 5000; ++k) {
        Genome x = random_genome(o, rng);
        const Decoded dx = decode(g, o, x);
        if (!dx.valid || dx.wraps > 0) continue;
        ++checked;
        Genome y = x;
        duplication(o, y, rng);
        const Decoded dy = decode(g, o, y);
        check(dy.valid && params_of(dy.tree) == params_of(dx.tree) && dy.tree.nodes.size() == dx.tree.nodes.size() &&
                  dy.codons_read == dx.codons_read,
              "duplication keeps the derivation");
        Genome z = y;
        prune(z, dy);
        const Decoded dz = decode(g, o, z);
        check(dz.valid && dz.tree.nodes.size() == dx.tree.nodes.size() && z.codons.size() <= y.codons.size(),
              "pruning keeps the derivation");
        for (std::size_t n = 0; n < dz.tree.nodes.size(); ++n)
            check(dz.tree.nodes[n].rule == dx.tree.nodes[n].rule, "pruning keeps every choice");
    }
    check(checked > 100, "enough unwrapped genomes for the duplication check");

    // the operators keep codons in range and lengths within max_len
    GEOptions lim = o;
    lim.max_len = 40;
    std::vector<Genome> pop;
    for (int k = 0; k < 40; ++k) pop.push_back(random_genome(lim, rng));
    int bad = 0;
    for (int it = 0; it < 20000; ++it) {
        Genome& a = pop[static_cast<std::size_t>(mootation::rnd::below(rng, pop.size()))];
        Genome& b = pop[static_cast<std::size_t>(mootation::rnd::below(rng, pop.size()))];
        if (&a != &b) one_point_crossover(lim, a, b, rng);
        mutation(lim, a, rng);
        duplication(lim, a, rng);
        for (const Genome* x : {&a, &b}) {
            if (x->codons.empty() || x->codons.size() > 40) ++bad;
            for (int c : x->codons)
                if (c < 0 || c > 255) ++bad;
        }
    }
    check(bad == 0, "GE operators: codons in [0, 255], lengths in [1, max_len]");
    GEOptions codon = o;
    codon.mutation_codon = true;
    codon.p_mutation = 1.0;
    Genome m{{1, 2, 3, 4, 5, 6, 7, 8}, {}};
    mutation(codon, m, rng);
    for (int c : m.codons) check(c >= 0 && c <= 255, "codon mutation in range");
}

// ── the same seed, the same results; refusals ───────────────────────────────
static void determinism_and_refusals()
{
    const Grammar g = series_parallel(3);
    const DepthIndex ix = depth_index(g);
    TreeOptions o;
    o.max_depth = 5;
    o.init_depth = 0;
    std::mt19937 r1(42), r2(42);
    for (int k = 0; k < 200; ++k) {
        Tree a = random_tree(g, ix, o, r1), b = random_tree(g, ix, o, r2);
        check(canon_key(g, a) == canon_key(g, b) && params_of(a) == params_of(b), "same seed, same tree");
        subtree_mutation(g, ix, o, a, r1);
        subtree_mutation(g, ix, o, b, r2);
        check(canon_key(g, a) == canon_key(g, b) && params_of(a) == params_of(b), "same seed, same mutation");
    }
    GEOptions ge;
    for (int k = 0; k < 200; ++k)
        check(random_genome(ge, r1).codons == random_genome(ge, r2).codons, "same seed, same genome");

    Grammar loop;
    loop.symbols = {"A", "B"};
    loop.rules = {{rule("a", {1})}, {rule("b", {0})}};
    bool refused = false;
    try {
        depth_index(loop);
    } catch (const std::invalid_argument&) {
        refused = true;
    }
    check(refused, "a symbol with no finite tree refused");
    refused = false;
    try {
        decode(g, ge, Genome{{1, 256}, {}});
    } catch (const std::invalid_argument&) {
        refused = true;
    }
    check(refused, "a codon out of range refused");
}

int main()
{
    ratle_sebag();
    montana();
    operators_keep_types();
    correct_by_construction();
    canonical_key();
    parameters();
    grammatical_evolution();
    determinism_and_refusals();
    if (g_failed) {
        std::printf("%d check(s) failed\n", g_failed);
        return 1;
    }
    std::printf("all checks passed\n");
    return 0;
}

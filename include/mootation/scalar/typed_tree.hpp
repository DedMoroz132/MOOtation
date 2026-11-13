#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// A structural genotype: the derivation tree of a typed grammar, with
// operators that keep the types (task 27, item 24.1.2 of the 2026-11-13
// package). Nothing here is used by a method unless asked for: the header
// adds types and functions and changes no run.
//
// THE GRAMMAR. A context-free grammar in Backus-Naur form (Ratle & Sebag 2001,
// §4: the four-tuple {S, N, T, P}) whose non-terminals are the types of
// strongly typed genetic programming (Montana 1995, §2.1: every node returns
// the type its parent's argument wants). Grammar::symbols are the types;
// rules[s] are the productions of symbol s; a Production names the types of
// its arguments (children, in order; none: a terminal production, a leaf),
// the label it writes into the phenotype, the real parameters it carries
// (with the range a new one is drawn from) and whether the order of its
// children of one type is immaterial (commutative: a series or a parallel
// connection, a sum).
//
// THE TREE. Tree::nodes in preorder; a node is an occurrence of a symbol and
// the production that rewrote it, with that production's parameters. A
// subtree is a contiguous run of nodes, so crossover and mutation are splices
// of runs. Every node carries a non-terminal: the terminals of the grammar
// are inside the productions, as in a derivation tree, which is the genotype
// in Ratle & Sebag's G3P (§6.1); the user's decoder reads the phenotype off
// the labels and parameters.
//
// THE OPERATORS (sources strongly-typed-genetic-programming_montana1995,
// grammar-guided-genetic_ratle2001):
//   random_tree — Ratle & Sebag §6.2: every symbol and production gets the
//     index "depth of the smallest tree that resolves it" (a terminal
//     production 1; a symbol the least of its productions; a production 1 +
//     the largest of its arguments); an occurrence with depth budget r is
//     rewritten by a production drawn uniformly among those whose index is
//     <= r (the others are masked), its arguments get r - 1. Every tree so
//     built obeys the grammar and the depth limit. (Montana's "types
//     possibilities table" of the grow method, §2.1, is the same index read
//     the other way: type t is in table_entry(i) iff index(t) <= i.)
//   crossover — Ratle & Sebag §6.1 and Montana §2.1: a node of the first
//     parent drawn uniformly, a node of the second drawn uniformly among
//     those of the same symbol; none: refused; else the two subtrees are
//     swapped, giving two children.
//   subtree_mutation — Ratle & Sebag §6.1, Montana §2.1 (Fig. 7): a node
//     drawn uniformly and its subtree replaced by a new derivation of the
//     same symbol (random_tree with the node's remaining depth).
//   point_mutation — Ratle & Sebag §7.2 (Table 2): a leaf drawn uniformly and
//     its terminal production replaced by another terminal production of the
//     same symbol.
//   canon_key, canonicalize — the structure's key, unchanged by permuting
//     the children of a commutative node; canonicalize puts the children in
//     the key's order, so a genome keeps one form (written back, Lamarckian).
//
// The random numbers are rnd:: (random.hpp) from the generator handed in: the
// same seed gives the same trees on every platform.
//
// Readings (OURS):
//   TREE-1. Depth. Ratle & Sebag §6.2 allow at level m "derivations with index
//     less than D_max - m"; their own examples (D_max = 2 leaves <E> only
//     <op><V><V>, index 2; Fig. 8 with D_max = 4) agree only with "index <=
//     the remaining budget", the root's budget D_max and a child's one less,
//     and with the start rule S := <E> adding no level. Here the depth is
//     counted on the derivation tree, every node a level (a lone terminal
//     production: depth 1, as Montana's tree of one terminal, §1.1); a
//     production is admissible when its index <= the budget. A grammar that
//     wants Ratle & Sebag's count makes <E> its start symbol (the test does
//     so for Fig. 7-8).
//   TREE-2. Size. The masking of §6.2 bounds the depth only. With max_size >
//     0 random_tree draws again (init_tries times at most, then refuses);
//     crossover and subtree_mutation keep the parent where the child would
//     exceed max_depth or max_size (Montana §1.1: "use the child if its
//     maximum depth is less than or equal to MAX-TREE-DEPTH").
//   TREE-3. Crossover without a partner. No node of the symbol in the second
//     parent: both parents stay as they are and crossover returns false
//     (Montana §2.1: "returns either the parents or nothing" — the parents).
//     The first parent's node is drawn among all its nodes, the root
//     included (every node is an occurrence of a non-terminal: Ratle &
//     Sebag's footnote 2 holds).
//   TREE-4. Mutation. The new subtree's budget is max_depth - level + 1 (the
//     root at level 1), and at most mutation_depth when that is > 0
//     (Montana's MAX-MUTATION-TREE-DEPTH, §1.1). Point mutation draws the
//     new production uniformly among the symbol's other terminal
//     productions; with none it changes nothing and returns false. New
//     parameters are uniform in the production's range.
//   TREE-5. The key is the structure without the parameters: a leaf's label;
//     a node's label and its children's keys in brackets, sorted for a
//     commutative production. canonicalize sorts the children of a
//     commutative node that have one symbol (the positions of other symbols
//     stay: the production fixes them) by key, equal keys by their
//     parameters (lexicographically), equal both: as they were.
//   TREE-6. Defaults: max_depth 7 and init_depth 5 are Montana's §3.3-3.4
//     settings (maximum depth 7, maximum initial depth 5); Ratle & Sebag
//     give no D_max. A problem sets its own.
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../random.hpp"

namespace mootation::scalar::structure {

struct Production {
    std::string label;                  // written into the phenotype ("S", "+", "x")
    std::vector<int> children;          // the symbols of its arguments; empty: a leaf
    std::vector<double> param_lo, param_hi;   // its real parameters' range (one pair each)
    bool commutative = false;           // the order of its children of one symbol is immaterial

    std::size_t n_params() const { return param_lo.size(); }
};

struct Grammar {
    std::vector<std::string> symbols;            // the types (non-terminals)
    std::vector<std::vector<Production>> rules;  // rules[s]: the productions of symbol s
    int start = 0;
};

struct Node {
    int symbol = 0;              // the type of this occurrence
    int rule = 0;                // which of rules[symbol] rewrote it
    std::vector<double> params;  // that production's parameters
};

struct Tree {
    std::vector<Node> nodes;     // preorder
};

struct TreeOptions {
    int max_depth = 7;           // every tree (TREE-1, TREE-6)
    int init_depth = 5;          // random_tree's budget; 0: max_depth
    int max_size = 0;            // nodes; 0: no limit (TREE-2)
    int mutation_depth = 0;      // a mutated subtree's depth; 0: no limit beyond max_depth (TREE-4)
    int init_tries = 100;        // random_tree's draws under max_size before it refuses
};

constexpr int no_tree = std::numeric_limits<int>::max();   // an index: no finite tree

// Ratle & Sebag's indices (§6.2): of[s] the symbol's, rule[s][r] the production's.
struct DepthIndex {
    std::vector<int> of;
    std::vector<std::vector<int>> rule;
};

inline const Production& production(const Grammar& g, const Node& n)
{
    return g.rules[static_cast<std::size_t>(n.symbol)][static_cast<std::size_t>(n.rule)];
}

// The grammar checked (refusal with a message naming what is wrong) and its
// indices, by fixed-point iteration: a production's index is known once its
// arguments' are; a symbol none of whose productions ever gets one derives no
// finite tree and is refused.
inline DepthIndex depth_index(const Grammar& g)
{
    const std::size_t ns = g.symbols.size();
    if (ns == 0 || g.rules.size() != ns)
        throw std::invalid_argument("grammar: one list of productions per symbol");
    if (g.start < 0 || static_cast<std::size_t>(g.start) >= ns)
        throw std::invalid_argument("grammar: the start symbol is not a symbol");
    for (std::size_t s = 0; s < ns; ++s) {
        if (g.rules[s].empty())
            throw std::invalid_argument("grammar: symbol " + g.symbols[s] + " has no production");
        for (const Production& p : g.rules[s]) {
            for (int c : p.children)
                if (c < 0 || static_cast<std::size_t>(c) >= ns)
                    throw std::invalid_argument("grammar: a production of " + g.symbols[s] +
                                                " names a symbol that is not one");
            if (p.param_hi.size() != p.param_lo.size())
                throw std::invalid_argument("grammar: a production of " + g.symbols[s] +
                                            " has param_lo and param_hi of different lengths");
            for (std::size_t k = 0; k < p.param_lo.size(); ++k)
                if (!(p.param_lo[k] <= p.param_hi[k]) || !std::isfinite(p.param_lo[k]) ||
                    !std::isfinite(p.param_hi[k]))
                    throw std::invalid_argument("grammar: a production of " + g.symbols[s] +
                                                " has a parameter range that is not lo <= hi, finite");
        }
    }
    DepthIndex ix;
    ix.of.assign(ns, no_tree);
    ix.rule.resize(ns);
    for (std::size_t s = 0; s < ns; ++s) ix.rule[s].assign(g.rules[s].size(), no_tree);
    for (bool changed = true; changed;) {
        changed = false;
        for (std::size_t s = 0; s < ns; ++s) {
            for (std::size_t r = 0; r < g.rules[s].size(); ++r) {
                int deepest = 0;
                for (int c : g.rules[s][r].children)
                    deepest = std::max(deepest, ix.of[static_cast<std::size_t>(c)]);
                const int index = deepest == no_tree ? no_tree : 1 + deepest;
                if (index < ix.rule[s][r]) {
                    ix.rule[s][r] = index;
                    changed = true;
                }
                if (index < ix.of[s]) {
                    ix.of[s] = index;
                    changed = true;
                }
            }
        }
    }
    for (std::size_t s = 0; s < ns; ++s)
        if (ix.of[s] == no_tree)
            throw std::invalid_argument("grammar: symbol " + g.symbols[s] + " derives no finite tree");
    return ix;
}

// One past the last node of the subtree that starts at i.
inline std::size_t subtree_end(const Grammar& g, const Tree& t, std::size_t i)
{
    std::size_t open = 1;     // subtrees begun and not yet finished
    while (open > 0) {
        open += production(g, t.nodes[i]).children.size();
        --open;
        ++i;
    }
    return i;
}

// The level of every node, the root's 1; the tree's depth is the largest.
inline std::vector<int> levels(const Grammar& g, const Tree& t)
{
    std::vector<int> level(t.nodes.size(), 0);
    std::vector<std::pair<int, std::size_t>> open;   // (a node's level, its children still to come)
    for (std::size_t i = 0; i < t.nodes.size(); ++i) {
        while (!open.empty() && open.back().second == 0) open.pop_back();
        int lv = 1;
        if (!open.empty()) {
            lv = open.back().first + 1;
            --open.back().second;
        }
        level[i] = lv;
        const std::size_t k = production(g, t.nodes[i]).children.size();
        if (k > 0) open.emplace_back(lv, k);
    }
    return level;
}

inline int depth(const Grammar& g, const Tree& t)
{
    int d = 0;
    for (int lv : levels(g, t)) d = std::max(d, lv);
    return d;
}

// Does t derive `symbol` by the grammar: every node's production one of its
// symbol's, every child of the symbol its parent's production wants, the
// parameters as many as the production's, no node missing or left over?
inline bool obeys(const Grammar& g, const Tree& t, int symbol)
{
    std::vector<int> want{symbol};                    // the symbols still to come, the next on top
    for (const Node& n : t.nodes) {
        if (want.empty() || n.symbol != want.back()) return false;
        want.pop_back();
        if (n.symbol < 0 || static_cast<std::size_t>(n.symbol) >= g.rules.size()) return false;
        const auto& rs = g.rules[static_cast<std::size_t>(n.symbol)];
        if (n.rule < 0 || static_cast<std::size_t>(n.rule) >= rs.size()) return false;
        const Production& p = rs[static_cast<std::size_t>(n.rule)];
        if (n.params.size() != p.n_params()) return false;
        for (auto c = p.children.rbegin(); c != p.children.rend(); ++c) want.push_back(*c);
    }
    return want.empty();
}

inline bool obeys(const Grammar& g, const Tree& t) { return obeys(g, t, g.start); }

inline bool fits(const Grammar& g, const TreeOptions& o, const Tree& t)
{
    if (o.max_size > 0 && t.nodes.size() > static_cast<std::size_t>(o.max_size)) return false;
    return depth(g, t) <= o.max_depth;
}

template <typename G>
inline std::vector<double> draw_params(const Production& p, G& rng)
{
    std::vector<double> v(p.n_params());
    for (std::size_t k = 0; k < v.size(); ++k)
        v[k] = p.param_lo[k] + (p.param_hi[k] - p.param_lo[k]) * rnd::uniform01(rng);
    return v;
}

// A derivation of `symbol` within depth `budget`, appended to out (Ratle &
// Sebag §6.2). The occurrences are rewritten in preorder: the rule, then its
// parameters, are drawn for each in turn.
template <typename G>
inline void grow(const Grammar& g, const DepthIndex& ix, int symbol, int budget, std::vector<Node>& out,
                 G& rng)
{
    if (ix.of[static_cast<std::size_t>(symbol)] > budget)
        throw std::invalid_argument("typed tree: symbol " + g.symbols[static_cast<std::size_t>(symbol)] +
                                    " has no tree within depth " + std::to_string(budget));
    std::vector<std::pair<int, int>> todo{{symbol, budget}};   // (symbol, budget), the next on top
    std::vector<int> admissible;
    while (!todo.empty()) {
        const auto [s, r] = todo.back();
        todo.pop_back();
        const auto su = static_cast<std::size_t>(s);
        admissible.clear();
        for (std::size_t k = 0; k < g.rules[su].size(); ++k)
            if (ix.rule[su][k] <= r) admissible.push_back(static_cast<int>(k));
        const int rule = admissible[static_cast<std::size_t>(rnd::below(rng, admissible.size()))];
        const Production& p = g.rules[su][static_cast<std::size_t>(rule)];
        out.push_back(Node{s, rule, draw_params(p, rng)});
        for (auto c = p.children.rbegin(); c != p.children.rend(); ++c) todo.emplace_back(*c, r - 1);
    }
}

// A tree of the start symbol within init_depth (0: max_depth) and max_size.
template <typename G>
inline Tree random_tree(const Grammar& g, const DepthIndex& ix, const TreeOptions& o, G& rng)
{
    const int budget = o.init_depth > 0 ? std::min(o.init_depth, o.max_depth) : o.max_depth;
    Tree t;
    for (int k = 0; k < std::max(1, o.init_tries); ++k) {
        t.nodes.clear();
        grow(g, ix, g.start, budget, t.nodes, rng);
        if (o.max_size <= 0 || t.nodes.size() <= static_cast<std::size_t>(o.max_size)) return t;
    }
    throw std::runtime_error("typed tree: no tree within max_size = " + std::to_string(o.max_size) +
                             " in " + std::to_string(o.init_tries) + " draws");
}

// The nodes [0, i) of a, then sub, then [e, end) of a.
inline Tree splice(const Tree& a, std::size_t i, std::size_t e, const Node* sub_first, const Node* sub_last)
{
    Tree c;
    c.nodes.reserve(a.nodes.size() - (e - i) + static_cast<std::size_t>(sub_last - sub_first));
    c.nodes.insert(c.nodes.end(), a.nodes.begin(), a.nodes.begin() + static_cast<std::ptrdiff_t>(i));
    c.nodes.insert(c.nodes.end(), sub_first, sub_last);
    c.nodes.insert(c.nodes.end(), a.nodes.begin() + static_cast<std::ptrdiff_t>(e), a.nodes.end());
    return c;
}

// Subtrees of one symbol swapped between a and b (TREE-3): true if either
// changed; a child beyond the limits leaves its parent as it was (TREE-2).
template <typename G>
inline bool crossover(const Grammar& g, const TreeOptions& o, Tree& a, Tree& b, G& rng)
{
    if (a.nodes.empty() || b.nodes.empty()) return false;
    const std::size_t i = static_cast<std::size_t>(rnd::below(rng, a.nodes.size()));
    std::vector<std::size_t> same;
    for (std::size_t k = 0; k < b.nodes.size(); ++k)
        if (b.nodes[k].symbol == a.nodes[i].symbol) same.push_back(k);
    if (same.empty()) return false;
    const std::size_t j = same[static_cast<std::size_t>(rnd::below(rng, same.size()))];
    const std::size_t ie = subtree_end(g, a, i), je = subtree_end(g, b, j);
    Tree ca = splice(a, i, ie, b.nodes.data() + j, b.nodes.data() + je);
    Tree cb = splice(b, j, je, a.nodes.data() + i, a.nodes.data() + ie);
    const bool take_a = fits(g, o, ca), take_b = fits(g, o, cb);
    if (take_a) a = std::move(ca);
    if (take_b) b = std::move(cb);
    return take_a || take_b;
}

// A node's subtree replaced by a new derivation of its symbol (TREE-4).
template <typename G>
inline bool subtree_mutation(const Grammar& g, const DepthIndex& ix, const TreeOptions& o, Tree& t, G& rng)
{
    if (t.nodes.empty()) return false;
    const std::size_t i = static_cast<std::size_t>(rnd::below(rng, t.nodes.size()));
    int budget = o.max_depth - levels(g, t)[i] + 1;
    if (o.mutation_depth > 0) budget = std::min(budget, o.mutation_depth);
    const int s = t.nodes[i].symbol;
    if (ix.of[static_cast<std::size_t>(s)] > budget) return false;
    std::vector<Node> sub;
    grow(g, ix, s, budget, sub, rng);
    Tree c = splice(t, i, subtree_end(g, t, i), sub.data(), sub.data() + sub.size());
    if (!fits(g, o, c)) return false;
    t = std::move(c);
    return true;
}

// A leaf's terminal production replaced by another of its symbol (TREE-4).
template <typename G>
inline bool point_mutation(const Grammar& g, Tree& t, G& rng)
{
    std::vector<std::size_t> leaves;
    for (std::size_t k = 0; k < t.nodes.size(); ++k)
        if (production(g, t.nodes[k]).children.empty()) leaves.push_back(k);
    if (leaves.empty()) return false;
    Node& n = t.nodes[leaves[static_cast<std::size_t>(rnd::below(rng, leaves.size()))]];
    const auto& rs = g.rules[static_cast<std::size_t>(n.symbol)];
    std::vector<int> others;
    for (std::size_t r = 0; r < rs.size(); ++r)
        if (static_cast<int>(r) != n.rule && rs[r].children.empty()) others.push_back(static_cast<int>(r));
    if (others.empty()) return false;
    n.rule = others[static_cast<std::size_t>(rnd::below(rng, others.size()))];
    n.params = draw_params(production(g, n), rng);
    return true;
}

// The parameters of all nodes in preorder (what a local fit adjusts) and
// their write-back (Lamarckian; the count must match).
inline std::vector<double> params_of(const Tree& t)
{
    std::vector<double> v;
    for (const Node& n : t.nodes) v.insert(v.end(), n.params.begin(), n.params.end());
    return v;
}

inline void set_params(Tree& t, const std::vector<double>& v)
{
    std::size_t k = 0;
    for (const Node& n : t.nodes) k += n.params.size();
    if (k != v.size())
        throw std::invalid_argument("typed tree: " + std::to_string(v.size()) + " parameters for a tree of " +
                                    std::to_string(k));
    k = 0;
    for (Node& n : t.nodes)
        for (double& p : n.params) p = v[k++];
}

namespace detail {

// The canonical form of the subtree at i (TREE-5): its nodes, its key, its
// parameters in preorder; *end is one past the subtree in t.
struct Canonical {
    std::vector<Node> nodes;
    std::string key;
    std::vector<double> params;
};

inline const std::string& label_of(const Grammar& g, const Node& n)
{
    const Production& p = production(g, n);
    return p.label.empty() ? g.symbols[static_cast<std::size_t>(n.symbol)] : p.label;
}

inline Canonical canonical(const Grammar& g, const Tree& t, std::size_t i, std::size_t* end)
{
    const Node& n = t.nodes[i];
    const Production& p = production(g, n);
    std::vector<Canonical> kids;
    std::size_t at = i + 1;
    for (std::size_t c = 0; c < p.children.size(); ++c) kids.push_back(canonical(g, t, at, &at));
    *end = at;
    if (p.commutative) {
        // the children of each symbol sorted among the positions of that symbol
        auto before = [](const Canonical& x, const Canonical& y) {
            if (x.key != y.key) return x.key < y.key;
            return std::lexicographical_compare(x.params.begin(), x.params.end(), y.params.begin(),
                                                y.params.end());
        };
        for (std::size_t c = 0; c < kids.size(); ++c) {
            std::vector<std::size_t> pos;
            for (std::size_t d = c; d < kids.size(); ++d)
                if (p.children[d] == p.children[c]) pos.push_back(d);
            if (pos.front() != c) continue;                    // that symbol's group is done
            std::vector<Canonical> group;
            for (std::size_t d : pos) group.push_back(std::move(kids[d]));
            std::stable_sort(group.begin(), group.end(), before);
            for (std::size_t k = 0; k < pos.size(); ++k) kids[pos[k]] = std::move(group[k]);
        }
    }
    Canonical out;
    out.nodes.push_back(n);
    out.params = n.params;
    std::vector<std::string> keys;
    for (Canonical& k : kids) {
        out.nodes.insert(out.nodes.end(), k.nodes.begin(), k.nodes.end());
        out.params.insert(out.params.end(), k.params.begin(), k.params.end());
        keys.push_back(std::move(k.key));
    }
    if (p.commutative) std::sort(keys.begin(), keys.end());
    out.key = label_of(g, n);
    if (!keys.empty()) {
        out.key += '(';
        for (std::size_t k = 0; k < keys.size(); ++k) {
            if (k > 0) out.key += ',';
            out.key += keys[k];
        }
        out.key += ')';
    }
    return out;
}

}  // namespace detail

inline std::string canon_key(const Grammar& g, const Tree& t)
{
    if (t.nodes.empty()) return std::string();
    std::size_t end = 0;
    return detail::canonical(g, t, 0, &end).key;
}

inline void canonicalize(const Grammar& g, Tree& t)
{
    if (t.nodes.empty()) return;
    std::size_t end = 0;
    t.nodes = detail::canonical(g, t, 0, &end).nodes;
}

}  // namespace mootation::scalar::structure

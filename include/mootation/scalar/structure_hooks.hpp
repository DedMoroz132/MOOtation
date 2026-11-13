#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// The structural genotypes (typed_tree.hpp, grammatical_evolution.hpp) as a
// problem's genome operators (task 27 on task 24's contract, item 24.1.2):
// the methods keep a vector of fixed length n, so a tree or a GE genome is
// packed into one, and make_tree_hooks / make_ge_hooks fill the Problem's
// sample, crossover, mutate, decode, canon_key (and lo, hi) with functions that
// unpack, apply the operators of the genotype and pack again. The user's
// evaluate receives the packed genome and unpacks it with the codec
// (TreeCodec::unpack, GECodec::derivation); the real parameters it fits are
// the active coordinates of decode (a local search, 24.1.1, moves them only)
// and come back into the genome by decode's x_canon (Lamarckian).
//
// THE LAYOUT.
//   tree (M = max_size nodes, K = the most parameters of one production):
//     x[0]              the number of nodes, in [1, M]
//     x[1 .. M]         each node's production (its index among its symbol's
//                       productions), preorder; -1 past the last node
//     x[1+M .. 1+M+MK)  the parameters in preorder; param_lo past the last
//   GE (L = max_len codons, P = max_params parameters):
//     x[0]              the number of codons, in [1, L]
//     x[1]              the number of parameters kept, in [0, P]
//     x[2 .. 2+L)       the codons; -1 past the last
//     x[2+L .. 2+L+P)   the parameters (GE-7); param_lo past the last
//   The symbols are not stored: a preorder list of productions determines
//   them (the grammar names each child's symbol).
//
// Readings (OURS):
//   HOOK-1. A malformed vector (a count or a production out of range, a
//     derivation that ends before or after the count, a non-integer where an
//     integer belongs) is refused with std::invalid_argument naming the
//     coordinate; the operators here never make one, so a refusal means a
//     foreign operator or bound repair changed the structure's coordinates.
//   HOOK-2. Parameters live in one box [param_lo, param_hi] for every slot
//     (the vector's bounds cannot depend on the structure); by default the
//     union of the productions' ranges. A production's own range is where a
//     new parameter is drawn (TREE-4), the box is where a fit may move it.
//   HOOK-3. The probabilities inside the hooks. The GA calls crossover for
//     every pair in SBX's place (ga.hpp GA-9), so the hook draws its own
//     coin: a pair is crossed with p_crossover, else the children are the
//     parents — 0.8 for trees (Ratle & Sebag 2001, Table 2), 0.9 for GE
//     (O'Neill & Ryan 2001, Tables II-IV). mutate applies, each with its own
//     probability, subtree mutation (0.2) and point mutation (0.8) to a tree
//     (Ratle & Sebag, Table 2), and to a GE genome the mutation of every bit
//     (p_mutation 0.01), then duplication (0.01) (O'Neill & Ryan, Tables
//     II-IV), then pruning (default 0: Ryan, Collins & O'Neill 1998 apply
//     it "with a probability" and give none, and O'Neill & Ryan 2001 do not
//     use it; GE-3).
//   HOOK-4. decode of a tree writes back its canonical form (canonicalize):
//     the children of a commutative node in the key's order. A GE genome's
//     codons fix the derivation's order, so its decode writes back only the
//     parameters (the derivation's, in preorder) and keeps the codons; an
//     invalid genome (GE's wrap limit) decodes to valid = false with x as
//     it was.
//   HOOK-5. A GE derivation with more parameters than P keeps the first P in
//     the genome; the others take the middle of their production's range on
//     every decode and are not active (GE-7). Choose P large enough.
//   HOOK-6. canon_key (the cache of 24.1.4 with cache = canon) is the
//     structure's key without the parameters (TREE-5): one evaluation per
//     structure, right where evaluate fits the parameters itself (a local
//     fit inside the evaluation, 04 Р3, Р5). A problem whose f depends on
//     the parameters stored in the genome keeps cache = off or exact, or sets
//     its own canon_key. A GE genome that does not decode has the key
//     "(invalid)" (it is never evaluated).
// ============================================================================

#include <algorithm>
#include <cstddef>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../random.hpp"
#include "evaluator.hpp"
#include "grammatical_evolution.hpp"
#include "typed_tree.hpp"

namespace mootation::scalar::structure {

namespace detail {

// x[k] as an integer in [lo, hi], or a refusal (HOOK-1).
inline int slot_int(const std::vector<double>& x, std::size_t k, int lo, int hi, const char* what)
{
    const double v = x[k];
    if (!(v >= lo && v <= hi) || static_cast<double>(static_cast<int>(v)) != v)
        throw std::invalid_argument(std::string("structure genome: coordinate ") + std::to_string(k) + " (" +
                                    what + ") = " + std::to_string(v) + " is not an integer in [" +
                                    std::to_string(lo) + ", " + std::to_string(hi) + "]");
    return static_cast<int>(v);
}

inline std::pair<double, double> param_union(const Grammar& g)
{
    bool any = false;
    double lo = 0.0, hi = 0.0;
    for (const auto& rs : g.rules)
        for (const Production& p : rs)
            for (std::size_t k = 0; k < p.n_params(); ++k) {
                lo = any ? std::min(lo, p.param_lo[k]) : p.param_lo[k];
                hi = any ? std::max(hi, p.param_hi[k]) : p.param_hi[k];
                any = true;
            }
    return {lo, hi};
}

inline int most_rules(const Grammar& g)
{
    std::size_t r = 1;
    for (const auto& rs : g.rules) r = std::max(r, rs.size());
    return static_cast<int>(r);
}

}  // namespace detail

// ── the tree ────────────────────────────────────────────────────────────────

struct TreeHookOptions {
    TreeOptions tree;               // tree.max_size > 0 is the layout's M
    double p_crossover = 0.8;       // HOOK-3
    double p_subtree = 0.2, p_point = 0.8;
    bool param_box_given = false;   // false: the union of the productions' ranges (HOOK-2)
    double param_lo = 0.0, param_hi = 0.0;
};

struct TreeCodec {
    Grammar grammar;
    DepthIndex index;
    TreeHookOptions o;
    std::size_t M = 0, K = 0;       // nodes, the most parameters of one production
    double param_lo = 0.0, param_hi = 0.0;

    TreeCodec(Grammar g, const TreeHookOptions& opt) : grammar(std::move(g)), o(opt)
    {
        index = depth_index(grammar);
        if (o.tree.max_size <= 0)
            throw std::invalid_argument("tree genome: tree.max_size must be > 0 (the vector's length)");
        M = static_cast<std::size_t>(o.tree.max_size);
        for (const auto& rs : grammar.rules)
            for (const Production& p : rs) K = std::max(K, p.n_params());
        const auto box = detail::param_union(grammar);
        param_lo = o.param_box_given ? o.param_lo : box.first;
        param_hi = o.param_box_given ? o.param_hi : box.second;
        if (!(param_lo <= param_hi))
            throw std::invalid_argument("tree genome: param_lo > param_hi");
    }

    std::size_t n() const { return 1 + M + M * K; }

    void bounds(std::vector<double>& lo, std::vector<double>& hi) const
    {
        lo.assign(n(), param_lo);
        hi.assign(n(), param_hi);
        lo[0] = 1.0;
        hi[0] = static_cast<double>(M);
        for (std::size_t i = 1; i <= M; ++i) {
            lo[i] = -1.0;
            hi[i] = static_cast<double>(detail::most_rules(grammar) - 1);
        }
    }

    std::vector<double> pack(const Tree& t) const
    {
        const std::vector<double> p = params_of(t);
        if (t.nodes.empty() || t.nodes.size() > M || p.size() > M * K)
            throw std::invalid_argument("tree genome: a tree of " + std::to_string(t.nodes.size()) +
                                        " nodes does not fit max_size = " + std::to_string(M));
        std::vector<double> x(n(), param_lo);
        x[0] = static_cast<double>(t.nodes.size());
        for (std::size_t i = 0; i < M; ++i) x[1 + i] = i < t.nodes.size() ? t.nodes[i].rule : -1.0;
        std::copy(p.begin(), p.end(), x.begin() + static_cast<std::ptrdiff_t>(1 + M));
        return x;
    }

    Tree unpack(const std::vector<double>& x) const
    {
        if (x.size() != n())
            throw std::invalid_argument("tree genome: a vector of " + std::to_string(x.size()) + " for " +
                                        std::to_string(n()));
        const auto count = static_cast<std::size_t>(detail::slot_int(x, 0, 1, static_cast<int>(M), "node count"));
        Tree t;
        std::vector<int> want{grammar.start};
        std::size_t at = 1 + M;
        for (std::size_t i = 0; i < count; ++i) {
            if (want.empty())
                throw std::invalid_argument("tree genome: the derivation ends before node " + std::to_string(i));
            const int s = want.back();
            want.pop_back();
            const auto& rs = grammar.rules[static_cast<std::size_t>(s)];
            const int r = detail::slot_int(x, 1 + i, 0, static_cast<int>(rs.size()) - 1, "production");
            const Production& p = rs[static_cast<std::size_t>(r)];
            Node nd{s, r, std::vector<double>(x.begin() + static_cast<std::ptrdiff_t>(at),
                                              x.begin() + static_cast<std::ptrdiff_t>(at + p.n_params()))};
            at += p.n_params();
            t.nodes.push_back(std::move(nd));
            for (auto c = p.children.rbegin(); c != p.children.rend(); ++c) want.push_back(*c);
        }
        if (!want.empty())
            throw std::invalid_argument("tree genome: the derivation is incomplete after " + std::to_string(count) +
                                        " nodes");
        return t;
    }

    // decode: the canonical form written back (HOOK-4), the used parameters active
    scalar::Decoded decode(const std::vector<double>& x) const
    {
        Tree t = unpack(x);
        canonicalize(grammar, t);
        scalar::Decoded d;
        d.x_canon = pack(t);
        d.active.assign(n(), 0);
        const std::size_t used = params_of(t).size();
        for (std::size_t k = 0; k < used; ++k) d.active[1 + M + k] = 1;
        d.valid = true;
        return d;
    }
};

// The Problem's genome hooks for trees of grammar g; n, lo and hi set too.
inline std::shared_ptr<const TreeCodec> make_tree_hooks(Problem& P, Grammar g, const TreeHookOptions& o)
{
    auto c = std::make_shared<const TreeCodec>(std::move(g), o);
    c->bounds(P.lo, P.hi);
    P.sample = [c](std::vector<double>& x, std::mt19937& rng) {
        x = c->pack(random_tree(c->grammar, c->index, c->o.tree, rng));
    };
    P.crossover = [c](const std::vector<double>& p1, const std::vector<double>& p2, std::vector<double>& c1,
                      std::vector<double>& c2, std::mt19937& rng) {
        Tree a = c->unpack(p1), b = c->unpack(p2);
        if (rnd::uniform01(rng) < c->o.p_crossover) crossover(c->grammar, c->o.tree, a, b, rng);
        c1 = c->pack(a);
        c2 = c->pack(b);
    };
    P.mutate = [c](std::vector<double>& x, std::mt19937& rng) {
        Tree t = c->unpack(x);
        if (rnd::uniform01(rng) < c->o.p_subtree) subtree_mutation(c->grammar, c->index, c->o.tree, t, rng);
        if (rnd::uniform01(rng) < c->o.p_point) point_mutation(c->grammar, t, rng);
        x = c->pack(t);
    };
    P.repair = nullptr;
    P.decode = [c](const std::vector<double>& x) { return c->decode(x); };
    P.canon_key = [c](const std::vector<double>& x) { return canon_key(c->grammar, c->unpack(x)); };   // HOOK-6
    return c;
}

// ── grammatical evolution ───────────────────────────────────────────────────

struct GEHookOptions {
    GEOptions ge;                   // ge.max_len > 0 is the layout's L
    int max_params = 0;             // the layout's P (HOOK-5)
    double p_crossover = 0.9;       // HOOK-3
    double p_duplication = 0.01, p_prune = 0.0;
    bool param_box_given = false;   // HOOK-2
    double param_lo = 0.0, param_hi = 0.0;
};

struct GECodec {
    Grammar grammar;
    GEHookOptions o;
    std::size_t L = 0, P = 0;
    double param_lo = 0.0, param_hi = 0.0;

    GECodec(Grammar g, const GEHookOptions& opt) : grammar(std::move(g)), o(opt)
    {
        depth_index(grammar);   // the grammar's checks
        if (o.ge.max_len <= 0)
            throw std::invalid_argument("GE genome: ge.max_len must be > 0 (the vector's length)");
        if (o.max_params < 0) throw std::invalid_argument("GE genome: max_params < 0");
        if (o.ge.init_len_lo < 1 || o.ge.init_len_hi > o.ge.max_len || o.ge.init_len_lo > o.ge.init_len_hi)
            throw std::invalid_argument("GE genome: initial lengths must lie in [1, max_len]");
        L = static_cast<std::size_t>(o.ge.max_len);
        P = static_cast<std::size_t>(o.max_params);
        const auto box = detail::param_union(grammar);
        param_lo = o.param_box_given ? o.param_lo : box.first;
        param_hi = o.param_box_given ? o.param_hi : box.second;
        if (!(param_lo <= param_hi)) throw std::invalid_argument("GE genome: param_lo > param_hi");
    }

    std::size_t n() const { return 2 + L + P; }

    void bounds(std::vector<double>& lo, std::vector<double>& hi) const
    {
        lo.assign(n(), param_lo);
        hi.assign(n(), param_hi);
        lo[0] = 1.0;
        hi[0] = static_cast<double>(L);
        lo[1] = 0.0;
        hi[1] = static_cast<double>(P);
        for (std::size_t i = 0; i < L; ++i) {
            lo[2 + i] = -1.0;
            hi[2 + i] = static_cast<double>((1 << o.ge.codon_bits) - 1);
        }
    }

    std::vector<double> pack(const Genome& g) const
    {
        if (g.codons.empty() || g.codons.size() > L)
            throw std::invalid_argument("GE genome: " + std::to_string(g.codons.size()) +
                                        " codons do not fit max_len = " + std::to_string(L));
        const std::size_t kept = std::min(g.params.size(), P);
        std::vector<double> x(n(), param_lo);
        x[0] = static_cast<double>(g.codons.size());
        x[1] = static_cast<double>(kept);
        for (std::size_t i = 0; i < L; ++i) x[2 + i] = i < g.codons.size() ? g.codons[i] : -1.0;
        std::copy(g.params.begin(), g.params.begin() + static_cast<std::ptrdiff_t>(kept),
                  x.begin() + static_cast<std::ptrdiff_t>(2 + L));
        return x;
    }

    Genome unpack(const std::vector<double>& x) const
    {
        if (x.size() != n())
            throw std::invalid_argument("GE genome: a vector of " + std::to_string(x.size()) + " for " +
                                        std::to_string(n()));
        const auto len = static_cast<std::size_t>(detail::slot_int(x, 0, 1, static_cast<int>(L), "codon count"));
        const auto kept = static_cast<std::size_t>(detail::slot_int(x, 1, 0, static_cast<int>(P), "parameter count"));
        Genome g;
        const int top = (1 << o.ge.codon_bits) - 1;
        for (std::size_t i = 0; i < len; ++i) g.codons.push_back(detail::slot_int(x, 2 + i, 0, top, "codon"));
        g.params.assign(x.begin() + static_cast<std::ptrdiff_t>(2 + L),
                        x.begin() + static_cast<std::ptrdiff_t>(2 + L + kept));
        return g;
    }

    // the derivation a user's evaluate works on (valid = false: not evaluated)
    structure::Mapping derivation(const std::vector<double>& x) const { return structure::decode(grammar, o.ge, unpack(x)); }

    // decode: the derivation's parameters written back (HOOK-4, HOOK-5)
    scalar::Decoded decode(const std::vector<double>& x) const
    {
        Genome g = unpack(x);
        const structure::Mapping d = structure::decode(grammar, o.ge, g);
        scalar::Decoded out;
        if (!d.valid) {
            out.x_canon = x;
            out.valid = false;
            return out;
        }
        write_params(g, d.tree);
        out.x_canon = pack(g);
        out.active.assign(n(), 0);
        const std::size_t used = std::min(g.params.size(), P);
        for (std::size_t k = 0; k < used; ++k) out.active[2 + L + k] = 1;
        out.valid = true;
        return out;
    }
};

inline std::shared_ptr<const GECodec> make_ge_hooks(Problem& Pb, Grammar g, const GEHookOptions& o)
{
    auto c = std::make_shared<const GECodec>(std::move(g), o);
    c->bounds(Pb.lo, Pb.hi);
    Pb.sample = [c](std::vector<double>& x, std::mt19937& rng) { x = c->pack(random_genome(c->o.ge, rng)); };
    Pb.crossover = [c](const std::vector<double>& p1, const std::vector<double>& p2, std::vector<double>& c1,
                       std::vector<double>& c2, std::mt19937& rng) {
        Genome a = c->unpack(p1), b = c->unpack(p2);
        if (rnd::uniform01(rng) < c->o.p_crossover) one_point_crossover(c->o.ge, a, b, rng);
        c1 = c->pack(a);
        c2 = c->pack(b);
    };
    Pb.mutate = [c](std::vector<double>& x, std::mt19937& rng) {
        Genome a = c->unpack(x);
        mutation(c->o.ge, a, rng);
        if (rnd::uniform01(rng) < c->o.p_duplication) duplication(c->o.ge, a, rng);
        if (rnd::uniform01(rng) < c->o.p_prune) prune(a, structure::decode(c->grammar, c->o.ge, a));
        x = c->pack(a);
    };
    Pb.repair = nullptr;
    Pb.decode = [c](const std::vector<double>& x) { return c->decode(x); };
    Pb.canon_key = [c](const std::vector<double>& x) {                                                // HOOK-6
        const structure::Mapping d = c->derivation(x);
        return d.valid ? canon_key(c->grammar, d.tree) : std::string("(invalid)");
    };
    return c;
}

}  // namespace mootation::scalar::structure

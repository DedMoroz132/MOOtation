#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// Grammatical evolution (task 27, item 24.1.2 of the 2026-11-13 package):
// M. O'Neill, C. Ryan, "Grammatical evolution", IEEE Trans. Evol. Comput.
// 5(4):349-358, 2001 (source grammatical-evolution_oneill2001). A genome of
// integer codons of variable length is mapped onto a derivation of a grammar
// in Backus-Naur form (typed_tree.hpp's Grammar; the derivation comes out as
// a typed_tree.hpp Tree, so the canonical key, the parameters and the user's
// decoder are the same for both structural genotypes). Nothing here is used
// by a method unless asked for.
//
//   decode — §IV-A, IV-B: the leftmost non-terminal is rewritten by
//     production (codon value) MOD (number of productions of that
//     non-terminal), the codons read one after another; a non-terminal with
//     one production reads no codon. Run out of codons: the reading wraps to
//     the first codon; still incomplete after max_wraps wraps (or beyond
//     max_nodes nodes): the individual is invalid (valid = false — "given
//     the lowest possible fitness value", §IV-A; never a NaN). The mapping
//     is a function of the genome: no random numbers.
//   random_genome — §IV-D: a length drawn in [init_len_lo, init_len_hi],
//     codons uniform on 0 .. 2^codon_bits - 1 (8 bits in the paper).
//   one_point_crossover, mutation (point, per bit), duplication — §IV-D;
//     prune — see GE-3.
//
// Readings (OURS):
//   GE-1. One-point crossover of variable length (§IV-D names it only): a
//     cut point drawn in each parent independently, c1 in [1, L1], c2 in [1,
//     L2]; the children are a[0, c1) + b[c2, L2) and b[0, c2) + a[c1, L1).
//     A child longer than max_len (> 0) leaves its parent as it was.
//   GE-2. Mutation: the paper's is point mutation of the binary string,
//     probability 0.01 a bit (Tables II-IV): each bit of each codon flips
//     with probability p. mutation_codon = true instead redraws a whole
//     codon uniformly with probability p (off by default).
//   GE-3. Pruning is not in O'Neill & Ryan 2001 (§IV-D has duplication
//     only); it comes from Ryan, Collins & O'Neill 1998 (EuroGP, LNCS 1391),
//     not in the corpus. Here: the codons after the last one read are
//     removed, for a valid individual whose reading did not wrap (one codon
//     is kept at least). Whether and how often to prune is the caller's.
//   GE-4. Duplication: "randomly selecting a number of codons to duplicate
//     and the starting position of the first codon in this set ... placed
//     at the end": the start s uniform in [0, L), the number k uniform in
//     [1, L - s]; nothing is done when L + k would exceed max_len (> 0).
//   GE-5. The wrap limit: the paper gives none ("even after several
//     wrapping events"), nor do its experiments (Tables II-IV list
//     population, generations and the operators' probabilities only).
//     max_wraps counts the times the reading restarts from the first codon
//     (0: no wrapping); the default 10 is ours: with the initial lengths of
//     GE-6 (up to ten codons) it lets a derivation read up to about a
//     hundred codons before it is declared invalid.
//   GE-6. "The initialization range of ten codons" (§IV-D) read as lengths
//     uniform in [1, 10].
//   GE-7. Real parameters: the codons give the structure only. The genome
//     keeps a vector of parameters for the derivation's parametric
//     productions in preorder; a slot beyond its end takes the middle of the
//     production's range, values beyond the slots in use are ignored; a local
//     fit writes its values back (write_params). The operators move codons
//     only: a child keeps the parameters of the parent whose head it has.
//   GE-8. Codons are ints in [0, 2^codon_bits); a genome with a codon outside
//     is refused by decode (with the codon's position in the message).
//   GE-9. The worked example of §IV-B: its steps with the first nine codons
//     of Fig. 2 (1.0 - Sin(X), <pre-op> reading no codon) are what decode
//     does (tests/test_structure.cpp). The paper's final expression does not
//     follow from the rest of Fig. 2 under Table I's grammar (the eleventh
//     codon, 220, chooses <expr><op><expr> again; the individual does not
//     complete within 10 wraps), so the test closes the nine codons with
//     its own.
// ============================================================================

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../random.hpp"
#include "typed_tree.hpp"

namespace mootation::scalar::structure {

struct Genome {
    std::vector<int> codons;
    std::vector<double> params;   // GE-7
};

struct GEOptions {
    int codon_bits = 8;           // codons 0 .. 2^codon_bits - 1 (§IV-A)
    int max_wraps = 10;           // GE-5
    int init_len_lo = 1, init_len_hi = 10;   // GE-6
    int max_len = 0;              // codons; 0: no limit (GE-1, GE-4)
    int max_nodes = 100000;       // a derivation larger is invalid
    double p_mutation = 0.01;     // a bit's (or with mutation_codon a codon's) chance (GE-2)
    bool mutation_codon = false;  // GE-2
};

struct Mapping {
    bool valid = false;
    Tree tree;                    // empty when not valid
    std::size_t codons_read = 0;  // codon reads, wraps included
    int wraps = 0;
};

// O'Neill & Ryan §IV-A: the leftmost non-terminal first, one codon a choice.
inline Mapping decode(const Grammar& g, const GEOptions& o, const Genome& x)
{
    if (o.codon_bits < 1 || o.codon_bits > 30)
        throw std::invalid_argument("grammatical evolution: codon_bits must be in [1, 30]");
    const int top = 1 << o.codon_bits;
    for (std::size_t k = 0; k < x.codons.size(); ++k)
        if (x.codons[k] < 0 || x.codons[k] >= top)
            throw std::invalid_argument("grammatical evolution: codon " + std::to_string(k) + " = " +
                                        std::to_string(x.codons[k]) + " is not in [0, " +
                                        std::to_string(top) + ")");
    Mapping d;
    const auto invalid = [&d]() {
        d.tree.nodes.clear();
        return d;                                              // valid stays false
    };
    std::size_t at = 0, slot = 0;
    std::vector<int> todo{g.start};      // the non-terminals still to rewrite, the leftmost on top
    while (!todo.empty()) {
        const int s = todo.back();
        todo.pop_back();
        const auto& rs = g.rules[static_cast<std::size_t>(s)];
        int rule = 0;
        if (rs.size() > 1) {               // a choice reads a codon; one production reads none (§IV-B)
            if (x.codons.empty()) return invalid();
            if (at == x.codons.size()) {
                if (d.wraps == o.max_wraps) return invalid();
                ++d.wraps;
                at = 0;
            }
            rule = x.codons[at++] % static_cast<int>(rs.size());
            ++d.codons_read;
        }
        const Production& p = rs[static_cast<std::size_t>(rule)];
        Node n{s, rule, std::vector<double>(p.n_params())};
        for (std::size_t k = 0; k < n.params.size(); ++k, ++slot)
            n.params[k] = slot < x.params.size() ? x.params[slot] : 0.5 * (p.param_lo[k] + p.param_hi[k]);
        d.tree.nodes.push_back(std::move(n));
        if (d.tree.nodes.size() > static_cast<std::size_t>(o.max_nodes)) return invalid();
        for (auto c = p.children.rbegin(); c != p.children.rend(); ++c) todo.push_back(*c);
    }
    d.valid = true;
    return d;
}

template <typename G>
inline Genome random_genome(const GEOptions& o, G& rng)
{
    Genome x;
    const int len = rnd::integer(rng, o.init_len_lo, o.init_len_hi);
    x.codons.resize(static_cast<std::size_t>(len));
    const std::uint64_t top = std::uint64_t{1} << o.codon_bits;
    for (int& c : x.codons) c = static_cast<int>(rnd::below(rng, top));
    return x;
}

// One-point crossover (reading GE-1); true if either child was taken.
template <typename G>
inline bool one_point_crossover(const GEOptions& o, Genome& a, Genome& b, G& rng)
{
    if (a.codons.empty() || b.codons.empty()) return false;
    const std::size_t ca = 1 + static_cast<std::size_t>(rnd::below(rng, a.codons.size()));
    const std::size_t cb = 1 + static_cast<std::size_t>(rnd::below(rng, b.codons.size()));
    std::vector<int> na(a.codons.begin(), a.codons.begin() + static_cast<std::ptrdiff_t>(ca));
    na.insert(na.end(), b.codons.begin() + static_cast<std::ptrdiff_t>(cb), b.codons.end());
    std::vector<int> nb(b.codons.begin(), b.codons.begin() + static_cast<std::ptrdiff_t>(cb));
    nb.insert(nb.end(), a.codons.begin() + static_cast<std::ptrdiff_t>(ca), a.codons.end());
    const auto fits_len = [&](const std::vector<int>& v) {
        return o.max_len <= 0 || v.size() <= static_cast<std::size_t>(o.max_len);
    };
    const bool take_a = fits_len(na), take_b = fits_len(nb);
    if (take_a) a.codons = std::move(na);
    if (take_b) b.codons = std::move(nb);
    return take_a || take_b;
}

// Mutation (reading GE-2); the number of codons changed.
template <typename G>
inline int mutation(const GEOptions& o, Genome& x, G& rng)
{
    int changed = 0;
    const std::uint64_t top = std::uint64_t{1} << o.codon_bits;
    for (int& c : x.codons) {
        const int before = c;
        if (o.mutation_codon) {
            if (rnd::uniform01(rng) < o.p_mutation) c = static_cast<int>(rnd::below(rng, top));
        } else {
            for (int bit = 0; bit < o.codon_bits; ++bit)
                if (rnd::uniform01(rng) < o.p_mutation) c ^= 1 << bit;
        }
        if (c != before) ++changed;
    }
    return changed;
}

// Duplication (reading GE-4); true if codons were appended.
template <typename G>
inline bool duplication(const GEOptions& o, Genome& x, G& rng)
{
    const std::size_t len = x.codons.size();
    if (len == 0) return false;
    const std::size_t s = static_cast<std::size_t>(rnd::below(rng, len));
    const std::size_t k = 1 + static_cast<std::size_t>(rnd::below(rng, len - s));
    if (o.max_len > 0 && len + k > static_cast<std::size_t>(o.max_len)) return false;
    x.codons.insert(x.codons.end(), x.codons.begin() + static_cast<std::ptrdiff_t>(s),
                    x.codons.begin() + static_cast<std::ptrdiff_t>(s + k));
    return true;
}

// Pruning (reading GE-3); d must be decode(x); true if codons were removed.
inline bool prune(Genome& x, const Mapping& d)
{
    if (!d.valid || d.wraps > 0) return false;
    const std::size_t keep = std::max<std::size_t>(d.codons_read, 1);
    if (keep >= x.codons.size()) return false;
    x.codons.resize(keep);
    return true;
}

// The derivation's parameters (after a local fit) into the genome (reading GE-7).
inline void write_params(Genome& x, const Tree& t) { x.params = params_of(t); }

}  // namespace mootation::scalar::structure

#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// THE OPERATOR PORTFOLIO AND THE CHILD'S TAG (task 31, layer 2; task 30's
// §30.7 and §30.15 made general). The portfolio was cluster_ga's (CGA-7); it
// is here so that every strategy whose move is a portfolio (moves.hpp:
// basin_hopping, ils, pbh, cluster_ga; ga.hpp: ga_sbx, ssga) has the same
// one. Included by evaluator.hpp (the tag, for Problem::on_child) and ga.hpp.
//
// THE PORTFOLIO. Named operators of one form, one of them chosen for each
// application by ProportionalSelectOperator(P) (one uniform of the strategy's
// stream; none with one operator, so a portfolio of one operator is the
// strategy's user move with it, bit for bit), the allocation rule op_select
// (D. Thierens, "An adaptive pursuit strategy for allocating operator
// probabilities", GECCO 2005, source thierens2005; probability matching
// after D. E. Goldberg, Machine Learning 5, 1990): fixed P = 1/K (§4, the
// equal-probability rule); probability_matching (§2, (1) and (2));
// adaptive_pursuit (§3, (1) and (3)-(5)); P(i) = 1/K and Q(i) = 1.0 at the
// start, the algorithms' line 2; op_alpha (0.8), op_beta (0.8) and op_pmin
// (1/(2K), §3's value, the one of §4's runs) as in §4; op_pmin < 1/K refused
// otherwise (footnote 2). The readings TH-1 ... TH-4 are cluster_ga.hpp's
// CGA-7, the reward and its moment each strategy's (moves.hpp MOVE-5).
//
// THE TAG (MOVE-4 of moves.hpp): what the library tells Problem::on_child
// with a child's outcome — the portfolio's index of the crossover and of the
// move that made it (-1 where none, or the built-in one; 0 the problem's one
// crossover or mutate hook) and whether it was taken.
// ============================================================================

#include <algorithm>
#include <cstddef>
#include <functional>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "../json.hpp"
#include "../random.hpp"

namespace mootation::scalar {

// a child's operators and fate (moves.hpp MOVE-4)
struct ChildTag {
    long long crossover = -1;    // the crossover portfolio's index; 0 the crossover hook; -1 none or the built-in
    long long mutation = -1;     // the move's: the portfolio's index; 0 the mutate hook; -1 none or the built-in
    bool taken = false;          // the child entered the population, or the walk moved to it
};

// on_child(parents_f, child_f, tag): the parents' f, the child's f after its local search (where the
// strategy has one), its tag
using OnChild = std::function<void(const std::vector<double>& parents_f, double child_f, const ChildTag& tag)>;

namespace moves {

// A portfolio of operators and its allocation rule (Thierens 2005, §2-3):
// fixed, probability matching (2), adaptive pursuit (3)-(5)
struct Portfolio {
    std::vector<std::string> names;
    std::vector<double> P, Q;                 // the probabilities, the running estimates of the reward
    std::vector<long long> calls, taken;      // applications; children taken
    std::vector<double> gain;                 // the sum of (the better parent's f - the child's f)
    std::vector<long long> scored;            // the children whose f came back
    int rule = 0;                             // 0 fixed, 1 probability matching, 2 adaptive pursuit
    double pmin = 0.0, alpha = 0.8, beta = 0.8;

    bool on() const { return !names.empty(); }

    void start(std::vector<std::string> ns, int r, double p_min, double a, double b)
    {
        names = std::move(ns);
        const std::size_t K = names.size();
        P.assign(K, 1.0 / static_cast<double>(K));   // the algorithms' line 2: P(i) = 1/K, Q(i) = 1.0
        Q.assign(K, 1.0);
        calls.assign(K, 0);
        taken.assign(K, 0);
        gain.assign(K, 0.0);
        scored.assign(K, 0);
        rule = r;
        pmin = p_min;
        alpha = a;
        beta = b;
    }

    // ProportionalSelectOperator(P): one uniform of g, none with one operator
    std::size_t pick(std::mt19937& g) const
    {
        if (P.size() == 1) return 0;
        const double u = rnd::uniform01(g);
        double s = 0.0;
        for (std::size_t q = 0; q + 1 < P.size(); ++q) {
            s += P[q];
            if (u < s) return q;
        }
        return P.size() - 1;
    }

    // the reward R of operator a: Q by (1), then P by (2) or (3)-(5); ties
    // of the best Q broken at random, by one uniform of g (TH-4)
    void reward(std::size_t a, double R, std::mt19937& g)
    {
        if (rule == 0) return;
        const std::size_t K = P.size();
        Q[a] = Q[a] + alpha * (R - Q[a]);
        if (rule == 1) {
            double sum = 0.0;
            for (double q : Q) sum += q;
            for (std::size_t i = 0; i < K; ++i)
                P[i] = sum > 0.0 ? pmin + (1.0 - static_cast<double>(K) * pmin) * Q[i] / sum : 1.0 / static_cast<double>(K);
            return;
        }
        double top = Q[0];
        for (double q : Q) top = (std::max)(top, q);
        std::vector<std::size_t> best;
        for (std::size_t i = 0; i < K; ++i)
            if (Q[i] == top) best.push_back(i);
        const std::size_t star =
            best.size() == 1 ? best[0]
                             : best[(std::min)(best.size() - 1,
                                             static_cast<std::size_t>(rnd::uniform01(g) * static_cast<double>(best.size())))];
        const double pmax = 1.0 - static_cast<double>(K - 1) * pmin;                     // (5)
        for (std::size_t i = 0; i < K; ++i)
            P[i] = i == star ? P[i] + beta * (pmax - P[i]) : P[i] + beta * (pmin - P[i]);   // (3), (4)
    }

    // a child of operator a came back: taken or not, its gain over the
    // better parent; the reward 1 where taken, else 0 (TH-1)
    void outcome(std::size_t a, bool in, double d, std::mt19937& g)
    {
        taken[a] += in ? 1 : 0;
        gain[a] += d;
        ++scored[a];
        reward(a, in ? 1.0 : 0.0, g);
    }

    // the settings' rows: [name, kind, calls, children taken, mean gain, P at the end]
    void rows(json::Value& out, const char* kind) const
    {
        for (std::size_t o = 0; o < names.size(); ++o) {
            json::Value row = json::Value::array();
            row.items.push_back(json::Value::string(names[o]));
            row.items.push_back(json::Value::string(kind));
            row.items.push_back(json::Value::integer(calls[o]));
            row.items.push_back(json::Value::integer(taken[o]));
            row.items.push_back(json::Value::number(scored[o] ? gain[o] / static_cast<double>(scored[o]) : 0.0));
            row.items.push_back(json::Value::number(P[o]));
            out.items.push_back(row);
        }
    }
};

// A child as its strategy records it where it is made (moves.hpp MOVE-4)
struct Child {
    long long crossover = -1, mutation = -1;
    std::vector<double> parents_f;

    double best_parent() const { return parents_f.size() == 1 ? parents_f[0] : (std::min)(parents_f[0], parents_f[1]); }
};

// The child's outcome: its operators rewarded (moves.hpp MOVE-5), then on_child (MOVE-4)
inline void told(const OnChild& on_child, Portfolio* xport, Portfolio* mport, const Child& c, double f, bool in,
                 std::mt19937& g)
{
    const double d = c.best_parent() - f;
    if (xport && c.crossover >= 0 && xport->on()) xport->outcome(static_cast<std::size_t>(c.crossover), in, d, g);
    if (mport && c.mutation >= 0 && mport->on()) mport->outcome(static_cast<std::size_t>(c.mutation), in, d, g);
    if (on_child) {
        ChildTag t;
        t.crossover = c.crossover;
        t.mutation = c.mutation;
        t.taken = in;
        on_child(c.parents_f, f, t);
    }
}

// the settings' row of the portfolios (MOVE-5, CGA-7)
inline std::string operators_row(const Portfolio* xport, const Portfolio* mport)
{
    json::Value rows = json::Value::array();
    if (xport) xport->rows(rows, "crossover");
    if (mport) mport->rows(rows, "mutation");
    return json::dump(rows, -1, ",", ":");
}

}   // namespace moves

}   // namespace mootation::scalar

#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// LOCAL SEARCH OF THE POPULATION'S MEMBERS (task 34.1): the memetic mode of
// the multi-objective cores, one layer for all of them. The run (run_core.hpp)
// calls the user's local search on some members of the population between
// the core's steps and writes what it gives back into the vault, where the
// core reads its population at its next step. No core is changed.
//
//   local_search = off      (the default, even with a hook given) nothing
//                           here runs: the run is the one it was, bit for bit;
//                  lamarck  the improved point replaces the member: its
//                           variables x', its objectives f', its constraint
//                           values g';
//                  baldwin  the member keeps its variables x and takes the
//                           values f', g' of the improved point.
//                           The two as Whitley, Gordon and Mathias define
//                           them (PPSN III, 1994, §1 and §1.1): the
//                           improvement coded back onto the genome
//                           (Lamarckian), or only its fitness value given to
//                           the individual, its genome unchanged (Baldwinian).
//   ls_every  after every ls_every-th step of the core (1: every step);
//   ls_share  the share of the population improved each time: k =
//             ceil(ls_share x pop_size) members (0 < ls_share <= 1);
//   ls_pick   which members (MLS-4): spread (the default), random, objective,
//             feasible.
//
// The hook: (x, f, g) -> {x', f', g', cost}. It is handed a member's
// variables, objectives and constraint values (g empty without constraints)
// and gives back the point it found, that point's values, and what it spent,
// in evaluations of the problem (declared: the layer does not evaluate x').
//
// READINGS:
//   MLS-1. Off by default. local_search = off: the hook is never called and
//     the run is the one it was, bit for bit; the knobs below are not read.
//   MLS-2. The schedule. Under n_gen: after the core's step s with s a
//     multiple of ls_every (not after its setup), the last step included: a
//     run of n_gen steps ends with its population searched. Under
//     max_evaluations: after the first step at which the core's evaluations
//     since its setup reach the next multiple of ls_every x pop_size, once a
//     step — a generation of pop_size evaluations, as the observer's stride
//     (run_core.hpp, RunConfig::max_evaluations): a steady-state core's step
//     is one offspring (NIMMO, SMS-EMOA), and a schedule in steps would
//     search its members pop_size times as often as a generational core's.
//     The hook's declared
//     costs are spent from max_evaluations as the evaluations are: the run's
//     spending is evaluations + ceil(the costs' sum), checked before every
//     call, and no call is made once it reaches the budget. So with a cost of
//     c a call, the run makes at most ceil(budget / c) calls, and the check
//     between the core's steps ends it as it ends every run.
//   MLS-3. The members are the population's slots [0, pop_size) that are
//     evaluated and did not fail; a core's scratch slot (data_vault.hpp,
//     SCRATCH SLOTS) is not one. k = ceil(ls_share x pop_size), at most all
//     of them.
//   MLS-4. ls_pick, the members searched, in this order:
//       spread    the nondominated members first (constraint domination:
//                 the smaller violation, at the same violation Pareto
//                 dominance), by crowding distance, the largest first (the
//                 extremes of each objective infinite; Deb et al., IEEE TEVC
//                 6(2), 2002, §III-B), then the next front;
//       random    k members drawn without replacement, uniformly;
//       objective the c-th of a time's picks the best of the rest on
//                 objective (t + c) mod M, t the number of the time;
//       feasible  the smallest violation first.
//     Ties go to the lower slot. random draws from the layer's own generator
//     (seeded with the run's seed), so the core's draws are the same as with
//     local_search = off until the first member is replaced.
//   MLS-5. What the hook gives back is checked: x' of n finite values in the
//     box, f' of M finite values, g' of n_cons finite values (empty without
//     constraints), a cost finite and >= 0; otherwise, and on an exception of
//     the hook, the run ends with std::runtime_error naming the member (as
//     scalar.hpp's ULS-4).
//   MLS-6. The improvement is taken when the point is better than the member:
//     a smaller violation (the sum of the positive constraint values), or the
//     same violation and f' <= f in every objective, < in one (a Pareto
//     improvement). A point that is not is left: the member stays as it was;
//     the call and its cost count either way.
//   MLS-7. Taken, the member's slot is written whole (data_vault.hpp,
//     WRITING BACK: put_point brings it to the vault's present): with
//     lamarck x', f', g'; with baldwin x, f', g'. Every slot of the vault's
//     archive that holds the member's old point (its variables and objectives
//     bit for bit) is written the same way: the cores whose answer is a copy
//     of their archive (SPEA2: the archive published into the population
//     after every step) breed from the archive, not from the copy.
//   MLS-8. baldwin: a member answers with the point its values are of. The
//     run's answer (and current_variables() of an observer) gives, for a
//     member holding x with the values f' given to it, x' — the pair (x', f')
//     is the problem's, (x, f') is not. A member the core evaluated again (f
//     of x) answers as it is.
//   MLS-9. Refused with local_search on: binary, integer or discrete
//     variables (x' would have to be rounded, and its f' with it), the
//     population-dependent objectives (aux: their k values come from the
//     population, not from the hook), equality constraints (a member keeps
//     |h|, not h, to hand over), and the cores of refusal() below, each
//     with its reason: the cores that keep their population between steps
//     in copies of their own and write the vault from them (data_vault.hpp,
//     WRITING BACK: a member written between their steps is overwritten
//     unread, or answers without being bred from), and MO-CMA-ES, which
//     breeds from each member's own point before repair and selects by the
//     penalised values it keeps in the member. Every other core reads its
//     population from the vault at its next step and so takes the member;
//     what such a core keeps besides (an ideal point, a normalization, the
//     associations to its reference vectors, a fitness) it updates from the
//     vault at that step as it does after its own writes, or keeps as it was
//     where it does not.
//   MLS-10. The run's settings (RunResult::settings, embed's Result::settings)
//     say, with local_search on: ls_calls (the calls made), ls_accepted (the
//     points taken, MLS-6), ls_cost (the declared costs' sum) and used (the
//     evaluations + ls_cost: what max_evaluations counts). The evaluations
//     themselves stay the problem's calls by the core.
//   MLS-11. The hook is called on the run's thread, a call at a time, in the
//     order of the picks; the layer keeps no state outside the run, so runs
//     on several threads at once are each the run they are on one.
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "random.hpp"

namespace mootation {

// What the hook gives back (the module comment).
struct LocalStep {
    std::vector<double> x;      // x', n values
    std::vector<double> f;      // f(x'), M values
    std::vector<double> g;      // g(x'), n_cons values; empty without constraints
    double cost = 0.0;          // what the call spent, in evaluations (declared)
};

// The user's local search: a member's x, f, g in, the point it found out.
using LocalSearch = std::function<LocalStep(const std::vector<double>& x, const std::vector<double>& f,
                                            const std::vector<double>& g)>;

namespace member_search {

inline bool known_mode(const std::string& m) { return m == "off" || m == "lamarck" || m == "baldwin"; }
inline bool known_pick(const std::string& p)
{
    return p == "spread" || p == "random" || p == "objective" || p == "feasible";
}

// The cores that cannot take a member written between their steps (MLS-9),
// with the reason; nullptr for the others.
inline const char* refusal(const std::string& algorithm)
{
    static const char* copies =
        "it keeps its population between steps in copies of its own and writes the vault from them "
        "(data_vault.hpp, WRITING BACK): a member written between its steps is not what it breeds from";
    static const std::pair<const char*, const char*> list[] = {
        {"aprd", copies},       {"clia", copies},        {"dcea", copies},      {"dea_gng", copies},
        {"dhea", copies},       {"dms", copies},         {"dmulti_mads", copies}, {"etea", copies},
        {"hcca", copies},       {"hlmea", copies},       {"if_maoea", copies},  {"isde_rd", copies},
        {"lis_lcs", copies},    {"liu_gu2011", copies},  {"maoea_3c", copies},  {"moead_am2m", copies},
        {"moead_ds", copies},   {"mosa", copies},        {"multiglods", copies}, {"naemo", copies},
        {"paes", copies},       {"rd_emo", copies},      {"smpso", copies},     {"two_arch2", copies},
        {"mo_cma_es", "it breeds from each member's own point before repair (x) and selects by the penalised "
                      "values it keeps in the member (fpen): a member written between its steps keeps the old ones"},
    };
    for (const auto& e : list)
        if (algorithm == e.first) return e.second;
    return nullptr;
}

// One member of the population as the layer sees it.
struct Member {
    std::size_t slot = 0;
    const std::vector<double>* f = nullptr;
    double cv = 0.0;            // the violation, whatever the vault's level
};

// a dominates b: the smaller violation, at the same violation Pareto (MLS-4)
inline bool dominates(const std::vector<double>& fa, double cva, const std::vector<double>& fb, double cvb)
{
    if (cva != cvb) return cva < cvb;
    bool strict = false;
    for (std::size_t j = 0; j < fa.size(); ++j) {
        if (fa[j] > fb[j]) return false;
        if (fa[j] < fb[j]) strict = true;
    }
    return strict;
}

// MLS-6: the new point is better than the member
inline bool improves(const std::vector<double>& f_new, double cv_new, const std::vector<double>& f_old,
                     double cv_old)
{
    return dominates(f_new, cv_new, f_old, cv_old);
}

inline double violation(const std::vector<double>& g)
{
    double s = 0.0;
    for (double v : g)
        if (v > 0.0) s += v;
    return s;
}

// spread (MLS-4): fronts by constraint domination, each by crowding distance
inline std::vector<std::size_t> by_spread(const std::vector<Member>& P)
{
    const std::size_t n = P.size();
    std::vector<std::size_t> order, rest(n);
    std::iota(rest.begin(), rest.end(), std::size_t(0));
    while (!rest.empty()) {
        std::vector<std::size_t> front, next;
        for (std::size_t a : rest) {
            bool dominated = false;
            for (std::size_t b : rest)
                if (b != a && dominates(*P[b].f, P[b].cv, *P[a].f, P[a].cv)) { dominated = true; break; }
            (dominated ? next : front).push_back(a);
        }
        // crowding distance over the front (Deb et al. 2002, §III-B)
        std::vector<double> dist(front.size(), 0.0);
        const std::size_t m = P[front.front()].f->size();
        for (std::size_t j = 0; j < m; ++j) {
            std::vector<std::size_t> idx(front.size());
            std::iota(idx.begin(), idx.end(), std::size_t(0));
            std::stable_sort(idx.begin(), idx.end(), [&](std::size_t u, std::size_t v) {
                return (*P[front[u]].f)[j] < (*P[front[v]].f)[j];
            });
            const double lo = (*P[front[idx.front()]].f)[j], hi = (*P[front[idx.back()]].f)[j];
            dist[idx.front()] = dist[idx.back()] = std::numeric_limits<double>::infinity();
            if (!(hi > lo)) continue;
            for (std::size_t q = 1; q + 1 < idx.size(); ++q)
                dist[idx[q]] += ((*P[front[idx[q + 1]]].f)[j] - (*P[front[idx[q - 1]]].f)[j]) / (hi - lo);
        }
        std::vector<std::size_t> idx(front.size());
        std::iota(idx.begin(), idx.end(), std::size_t(0));
        std::stable_sort(idx.begin(), idx.end(), [&](std::size_t u, std::size_t v) { return dist[u] > dist[v]; });
        for (std::size_t q : idx) order.push_back(front[q]);
        rest = std::move(next);
    }
    return order;
}

// The k members to search, as indices into P (MLS-4); t: the number of the time
inline std::vector<std::size_t> pick(const std::vector<Member>& P, std::size_t k, const std::string& how,
                                     long long t, std::mt19937& rng)
{
    const std::size_t n = P.size();
    k = (std::min)(k, n);
    std::vector<std::size_t> all(n);
    std::iota(all.begin(), all.end(), std::size_t(0));
    if (how == "random") {
        for (std::size_t c = 0; c < k; ++c) {          // a partial Fisher-Yates shuffle
            const std::size_t r = c + static_cast<std::size_t>(rnd::below(rng, n - c));
            std::swap(all[c], all[r]);
        }
        all.resize(k);
        return all;
    }
    if (how == "feasible") {
        std::stable_sort(all.begin(), all.end(), [&](std::size_t a, std::size_t b) { return P[a].cv < P[b].cv; });
        all.resize(k);
        return all;
    }
    if (how == "objective") {
        const std::size_t m = n > 0 ? P.front().f->size() : 1;
        std::vector<char> taken(n, 0);
        std::vector<std::size_t> out;
        for (std::size_t c = 0; c < k; ++c) {
            const std::size_t j = static_cast<std::size_t>((t + static_cast<long long>(c)) %
                                                           static_cast<long long>(m));
            std::size_t best = n;
            for (std::size_t i = 0; i < n; ++i)
                if (!taken[i] && (best == n || (*P[i].f)[j] < (*P[best].f)[j])) best = i;
            taken[best] = 1;
            out.push_back(best);
        }
        return out;
    }
    std::vector<std::size_t> order = by_spread(P);     // spread
    order.resize(k);
    return order;
}

// A number of the settings (MLS-10): the shortest of %.15g ... %.17g that
// reads back as the same double.
inline std::string number(double v)
{
    char buf[40];
    for (int digits = 15; digits <= 17; ++digits) {
        std::snprintf(buf, sizeof buf, "%.*g", digits, v);
        if (std::strtod(buf, nullptr) == v) break;
    }
    return buf;
}

// The bytes of a point's variables and values, the key of MLS-8.
inline std::string key_of(const std::vector<double>& x, const std::vector<double>& f)
{
    std::string k(reinterpret_cast<const char*>(x.data()), x.size() * sizeof(double));
    k.append(reinterpret_cast<const char*>(f.data()), f.size() * sizeof(double));
    return k;
}

}   // namespace member_search
}   // namespace mootation

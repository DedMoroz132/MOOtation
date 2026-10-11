#pragma once
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

#include "../constraint_mode.hpp"
#include "../data_vault.hpp"
#include "../operators/binary_crossover.hpp"
#include "../operators/bit_flip.hpp"
#include "../operators/poly_mutation.hpp"
#include "../operators/sbx.hpp"

namespace mootation {

// ============================================================================
// CCMO — Coevolutionary Constrained Multiobjective Optimization
// Y. Tian, T. Zhang, J. Xiao, X. Zhang, Y. Jin — IEEE TEVC 25(1):102-116, 2021
// doi:10.1109/TEVC.2020.3004012           (source: ccmo_tian2021_accepted)
// (task 7 of 2026-10-10, supplement 3, §4.2)
//
// Two populations of N (Algorithm 1): Population1 (the active population,
// the answer) solves the problem, Population2 (the vault's archive) the
// helper problem — the same objectives without the constraints, the paper's
// setting (§IV-A). Every generation each one gives N/2 parents by the
// mating selection of the employed MOEA, each parent set N/2 children, and
// BOTH populations are cut from themselves and ALL N children by the
// environmental selection: the weak cooperation, the children shared and
// nothing else. N evaluations a generation (N/2 + N/2), 2N at the start.
//
// The employed MOEA is the paper's: constrained NSGA-II (§IV-A) — the binary
// tournament by rank and crowding distance, non-dominated sorting by Deb's
// constrained domination for Population1 and plain Pareto domination for
// Population2 — "with the truncation strategy in SPEA2 adopted in the
// environmental selection instead of crowding distance": the last front is
// cut by removing, one at a time, the member nearest to its nearest
// neighbours (Zitzler et al.'s k-th nearest in turn). SBX (pc 1) and
// polynomial mutation (pm 1/D), both eta 20 (§IV-A).
//
// READINGS (declared):
//   CCMO-1. The supplement asks for a wrapper over any core; CCMO needs a
//     core that hands over its children before its environmental selection
//     and takes the other population's into it, which none of the others does
//     (each step breeds, evaluates and selects at once). So this is CCMO as
//     the paper instantiates it, on NSGA-II.
//   CCMO-2. The SPEA2 truncation runs within the last front, as the crowding
//     distance it replaces does, on the objective vectors as they are.
//   CCMO-3. A seed population (setup_seeded) is both populations: the helper
//     starts as its copy, so that a warm start spends nothing.
// ============================================================================

template <typename Ind_t>
class CCMOCore {
public:
    ConstraintMode constraint_mode = ConstraintMode::NONE;

private:
    double eta_c_ = 20.0, pc_ = 1.0, eta_m_ = 20.0, pm_ = -1.0;
    std::mt19937 rng_{std::random_device{}()};

    using Vec = std::vector<double>;

    struct Pt {
        Vec f;
        double cv;
        int src;          // >= 0: active slot; < 0: archive entry -(src + 1)
        int rank = 0;
        double crowd = 0.0;
    };

    static bool dominates(const Vec& a, const Vec& b)
    {
        bool better = false;
        for (std::size_t j = 0; j < a.size(); ++j) {
            if (a[j] > b[j]) return false;
            if (a[j] < b[j]) better = true;
        }
        return better;
    }

    // Deb's constrained domination where `constrained`, Pareto otherwise
    static bool beats(const Pt& a, const Pt& b, bool constrained)
    {
        if (constrained) {
            const bool fa = a.cv <= 0.0, fb = b.cv <= 0.0;
            if (fa && !fb) return true;
            if (!fa && fb) return false;
            if (!fa && !fb) return a.cv < b.cv;
        }
        return dominates(a.f, b.f);
    }

    // the fronts of P (indices), ranks written into P
    static std::vector<std::vector<int>> sort(std::vector<Pt>& P, bool constrained)
    {
        const std::size_t n = P.size();
        std::vector<std::vector<int>> S(n);
        std::vector<int> count(n, 0);
        for (std::size_t a = 0; a < n; ++a)
            for (std::size_t b = a + 1; b < n; ++b) {
                if (beats(P[a], P[b], constrained))      { S[a].push_back(static_cast<int>(b)); ++count[b]; }
                else if (beats(P[b], P[a], constrained)) { S[b].push_back(static_cast<int>(a)); ++count[a]; }
            }
        std::vector<std::vector<int>> fronts;
        std::vector<int> cur;
        for (std::size_t a = 0; a < n; ++a) if (count[a] == 0) cur.push_back(static_cast<int>(a));
        for (int r = 0; !cur.empty(); ++r) {
            std::vector<int> next;
            for (int a : cur) {
                P[static_cast<std::size_t>(a)].rank = r;
                for (int b : S[static_cast<std::size_t>(a)])
                    if (--count[static_cast<std::size_t>(b)] == 0) next.push_back(b);
            }
            std::sort(next.begin(), next.end());
            fronts.push_back(std::move(cur));
            cur = std::move(next);
        }
        return fronts;
    }

    // NSGA-II's crowding distance within each front, for the tournament
    static void crowding(std::vector<Pt>& P, const std::vector<std::vector<int>>& fronts)
    {
        for (const auto& front : fronts) {
            for (int i : front) P[static_cast<std::size_t>(i)].crowd = 0.0;
            if (front.empty()) continue;
            std::vector<int> s = front;
            for (std::size_t k = 0; k < P[static_cast<std::size_t>(s[0])].f.size(); ++k) {
                auto f = [&](int i) { return P[static_cast<std::size_t>(i)].f[k]; };
                std::stable_sort(s.begin(), s.end(), [&](int a, int b) { return f(a) < f(b); });
                const double inf = std::numeric_limits<double>::infinity();
                P[static_cast<std::size_t>(s.front())].crowd = inf;
                P[static_cast<std::size_t>(s.back())].crowd = inf;
                const double range = f(s.back()) - f(s.front());
                if (range < 1e-14) continue;
                for (std::size_t i = 1; i + 1 < s.size(); ++i)
                    P[static_cast<std::size_t>(s[i])].crowd += (f(s[i + 1]) - f(s[i - 1])) / range;
            }
        }
    }

    // SPEA2's truncation of `front` down to `keep` members (CCMO-2)
    static std::vector<int> truncate(const std::vector<Pt>& P, std::vector<int> front, std::size_t keep)
    {
        while (front.size() > keep) {
            // each member's distances to the others, ascending
            std::vector<std::vector<double>> d(front.size());
            for (std::size_t a = 0; a < front.size(); ++a) {
                for (std::size_t b = 0; b < front.size(); ++b) {
                    if (a == b) continue;
                    double s = 0.0;
                    const Vec& fa = P[static_cast<std::size_t>(front[a])].f;
                    const Vec& fb = P[static_cast<std::size_t>(front[b])].f;
                    for (std::size_t j = 0; j < fa.size(); ++j) s += (fa[j] - fb[j]) * (fa[j] - fb[j]);
                    d[a].push_back(std::sqrt(s));
                }
                std::sort(d[a].begin(), d[a].end());
            }
            // the one whose nearest is nearest, then second nearest, ...
            std::size_t worst = 0;
            for (std::size_t a = 1; a < front.size(); ++a)
                if (std::lexicographical_compare(d[a].begin(), d[a].end(), d[worst].begin(), d[worst].end()))
                    worst = a;
            front.erase(front.begin() + static_cast<std::ptrdiff_t>(worst));
        }
        return front;
    }

    // the environmental selection: N of P by fronts, the last one truncated
    static std::vector<int> select(std::vector<Pt>& P, int N, bool constrained)
    {
        const auto fronts = sort(P, constrained);
        std::vector<int> out;
        for (const auto& front : fronts) {
            const std::size_t room = static_cast<std::size_t>(N) - out.size();
            if (room == 0) break;
            if (front.size() <= room) { out.insert(out.end(), front.begin(), front.end()); continue; }
            const auto cut = truncate(P, front, room);
            out.insert(out.end(), cut.begin(), cut.end());
        }
        return out;
    }

    // NSGA-II's binary tournament: the lower rank, then the larger crowding
    int tournament(const std::vector<Pt>& P)
    {
        std::uniform_int_distribution<int> pick(0, static_cast<int>(P.size()) - 1);
        const int a = pick(rng_), b = pick(rng_);
        const Pt& x = P[static_cast<std::size_t>(a)];
        const Pt& y = P[static_cast<std::size_t>(b)];
        if (x.rank != y.rank) return x.rank < y.rank ? a : b;
        if (x.crowd != y.crowd) return x.crowd > y.crowd ? a : b;
        std::uniform_int_distribution<int> coin(0, 1);
        return coin(rng_) ? a : b;
    }

    bool constrained() const { return constraint_mode != ConstraintMode::NONE; }

    Pt active_pt(DataVault<Ind_t>& vault, int v)
    {
        return Pt{vault.objectives_of(static_cast<std::size_t>(v)), vault.get_cv(static_cast<std::size_t>(v)), v};
    }
    Pt archive_pt(DataVault<Ind_t>& vault, std::size_t i)
    {
        return Pt{vault.archive_objectives_of(i), vault.archive_cv(i), -static_cast<int>(i) - 1};
    }

    // random variables into active slots [from, from + n), then evaluated
    void randomize(DataVault<Ind_t>& vault, int from, int n)
    {
        const auto& bounds = vault.get_bounds();
        std::uniform_real_distribution<double> U01(0.0, 1.0);
        Vec x(static_cast<std::size_t>(vault.vars_n()));
        std::vector<int> bx(static_cast<std::size_t>(vault.bin_vars_n()));
        for (int i = from; i < from + n; ++i) {
            for (int j = 0; j < vault.vars_n(); ++j) {
                const double lo = bounds[j].first.value_or(0.0);
                const double hi = bounds[j].second.value_or(1.0);
                x[static_cast<std::size_t>(j)] = lo + U01(rng_) * (hi - lo);
            }
            for (int j = 0; j < vault.bin_vars_n(); ++j)
                bx[static_cast<std::size_t>(j)] = ops::random_gene(j, rng_);
            if (vault.bin_vars_n() > 0) vault.set_all_variables(static_cast<std::size_t>(i), x, bx);
            else                        vault.set_variables(static_cast<std::size_t>(i), x);
        }
        vault.sync();
    }

public:
    CCMOCore() = default;

    void set_seed(unsigned s) { rng_.seed(s); }
    void set_eta_crossover(double e) { eta_c_ = e; }
    void set_eta_mutation(double e)  { eta_m_ = e; }
    void set_pc(double p)            { pc_ = p; }
    void set_pm(double p)            { pm_ = p; }

    // Algorithm 1, lines 1-4: both populations at random, 2N evaluations
    void setup(DataVault<Ind_t>& vault)
    {
        const int N = vault.pop_size();
        randomize(vault, 0, N);
        const int base = vault.expand(N);
        randomize(vault, base, N);
        vault.archive_clear();
        for (int i = 0; i < N; ++i) vault.archive_push(static_cast<std::size_t>(base + i));
        vault.reduce(N);
    }

    void setup_seeded(DataVault<Ind_t>& vault)     // CCMO-3
    {
        vault.archive_clear();
        for (int i = 0; i < vault.pop_size(); ++i) vault.archive_push(static_cast<std::size_t>(i));
    }

    void step(DataVault<Ind_t>& vault)
    {
        const int N = vault.pop_size();
        const auto& bounds = vault.get_bounds();
        const int nv = vault.vars_n(), nb = vault.bin_vars_n();
        const double pm = pm_ >= 0.0 ? pm_ : (nv > 0 ? 1.0 / nv : 0.0);

        std::vector<Pt> p1, p2;
        for (int i = 0; i < N; ++i) p1.push_back(active_pt(vault, i));
        for (std::size_t i = 0; i < vault.archive_size(); ++i) p2.push_back(archive_pt(vault, i));
        crowding(p1, sort(p1, constrained()));
        crowding(p2, sort(p2, false));

        auto vars_of = [&](const Pt& p) -> const Vec& {
            return p.src >= 0 ? vault.variables_of(static_cast<std::size_t>(p.src))
                              : vault.archive_variables_of(static_cast<std::size_t>(-p.src - 1));
        };
        auto bits_of = [&](const Pt& p) -> const std::vector<int>& {
            return p.src >= 0 ? vault.binary_variables_of(static_cast<std::size_t>(p.src))
                              : vault.archive_bin_variables_of(static_cast<std::size_t>(-p.src - 1));
        };

        // lines 6-9: N/2 children from each population's parents
        std::vector<Vec> qx;
        std::vector<std::vector<int>> qb;
        auto breed = [&](const std::vector<Pt>& P, int n) {
            for (int made = 0; made < n; made += 2) {
                const Pt& a = P[static_cast<std::size_t>(tournament(P))];
                const Pt& b = P[static_cast<std::size_t>(tournament(P))];
                Vec c1, c2;
                ops::sbx(vars_of(a), vars_of(b), c1, c2, bounds, eta_c_, pc_, rng_);
                ops::polynomial_mutation(c1, bounds, eta_m_, pm, rng_);
                ops::polynomial_mutation(c2, bounds, eta_m_, pm, rng_);
                qx.push_back(std::move(c1));
                if (made + 1 < n) qx.push_back(std::move(c2));
                if (nb > 0) {
                    std::vector<int> b1, b2;
                    ops::binary_crossover(bits_of(a), bits_of(b), b1, b2, rng_);
                    ops::bit_flip_mutation(b1, nb, rng_);
                    ops::bit_flip_mutation(b2, nb, rng_);
                    qb.push_back(std::move(b1));
                    if (made + 1 < n) qb.push_back(std::move(b2));
                }
            }
        };
        breed(p1, N / 2);
        breed(p2, N - N / 2);

        const int base = vault.expand(N);
        for (int i = 0; i < N; ++i) {
            if (nb > 0) vault.set_all_variables(static_cast<std::size_t>(base + i), qx[static_cast<std::size_t>(i)],
                                                qb[static_cast<std::size_t>(i)]);
            else        vault.set_variables(static_cast<std::size_t>(base + i), qx[static_cast<std::size_t>(i)]);
        }
        vault.sync();                                       // N evaluations

        // lines 10-15: each population from itself and all the children
        std::vector<Pt> h1, h2 = p2;
        for (int i = 0; i < base + N; ++i) h1.push_back(active_pt(vault, i));
        for (int i = 0; i < N; ++i) h2.push_back(active_pt(vault, base + i));
        const std::vector<int> keep1 = select(h1, N, constrained());
        const std::vector<int> keep2 = select(h2, N, false);

        // Population2: the archive keeps its survivors, gains the children chosen
        std::vector<char> stay(p2.size(), 0);
        std::vector<int> add;
        for (int i : keep2) {
            if (static_cast<std::size_t>(i) < p2.size()) stay[static_cast<std::size_t>(i)] = 1;
            else add.push_back(h2[static_cast<std::size_t>(i)].src);
        }
        for (std::size_t i = p2.size(); i-- > 0;) if (!stay[i]) vault.archive_erase(i);
        for (int v : add) vault.archive_push(static_cast<std::size_t>(v));

        // Population1 into slots [0, N)
        std::vector<int> pos(static_cast<std::size_t>(base + N)), at(static_cast<std::size_t>(base + N));
        std::iota(pos.begin(), pos.end(), 0);
        std::iota(at.begin(), at.end(), 0);
        for (int i = 0; i < N; ++i) {
            const int want = keep1[static_cast<std::size_t>(i)];
            const int cur  = pos[static_cast<std::size_t>(want)];
            if (cur == i) continue;
            const int other = at[static_cast<std::size_t>(i)];
            vault.swap_active(static_cast<std::size_t>(i), static_cast<std::size_t>(cur));
            pos[static_cast<std::size_t>(want)]  = i;
            pos[static_cast<std::size_t>(other)] = cur;
            at[static_cast<std::size_t>(i)]      = want;
            at[static_cast<std::size_t>(cur)]    = other;
        }
        vault.reduce(N);
    }
};

}   // namespace mootation

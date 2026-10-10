#pragma once
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstddef>
#include <cmath>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "../constraint_mode.hpp"
#include "../data_vault.hpp"
#include "../operators/binary_crossover.hpp"
#include "../operators/bit_flip.hpp"
#include "../operators/inner_archive.hpp"
#include "../operators/poly_mutation.hpp"
#include "../operators/real_crossover.hpp"
#include "../operators/real_mutation.hpp"
#include "../operators/variation.hpp"
#include "../operators/sbx.hpp"

namespace mootation {

// Where NSGA-II measures crowding: in the objectives (the paper) or in the
// decision variables (see the header below, "crowding_space").
enum class CrowdingSpace { Objectives, Decision };

inline const char* crowding_space_name(CrowdingSpace c)
{
    return c == CrowdingSpace::Decision ? "decision" : "objectives";
}

inline std::optional<CrowdingSpace> parse_crowding_space(const std::string& s)
{
    if (s == "objectives") return CrowdingSpace::Objectives;
    if (s == "decision")   return CrowdingSpace::Decision;
    return std::nullopt;
}

// ============================================================================
// NSGA-II — A Fast and Elitist Multiobjective Genetic Algorithm
// K. Deb, A. Pratap, S. Agarwal, T. Meyarivan — IEEE TEVC 6(2), 2002
// doi:10.1109/4235.996017          (source: nsga2_deb2002)
//
// Generational scheme:
//   1. Binary tournament by ≺_n (rank, then crowding distance; CDP — §VI)
//   2. SBX (eta_c=20, pc=0.9) + polynomial mutation (eta_m=20, pm=1/n)
//   3. R_t = P_t ∪ Q_t (2N) → fast non-dominated sort (§III-A)
//   4. Whole fronts are accepted; the last front — selection by descending CD (§III-B)
//   5. Permute survivors into [0,N), reduce to N
//
// Defaults = §IV-A: pc=0.9, pm=1/n, eta_c=20, eta_m=20.
// Deviations (MINOR, deliberately left unchanged; internal audit):
//   - CD normalization uses min/max of the current FRONT (the paper is ambiguous);
//   - tournament "with replacement" (a==b is possible) instead of two permutations;
//   - deterministic tournament tie-breaks (on a tie — `b`);
//   - binary genome: uniform crossover p=1 instead of single-point pc=0.9.
// Extensions beyond the paper: constraint_mode FEASIBILITY (off by default;
//   CDP follows §VI of the paper); mixed real+binary genome.
//
// crowding_space (task 2 of 2026-09-23, C2): `decision` computes the crowding
//   distance of §III-B over the decision variables instead of the objectives
//   — the same formula, each variable's gaps divided by its range on the
//   front, which is the normalisation — so that among equally ranked
//   solutions the ones far from others IN THE PARAMETERS survive: different
//   parameter sets with the same residual, what a calibration wants to see.
//   It is one of the two changes of DN-NSGA-II (J. J. Liang, C. T. Yue,
//   B. Y. Qu, "Multimodal multi-objective optimization: A preliminary
//   study", IEEE CEC 2016, pp. 2454-2461, doi:10.1109/CEC.2016.7744093;
//   source dn-nsga2_liang2016). Its §IV says only that "the crowding in objective
//   space is replaced by crowding in decision space", with no formula, so
//   §III-B over the (real) variables is this library's reading. The other
//   change is not here: DN-NSGA-II fills the mating pool by niching — a
//   random solution meets the nearest, by Euclidean distance in the
//   variables, of CF random others and the better of the two enters (CF
//   and "better" left open) — where this tournament draws its rival at
//   random. So this is not DN-NSGA-II. Default `objectives`.
//
// survival = "stochastic" (task 7 of 2026-10-10, supplement 2): Bian, Zhou,
//   Li & Qian, "Stochastic Population Update Can Provably Be Helpful in
//   Multi-Objective Evolutionary Algorithms", IJCAI 2023,
//   doi:10.24963/ijcai.2023/612 (arXiv 2306.02611 v3), Algorithm 6 — written
//   from the task, the paper to be checked against: Q' of ⌊spu_frac·2μ⌋
//   members of Q = P ∪ P' (spu_frac 3/4, the paper's) drawn uniformly
//   without replacement, by a partial Fisher-Yates shuffle of the slots into
//   [0, |Q'|); the usual selection — fronts and crowding distance over Q'
//   alone, crowding_space as set — keeps |Q'| − μ of them, and every member
//   of Q \ Q' survives without a comparison: P_{t+1} = (Q \ Q') ∪ O, exactly
//   μ. A member survives with probability at least 1/4 (Lemma 1). The ranks
//   and crowding distances the next tournament reads are then computed over
//   P_{t+1}; the greedy selection keeps those of its sort of Q, as always.
//   spu_frac = 1 draws nothing and is greedy bit for bit; |Q'| < μ is
//   refused. "greedy" is the default.
// inner_archive, inner_archive_cap, archive_reuse_p (operators/
//   inner_archive.hpp): an archive of every non-dominated point evaluated, the
//   initial population included; each parent, with probability
//   archive_reuse_p, a member drawn uniformly instead of the tournament's
//   winner. bin_crossover = "none": the binary part is not crossed. mating =
//   "each" (the paper's Algorithm 3, for its theory runs): every member of P
//   is a parent exactly once, in the order of the population, its child its
//   mutation alone — so it needs no crossover on any part of the genome
//   (crossover = "none" with real variables, bin_crossover = "none" with
//   binary ones), and is refused otherwise.
// ============================================================================
template <typename Ind_t>
class NSGAIICore {
public:
    // Supported: NONE, FEASIBILITY, CDP
    ConstraintMode constraint_mode = ConstraintMode::NONE;

private:
    double       eta_c_ = 20.0;
    double       eta_m_ = 20.0;
    double       pc_    = 0.9;   // §IV-A: "crossover probability of p_c = 0.9"
    // Switchable operators (task 2 of 2026-09-23, B3): crossover and mutation
    // default to the paper's SBX and polynomial mutation, called exactly as
    // before; the alternatives (real_crossover.hpp, real_mutation.hpp) are for
    // ablations, and bound_repair applies to those that can leave the box.
    ops::CrossoverSpec xover_;
    std::optional<double> pm_;
    ops::MutationSpec  mut_;
    CrowdingSpace      crowding_space_ = CrowdingSpace::Objectives;
    ops::InnerArchive  arch_;
    bool   stochastic_ = false;           // survival
    double spu_frac_ = 0.75;              // |Q'| / |Q|, the paper's 3/4
    bool   bin_cross_ = true;             // bin_crossover = uniform
    bool   mating_each_ = false;          // mating = each
    std::mt19937 rng_{std::random_device{}()};

    // One evaluated point into the inner archive (feasible only under
    // constraint handling, as the answer's archive).
    void archive_point(DataVault<Ind_t>& vault, std::size_t i)
    {
        if (!arch_.on) return;
        if (constraint_mode != ConstraintMode::NONE && vault.get_cv(i) > 0.0) return;
        arch_.add(vault.objectives_of(i), vault.variables_of(i),
                  vault.bin_vars_n() > 0 ? vault.binary_variables_of(i) : std::vector<int>{});
    }

    void start_archive(DataVault<Ind_t>& vault)
    {
        arch_.start(vault.pop_size(), vault.vars_n() == 0 && vault.bin_vars_n() > 0);
        for (std::size_t i = 0; i < static_cast<std::size_t>(vault.pop_size()); ++i)
            archive_point(vault, i);
        if (mating_each_ && ((vault.vars_n() > 0 && xover_.kind != ops::Crossover::None) ||
                             (vault.bin_vars_n() > 0 && bin_cross_)))
            throw std::invalid_argument("mating = \"each\" makes every child its parent's "
                                        "mutation: it needs crossover = \"none\" with real "
                                        "variables and bin_crossover = \"none\" with binary ones");
    }

    bool dominates_plain(const std::vector<double>& a,
                         const std::vector<double>& b) const
    {
        bool any_better = false;
        for (std::size_t i = 0; i < a.size(); ++i) {
            if (a[i] > b[i]) return false;
            if (a[i] < b[i]) any_better = true;
        }
        return any_better;
    }

    // CDP: Constrained Domination
    bool dominates_cdp(const std::vector<double>& fa, double cva,
                       const std::vector<double>& fb, double cvb) const
    {
        bool a_feas = (cva <= 0.0);
        bool b_feas = (cvb <= 0.0);
        if ( a_feas && !b_feas) return true;
        if (!a_feas &&  b_feas) return false;
        if (!a_feas && !b_feas) return cva < cvb;
        return dominates_plain(fa, fb);
    }

    std::vector<std::vector<int>>
    fast_nondominated_sort(DataVault<Ind_t>& vault, int n)
    {
        std::vector<double> cvs(n, 0.0);
        if (constraint_mode == ConstraintMode::CDP ||
            constraint_mode == ConstraintMode::FEASIBILITY)
            for (int i = 0; i < n; ++i) cvs[i] = vault.get_cv(i);

        std::vector<std::vector<int>> S(n);
        std::vector<int>              np(n, 0);

        for (int i = 0; i < n; ++i) {
            const auto& fi = vault.objectives_of(i);
            for (int j = 0; j < n; ++j) {
                if (i == j) continue;
                const auto& fj = vault.objectives_of(j);
                bool i_dom_j, j_dom_i;
                if (constraint_mode == ConstraintMode::CDP ||
                    constraint_mode == ConstraintMode::FEASIBILITY) {
                    i_dom_j = dominates_cdp(fi, cvs[i], fj, cvs[j]);
                    j_dom_i = dominates_cdp(fj, cvs[j], fi, cvs[i]);
                } else {
                    i_dom_j = dominates_plain(fi, fj);
                    j_dom_i = dominates_plain(fj, fi);
                }
                if (i_dom_j)      S[i].push_back(j);
                else if (j_dom_i) ++np[i];
            }
            if (np[i] == 0) vault.get_ind(i).rank = 0;
        }

        std::vector<std::vector<int>> fronts;
        std::vector<int> front0;
        for (int i = 0; i < n; ++i)
            if (np[i] == 0) front0.push_back(i);
        fronts.push_back(front0);

        int k = 0;
        while (!fronts[k].empty()) {
            std::vector<int> next_front;
            for (int i : fronts[k]) {
                for (int j : S[i]) {
                    if (--np[j] == 0) {
                        vault.get_ind(j).rank = k + 1;
                        next_front.push_back(j);
                    }
                }
            }
            fronts.push_back(next_front);
            ++k;
        }
        fronts.pop_back();
        return fronts;
    }

    void assign_crowding_distance(DataVault<Ind_t>& vault,
                                  const std::vector<int>& front)
    {
        const bool in_x = crowding_space_ == CrowdingSpace::Decision;
        int m = in_x ? vault.vars_n() : vault.objs_n();
        int l = static_cast<int>(front.size());
        if (l == 0) return;
        // the coordinate the distance is measured in: an objective, or a variable
        auto at = [&](int v, int k) -> double {
            return in_x ? vault.variables_of(v)[k] : vault.objectives_of(v)[k];
        };

        for (int v : front) vault.get_ind(v).crowding_distance = 0.0;

        for (int obj = 0; obj < m; ++obj) {
            std::vector<int> sorted = front;
            std::sort(sorted.begin(), sorted.end(), [&](int a, int b) {
                return at(a, obj) < at(b, obj);
            });

            vault.get_ind(sorted.front()).crowding_distance =
                std::numeric_limits<double>::infinity();
            vault.get_ind(sorted.back()).crowding_distance  =
                std::numeric_limits<double>::infinity();

            double f_min = at(sorted.front(), obj);
            double f_max = at(sorted.back(), obj);
            double range = f_max - f_min;
            if (range < 1e-14) continue;

            for (int i = 1; i < l - 1; ++i) {
                double prev = at(sorted[i - 1], obj);
                double next = at(sorted[i + 1], obj);
                vault.get_ind(sorted[i]).crowding_distance += (next - prev) / range;
            }
        }
    }

    int tournament(DataVault<Ind_t>& vault,
                   std::uniform_int_distribution<int>& dist)
    {
        int a = dist(rng_), b = dist(rng_);
        const auto& ia = vault.get_ind(a);
        const auto& ib = vault.get_ind(b);

        if (constraint_mode == ConstraintMode::FEASIBILITY ||
            constraint_mode == ConstraintMode::CDP) {
            double cva = vault.get_cv(a);
            double cvb = vault.get_cv(b);
            bool a_feas = (cva <= 0.0);
            bool b_feas = (cvb <= 0.0);
            if ( a_feas && !b_feas) return a;
            if (!a_feas &&  b_feas) return b;
            if (!a_feas && !b_feas) return (cva < cvb) ? a : b;
        }
        if (ia.rank < ib.rank) return a;
        if (ib.rank < ia.rank) return b;
        return (ia.crowding_distance > ib.crowding_distance) ? a : b;
    }

public:
    NSGAIICore() = default;

    void set_eta_crossover(double e) { eta_c_ = e; }
    void set_eta_mutation (double e) { eta_m_ = e; }
    void set_pc(double p)            { pc_ = p; }
    void set_seed(unsigned s)        { rng_.seed(s); }
    void set_crossover(ops::Crossover c)      { xover_.kind = c; }
    void set_mutation(ops::Mutation m)        { mut_.kind = m; }
    void set_mutation_scale(double s)         { mut_.scale = ops::checked_mutation_scale(s); }
    void set_mixture_q(double q)              { mut_.q = q; }
    void set_blx_alpha(double a)              { xover_.alpha = a; }
    void set_crowding_space(CrowdingSpace c) { crowding_space_ = c; }
    // p_m; 1/n when not set (§IV-A). A knob since task 7 (§5.1): it used to be
    // reported as ignored.
    void set_pm(double p)                     { pm_ = p; }
    // crossover = de / de_pbest / portfolio (de_crossover.hpp, real_crossover.hpp)
    void set_F(double f)                      { xover_.de.set.F  = ops::checked_de_F(f); }
    void set_CR(double c)                     { xover_.de.set.CR = ops::checked_de_CR(c); }
    bool set_param(const std::string& k, double v)
    {
        if (k == "spu_frac") {
            if (!(v > 0.0 && v <= 1.0)) throw std::invalid_argument("spu_frac must be in (0, 1]");
            spu_frac_ = v;
            return true;
        }
        return arch_.set_param(k, v) || xover_.set_param(k, v);
    }
    bool set_word(const std::string& k, const std::string& w)
    {
        if (k == "survival") {
            if (w != "greedy" && w != "stochastic")
                throw std::invalid_argument("survival = '" + w + "': greedy or stochastic");
            stochastic_ = w == "stochastic";
            return true;
        }
        if (k == "bin_crossover") {
            if (w != "uniform" && w != "none")
                throw std::invalid_argument("bin_crossover = '" + w + "': uniform or none");
            bin_cross_ = w == "uniform";
            return true;
        }
        if (k == "mating") {
            if (w != "tournament" && w != "each")
                throw std::invalid_argument("mating = '" + w + "': tournament or each");
            mating_each_ = w == "each";
            return true;
        }
        return arch_.set_word(k, w) || xover_.set_word(k, w);
    }

    // The inner archive's members, for the binding's result.
    const std::vector<ops::InnerArchive::Member>& inner_archive() const { return arch_.members(); }

    // The objectives of Q \ Q' at the last stochastic selection (tests).
    const std::vector<std::vector<double>>& last_untouched() const { return untouched_; }

    void set_bound_repair(ops::BoundRepair r) {
        ops::require_repair(r, "nsga2", true, false);
        xover_.repair = r; mut_.repair = r;
    }

    void setup(DataVault<Ind_t>& vault)
    {
        int n = vault.pop_size();
        const auto& bounds = vault.get_bounds();
        std::uniform_real_distribution<double> dist_real(0.0, 1.0);
        std::vector<double> vars(vault.vars_n());
        std::vector<int>    bvars(vault.bin_vars_n());

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < vault.vars_n(); ++j) {
                double lo = bounds[j].first .value_or(0.0);
                double hi = bounds[j].second.value_or(1.0);
                vars[j] = lo + dist_real(rng_) * (hi - lo);
            }
            for (int j = 0; j < vault.bin_vars_n(); ++j)
                bvars[j] = ops::random_gene(j, rng_);

            if (vault.bin_vars_n() > 0)
                vault.set_all_variables(i, vars, bvars);
            else
                vault.set_variables(i, vars);
        }
        vault.sync();

        auto fronts = fast_nondominated_sort(vault, n);
        for (auto& front : fronts) assign_crowding_distance(vault, front);
        start_archive(vault);
    }

    // ============================================================
    //  setup_seeded — for resume via Optimizer::setup_with_seed.
    //  The vault is already seeded with the seed_individual → skip the
    //  random generation, only recompute ranks + crowding distance.
    // ============================================================
    void setup_seeded(DataVault<Ind_t>& vault)
    {
        int n = vault.pop_size();
        auto fronts = fast_nondominated_sort(vault, n);
        for (auto& front : fronts) assign_crowding_distance(vault, front);
        start_archive(vault);
    }

    void step(DataVault<Ind_t>& vault)
    {
        int n = vault.pop_size();
        const auto& bounds = vault.get_bounds();
        std::uniform_int_distribution<int> dist_int(0, n - 1);

        xover_.observe(vault, n, rng_);  // de, de_pbest, portfolio: the last step's verdict
        // de_pbest: a random one of the best ⌈p·n⌉, by rank, then crowding distance
        std::vector<int> order;
        auto best = [&](double p) {
            if (order.empty()) {
                order.resize(static_cast<std::size_t>(n));
                std::iota(order.begin(), order.end(), 0);
                std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
                    const auto& ia = vault.get_ind(a);
                    const auto& ib = vault.get_ind(b);
                    if (ia.rank != ib.rank) return ia.rank < ib.rank;
                    return ia.crowding_distance > ib.crowding_distance;
                });
            }
            const int k = std::max(1, static_cast<int>(std::ceil(p * n)));
            std::uniform_int_distribution<int> top(0, k - 1);
            return vault.variables_of(order[static_cast<std::size_t>(top(rng_))]);
        };

        int off_base = vault.expand(n);  // [off_base, off_base+n) — offspring slots
        // one more parent, for the crossovers that take more than two (B2)
        auto draw = [&] {
            if (arch_.reuse(rng_)) return arch_.pick(rng_).x;
            return vault.variables_of(tournament(vault, dist_int));
        };
        // a parent: the tournament's winner (mating = each: member k itself),
        // or a member of the inner archive
        const bool bin = vault.bin_vars_n() > 0;
        auto parent = [&](int k, std::vector<double>& x, std::vector<int>& bv) {
            if (arch_.reuse(rng_)) {
                const auto& m = arch_.pick(rng_);
                x = m.x;
                bv = m.b;
                return;
            }
            const int p = mating_each_ ? std::min(k, n - 1) : tournament(vault, dist_int);
            for (int j = 0; j < vault.vars_n(); ++j) x[j] = vault.get_variable(p, j);
            if (bin) bv = vault.binary_variables_of(static_cast<std::size_t>(p));
        };

        std::vector<double> pv1(vault.vars_n()), pv2(vault.vars_n()), c1, c2;
        std::vector<int> bv1, bv2;
        for (int i = 0; i < n; i += 2) {
            parent(i, pv1, bv1);
            parent(i + 1, pv2, bv2);
            // §IV-A: pc=0.9, pm=1/n
            double pm = pm_ ? *pm_ : (vault.vars_n() > 0) ? 1.0 / vault.vars_n() : 0.0;
            ops::vary(xover_, mut_, pv1, pv2, draw, best, n, c1, c2, bounds,
                      eta_c_, pc_, eta_m_, pm, rng_, [] { return -1; });
            xover_.finish(c1, c2, i + 1 < n ? -1 : 0);

            if (bin) {
                std::vector<int> bc1 = bv1, bc2 = bv2;
                if (bin_cross_) ops::binary_crossover(bv1, bv2, bc1, bc2, rng_);
                ops::bit_flip_mutation(bc1, vault.bin_vars_n(), rng_);
                if (i + 1 < n) ops::bit_flip_mutation(bc2, vault.bin_vars_n(), rng_);
                vault.set_all_variables(off_base + i, c1, bc1);
                if (i + 1 < n) vault.set_all_variables(off_base + i + 1, c2, bc2);
            } else {
                vault.set_variables(off_base + i, c1);
                if (i + 1 < n) vault.set_variables(off_base + i + 1, c2);
            }
        }
        vault.sync();
        for (int i = 0; i < n; ++i) archive_point(vault, static_cast<std::size_t>(off_base + i));
        select(vault);
    }

    // The environmental selection on the 2μ active points: greedy, the
    // paper's, or stochastic (above). Public for the tests.
    void select(DataVault<Ind_t>& vault)
    {
        const int n = vault.pop_size();
        int m = 2 * n, keep = n;              // Q' and how many of it survive
        if (stochastic_ && spu_frac_ < 1.0) {
            m = static_cast<int>(std::floor(spu_frac_ * 2 * n));
            if (m < n)
                throw std::invalid_argument("spu_frac = " + std::to_string(spu_frac_) +
                                            " makes |Q'| = floor(spu_frac 2n) smaller than the "
                                            "population " + std::to_string(n));
            for (int i = 0; i < m; ++i) {      // Q' into [0, m)
                std::uniform_int_distribution<int> any(i, 2 * n - 1);
                const int j = any(rng_);
                if (j != i) vault.swap_active(i, j);
            }
            keep = m - n;
            untouched_.clear();
            for (int v = m; v < 2 * n; ++v) untouched_.push_back(vault.objectives_of(v));
        }

        auto fronts = fast_nondominated_sort(vault, m);
        for (auto& front : fronts) assign_crowding_distance(vault, front);

        std::vector<int> survivors;
        survivors.reserve(n);
        for (auto& front : fronts) {
            if (static_cast<int>(survivors.size()) == keep) break;
            if (static_cast<int>(survivors.size()) + static_cast<int>(front.size()) <= keep) {
                for (int v : front) survivors.push_back(v);
            } else {
                int needed = keep - static_cast<int>(survivors.size());
                std::partial_sort(front.begin(), front.begin() + needed, front.end(),
                    [&](int a, int b) {
                        return vault.get_ind(a).crowding_distance
                             > vault.get_ind(b).crowding_distance;
                    });
                for (int i = 0; i < needed; ++i) survivors.push_back(front[i]);
                break;
            }
            if (static_cast<int>(survivors.size()) == keep) break;
        }
        for (int v = m; v < 2 * n; ++v) survivors.push_back(v);     // Q \ Q'

        // Permutation: survivors[i] goes to position i.
        // pos[v] = current virtual position of the individual originally at v.
        // at_pos[p] = virtual "original id" currently located at position p.
        // Both maps are updated in O(1) per swap → O(N) total instead of O(N²).
        std::vector<int> pos   (n * 2);
        std::vector<int> at_pos(n * 2);
        std::iota(pos.begin(),    pos.end(),    0);
        std::iota(at_pos.begin(), at_pos.end(), 0);

        for (int i = 0; i < n; ++i) {
            int want = survivors[i];
            int cur  = pos[want];
            if (cur == i) continue;

            int other = at_pos[i];
            vault.swap_active(i, cur);
            pos[want]    = i;
            pos[other]   = cur;
            at_pos[i]    = want;
            at_pos[cur]  = other;
        }

        vault.reduce(n);
        if (m < 2 * n) {                      // the next tournament: ranks over P_{t+1}
            auto again = fast_nondominated_sort(vault, n);
            for (auto& front : again) assign_crowding_distance(vault, front);
        }
    }

private:
    std::vector<std::vector<double>> untouched_;
};

} // namespace mootation

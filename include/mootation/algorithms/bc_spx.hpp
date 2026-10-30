#pragma once
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "../constraint_mode.hpp"
#include "../data_vault.hpp"
#include "../detail/constrained.hpp"
#include "../operators/bound_repair.hpp"
#include "../operators/multi_parent.hpp"
#include "../operators/real_crossover.hpp"
#include "../random.hpp"

namespace mootation {

// ============================================================================
// BC — the non-elitist best-children selection, with SPX or REX
// R. Tanabe, H. Ishibuchi, "Non-elitist Evolutionary Multi-objective
// Optimizers Revisited", GECCO 2019, pp. 612-619, doi:10.1145/3321707.3321754
// (arXiv:2009.14717, the authors' version: bc_tanabe2019; Algorithms 1-4,
// §2.2-2.3, Table 1, §3).
//
// The simple EMOA (Algorithm 1), one step:
//   R ← k parents drawn from P, all different;
//   Q ← λ children, each by the crossover applied to the same R;
//   BC (Algorithm 4): P ← P \ R; rank every point of P ∪ Q; the k best of
//      Q join P — the parents go whatever their quality: non-elitist.
// The other two selections of the paper, for its comparisons (bc_selection):
//   BF (Algorithm 3): rank P ∪ Q; P ← P \ R; the k best of Q ∪ R join P;
//   BA (Algorithm 2): rank P ∪ Q; the μ best of it are P.
// The ranks are assigned once and the best taken one after another (lines
// 3-5). Ranking "NS" (§2.3): the non-domination level, then the crowding
// distance within it (NSGA-II's).
//
// Parameters (§3, Table 1): λ = 10n (bc_lambda; 0: 10n); k = n + 1; SPX with
// ε = √(n + 2), REX with ξ ~ N(0, 1/(k − 1)) (operators/multi_parent.hpp,
// which already follows Table 1). The paper's μ = ⌊100 ln n⌋ is the run's
// pop_size here (BC-1). No mutation (§2.2: SPX and REX need none).
// The two registry names: bc_spx (crossover spx, this header) and bc_rex
// (rex, bc_rex.hpp); the knob crossover switches between them.
//
// READINGS (declared):
//   BC-1. μ is pop_size; ⌊100 ln n⌋ (Tušar & Filipič, as §3 takes it) is the
//     paper's, worth passing; k = n + 1 > μ is refused.
//   BC-2. A child outside the box: bound_repair (reflect by default, as the
//     library's multi-parent crossovers, multi_parent.hpp MP-4); the paper's
//     BBOB problems are unbounded.
//   BC-3. Constraints: Deb's constrained domination in the sorting
//     (constraint_mode not NONE); the paper has none.
//   BC-4. The answer is P, which BC does not keep elitist: the paper measures
//     an unbounded archive of every non-dominated point (COCO's), not P; a
//     campaign's archive scenario is that measure here.
//   BC-5. A tie of level and crowding distance: the earlier candidate (BC:
//     Q in order; BF: Q, then R; BA: P, then Q).
// Real variables only.
// ============================================================================

template <typename Ind_t>
class BCCore {
public:
    ConstraintMode constraint_mode = ConstraintMode::NONE;

    explicit BCCore(ops::Crossover c = ops::Crossover::SPX) : xover_(c) {}

    void set_seed(unsigned s) { rng_.seed(s); }
    void set_crossover(ops::Crossover c)
    {
        if (c != ops::Crossover::SPX && c != ops::Crossover::REX)
            throw std::invalid_argument("bc: crossover = spx or rex");
        xover_ = c;
    }
    void set_bound_repair(ops::BoundRepair r)
    {
        ops::require_repair(r, "bc", false, false);
        repair_ = r;
    }

    bool set_param(const std::string& k, double v)
    {
        if (k == "bc_lambda") {
            if (!(v >= 0.0) || v != std::floor(v) || v > 1e8)
                throw std::invalid_argument("bc_lambda must be a whole number >= 0 (0: 10n)");
            lambda_ = static_cast<int>(v);
            return true;
        }
        return false;
    }

    bool set_word(const std::string& k, const std::string& w)
    {
        if (k == "bc_selection") {
            if (w == "bc") sel_ = Sel::BC;
            else if (w == "bf") sel_ = Sel::BF;
            else if (w == "ba") sel_ = Sel::BA;
            else throw std::invalid_argument("bc_selection = '" + w + "': bc, bf or ba");
            return true;
        }
        return false;
    }

    void setup(DataVault<Ind_t>& vault)
    {
        check(vault);
        const auto& b = vault.get_bounds();
        rnd::uniform_real_distribution<double> U(0.0, 1.0);
        std::vector<double> x(b.size());
        for (int i = 0; i < vault.pop_size(); ++i) {
            for (std::size_t j = 0; j < b.size(); ++j)
                x[j] = *b[j].first + U(rng_) * (*b[j].second - *b[j].first);
            vault.set_variables(static_cast<std::size_t>(i), x);
        }
        vault.sync();
    }

    void setup_seeded(DataVault<Ind_t>& vault) { check(vault); }

    void step(DataVault<Ind_t>& vault)
    {
        const int mu = vault.pop_size();
        const int n = vault.vars_n();
        const int k = n + 1;
        const int lam = lambda_ > 0 ? lambda_ : 10 * n;
        // R: k different members of P (a partial Fisher-Yates draw)
        std::vector<int> idx(static_cast<std::size_t>(mu));
        std::iota(idx.begin(), idx.end(), 0);
        for (int t = 0; t < k; ++t) {
            rnd::uniform_int_distribution<int> any(t, mu - 1);
            std::swap(idx[static_cast<std::size_t>(t)], idx[static_cast<std::size_t>(any(rng_))]);
        }
        std::vector<int> R(idx.begin(), idx.begin() + k);
        ops::ParentSet parents;
        for (int r : R) parents.push_back(vault.variables_of(static_cast<std::size_t>(r)));
        const auto g = ops::mp_detail::centre(parents);
        // Q: λ children of the same R, in the slots after P
        const auto& bounds = vault.get_bounds();
        const int q0 = vault.expand(lam);
        ops::note_operator(xover_ == ops::Crossover::SPX ? "spx" : "rex",
                           ops::bound_repair_name(repair_));
        for (int c = 0; c < lam; ++c) {
            auto y = xover_ == ops::Crossover::SPX
                         ? ops::spx_child(parents, std::sqrt(static_cast<double>(n) + 2.0), rng_)
                         : ops::rex_child(parents, rng_);
            ops::mp_detail::repair(y, g, bounds, repair_, rng_);
            vault.set_variables(static_cast<std::size_t>(q0 + c), y);
        }
        vault.sync();
        std::vector<int> Q(static_cast<std::size_t>(lam));
        std::iota(Q.begin(), Q.end(), q0);
        std::vector<char> inR(static_cast<std::size_t>(mu), 0);
        for (int r : R) inR[static_cast<std::size_t>(r)] = 1;

        // the ranked set and the candidates, by the selection
        std::vector<int> ranked, cand;
        for (int i = 0; i < mu; ++i)
            if (sel_ != Sel::BC || !inR[static_cast<std::size_t>(i)]) ranked.push_back(i);
        ranked.insert(ranked.end(), Q.begin(), Q.end());
        if (sel_ == Sel::BC) cand = Q;
        else if (sel_ == Sel::BF) { cand = Q; cand.insert(cand.end(), R.begin(), R.end()); }
        else cand = ranked;
        const auto key = rank(vault, ranked);              // (level, −crowding) per member
        std::vector<std::size_t> pos(cand.size());
        std::iota(pos.begin(), pos.end(), std::size_t{0});
        std::vector<std::size_t> at(static_cast<std::size_t>(mu + lam));   // slot -> place in ranked
        for (std::size_t t = 0; t < ranked.size(); ++t) at[static_cast<std::size_t>(ranked[t])] = t;
        std::stable_sort(pos.begin(), pos.end(), [&](std::size_t a, std::size_t b) {
            return key[at[static_cast<std::size_t>(cand[a])]] < key[at[static_cast<std::size_t>(cand[b])]];
        });
        const int take = sel_ == Sel::BA ? mu : k;
        std::vector<VaultPoint> chosen;
        for (int t = 0; t < take; ++t)
            chosen.push_back(vault.point_of(static_cast<std::size_t>(cand[pos[static_cast<std::size_t>(t)]])));
        if (sel_ == Sel::BA) {
            for (int i = 0; i < mu; ++i) vault.put_point(static_cast<std::size_t>(i), chosen[static_cast<std::size_t>(i)]);
        } else {
            for (int t = 0; t < k; ++t) vault.put_point(static_cast<std::size_t>(R[static_cast<std::size_t>(t)]), chosen[static_cast<std::size_t>(t)]);
        }
        vault.reduce(mu);
    }

private:
    enum class Sel { BC, BF, BA };
    ops::Crossover xover_;
    ops::BoundRepair repair_ = ops::BoundRepair::Reflect;   // BC-2
    Sel sel_ = Sel::BC;
    int lambda_ = 0;                                         // 0: 10n
    std::mt19937 rng_{0};

    void check(DataVault<Ind_t>& vault) const
    {
        if (vault.bin_vars_n() > 0)
            throw std::invalid_argument("bc: continuous variables only");
        for (const auto& b : vault.get_bounds())
            if (!b.first || !b.second)
                throw std::invalid_argument("bc: every variable needs finite bounds");
        const int n = vault.vars_n();
        if (vault.pop_size() < n + 1)
            throw std::invalid_argument(
                "bc: incompatible by construction: one crossover takes k = n + 1 = " +
                std::to_string(n + 1) + " different parents from the population, pop_size = " +
                std::to_string(vault.pop_size()) + " is fewer (pass pop_size >= n + 1; the "
                "paper's μ is floor(100 ln n) = " +
                std::to_string(static_cast<int>(std::floor(100.0 * dm::log(static_cast<double>(n))))) +
                ")");
    }

    // NS: the non-domination level, then minus the crowding distance within
    // it, of every member of S (BC-3: Deb's constrained domination).
    std::vector<std::pair<int, double>> rank(DataVault<Ind_t>& vault, const std::vector<int>& S)
    {
        const std::size_t N = S.size();
        const bool cons = constraint_mode != ConstraintMode::NONE;
        std::vector<std::vector<double>> F(N);
        std::vector<double> cv(N, 0.0);
        for (std::size_t i = 0; i < N; ++i) {
            F[i] = vault.objectives_of(static_cast<std::size_t>(S[i]));
            if (cons) cv[i] = vault.get_cv(static_cast<std::size_t>(S[i]));
        }
        std::vector<std::vector<std::size_t>> dom(N);
        std::vector<int> count(N, 0);
        for (std::size_t i = 0; i < N; ++i)
            for (std::size_t j = i + 1; j < N; ++j) {
                if (detail::dominates(constraint_mode, F[i], cv[i], F[j], cv[j])) {
                    dom[i].push_back(j); ++count[j];
                } else if (detail::dominates(constraint_mode, F[j], cv[j], F[i], cv[i])) {
                    dom[j].push_back(i); ++count[i];
                }
            }
        std::vector<std::pair<int, double>> key(N, {0, 0.0});
        std::vector<std::size_t> front;
        for (std::size_t i = 0; i < N; ++i) if (count[i] == 0) front.push_back(i);
        for (int level = 0; !front.empty(); ++level) {
            crowding(F, front, key, level);
            std::vector<std::size_t> next;
            for (std::size_t i : front)
                for (std::size_t j : dom[i]) if (--count[j] == 0) next.push_back(j);
            std::stable_sort(next.begin(), next.end());
            front = std::move(next);
        }
        return key;
    }

    static void crowding(const std::vector<std::vector<double>>& F,
                         const std::vector<std::size_t>& front,
                         std::vector<std::pair<int, double>>& key, int level)
    {
        const double inf = std::numeric_limits<double>::infinity();
        std::vector<double> d(front.size(), 0.0);
        if (front.size() <= 2) std::fill(d.begin(), d.end(), inf);
        else {
            std::vector<std::size_t> o(front.size());
            for (std::size_t m = 0; m < F.front().size(); ++m) {
                std::iota(o.begin(), o.end(), std::size_t{0});
                std::stable_sort(o.begin(), o.end(), [&](std::size_t a, std::size_t b) {
                    return F[front[a]][m] < F[front[b]][m];
                });
                const double lo = F[front[o.front()]][m], hi = F[front[o.back()]][m];
                d[o.front()] = d[o.back()] = inf;
                if (!(hi > lo)) continue;
                for (std::size_t t = 1; t + 1 < o.size(); ++t)
                    d[o[t]] += (F[front[o[t + 1]]][m] - F[front[o[t - 1]]][m]) / (hi - lo);
            }
        }
        for (std::size_t t = 0; t < front.size(); ++t) key[front[t]] = {level, -d[t]};
    }
};

template <typename Ind_t>
class BCSPXCore : public BCCore<Ind_t> {
public:
    BCSPXCore() : BCCore<Ind_t>(ops::Crossover::SPX) {}
};

}   // namespace mootation

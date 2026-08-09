#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// NAEMO — Neighborhood-sensitive Archived Evolutionary Many-objective
// Optimization algorithm.
// R. Sengupta, M. Pal, S. Saha, S. Bandyopadhyay — Swarm and Evolutionary
// Computation 46 (2019) 201-218.
// doi:10.1016/j.swevo.2018.12.002
//
// IDEA. The archive is split into sub-archives by reference lines
// (Das–Dennis). Every reference line always retains >=1 associated solution
// (monotone growth of diversity, Thm 3-4). Mating is local, within the k
// nearest NON-EMPTY neighbouring lines. The archive size is held between
// L_hard=n and L_soft by periodic filtering: a convergence filter (removes
// solutions dominated by a new child, but never empties a line) and a diversity
// filter (removes the solution with the LARGEST PBI from the most populated
// line). Variation switches probabilistically between SBX and DE, with optional
// polynomial mutation.
//
// SCHEME (Algorithm 1):
//   1. W <- Das–Dennis; arch <- L_soft random solutions; association ->
//      sub_arch_i (Eq.15).
//   for itr=1..tot_itr:
//     S_ηc=S_F=S_CR=∅
//     for j=1..n:
//       ind=j; if sub_arch[ind]=∅: ind <- a random line from nbr_j (the k
//       nearest non-empty ones); parent <- a random member of sub_arch[ind]
//       η_c=N(μ_ηc,5); F=N(μ_F,0.1); CR=N(μ_CR,0.1)        (Eq.21-23)
//       child ← Mutate(parent)                              (Algorithm 2)
//       if parent does NOT dominate child:
//         line <- association(child); sub_arch[line] ∪= child
//         convergence-filter (Algorithm 3)
//         if |arch|>L_soft: diversity-filter → L_hard (Algorithm 4)
//         S_ηc∪=η_c; S_F∪=F; S_CR∪=CR
//     μ_ηc=mean(S_ηc); μ_F=mean(S_F); μ_CR=mean(S_CR)       (Eq. adapt)
//
// MUTATE (Algorithm 2): rand>mut_prob -> DE (Eq.18, rand/1/bin with CR); if
//   flag2, polynomial mutation follows. Otherwise -> SBX (Eq.16-17, the first
//   child is taken); if flag1, polynomial mutation follows. Partners are drawn
//   from the neighbourhood nbr.
//
// ASSOCIATION (Eq.15): d2(x,w_i)=||x − (x·w_i/||w_i||²)·w_i||; line = argmin d2.
// PBI (Eq.2): g=d1+θ·d2, d1=(x·w)/||w|| (x is translated by subtracting the
//   ideal point z*).
//
// PAPER DEFAULTS (§4.5, §5.2):
//   • θ = 5 («standard value», §3.2). • mut_prob = 0.75.
//   • μ_ηc init=30 (var 5); μ_F init=0.5 (var 0.1); μ_CR init=0.2 (var 0.1).
//   • flag1 = flag2 = FALSE (§5.2: "The values of flag1 and flag2 are false in
//     all cases except for DTLZ1 and DTLZ3 ... flag2 is set as true for DTLZ1
//     while flag1 is set as true for DTLZ3") — i.e. per problem, via
//     set_flag1/set_flag2. The code defaults match. • η_m = 20.
//   • k = max(2, ⌊0.2·n⌋) («The neighborhood size is set as 20% of the total
//     number of reference lines», §5.2; §4.9 gives the 10-20% range).
//   * L_hard = n (the number of reference lines) = set_K;
//     L_soft = pop_size (= C·n, C>1).
//   In this library: pop_size = L_soft; n = set_K <= pop_size; L_hard = n.
//   The resulting active_n lies in [L_hard, L_soft].
//
// DECLARED DEVIATIONS:
//   NAEMO-1 (DEVIATION). Eq.19 formally applies polynomial mutation to "all j";
//     implemented as Deb's CANONICAL polynomial mutation with pm=1/n_vars. The
//     literal "every variable every time" is excessive mutation and contradicts
//     both the convergence reported in the paper and common practice.
//     Overridable via set_pm.
//   NAEMO-2 (MINOR). PBI and association operate in the translated space
//     x' = f − z* (z* being the archive ideal point); the paper draws the lines
//     through the origin in the raw space (for DTLZ, f>=0 and z*~0) — this is
//     the MOEA/D-PBI convention.
//   NAEMO-3 (MINOR). §4.4 takes the DE points "from the archive", while
//     §4.1/4.9 restrict reproduction to the neighbourhood. §4.1/4.9 is followed
//     (partners from nbr), with a fallback to the whole archive when the pool
//     is too small.
//   NAEMO-4 (MINOR). Gaussian(μ,v): v is read as the VARIANCE (text of §4.5),
//     so std = sqrt(v). F and CR are truncated to [0,1] (§4.5: "sampled values
//     ... truncated to [0,1]", said of F and CR); η_c is clamped from below to
//     >=1e-6, a guard beyond the paper that is harmless under N(30, sqrt(5)).
//   NAEMO-5 (MINOR). k=⌊0.2·n⌋ (§5.2: «20% of the total number of
//     reference lines"); the soft limit is pop_size (the paper: C·n, with C a
//     small constant, see Table 2). The header previously declared 10% while
//     the code used 20% (n_/5) — the code matched the paper, the declaration
//     did not; the declaration is now correct.
//   NAEMO-6 (MINOR). z* and PBI are recomputed once per generation (at the
//     start of step) and held fixed for the inner loop.
//   NAEMO-7 (MINOR, resolved). The reference-line count is chosen as the
//     LARGEST attainable Das-Dennis lattice <= pop_size on the default path.
//     das_dennis::generate_auto rounds UP, and since every line keeps at least
//     one member, a lattice larger than pop_size violates the invariant stated
//     above (n = L_hard <= L_soft = pop_size) and inflates the per-generation
//     function-evaluation count, because one offspring is bred per line. An
//     explicit set_K is honoured as given; if it resolves above pop_size the
//     warn channel says so rather than silently changing the budget.
//
// GENOME: real-valued only. Both reproduction branches (SBX and DE) are
//   real-valued, so setup()/setup_seeded() THROW std::invalid_argument when the
//   vault carries binary variables. An earlier version of this header advertised
//   a binary genome "beyond the paper"; no such code path ever existed, and the
//   binaries of every child were silently dropped.
// CONSTRAINTS (beyond the paper, off by default). NAEMO's only acceptance test
//   is "the parent does not dominate the child", and its only removal test is
//   the convergence filter's "the child dominates this member". constraint_mode
//   FEASIBILITY/CDP makes both of them Deb's constrained domination, so an
//   infeasible child cannot enter over a feasible parent and cannot evict a
//   feasible archive member. The diversity filter (largest PBI on the most
//   populated line) is geometric and is left alone. The paper is unconstrained.
// ============================================================================

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "../constraint_mode.hpp"
#include "../detail/constrained.hpp"
#include "../das_dennis.hpp"
#include "../data_vault.hpp"
#include "../warn.hpp"
#include "../operators/poly_mutation.hpp"
#include "../operators/sbx.hpp"

namespace mootation {

template <typename Ind_t>
class NAEMOCore {
public:
    ConstraintMode constraint_mode = ConstraintMode::NONE;  // beyond the paper

private:
    // ── parameters ──────────────────────────────────────────────────────────
    int    n_req_    = 0;        // requested reference lines (=L_hard); 0 -> auto
    double theta_    = 5.0;      // PBI penalty (§3.2 standard)
    double mut_prob_ = 0.75;     // §4.5
    double eta_m_    = 20.0;     // polynomial mutation
    double pm_       = -1.0;     // <0 → 1/n_vars (NAEMO-1)
    // The paper: "flag1 and flag2 are FALSE in all cases except DTLZ1/DTLZ3".
    // Always-on polynomial mutation suppressed convergence on every other
    // problem, so the default is false. For DTLZ1 set flag2=true, for DTLZ3 set
    // flag1=true — per-problem, via the setters.
    bool   flag1_    = false;    // polynomial mutation after SBX (DTLZ3 -> true)
    bool   flag2_    = false;    // polynomial mutation after DE  (DTLZ1 -> true)
    std::mt19937 rng_{std::random_device{}()};

    // ── state ───────────────────────────────────────────────────────────────
    std::vector<std::vector<double>> W_;        // reference lines (Das-Dennis) [n][m]
    // FIX 2026-07-08:
    // The line neighbourhood is DYNAMIC. line_rank_[i] is the FULL list of all
    // other lines, sorted once by geometric proximity. The neighbourhood itself
    // nbr_i (: «set of NON-EMPTY reference lines, neighboring the i-th») —
    // is the first k NON-EMPTY lines of line_rank_[i], computed on the fly from
    // cnt each generation (see neighbors_nonempty()). Storing the k
    // geometrically nearest lines statically, ignoring occupancy, narrowed the
    // search when neighbours were empty instead of widening it to the k nearest
    // non-empty ones.
    std::vector<std::vector<int>>    line_rank_; // all other lines, by proximity
    int    n_ = 0;                               // reference lines = L_hard
    int    L_soft_ = 0;                          // = pop_size
    int    k_ = 2;                               // neighbourhood size
    // adaptive means
    double mu_etac_ = 30.0, mu_F_ = 0.5, mu_CR_ = 0.2;

    struct Sol { std::vector<double> vars, objs; std::vector<int> bvars; double cv=0.0; };
    std::vector<Sol> arch_;                      // source of truth (the archive)

    double pm_eff(int nv) const {
        return (pm_ > 0.0) ? pm_ : (nv > 0 ? 1.0 / nv : 0.0);
    }

    static bool dominates(const std::vector<double>& a,
                          const std::vector<double>& b) {
        return detail::pareto_dominates(a, b);
    }
    // Constrained form: used everywhere the archive decides what survives.
    bool dominates_c(const std::vector<double>& a, double cva,
                     const std::vector<double>& b, double cvb) const {
        return detail::dominates(constraint_mode, a, cva, b, cvb);
    }

    // d2 (Eq.15) and PBI (Eq.2) in the translated space x' = f - z*
    void pbi_components(const std::vector<double>& f,
                        const std::vector<double>& z,
                        const std::vector<double>& w,
                        double& d1, double& d2) const {
        int m = (int)f.size();
        double dot = 0.0, wn2 = 0.0;
        for (int k = 0; k < m; ++k) {
            double xk = f[k] - z[k];
            dot += xk * w[k];
            wn2 += w[k] * w[k];
        }
        double wn = std::sqrt(std::max(wn2, 1e-300));
        double proj = dot / wn2;                 // projection coefficient
        d1 = dot / wn;                           // = proj*||w||
        double s2 = 0.0;
        for (int k = 0; k < m; ++k) {
            double xk = f[k] - z[k];
            double perp = xk - proj * w[k];
            s2 += perp * perp;
        }
        d2 = std::sqrt(std::max(s2, 0.0));
    }

    int associate(const std::vector<double>& f, const std::vector<double>& z) const {
        int best = 0; double best_d2 = std::numeric_limits<double>::max();
        for (int i = 0; i < n_; ++i) {
            double d1, d2; pbi_components(f, z, W_[i], d1, d2);
            if (d2 < best_d2) { best_d2 = d2; best = i; }
        }
        return best;
    }

    double pbi(const std::vector<double>& f, const std::vector<double>& z,
               int line) const {
        double d1, d2; pbi_components(f, z, W_[line], d1, d2);
        return d1 + theta_ * d2;
    }

    // NAEMO has no binary path: the SBX and DE branches both produce real
    // vectors only, and the child is written with set_variables, which leaves
    // the scratch slot's binaries at their init_slot zeros. Rather than lose a
    // caller's binary genome silently, refuse the configuration outright.
    static void require_real_only(const DataVault<Ind_t>& vault) {
        if (vault.bin_vars_n() > 0)
            throw std::invalid_argument(
                "NAEMO: binary variables are not supported "
                "(reproduction is real-valued only)");
    }

    void build_directions(int m) {
        if (n_req_ > 0) {
            W_ = das_dennis::generate_auto(m, n_req_);
            if ((int)W_.size() > L_soft_)
                warn_lazy([&]{ return "naemo: set_K(" + std::to_string(n_req_) +
                               ") resolved to " + std::to_string(W_.size()) +
                               " reference lines, which exceeds pop_size=" +
                               std::to_string(L_soft_); });
        } else {
            // L_hard = n must not exceed L_soft = pop_size: the archive is held
            // in [L_hard, L_soft] and every line keeps at least one member, so a
            // lattice larger than pop_size breaks that invariant and inflates
            // the per-generation FE count. generate_auto rounds UP, so the
            // default path takes the largest attainable lattice <= pop_size
            // instead (NAEMO-7).
            const int H = das_dennis::find_H_le(m, L_soft_);
            W_ = das_dennis::generate(m, std::max(1, H));
            if ((int)W_.size() > L_soft_)   // only reachable at absurdly small N
                W_.resize(static_cast<std::size_t>(std::max(2, L_soft_)));
        }
        n_ = (int)W_.size();
        // FULL geometric neighbour ordering, computed once: for each line i,
        // every other line sorted by Euclidean distance between the simplex
        // reference points. The dynamic choice of the k nearest NON-EMPTY lines
        // (nbr_i) happens on the fly in neighbors_nonempty(), driven by cnt.
        line_rank_.assign(n_, {});
        for (int i = 0; i < n_; ++i) {
            std::vector<std::pair<double,int>> d;
            d.reserve(n_);
            for (int j = 0; j < n_; ++j) {
                if (j == i) continue;
                double s = 0.0;
                for (int t = 0; t < m; ++t) {
                    double diff = W_[i][t] - W_[j][t]; s += diff*diff;
                }
                d.emplace_back(std::sqrt(s), j);
            }
            std::sort(d.begin(), d.end());
            line_rank_[i].reserve(d.size());
            for (auto& pr : d) line_rank_[i].push_back(pr.second);
        }
    }

    // nbr_i: the k nearest NON-EMPTY lines to line i, derived DYNAMICALLY from
    // the geometric ranking line_rank_[i] and the current occupancy cnt. Walk
    // the ranking from the nearest and take the first k non-empty lines.
    std::vector<int> neighbors_nonempty(int i, const std::vector<int>& cnt) const {
        std::vector<int> nb;
        nb.reserve(k_);
        for (int j : line_rank_[i]) {
            if (cnt[j] > 0) {
                nb.push_back(j);
                if ((int)nb.size() >= k_) break;
            }
        }
        return nb;
    }

    std::vector<double> ideal_point() const {
        int m = arch_.empty() ? 0 : (int)arch_[0].objs.size();
        std::vector<double> z(m, std::numeric_limits<double>::max());
        for (const auto& s : arch_)
            for (int k = 0; k < m; ++k) z[k] = std::min(z[k], s.objs[k]);
        return z;
    }

    // polynomial mutation of the child
    void poly(std::vector<double>& v,
              const std::vector<std::pair<std::optional<double>,
                                          std::optional<double>>>& bounds,
              int nv) {
        ops::polynomial_mutation(v, bounds, eta_m_, pm_eff(nv), rng_);
    }

    // ── Mutate (Algorithm 2) over the local archive ────────────────────────
    // pool is a list of arch_ indices (the neighbourhood); returns child.vars.
    std::vector<double> mutate(const std::vector<double>& parent,
                               const std::vector<int>& pool,
                               double etac, double F, double CR,
                               const std::vector<std::pair<std::optional<double>,
                                                           std::optional<double>>>& bounds) {
        int nv = (int)parent.size();
        std::uniform_real_distribution<double> uni(0.0, 1.0);
        std::vector<double> child;

        if (uni(rng_) > mut_prob_ && pool.size() >= 3) {
            // DE /rand/1/bin (Eq.18) + binomial crossover
            std::uniform_int_distribution<int> pk(0, (int)pool.size()-1);
            int a=pk(rng_), b=pk(rng_), c=pk(rng_);
            for (int t=0; t<10 && (a==b||a==c||b==c); ++t){ b=pk(rng_); c=pk(rng_);}
            const auto& r1 = arch_[pool[a]].vars;
            const auto& r2 = arch_[pool[b]].vars;
            const auto& r3 = arch_[pool[c]].vars;
            child = parent;
            std::uniform_int_distribution<int> jr(0, nv-1);
            int jrand = jr(rng_);
            for (int j=0; j<nv; ++j) {
                if (uni(rng_) < CR || j == jrand)   // Eq.18 + binomial: j_rand ∈ [0,nv)
                    child[j] = r1[j] + F * (r2[j] - r3[j]);
                // FIX 2026-07-08:
                // When bounds are absent the DE branch SILENTLY substitutes
                // [0,1] (value_or(0.0)/value_or(1.0)). This is a deliberate
                // repair fallback for the DE mutant (Eq.18 can leave the
                // range), but it is laxer than the sbx_require_bound policy,
                // where SBX and PM demand explicit bounds. For DTLZ and WFG
                // the bounds are always given.
                double lo = bounds[j].first .value_or(0.0);
                double hi = bounds[j].second.value_or(1.0);
                child[j] = std::clamp(child[j], lo, hi);
            }
            if (flag2_) poly(child, bounds, nv);
        } else {
            // SBX (Eq.16-17), the second parent comes from the neighbourhood
            std::vector<double> p2;
            if (!pool.empty()) {
                std::uniform_int_distribution<int> pk(0, (int)pool.size()-1);
                p2 = arch_[pool[pk(rng_)]].vars;
            } else {
                p2 = parent;
            }
            std::vector<double> c1, c2;
            ops::sbx(parent, p2, c1, c2, bounds, etac, 1.0, rng_);
            child = c1;   // the first child is taken
            if (flag1_) poly(child, bounds, nv);
        }
        return child;
    }

    // load the archive from the vault's active population
    void load_arch(DataVault<Ind_t>& vault) {
        arch_.clear();
        int N = (int)vault.active_n();
        arch_.reserve(N);
        for (int i = 0; i < N; ++i) {
            Sol s;
            s.vars  = vault.variables_of(i);
            s.objs  = vault.objectives_of(i);
            s.bvars = vault.binary_variables_of(i);
            if (constraint_mode != ConstraintMode::NONE) s.cv = vault.get_cv(i);
            arch_.push_back(std::move(s));
        }
    }

    // write the archive back into the vault's active population
    void store_arch(DataVault<Ind_t>& vault) {
        vault.reduce(0);
        vault.expand((int)arch_.size());
        for (int i = 0; i < (int)arch_.size(); ++i) {
            vault.seed_individual((std::size_t)i, arch_[i].vars, arch_[i].objs,
                                  arch_[i].bvars, {});
        }
    }

public:
    NAEMOCore() = default;

    void set_K(int n)              { n_req_ = n; }
    void set_n_lines(int n)        { n_req_ = n; }
    void set_theta(double t)       { if (t > 0.0) theta_ = t; }
    void set_mut_prob(double p)    { mut_prob_ = p; }
    void set_eta_mutation(double e){ eta_m_ = e; }
    void set_eta_crossover(double){}   // η_c is adaptive — no-op shim for API uniformity
    void set_pm(double p)          { pm_ = p; }
    void set_flag1(bool f)         { flag1_ = f; }
    void set_flag2(bool f)         { flag2_ = f; }
    void set_seed(unsigned s)      { rng_.seed(s); }

    void setup(DataVault<Ind_t>& vault) {
        require_real_only(vault);
        int m = vault.objs_n();
        L_soft_ = vault.pop_size();
        // k depends on n, so build W first; estimate a draft n from n_req_ or
        // L_soft to size k
        int n_hint = (n_req_ > 0) ? n_req_ : (L_soft_ - 1);
        k_ = std::max(2, n_hint / 5);   // 20% of the reference lines (§4.5)
        build_directions(m);
        // n_ (the actual lattice size) is known only now; refine k_ from it.
        // build_directions does NOT read k_ — the neighbourhoods are the full
        // geometric ranking and k is applied on the fly — so it is not rerun.
        k_ = std::max(2, n_ / 5);

        // random initialization of L_soft individuals
        int N = vault.pop_size();
        const auto& bounds = vault.get_bounds();
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        std::vector<double> vars(vault.vars_n());
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < vault.vars_n(); ++j) {
                double lo = bounds[j].first .value_or(0.0);
                double hi = bounds[j].second.value_or(1.0);
                vars[j] = lo + dist(rng_)*(hi-lo);
            }
            vault.set_variables(i, vars);
        }
        vault.sync();
        load_arch(vault);
        mu_etac_ = 30.0; mu_F_ = 0.5; mu_CR_ = 0.2;
    }

    void setup_seeded(DataVault<Ind_t>& vault) {
        require_real_only(vault);
        int m = vault.objs_n();
        L_soft_ = vault.pop_size();
        int n_hint = (n_req_ > 0) ? n_req_ : (L_soft_ - 1);
        k_ = std::max(2, n_hint / 5);   // 20% of the reference lines (§4.5)
        build_directions(m);
        k_ = std::max(2, n_ / 5);       // see setup(): no second build needed
        load_arch(vault);
        mu_etac_ = 30.0; mu_F_ = 0.5; mu_CR_ = 0.2;
    }

    void step(DataVault<Ind_t>& vault) {
        const auto& bounds = vault.get_bounds();
        int L_hard = n_;

        // z* is fixed for the generation (NAEMO-6)
        std::vector<double> z = ideal_point();

        // associations of the current archive
        std::vector<int> line_of(arch_.size());
        std::vector<int> cnt(n_, 0);
        for (int i = 0; i < (int)arch_.size(); ++i) {
            line_of[i] = associate(arch_[i].objs, z);
            ++cnt[line_of[i]];
        }

        // scratch slot for evaluating offspring
        int scratch = vault.expand(1);

        std::normal_distribution<double> gec(mu_etac_, std::sqrt(5.0));
        std::normal_distribution<double> gF (mu_F_,    std::sqrt(0.1));
        std::normal_distribution<double> gCR(mu_CR_,   std::sqrt(0.1));
        std::uniform_real_distribution<double> uni(0.0, 1.0);

        std::vector<double> Setac, SF, SCR;

        for (int j = 0; j < n_; ++j) {
            int ind = j;
            if (cnt[ind] == 0) {
                // For an empty line the parent is taken from a neighbouring
                // NON-EMPTY line; nbr_j is the k nearest non-empty ones,
                // resolved dynamically from cnt.
                std::vector<int> ne = neighbors_nonempty(j, cnt);
                if (ne.empty()) continue;  // archive empty (should not happen)
                std::uniform_int_distribution<int> pe(0, (int)ne.size()-1);
                ind = ne[pe(rng_)];
            }
            // parent: a random member of sub_arch[ind]
            std::vector<int> members;
            for (int i = 0; i < (int)arch_.size(); ++i)
                if (line_of[i] == ind) members.push_back(i);
            if (members.empty()) continue;
            std::uniform_int_distribution<int> pm(0, (int)members.size()-1);
            int parent_idx = members[pm(rng_)];

            // mating neighbourhood pool: ind plus the k nearest NON-EMPTY
            // lines (nbr_ind, resolved dynamically from cnt).
            std::vector<int> pool = members;
            for (int nb : neighbors_nonempty(ind, cnt)) {
                for (int i = 0; i < (int)arch_.size(); ++i)
                    if (line_of[i] == nb) pool.push_back(i);
            }
            if (pool.size() < 3)   // fallback: the whole archive (NAEMO-3)
                for (int i = 0; i < (int)arch_.size(); ++i) pool.push_back(i);

            double etac = std::max(1e-6, gec(rng_));
            double F    = std::clamp(gF(rng_),  0.0, 1.0);
            double CR   = std::clamp(gCR(rng_), 0.0, 1.0);

            std::vector<double> cv = mutate(arch_[parent_idx].vars, pool,
                                            etac, F, CR, bounds);

            // evaluate the child
            vault.set_variables(scratch, cv);
            vault.refresh_objectives(scratch);
            std::vector<double> co = vault.objectives_of(scratch);
            double ccv = (constraint_mode==ConstraintMode::NONE) ? 0.0
                                                                : vault.get_cv(scratch);

            // accept if the parent does NOT dominate the child
            if (dominates_c(arch_[parent_idx].objs, arch_[parent_idx].cv, co, ccv)) continue;

            // insert the child
            Sol child; child.vars = cv; child.objs = co; child.cv = ccv;
            int cline = associate(co, z);
            arch_.push_back(std::move(child));
            line_of.push_back(cline);
            ++cnt[cline];
            int child_idx = (int)arch_.size() - 1;

            // convergence filter (Algorithm 3): remove solutions dominated by
            // the child without emptying a line. Iterate backwards, swap-erase.
            for (int i = (int)arch_.size() - 1; i >= 0; --i) {
                if (i == child_idx) continue;
                if (dominates_c(co, ccv, arch_[i].objs, arch_[i].cv) && cnt[line_of[i]] > 1) {
                    --cnt[line_of[i]];
                    int last = (int)arch_.size() - 1;
                    // swap i <- last
                    arch_[i]    = arch_[last];
                    line_of[i]  = line_of[last];
                    if (child_idx == last) child_idx = i;
                    arch_.pop_back(); line_of.pop_back();
                }
            }

            // diversity filter (Algorithm 4) when |arch| > L_soft, down to L_hard
            if ((int)arch_.size() > L_soft_) {
                while ((int)arch_.size() > L_hard) {
                    int lmax = 0;
                    for (int t = 1; t < n_; ++t) if (cnt[t] > cnt[lmax]) lmax = t;
                    if (cnt[lmax] <= 0) break;
                    // the member of line lmax with the LARGEST PBI
                    int worst = -1; double worst_g = -std::numeric_limits<double>::max();
                    for (int i = 0; i < (int)arch_.size(); ++i) {
                        if (line_of[i] != lmax) continue;
                        double g = pbi(arch_[i].objs, z, lmax);
                        if (g > worst_g) { worst_g = g; worst = i; }
                    }
                    if (worst < 0) break;
                    --cnt[lmax];
                    int last = (int)arch_.size() - 1;
                    arch_[worst]   = arch_[last];
                    line_of[worst] = line_of[last];
                    arch_.pop_back(); line_of.pop_back();
                }
            }

            Setac.push_back(etac); SF.push_back(F); SCR.push_back(CR);
        }

        // adapt the means
        auto mean = [](const std::vector<double>& v, double def){
            if (v.empty()) return def;
            double s = 0; for (double x : v) s += x; return s / v.size();
        };
        mu_etac_ = mean(Setac, mu_etac_);
        mu_F_    = mean(SF,    mu_F_);
        mu_CR_   = mean(SCR,   mu_CR_);

        store_arch(vault);
    }
};

} // namespace mootation

#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// SPEA2+SDE — Shift-Based Density Estimation for Pareto-Based Algorithms
//             in Many-Objective Optimization
// M. Li, S. Yang, X. Liu — IEEE TEVC 18(3), 2014
// doi:10.1109/TEVC.2013.2262178          (source: li2014)
// SPEA2 framework: Zitzler, Laumanns, Thiele — TIK-Report 103, 2001
// (eth-24689-01); SDE replaces ONLY the density estimation (§II-C).
//
// Generational scheme (= SPEA2 Algorithm 1 + SDE):
//   1. Fitness over the pool P_t ∪ P̄_t: S, R as in SPEA2;
//      D(x) = 1/(σ^SDE_k(x) + 2), k = ⌊√(N+N̄)⌋ (based on the archive
//      capacity, including at t=0); SDE shift (Eq.3): y'_i = max(y_i, x_i),
//      dist = ‖x − y'‖
//   2. Environmental selection: non-dominated → archive; truncation by
//      lexicographic comparison of the SDE distances (§II-C: SDE is applied
//      "in both the fitness assignment and archive truncation procedures")
//   3. Mating: binary tournament by F on the archive; SBX + poly-mutation
//
// Defaults = §III (the experimental-setup paragraph, before §III-A; §IV
//   restates no values): pc=1.0, pm=1/n, eta_c=20, eta_m=20, N̄ = N.
// Deviations: none (internal audit; the SDE math verified against Fig.2 of
//   the paper).
// NOTE (a reading, not a deviation): σ^k, the distance to the k-th nearest
//   neighbour, is computed SELF-EXCLUSIVELY. The SPEA2 report is internally
//   inconsistent — the same paragraph gives a procedural list over "all
//   individuals j in archive and population" (self-inclusive) and calls the
//   result "the distance to the k-th nearest neighbor" (self-exclusive) — but
//   this file's primary source is li2014, whose §II-C uses the self-excluded
//   form, and the two coincide arithmetically anyway: d(x,x)=0 is the unique
//   minimum, so excluding it shifts every order statistic by exactly one.
//   Known limitation (§V "Discussions"): the algorithm is
//   sensitive to objective scaling — the paper notes that for badly scaled
//   problems this "can be addressed by normalizing each
//   dimension … before estimating individuals' density"; normalization is
//   a recommended remedy, NOT part of the algorithm definition, and is not
//   performed by default (arbitration verdict #12a).
// Extensions beyond the paper: constraint_mode FEASIBILITY (CDP; off by
//   default); mixed real+binary genome.
//
//
// RESULT SET (read this before consuming the output). Per SPEA2 Algorithm 1
//   Step 4 the answer is the nondominated members of the ARCHIVE P̄_{t+1}, in
//   vault.archive_* (archive_size(), archive_objectives_of(), ...). The ACTIVE
//   population after step() is the freshly bred, un-selected offspring
//   generation P_{t+1} — it is NOT the algorithm's output. This differs from
//   most cores in the library, where the active population IS the answer.
//
// Layout: active [0,N) = P_t; vault.archive_* = P̄_t.
// ============================================================================

#include <algorithm>
#include <cmath>
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

template <typename Ind_t>
class SPEA2SDECore {
public:
    ConstraintMode constraint_mode = ConstraintMode::NONE;

private:
    double       eta_c_       = 20.0;  // §IV: distribution index 20
    double       eta_m_       = 20.0;  // §IV: distribution index 20
    double       pc_          = 1.0;   // §IV: "crossover probability pc = 1.0"
    int          archive_size_ = -1;   // Ā; -1 → use pop_size()
    std::mt19937 rng_{std::random_device{}()};
    int          eff_arch_    = 0;

    // ── CDP dominance (active × active) ───────────────────────────────────
    bool dominates(DataVault<Ind_t>& vault, int a, int b) const {
        if (constraint_mode == ConstraintMode::FEASIBILITY) {
            double ca = vault.get_cv(a), cb = vault.get_cv(b);
            bool af = (ca<=0.0), bf = (cb<=0.0);
            if( af&&!bf) return true;
            if(!af&& bf) return false;
            if(!af&&!bf) return ca<cb;
        }
        const auto& fa = vault.objectives_of(a);
        const auto& fb = vault.objectives_of(b);
        bool better = false;
        for (std::size_t i=0;i<fa.size();++i) {
            if(fa[i]>fb[i]) return false;
            if(fa[i]<fb[i]) better=true;
        }
        return better;
    }

    // ── SDE distance from x (active slot) to y objective vector ──────────
    double sde_dist(const std::vector<double>& fx,
                    const std::vector<double>& fy) const {
        double sum = 0.0;
        for (std::size_t i=0;i<fx.size();++i) {
            double yp = (fy[i]<fx[i]) ? fx[i] : fy[i];  // max(y_i, x_i)
            double d  = fx[i] - yp;
            sum += d*d;
        }
        return std::sqrt(sum);
    }

    // ── Fitness assignment over pool = active[0,n) ∪ archive ──────────────
    // SPEA2+SDE inherits the SPEA2 framework (Li, Yang, Liu 2014 — SDE only
    // replaces the density estimation, everything else comes from SPEA2).
    // Per SPEA2 (Zitzler, Laumanns, Thiele 2001, §3.1) S/R/D are computed
    // over pool = archive ∪ population. Previously only the n active members
    // were considered — a violation of the inherited SPEA2 framework; fixed.
    //
    // Pool indexing: [0,n) → active slots; [n,pool) → archive slot (i-n).
    void assign_fitness(DataVault<Ind_t>& vault, int n) {
        int arch_n = static_cast<int>(vault.archive_size());
        int pool   = n + arch_n;

        auto obj_of = [&](int i) -> const std::vector<double>& {
            return (i < n) ? vault.objectives_of(i)
                           : vault.archive_objectives_of(
                                 static_cast<std::size_t>(i - n));
        };
        auto cv_of = [&](int i) -> double {
            return (i < n) ? vault.get_cv(i)
                           : vault.archive_cv(static_cast<std::size_t>(i - n));
        };
        auto ind_of = [&](int i) -> Ind_t& {
            return (i < n) ? vault.get_ind(i)
                           : vault.archive_get(static_cast<std::size_t>(i - n));
        };
        auto dom = [&](int a, int b) -> bool {
            if (constraint_mode == ConstraintMode::FEASIBILITY) {
                double ca = cv_of(a), cb = cv_of(b);
                bool af=(ca<=0.0), bf=(cb<=0.0);
                if( af&&!bf) return true;
                if(!af&& bf) return false;
                if(!af&&!bf) return ca<cb;
            }
            const auto& fa = obj_of(a);
            const auto& fb = obj_of(b);
            bool better=false;
            for(std::size_t q=0;q<fa.size();++q){
                if(fa[q]>fb[q]) return false;
                if(fa[q]<fb[q]) better=true;
            }
            return better;
        };

        // Strength.
        std::vector<int> S(pool,0);
        for(int i=0;i<pool;++i)
            for(int j=0;j<pool;++j)
                if(i!=j && dom(i,j)) ++S[i];

        // Raw fitness.
        std::vector<double> R(pool,0.0);
        for(int i=0;i<pool;++i)
            for(int j=0;j<pool;++j)
                if(i!=j && dom(j,i)) R[i]+=S[j];

        // SDE density. SPEA2 §3.1: k = ⌊√(N+N̄)⌋ — based on the archive
        // capacity N̄ (including the empty archive at t=0), not on the actual
        // pool size.
        int k = static_cast<int>(std::floor(
                    std::sqrt(static_cast<double>(n + eff_arch_))));
        if(k<1) k=1;
        if(k>pool-1) k=pool-1;   // guard: k at most the number of neighbors
        for(int i=0;i<pool;++i) {
            std::vector<double> dists;
            dists.reserve(pool-1);
            for(int j=0;j<pool;++j) {
                if(i==j) continue;
                dists.push_back(sde_dist(obj_of(i), obj_of(j)));
            }
            double sigma_k = 0.0;   // guard: pool==1 → σ_k = 0
            if(!dists.empty()) {
                std::nth_element(dists.begin(), dists.begin()+k-1, dists.end());
                sigma_k = dists[k-1];
            }
            double density = 1.0/(sigma_k+2.0);
            ind_of(i).strength    = S[i];
            ind_of(i).raw_fitness = R[i];
            ind_of(i).density     = density;
            ind_of(i).fitness     = R[i] + density;
        }
    }

    // ── Archive update from pool = active[0,n) ∪ archive ──────────────────
    // Rebuilds the archive from the eff_arch_ best members of the pool.
    // Candidates are drawn both from active and from the old archive (per
    // the SPEA2 framework). Because archive_clear() releases the archive
    // slots, the data of the chosen archive members is snapshotted
    // beforehand.
    void update_archive(DataVault<Ind_t>& vault, int n) {
        int arch_n = static_cast<int>(vault.archive_size());
        int pool   = n + arch_n;

        auto obj_of = [&](int i) -> const std::vector<double>& {
            return (i < n) ? vault.objectives_of(i)
                           : vault.archive_objectives_of(
                                 static_cast<std::size_t>(i - n));
        };
        auto ind_of = [&](int i) -> Ind_t& {
            return (i < n) ? vault.get_ind(i)
                           : vault.archive_get(static_cast<std::size_t>(i - n));
        };

        // Non-dominated (R==0) and dominated.
        std::vector<int> nd, dom_idx;
        for(int i=0;i<pool;++i) {
            if(ind_of(i).raw_fitness < 0.5) nd.push_back(i);
            else                            dom_idx.push_back(i);
        }

        // Choose pool indices for the archive.
        std::vector<int> chosen;
        chosen.reserve(eff_arch_);

        if(static_cast<int>(nd.size()) <= eff_arch_) {
            chosen = nd;
            if(static_cast<int>(nd.size()) < eff_arch_) {
                std::sort(dom_idx.begin(),dom_idx.end(),[&](int a,int b){
                    return ind_of(a).fitness < ind_of(b).fitness;
                });
                int needed = eff_arch_ - static_cast<int>(nd.size());
                for(int i=0;i<needed && i<static_cast<int>(dom_idx.size());++i)
                    chosen.push_back(dom_idx[i]);
            }
        } else {
            // Truncation: iteratively remove the most crowded one by SDE.
            std::vector<int> cand = nd;
            while(static_cast<int>(cand.size()) > eff_arch_) {
                int sz = static_cast<int>(cand.size());
                std::vector<std::vector<double>> sdists(sz);
                for(int i=0;i<sz;++i) {
                    for(int j=0;j<sz;++j) {
                        if(i==j) continue;
                        sdists[i].push_back(
                            sde_dist(obj_of(cand[i]), obj_of(cand[j])));
                    }
                    std::sort(sdists[i].begin(), sdists[i].end());
                }
                int worst=0;
                for(int i=1;i<sz;++i) {
                    const auto& da=sdists[worst];
                    const auto& db=sdists[i];
                    int len=static_cast<int>(std::min(da.size(),db.size()));
                    for(int kk=0;kk<len;++kk) {
                        if(db[kk]<da[kk]){ worst=i; break; }
                        if(db[kk]>da[kk])   break;
                    }
                }
                cand[worst]=cand.back(); cand.pop_back();
            }
            chosen = cand;
        }

        // Snapshot the data of the chosen OLD archive members (idx >= n) —
        // archive_clear() below invalidates them.
        // IMPORTANT: also snapshot the fitness fields — archive_push_data
        // does not carry them over, while tournament_archive (SPEA2 Step 5)
        // compares exactly F(i) = R + D computed for the members of P̄_{t+1}
        // The fitness travels WITH the archive entry: otherwise a carried-over
        // member would be judged by the fitness of whatever previously occupied
        // its slot.
        struct ArchiveEntry {
            std::vector<double> vars, objs, lims;
            std::vector<int>    bvars;
            int    strength    = 0;
            double raw_fitness = 0.0;
            double density     = 0.0;
            double fitness     = 0.0;
        };
        std::vector<ArchiveEntry> old_arch;
        for(int idx : chosen) {
            if(idx >= n) {
                std::size_t ai = static_cast<std::size_t>(idx - n);
                ArchiveEntry e;
                e.vars = vault.archive_variables_of(ai);
                e.objs = vault.archive_objectives_of(ai);
                if(vault.bin_vars_n() > 0)
                    e.bvars = vault.archive_bin_variables_of(ai);
                if(vault.lims_n() > 0)
                    e.lims  = vault.archive_limits_of(ai);
                const auto& ind = vault.archive_get(ai);
                e.strength    = ind.strength;
                e.raw_fitness = ind.raw_fitness;
                e.density     = ind.density;
                e.fitness     = ind.fitness;
                old_arch.push_back(std::move(e));
            } else {
                old_arch.emplace_back();   // placeholder
            }
        }

        vault.archive_clear();
        int ai = 0;
        for(int idx : chosen) {
            if(idx < n) {
                vault.archive_push(idx);   // full slot copy, fitness fields OK
            } else {
                const auto& e = old_arch[ai];
                vault.archive_push_data(e.vars, e.objs, e.bvars, e.lims);
                // Restore the fitness fields (archive_push_data does not write them).
                auto& ind = vault.archive_get(vault.archive_size() - 1);
                ind.strength    = e.strength;
                ind.raw_fitness = e.raw_fitness;
                ind.density     = e.density;
                ind.fitness     = e.fitness;
            }
            ++ai;
        }
    }

    // ── Binary tournament on archive (lower fitness = better) ────────────
    // CDP branch (FEASIBILITY) — unified with spea2.hpp (extension beyond the paper).
    int tournament_archive(DataVault<Ind_t>& vault) {
        int sz = static_cast<int>(vault.archive_size());
        std::uniform_int_distribution<int> dist(0, sz-1);
        int ai = dist(rng_), bi = dist(rng_);
        if (constraint_mode == ConstraintMode::FEASIBILITY) {
            double ca = vault.archive_cv(ai), cb = vault.archive_cv(bi);
            bool af=(ca<=0.0), bf=(cb<=0.0);
            if(af&&!bf) return ai;
            if(!af&&bf) return bi;
            if(!af&&!bf) return (ca<cb)?ai:bi;
        }
        return (vault.archive_get(ai).fitness <= vault.archive_get(bi).fitness)
               ? ai : bi;
    }

    // ── Breed N offspring into active slots [0, N) from archive ───────────
    // FIX 2026-06: previously offspring were written into [N,2N) and then
    // reduce(N) discarded them → the active population stayed frozen at P0.
    // Now as in spea2.hpp: read from the archive (it is separate, no
    // aliasing), write into [0,N).
    void breed(DataVault<Ind_t>& vault, int N) {
        const auto& bounds = vault.get_bounds();
        std::vector<double> c1, c2;
        for (int i = 0; i < N; i += 2) {
            int ai = tournament_archive(vault);
            int bi = tournament_archive(vault);
            const auto& av = vault.archive_variables_of(static_cast<std::size_t>(ai));
            const auto& bv = vault.archive_variables_of(static_cast<std::size_t>(bi));
            // §IV: pc=1.0, pm=1/n
            double pm = (vault.vars_n() > 0) ? 1.0 / vault.vars_n() : 0.0;
            ops::sbx(av, bv, c1, c2, bounds, eta_c_, pc_, rng_);
            ops::polynomial_mutation(c1, bounds, eta_m_, pm, rng_);
            ops::polynomial_mutation(c2, bounds, eta_m_, pm, rng_);
            if (vault.bin_vars_n() > 0) {
                const auto& abv = vault.archive_bin_variables_of(static_cast<std::size_t>(ai));
                const auto& bbv = vault.archive_bin_variables_of(static_cast<std::size_t>(bi));
                std::vector<int> bc1, bc2;
                ops::binary_crossover(abv, bbv, bc1, bc2, rng_);
                ops::bit_flip_mutation(bc1, vault.bin_vars_n(), rng_);
                if (i+1<N) ops::bit_flip_mutation(bc2, vault.bin_vars_n(), rng_);
                vault.set_all_variables(i, c1, bc1);
                if (i+1<N) vault.set_all_variables(i+1, c2, bc2);
            } else {
                vault.set_variables(i, c1);
                if (i+1<N) vault.set_variables(i+1, c2);
            }
        }
    }

public:
    SPEA2SDECore() = default;

    void set_eta_crossover(double e) { eta_c_       = e; }
    void set_eta_mutation (double e) { eta_m_       = e; }
    void set_pc           (double p) { pc_          = p; }
    void set_archive_size (int sz)   { archive_size_= sz; }
    void set_seed(unsigned s)        { rng_.seed(s); }

    // ── setup ──────────────────────────────────────────────────────────────
    void setup(DataVault<Ind_t>& vault) {
        int N = vault.pop_size();
        eff_arch_ = (archive_size_>0) ? archive_size_ : N;

        const auto& bounds = vault.get_bounds();
        std::uniform_real_distribution<double> dr(0.0,1.0);
        std::uniform_int_distribution<int>     db(0,1);
        std::vector<double> vars(vault.vars_n());
        std::vector<int>    bvars(vault.bin_vars_n());
        for(int i=0;i<N;++i) {
            for(int j=0;j<vault.vars_n();++j) {
                double lo=bounds[j].first.value_or(0.0), hi=bounds[j].second.value_or(1.0);
                vars[j]=lo+dr(rng_)*(hi-lo);
            }
            for(int j=0;j<vault.bin_vars_n();++j) bvars[j]=db(rng_);
            if(vault.bin_vars_n()>0) vault.set_all_variables(i,vars,bvars);
            else                     vault.set_variables(i,vars);
        }
        vault.sync();
        assign_fitness(vault, N);
        update_archive(vault, N);
        breed(vault, N);          // FIX: first offspring generation into [0,N)
        vault.sync();
    }

    void setup_seeded(DataVault<Ind_t>& vault) {
        eff_arch_ = (archive_size_>0) ? archive_size_ : vault.pop_size();
        assign_fitness(vault, vault.pop_size());
        update_archive(vault, vault.pop_size());
        breed(vault, vault.pop_size());   // FIX: first offspring generation into [0,N)
        vault.sync();
    }

    // ── step ──────────────────────────────────────────────────────────────
    // FIX 2026-06: aligned with the structure of spea2.hpp.
    // On entry: active [0,N) = offspring from the previous step;
    // vault.archive_* = P̄t.
    // Fitness pool = active[0,N) ∪ archive (the SDE shift is inside
    // assign_fitness).
    // Previously: expand→breed into [N,2N)→assign(2N)→update(2N)→reduce(N)
    // discarded the offspring, leaving a frozen P0 in [0,N). Now without
    // expand/reduce.
    void step(DataVault<Ind_t>& vault) {
        int N = vault.pop_size();
        eff_arch_ = (archive_size_>0) ? archive_size_ : N;

        assign_fitness(vault, N);     // pool = active[0,N) ∪ archive
        update_archive(vault, N);     // P̄t+1 = best eff_arch_ from the pool
        breed(vault, N);              // offspring Pt+1 from archive into active [0,N)
        vault.sync();
    }
};

} // namespace mootation

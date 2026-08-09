#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// SPEA2 — Improving the Strength Pareto Evolutionary Algorithm
// E. Zitzler, M. Laumanns, L. Thiele — TIK-Report 103, ETH Zürich, 2001
// doi:10.3929/ethz-a-004284029          (source: eth-24689-01)
// NOTE: this DOI is registered with DataCite (ETH Zurich Research Collection),
//   not Crossref — api.crossref.org returns 404 for it. Verified via
//   api.datacite.org: "SPEA2: Improving the strength pareto evolutionary
//   algorithm", Zitzler/Laumanns/Thiele, ETH Zurich, 2001.
//
// Generational scheme (Algorithm 1):
//   1. Fitness over the pool P_t ∪ P̄_t: S(x) = |{y: x ≻ y}|; R(x) = Σ_{y≻x} S(y);
//      D(x) = 1/(σ^k_x + 2), k = ⌊√(N+N̄)⌋ (§3.1, based on the archive
//      capacity N̄ — including at t=0 with an empty archive); F = R + D
//      (lower = better)
//   2. Environmental selection (§3.2): non-dominated (F<1) → archive P̄_{t+1};
//      overflow → iterative truncation by lexicographic comparison of the
//      sorted k-NN distances; underfill → best dominated individuals by F
//   3. Mating: binary tournament with replacement by F on P̄_{t+1} (Step 5)
//   4. Variation: SBX-20 + polynomial mutation (pm=1/n) → P_{t+1}
//
// Defaults: N = N̄ = pop_size and eta_c = 20 come from §4.1 ("SBX-20
//   operator"; "population size and the archive size were set to 100"). The
//   paper specifies NEITHER p_c NOR eta_m NOR p_m for the continuous case; the
//   library uses Deb's reference-implementation values (p_c = 1.0 with
//   per-variable 0.5, eta_m = 20, p_m = 1/n). Attributing those three to §4
//   would be a false citation.
// Deviations (MINOR, deliberate; internal audit):
//   - the `<=` tournament tie-break deterministically awards the win to the
//     first contestant;
//   - k is FLOORED. §3.1 writes k = √(N+N̄) with no rounding rule, and it is
//     an index into a sorted distance list, so it must be made integral; this
//     port takes ⌊·⌋. At the paper's own N = N̄ = 100 the value is exact (14.14
//     -> 14 either way only under floor; ⌈·⌉ would give 15).
//   - σ^k is computed SELF-EXCLUSIVELY, and the report contradicts itself on
//     this point. §3.1 gives a procedural recipe — compute the distances "to
//     all individuals j in archive and population", sort ascending, "the k-th
//     element gives the distance sought" — whose list contains d(i,i)=0, so its
//     k-th element is the (k−1)-th nearest OTHER individual. The same sentence
//     nonetheless calls σ^k "the distance to the k-th nearest neighbor", and
//     §3.2's truncation defines it outright as "the distance of i to its k-th
//     nearest neighbor in P̄_{t+1}". Two of the three statements are
//     self-exclusive, so that is the reading taken. Since d(i,i)=0 is the
//     unique minimum of the list, the two readings differ by exactly one
//     position in the order statistic — never more.
// Extensions beyond the paper: constraint_mode FEASIBILITY (CDP; off by
//   default — the paper handled constraints differently, knapsack);
//   mixed real+binary genome (uniform crossover + bit-flip).
//
// RESULT SET (read this before consuming the output). Per Algorithm 1 Step 4
//   the answer is the nondominated members of the ARCHIVE P̄_{t+1}, which live
//   in vault.archive_* (archive_size(), archive_objectives_of(), ...). The
//   ACTIVE population after step() is the freshly bred, un-selected offspring
//   generation P_{t+1} — it is NOT the algorithm's output. A caller that reads
//   the active slots gets raw offspring, not the selected front. This differs
//   from most cores in the library, where the active population IS the answer.
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
class SPEA2Core {
public:
    ConstraintMode constraint_mode = ConstraintMode::NONE;

private:
    int          archive_size_ = -1;
    double       eta_c_        = 20.0;
    double       eta_m_        = 20.0;
    std::mt19937 rng_{std::random_device{}()};
    int          eff_arch_     = 0;

    // ── Fitness assignment over active pool [0, n) ────────────────────────
    // Note: archive members are NOT in active pool here — they were used as
    // parents to produce the current population. Pool = population only OR
    // we augment with archive objectives below (see assign_fitness_with_archive).
    //
    // SPEA2 paper: pool = P̄t ∪ Pt.
    // We implement this by building a combined view:
    //   active [0, n) = Pt
    //   vault.archive_* = P̄t
    // and computing S/R/D over all pool_size = n + arch_n individuals.
    // Because archive individuals are not in active slots, we access them
    // via vault.archive_objectives_of / archive_cv.
    // Results stored in:
    //   ind.strength, ind.raw_fitness, ind.density, ind.fitness for active [0,n)
    //   archive_get(i).strength etc. for archive members.

    void assign_fitness(DataVault<Ind_t>& vault, int n) {
        int arch_n = static_cast<int>(vault.archive_size());
        int pool   = n + arch_n;
        int m      = vault.objs_n();

        // Helper to get objectives of pool member i (0..n-1 = active, n..pool-1 = archive).
        auto obj_of = [&](int i) -> const std::vector<double>& {
            return (i < n) ? vault.objectives_of(i)
                           : vault.archive_objectives_of(static_cast<std::size_t>(i - n));
        };
        auto cv_of = [&](int i) -> double {
            return (i < n) ? vault.get_cv(i)
                           : vault.archive_cv(static_cast<std::size_t>(i - n));
        };
        auto ind_of = [&](int i) -> Ind_t& {
            return (i < n) ? vault.get_ind(i)
                           : vault.archive_get(static_cast<std::size_t>(i - n));
        };

        // Dominance helper over pool.
        auto dom = [&](int a, int b) -> bool {
            if (constraint_mode == ConstraintMode::FEASIBILITY) {
                double ca = cv_of(a), cb = cv_of(b);
                bool af=(ca<=0.0), bf=(cb<=0.0);
                if(af&&!bf) return true;
                if(!af&&bf) return false;
                if(!af&&!bf) return ca<cb;
            }
            const auto& fa = obj_of(a);
            const auto& fb = obj_of(b);
            bool better = false;
            for (int j=0;j<m;++j){
                if(fa[j]>fb[j]) return false;
                if(fa[j]<fb[j]) better=true;
            }
            return better;
        };

        // Step 1: Strength.
        for (int i=0;i<pool;++i) ind_of(i).strength = 0;
        for (int i=0;i<pool;++i)
            for (int j=0;j<pool;++j)
                if (i!=j && dom(i,j)) ++ind_of(i).strength;

        // Step 2: Raw fitness.
        for (int i=0;i<pool;++i) ind_of(i).raw_fitness = 0;
        for (int i=0;i<pool;++i)
            for (int j=0;j<pool;++j)
                if (i!=j && dom(j,i)) ind_of(i).raw_fitness += ind_of(j).strength;

        // Step 3: k-NN density.
        // §3.1: k = ⌊√(N+N̄)⌋ — based on the archive capacity N̄ (including
        // the empty archive at t=0), not on the actual pool size.
        int k = static_cast<int>(std::floor(
                    std::sqrt(static_cast<double>(n + eff_arch_))));
        if (k < 1) k = 1;
        if (k > pool - 1) k = pool - 1;   // guard: k at most the number of neighbors

        for (int i=0;i<pool;++i) {
            const auto& oi = obj_of(i);
            std::vector<double> dists;
            dists.reserve(pool-1);
            for (int j=0;j<pool;++j) {
                if (i==j) continue;
                const auto& oj = obj_of(j);
                double d=0.0;
                for (int q=0;q<m;++q){double diff=oi[q]-oj[q];d+=diff*diff;}
                dists.push_back(std::sqrt(d));
            }
            double sigma_k = 0.0;   // guard: pool==1 → σ_k = 0 (D = 0.5)
            if (!dists.empty()) {
                std::nth_element(dists.begin(), dists.begin()+k-1, dists.end());
                sigma_k = dists[k-1];
            }
            ind_of(i).density = 1.0 / (sigma_k + 2.0);
            ind_of(i).fitness = static_cast<double>(ind_of(i).raw_fitness)
                              + ind_of(i).density;
        }
    }

    // ── Archive update from active pool [0, n) + current archive ─────────
    // Rebuilds vault.archive_* with the Ā best individuals.
    void update_archive(DataVault<Ind_t>& vault, int n) {
        int arch_n = static_cast<int>(vault.archive_size());
        int pool   = n + arch_n;
        int m      = vault.objs_n();

        auto obj_of = [&](int i) -> const std::vector<double>& {
            return (i < n) ? vault.objectives_of(i)
                           : vault.archive_objectives_of(static_cast<std::size_t>(i - n));
        };
        auto ind_of = [&](int i) -> Ind_t& {
            return (i < n) ? vault.get_ind(i)
                           : vault.archive_get(static_cast<std::size_t>(i - n));
        };

        // Collect nondominated (raw_fitness == 0).
        std::vector<int> nd, dom_idx;
        for (int i=0;i<pool;++i) {
            if (ind_of(i).raw_fitness == 0) nd.push_back(i);
            else                             dom_idx.push_back(i);
        }

        std::vector<int> chosen;
        chosen.reserve(eff_arch_);

        if (static_cast<int>(nd.size()) <= eff_arch_) {
            chosen = nd;
            if (static_cast<int>(nd.size()) < eff_arch_) {
                // Fill with best dominated.
                std::sort(dom_idx.begin(), dom_idx.end(), [&](int a, int b){
                    return ind_of(a).fitness < ind_of(b).fitness;
                });
                int needed = eff_arch_ - static_cast<int>(nd.size());
                for (int i=0;i<needed && i<static_cast<int>(dom_idx.size());++i)
                    chosen.push_back(dom_idx[i]);
            }
        } else {
            // Truncate: iteratively remove most crowded.
            std::vector<int> cand = nd;
            while (static_cast<int>(cand.size()) > eff_arch_) {
                int sz = static_cast<int>(cand.size());
                // Build pairwise distances.
                std::vector<std::vector<double>> d(sz, std::vector<double>(sz, 0.0));
                for (int ci=0;ci<sz;++ci) {
                    const auto& oi = obj_of(cand[ci]);
                    for (int cj=ci+1;cj<sz;++cj) {
                        const auto& oj = obj_of(cand[cj]);
                        double dist=0.0;
                        for(int q=0;q<m;++q){double diff=oi[q]-oj[q];dist+=diff*diff;}
                        d[ci][cj]=d[cj][ci]=std::sqrt(dist);
                    }
                }
                // Sorted distances per candidate (lex compare to find worst diversity).
                std::vector<std::vector<double>> sd(sz);
                for(int ci=0;ci<sz;++ci){
                    sd[ci]=d[ci]; sd[ci][ci]=std::numeric_limits<double>::max();
                    std::sort(sd[ci].begin(),sd[ci].end());
                    sd[ci].pop_back();
                }
                int rm=0;
                for(int ci=1;ci<sz;++ci){
                    const auto& a=sd[ci]; const auto& b=sd[rm];
                    int len=static_cast<int>(std::min(a.size(),b.size()));
                    for(int kk=0;kk<len;++kk){
                        if(a[kk]<b[kk]){rm=ci;break;}
                        if(a[kk]>b[kk]) break;
                    }
                }
                cand[rm]=cand.back(); cand.pop_back();
            }
            chosen=cand;
        }

        // Rebuild archive from chosen pool indices.
        // First snapshot archive data for chosen members that come from
        // the old archive (idx >= n), because archive_clear() below will
        // invalidate those entries before we can read them.
        // IMPORTANT: also snapshot the fitness fields (strength/raw_fitness/
        // density/fitness) — archive_push_data does not carry them over, while
        // the Step 5 tournament (tournament_archive) compares exactly
        // F(i) = R + D computed in Step 2 for all members of P̄_{t+1}
        // The fitness travels WITH the archive entry: otherwise a carried-over
        // member would be judged by the fitness of whatever previously occupied
        // its slot.
        struct ArchiveEntry {
            std::vector<double> vars, objs, lims;
            std::vector<int>    bvars;
            int    strength    = 0;
            int    raw_fitness = 0;
            double density     = 0.0;
            double fitness     = 0.0;
        };
        std::vector<ArchiveEntry> old_arch_data;
        for (int idx : chosen) {
            if (idx >= n) {
                std::size_t ai = static_cast<std::size_t>(idx - n);
                ArchiveEntry e;
                e.vars  = vault.archive_variables_of(ai);
                e.objs  = vault.archive_objectives_of(ai);
                if (vault.bin_vars_n() > 0)
                    e.bvars = vault.archive_bin_variables_of(ai);
                if (vault.lims_n() > 0)
                    e.lims  = vault.archive_limits_of(ai);
                const auto& ind = vault.archive_get(ai);
                e.strength    = ind.strength;
                e.raw_fitness = ind.raw_fitness;
                e.density     = ind.density;
                e.fitness     = ind.fitness;
                old_arch_data.push_back(std::move(e));
            } else {
                old_arch_data.emplace_back(); // placeholder, not used
            }
        }

        vault.archive_clear();
        int arch_entry = 0;
        for (int idx : chosen) {
            if (idx < n) {
                vault.archive_push(idx);   // full slot copy, fitness fields OK
            } else {
                const auto& e = old_arch_data[arch_entry];
                vault.archive_push_data(e.vars, e.objs, e.bvars, e.lims);
                // Restore the fitness fields (archive_push_data does not write them).
                auto& ind = vault.archive_get(vault.archive_size() - 1);
                ind.strength    = e.strength;
                ind.raw_fitness = e.raw_fitness;
                ind.density     = e.density;
                ind.fitness     = e.fitness;
            }
            ++arch_entry;
        }
    }

    // ── Binary tournament on archive (lower fitness = better) ─────────────
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

    // ── Breed N offspring into active slots [0, N) ────────────────────────
    // Uses archive as mating pool via tournament_archive.
    void breed(DataVault<Ind_t>& vault, int N) {
        const auto& bounds = vault.get_bounds();
        std::vector<double> c1, c2;
        for (int i = 0; i < N; i += 2) {
            int ai = tournament_archive(vault);
            int bi = tournament_archive(vault);
            const auto& av = vault.archive_variables_of(static_cast<std::size_t>(ai));
            const auto& bv = vault.archive_variables_of(static_cast<std::size_t>(bi));
            ops::sbx(av, bv, c1, c2, bounds, eta_c_, rng_);
            ops::polynomial_mutation(c1, bounds, eta_m_, rng_);
            ops::polynomial_mutation(c2, bounds, eta_m_, rng_);
            if (vault.bin_vars_n() > 0) {
                const auto& abv = vault.archive_bin_variables_of(
                    static_cast<std::size_t>(ai));
                const auto& bbv = vault.archive_bin_variables_of(
                    static_cast<std::size_t>(bi));
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
    SPEA2Core() = default;

    void set_archive_size (int n)    { archive_size_ = n; }
    void set_eta_crossover(double e) { eta_c_        = e; }
    void set_eta_mutation (double e) { eta_m_        = e; }
    void set_seed         (unsigned s){ rng_.seed(s); }

    // ── setup ──────────────────────────────────────────────────────────────
    // Population Pt in active [0, N). Archive P̄0 = ∅ initially.
    // After setup: archive holds P̄1, active [0,N) holds new offspring Pt+1.
    void setup(DataVault<Ind_t>& vault) {
        int N = vault.pop_size();
        eff_arch_ = (archive_size_ > 0) ? archive_size_ : N;

        const auto& bounds = vault.get_bounds();
        std::uniform_real_distribution<double> dr(0.0,1.0);
        std::uniform_int_distribution<int>     db(0,1);
        std::vector<double> vars(vault.vars_n());
        std::vector<int>    bvars(vault.bin_vars_n());
        for (int i=0;i<N;++i) {
            for(int j=0;j<vault.vars_n();++j){
                double lo=bounds[j].first.value_or(0.0), hi=bounds[j].second.value_or(1.0);
                vars[j]=lo+dr(rng_)*(hi-lo);
            }
            for(int j=0;j<vault.bin_vars_n();++j) bvars[j]=db(rng_);
            if(vault.bin_vars_n()>0) vault.set_all_variables(i,vars,bvars);
            else vault.set_variables(i,vars);
        }
        vault.sync();

        // Archive is empty → assign_fitness uses only active pool [0,N).
        assign_fitness(vault, N);
        update_archive(vault, N);

        // Breed first offspring generation into [0,N) from archive.
        breed(vault, N);
        vault.sync();
    }

    void setup_seeded(DataVault<Ind_t>& vault) {
        int N = vault.pop_size();
        eff_arch_ = (archive_size_ > 0) ? archive_size_ : N;
        assign_fitness(vault, N);
        update_archive(vault, N);
        breed(vault, N);
        vault.sync();
    }

    // ── step ──────────────────────────────────────────────────────────────
    // On entry: active [0,N) = Pt (offspring from last step),
    //           vault.archive_* = P̄t.
    // Pool = P̄t ∪ Pt  (archive + active).
    void step(DataVault<Ind_t>& vault) {
        int N = vault.pop_size();
        eff_arch_ = (archive_size_ > 0) ? archive_size_ : N;

        // Fitness over pool = archive ∪ active.
        assign_fitness(vault, N);

        // Update archive P̄t+1 from pool.
        update_archive(vault, N);

        // Breed new offspring Pt+1 from archive into active [0,N).
        breed(vault, N);
        vault.sync();
    }
};

} // namespace mootation

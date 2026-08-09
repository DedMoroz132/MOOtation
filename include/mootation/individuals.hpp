#pragma once
// SPDX-License-Identifier: Apache-2.0

#include <vector>

namespace mootation {

// Forward declarations: DataVault and Problem — friends
template <typename Ind_type> class DataVault;
template <typename Ind_type> class Problem;

// Base individual.
// variables/objectives/limits are private, accessible only through DataVault.
// The algorithm sees only the public fields of derived structs (fitness/rank/cd).
struct Based_Individual {
private:
    std::vector<double> variables, objectives, limits;
    std::vector<int>    binary_variables;   // 0 or 1
    double              cv = 0.0;           // sum max(0, limits[k]) — cache

    template <typename T> friend class DataVault;
    template <typename T> friend class Problem;
};

struct IBEA_Individual : public Based_Individual {
    double fitness = 0.0;
};

struct NSGAII_Individual : public Based_Individual {
    double fitness = 0.0;
    int    rank = 0;
    double crowding_distance = 0.0;
};

// mIBEA (Li et al., CEC 2017)
// fitness — sum_{y!=x} -exp(-I_eps+(y,x)/(c*kappa)); higher is better
// rank    — 0 = non-dominated, 1 = dominated; set by dominance filter step only
struct mIBEA_Individual : public Based_Individual {
    double fitness = 0.0;
    int    rank    = 0;
};

// R2-IBEA (Phan & Suzuki, CEC 2013)
// fitness — sum_{y!=x} -exp(-IR2(y,x)/kappa); higher is better
struct R2IBEA_Individual : public Based_Individual {
    double fitness = 0.0;
};

// HypE (Bader & Zitzler, Evol. Comput. 2011)
// fitness — Monte-Carlo hypervolume contribution estimate; higher is better
struct HypE_Individual : public Based_Individual {
    double fitness = 0.0;
};

// SPEA2 (Zitzler et al., TIK-Report 2001)
// strength    — number of solutions this individual dominates
// raw_fitness — sum of strengths of all dominators (0 = nondominated)
// density     — k-th nearest neighbour distance in objective space
// fitness     = raw_fitness + density; lower is better (opposite convention from IBEA!)
struct SPEA2_Individual : public Based_Individual {
    int    strength    = 0;
    int    raw_fitness = 0;
    double density     = 0.0;
    double fitness     = 0.0;
};

// MOEA/D and MOEA/D-DE (Zhang & Li 2007; Li & Zhang 2009)
// Weight vectors and neighbourhood are stored in the algorithm (W_, B_),
// not in the individual — no duplication.
// scalar_fitness — current Tchebycheff scalar value (for reference/debug)
struct MOEAD_Individual : public Based_Individual {
    double scalar_fitness = 0.0;
};

// NSGA-III (Deb & Jain, IEEE TEVC 2014)
// rank              — non-dominated front index (0 = best)
// ref_point_idx     — index of associated reference point (set during niche assignment)
// niche_count       — number of solutions associated to the same reference point
// norm_distance     — perpendicular distance to associated reference line (for niche selection)
struct NSGAIII_Individual : public Based_Individual {
    int    rank          = 0;
    int    ref_point_idx = -1;
    int    niche_count   = 0;
    double norm_distance = 0.0;
};

// RVEA (Cheng et al., IEEE TEVC 2016)
// ref_vector_idx — index of associated reference vector
// apd            — angle-penalised distance fitness; lower is better
struct RVEA_Individual : public Based_Individual {
    int    ref_vector_idx = -1;
    double apd            = 0.0;
};

// AGE-MOEA (Panichella, GECCO 2019)
// rank             — non-dominated front index
// survival_score   — combined proximity + spread score; higher is better
struct AGEMOEA_Individual : public Based_Individual {
    int    rank           = 0;
    double survival_score = 0.0;
};

// NIMMO (Tanabe & Ishibuchi, Swarm Evol. Comput. 2020)
// Steady-state (µ+1) — no persistent per-individual fitness fields needed;
// IBEA fitness is computed locally per neighbourhood and not stored.
// nimmo_fitness — placeholder for the local I⁺ fitness; written transiently
//                 within each step() call but not used across steps.
struct NIMMO_Individual : public Based_Individual {
    double nimmo_fitness = 0.0;
};

// SRV-RVEA (Liu et al., IEEE TCYB 2022) — uses same fields as RVEA
// ref_vector_idx — index of the adaptive reference vector x is assigned to
// apd            — angle-penalised distance; lower is better
struct SRV_Individual : public Based_Individual {
    int    ref_vector_idx = -1;
    double apd            = 0.0;
};

// SRV-NSGA3 — uses same fields as NSGA-III
// (reference lines come from SRV; association logic is identical)
using SRVNSGA3_Individual = NSGAIII_Individual;

// SRV-MOEA/D — no persistent per-individual fields beyond MOEA/D
// (weight vectors and neighbourhood rebuilt from SRV; in-place update)
using SRVMOEAD_Individual = MOEAD_Individual;

// SPEA2+SDE (Li, Yang, Liu, IEEE TEVC 2014)
// Identical to SPEA2_Individual except raw_fitness is double (fractional sums of strengths).
// strength   — S(x): number of individuals that x dominates
// raw_fitness — R(x): sum of strengths of x's dominators; 0 = non-dominated
// density    — D(x) = 1 / (σ^SDE_k(x) + 2)
// fitness    = R(x) + D(x); lower is better
struct SPEA2SDE_Individual : public Based_Individual {
    int    strength    = 0;
    double raw_fitness = 0.0;
    double density     = 0.0;
    double fitness     = 0.0;
};

// IBEAhd (Zitzler & Künzli, PPSN 2004)
// fitness — F(x) = Σ -exp(-I_HD(y,x)/(c·κ)); higher is better
struct IBEAhd_Individual : public Based_Individual {
    double fitness = 0.0;
};

// MOEA/DD (Li, Deb, Zhang, Kwong, IEEE TEVC 2015)
// rank          — Pareto non-domination front index (0 = best)
// subregion_idx — index of the weight vector / subregion x is assigned to
// pbi_value     — PBI distance in the assigned subregion; lower is better
// niche_count   — number of solutions in the same subregion
struct MOEADD_Individual : public Based_Individual {
    int    rank          = 0;
    int    subregion_idx = -1;
    double pbi_value     = 0.0;
    int    niche_count   = 0;
};

// GrEA (Yang, Li, Liu, Zheng, IEEE TEVC 2013)
// grid_coord — [M] integer grid coordinate G_k(x) ∈ [0, div-1] per objective
// gr         — Grid Ranking GR = Σ_k G_k(x) (Eq.9); adjusted by
//              GR_adjustment (Alg.3) during environmental selection
// gcd        — Grid Crowding Distance (Eq.10); built dynamically
//              relative to the archive being selected (Alg.6)
// GCPD (Eq.11, real-valued) is not stored in the individual — computed
// locally in environmental selection (grea.hpp).
struct GrEA_Individual : public Based_Individual {
    std::vector<int> grid_coord;   // size M, set each generation
    int gr  = 0;
    int gcd = 0;
};

// θ-DEA (Yuan, Xu, Wang, Yao, IEEE TEVC 2016)
// rank          — θ-dominance front index (0 = best)
// ref_point_idx — index of associated reference point (angular association)
// d1            — projection length onto reference line (PBI component)
// d2            — perpendicular distance to reference line (PBI component)
struct ThetaDEA_Individual : public Based_Individual {
    int    rank          = 0;
    int    ref_point_idx = -1;
    double d1            = 0.0;
    double d2            = 0.0;
};

// MaOEA-ARV (Zhang et al., Information Sciences 2021)
// rank    — non-dominated front index
// fitness — ||f'(x)||: distance to ideal in normalised space; smaller is better
struct MaOEAARV_Individual : public Based_Individual {
    int    rank    = 0;
    double fitness = 0.0;
};

} // namespace mootation

#pragma once
// SPDX-License-Identifier: Apache-2.0

#include <cmath>
#include <stdexcept>
#include <string>

#include "nsga2.hpp"

namespace mootation {

// ============================================================================
// DN-NSGA-II — decision space based niching NSGA-II
// J. J. Liang, C. T. Yue, B. Y. Qu, "Multimodal multi-objective optimization:
// A preliminary study", IEEE CEC 2016, pp. 2454-2461,
// doi:10.1109/CEC.2016.7744093          (source: dn-nsga2_liang2016)
//
// NSGA-II (nsga2.hpp) with the paper's two changes (§IV), so that solutions
// far apart in the variables with one image on the front survive together:
//   1. the mating pool is filled by niching, after De Jong's crowding: "a
//      solution is randomly chosen from the population", "a constant number
//      (crowding factor (CF)) of solutions are randomly selected in the left
//      population", the closest of them to it by Euclidean distance is
//      taken, and "the superior one of the two solutions" enters the pool;
//      "repeat ... until the mating pool is full" (NSGAIICore::niche);
//   2. "the crowding in objective space is replaced by crowding in decision
//      space": crowding_space = decision, §III-B's formula over the variables
//      (nsga2.hpp), in the environmental selection and in the comparison of
//      change 1.
// Everything else is NSGA-II's: SBX and polynomial mutation at §IV-A's
// defaults, the fast non-dominated sort, whole fronts and the last one cut by
// crowding distance. The paper's runs: N = 800, 100 generations.
//
// Resolved ambiguities (the paper leaves them open; no code of the authors at
// hand):
//   DN-1 CF: the knob dn_cf, default 20; all the other members when
//        CF >= N - 1. The nearest of CF draws lies in the member's own part
//        of a Pareto set of K equally filled parts with probability
//        1 - (1 - 1/K)^CF: 0.905 for SYM-PART's nine, 1 - 2^-20 for the two
//        of the paper's own problems. Drawn without replacement.
//   DN-2 "superior": NSGA-II's ≺_n, the tournament's comparison — CDP's
//        feasibility first under constraint handling, then the lower rank,
//        then the larger crowding distance (in the variables, change 2); on
//        a tie the neighbour.
//   DN-3 the distance is in the raw variables, the paper's "Euclidean
//        distance", not divided by the bounds' ranges.
//   DN-4 the solution that meets its neighbour is drawn anew for every place
//        in the pool, with replacement, as the tournament's are; one place
//        is drawn when a parent is needed, which is the same pool.
// Not covered, refused at setup: a binary or categorical genome — the paper
// gives no distance for one. `mating` is reported ignored: the pool is the
// niching one. crowding_space can be set back to `objectives`, for an
// ablation of change 2.
// ============================================================================
template <typename Ind_t>
class DNNSGAIICore : public NSGAIICore<Ind_t> {
public:
    DNNSGAIICore()
    {
        this->set_crowding_space(CrowdingSpace::Decision);
        this->niche_cf_ = 20;                                   // DN-1
    }

    bool set_param(const std::string& k, double v)
    {
        if (k == "dn_cf") {
            if (!(v >= 1.0) || v != std::floor(v) || v > 1e9)
                throw std::invalid_argument("dn_cf must be a whole number >= 1");
            this->niche_cf_ = static_cast<int>(v);
            return true;
        }
        return NSGAIICore<Ind_t>::set_param(k, v);
    }

    bool set_word(const std::string& k, const std::string& w)
    {
        if (k == "mating") return false;                      // the niching pool
        return NSGAIICore<Ind_t>::set_word(k, w);
    }

    void setup(DataVault<Ind_t>& vault)
    {
        refuse_genes(vault);
        NSGAIICore<Ind_t>::setup(vault);
    }

    void setup_seeded(DataVault<Ind_t>& vault)
    {
        refuse_genes(vault);
        NSGAIICore<Ind_t>::setup_seeded(vault);
    }

private:
    static void refuse_genes(DataVault<Ind_t>& vault)
    {
        if (vault.bin_vars_n() > 0)
            throw std::invalid_argument("DN-NSGA-II: binary variables are not supported (its "
                                        "niche and crowding are distances in the real "
                                        "variables)");
    }
};

} // namespace mootation

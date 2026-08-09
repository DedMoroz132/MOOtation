#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// Shared constraint-handling primitives.
//
// Convention (constraint_mode.hpp): limits[k] <= 0 is satisfied, > 0 is a
// violation, and CV(x) = sum_k max(0, limits[k]). CV == 0 means feasible.
//
// These helpers exist so that every algorithm expresses the SAME two rules the
// same way, rather than each file re-deriving them:
//
//   * Constrained domination (Deb 2002, NSGA-II §VI). x constrain-dominates y
//     when: x is feasible and y is not; or both are infeasible and CV(x) <
//     CV(y); or both are feasible and x Pareto-dominates y.
//
//   * Feasibility-first scalarization. An infeasible point gets its scalar
//     value pushed above every feasible one by an additive penalty·CV term, so
//     the ordering is (feasible by g) < (infeasible by CV) without changing the
//     comparison operator at the call site.
//
// Neither helper draws random numbers, so switching constraint_mode does not
// move the RNG stream — only the comparisons change.
// ============================================================================

#include <algorithm>
#include <vector>

#include "../constraint_mode.hpp"

namespace mootation::detail {

inline bool cm_active(ConstraintMode m) { return m != ConstraintMode::NONE; }

// Plain Pareto domination on minimisation objectives.
inline bool pareto_dominates(const std::vector<double>& a,
                             const std::vector<double>& b) {
    bool strictly_better = false;
    const std::size_t n = a.size() < b.size() ? a.size() : b.size();
    for (std::size_t k = 0; k < n; ++k) {
        if (a[k] > b[k]) return false;
        if (a[k] < b[k]) strictly_better = true;
    }
    return strictly_better;
}

// Constrained domination (Deb 2002). Reduces to pareto_dominates when both
// points are feasible, so passing cv = 0 everywhere is the unconstrained case.
inline bool cdp_dominates(const std::vector<double>& a, double cva,
                          const std::vector<double>& b, double cvb) {
    const bool fa = (cva <= 0.0), fb = (cvb <= 0.0);
    if (fa && !fb) return true;
    if (!fa && fb) return false;
    if (!fa && !fb) return cva < cvb;
    return pareto_dominates(a, b);
}

// Dispatching form: CDP when the mode is on, plain Pareto otherwise.
inline bool dominates(ConstraintMode m,
                      const std::vector<double>& a, double cva,
                      const std::vector<double>& b, double cvb) {
    return cm_active(m) ? cdp_dominates(a, cva, b, cvb)
                        : pareto_dominates(a, b);
}

// Feasibility-first penalty for a scalarising function (smaller is better).
// The penalty is large enough that any infeasible point loses to any feasible
// one for the objective scales these algorithms work on.
inline double penalize(ConstraintMode m, double g, double cv,
                       double penalty = 1e6) {
    if (!cm_active(m) || cv <= 0.0) return g;
    return g + penalty * (1.0 + cv);
}

// Feasibility-first comparison of two scalar values (smaller is better).
// Returns true when (ga, cva) is preferred over (gb, cvb).
inline bool better_scalar(ConstraintMode m, double ga, double cva,
                          double gb, double cvb) {
    if (cm_active(m)) {
        const bool fa = (cva <= 0.0), fb = (cvb <= 0.0);
        if (fa && !fb) return true;
        if (!fa && fb) return false;
        if (!fa && !fb) return cva < cvb;
    }
    return ga < gb;
}

} // namespace mootation::detail

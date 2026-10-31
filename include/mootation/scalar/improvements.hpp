#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// The single-objective answer and its trajectory (task 18, stage 1, §1.2):
// the best point so far, and a record each time it improves.
//
// THE BEST POINT is the feasible point of least f; while there is no feasible
// point, the point of least violation: the comparison of the stage 4 cores,
// NSGA-II's constrained domination at M = 1 (evo.hpp, evo::better). A failed
// evaluation (a value not finite) is never the best point.
//
// THE RECORDS: a record each time a point evaluated is better than the best
// so far — the evaluations and grad_calls at that point (its own call
// counted), its f and its violation (0: feasible). The first point that is
// not failed is the first record. The variables of the best point are the
// answer's; along the run they stay on the sample grid of the trajectory.
//
// The second answer (§3.4: a log of distinct minima) is not kept here; it
// will sit beside this log, offered the same points.
//
// READINGS:
//   SO-4. The violation is judged as the point is evaluated, at the
//     tolerance of that moment (an eq_tol schedule: the tolerance then), and
//     the records are not judged again. The answer at the end is the
//     evaluator's, at the final tolerance (evaluator.hpp).
//   SO-5. A violation that is not finite (a constraint value NaN) is +inf
//     (EVO-1): such a point is the best only while every point so far is as
//     bad.
//   SO-6. A tie (equal f of two feasible points, equal violation of two
//     infeasible ones) is not an improvement: the earlier point stays best.
// ============================================================================

#include <vector>

#include "evo.hpp"

namespace mootation::scalar {

struct Improvement {
    long long evaluations = 0, grad_calls = 0;
    double f = 0.0, cv = 0.0;
};

class ImprovementLog {
public:
    // A point evaluated, not failed: true when it is the new best (a record kept).
    bool offer(double f, double cv, long long evaluations, long long grad_calls)
    {
        cv = evo::key(cv);                                              // SO-5
        if (!records_.empty() && !evo::better(f, cv, records_.back().f, records_.back().cv)) return false;
        records_.push_back({evaluations, grad_calls, f, cv});
        return true;
    }

    const std::vector<Improvement>& records() const { return records_; }
    bool empty() const { return records_.empty(); }
    const Improvement& best() const { return records_.back(); }

private:
    std::vector<Improvement> records_;
};

}   // namespace mootation::scalar

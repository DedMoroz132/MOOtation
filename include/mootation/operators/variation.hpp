#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// One mating's variation in the switchable algorithms: the crossover, then the
// mutation of the children the host keeps (task 7 of 2026-10-10, supplement
// 1). With coords = axis, or before the learned coordinates are ready, it is
// exactly the hosts' former three calls — apply_any, then the mutation of c1
// and of c2 — drawing the same random numbers in the same order. With learned
// coordinates (learned_coords.hpp) both act in y: the parents, every further
// parent the crossover draws, de_pbest's best and the DE archive's members
// mapped there, the crossovers without bounds, the mutation within the
// widened range of the projected population, then each kept child mapped
// back and repaired into the box.
//
// choose(), called between the crossover and the mutation, says which
// children the host keeps: −1 both, 0 the first, 1 the second (sms_emoa
// draws its coin there). A child that is not kept is mapped back but neither
// mutated nor counted.
// ============================================================================

#include <vector>

#include "learned_coords.hpp"
#include "real_crossover.hpp"
#include "real_mutation.hpp"

namespace mootation::ops {

template <typename RNG, typename Draw, typename Best, typename Choose>
int vary(CrossoverSpec& xo, const MutationSpec& mu, const std::vector<double>& p1,
         const std::vector<double>& p2, Draw&& draw, Best&& best, int N,
         std::vector<double>& c1, std::vector<double>& c2, const Bounds& bounds,
         double eta_c, double pc, double eta_m, double pm, RNG& rng, Choose&& choose)
{
    LearnedCoords& co = xo.coords;
    if (!co.active()) {
        xo.apply_any(p1, p2, draw, best, N, c1, c2, bounds, eta_c, pc, rng);
        const int which = choose();
        if (which != 1) mu.apply(c1, bounds, eta_m, pm, rng);
        if (which != 0) mu.apply(c2, bounds, eta_m, pm, rng);
        return which;
    }
    require_repair(xo.repair, coords_name(co.kind), false, false);
    note_operator(coords_name(co.kind), bound_repair_name(xo.repair));
    const std::vector<double> y1 = co.to_y(p1), y2 = co.to_y(p2);
    auto draw_y = [&] { return co.to_y(draw()); };
    auto best_y = [&](double p) { return co.to_y(best(p)); };
    xo.de.view = [&co](const std::vector<double>& x) { return co.to_y(x); };
    std::vector<double> d1, d2;
    xo.apply_any(y1, y2, draw_y, best_y, N, d1, d2, co.open(y1.size()), eta_c, pc, rng);
    xo.de.view = nullptr;
    const int which = choose();
    if (which != 1) mu.apply(d1, co.mutation_bounds(d1), eta_m, pm, rng, true);
    if (which != 0) mu.apply(d2, co.mutation_bounds(d2), eta_m, pm, rng, true);
    c1 = which != 1 ? co.back(d1, bounds, p1, xo.repair, rng) : co.to_x(d1);
    c2 = which != 0 ? co.back(d2, bounds, p2, xo.repair, rng) : co.to_x(d2);
    return which;
}

}   // namespace mootation::ops

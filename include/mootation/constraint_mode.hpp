#pragma once
// SPDX-License-Identifier: Apache-2.0

namespace mootation {

// NONE           — unconstrained
// FEASIBILITY    — Feasibility Rules (both algorithms)
// EPS_CONSTRAINT — CV in Iε+ (IBEA e+ only)
// CDP            — Constrained Domination Deb 2002 (NSGA-II only)
//
// Convention: limits[k] <= 0 — ok; > 0 — violation
// CV(x) = sum max(0, limits[k])
enum class ConstraintMode { NONE, FEASIBILITY, EPS_CONSTRAINT, CDP };

} // namespace mootation

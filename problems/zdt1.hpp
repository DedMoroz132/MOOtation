#pragma once
// SPDX-License-Identifier: Apache-2.0

// ZDT1 — the classical 2-objective unconstrained problem,
// 6 real variables in [0,1].
// Pareto optimum: f2 = 1 - sqrt(f1), x[1..5] = 0.
//
// This is just the Problem<T> default. The file is kept for documentation
// and as an example of an extension point.

#include "../include/mootation/problem.hpp"

// Usage:
//   mootation::Problem<mootation::IBEA_Individual>   prob;   // ZDT1 default
//   mootation::Problem<mootation::NSGAII_Individual> prob;

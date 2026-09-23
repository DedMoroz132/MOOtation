#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// BOUND REPAIR — what an operator does with a variable it put outside [lb, ub],
// and a count of how often that happens.
//
// Why a named, recorded choice: the repair is part of the algorithm. Kononova
// et al. (Evol. Comput. 32(1), 2024) show that results are not reproducible
// without it, and Kudela, van Stein, Bäck & Kononova (GECCO 2026) trace a
// systematic pull towards the bounds in 120 PlatEMO algorithms to its default
// clamp. No comparison of the options exists for the multi-objective case, so
// each operator names the one it uses (recorded per run in meta.json) and the
// existing algorithms keep theirs (tests/compat_dump.cpp).
//
//   clip      the violated bound
//   reflect   mirrored back across the violated bound, again and again while
//             it lands outside: a triangle wave of period 2(ub − lb), so any
//             overshoot ends inside
//   random    uniform in [lb, ub] — the draw MOEA/D-DE's Step 2.3 prescribes
//             ("reset to a randomly selected value inside the boundary")
//   midpoint  halfway between the parent's value and the violated bound
//   resample  the operator draws that variable again, at most K times
//             (resample_tries()), then clips; an operator that cannot redraw
//             one variable on its own refuses it
//   wrap      the box as a torus: lb + ((v − lb) mod (ub − lb))
//   native    the rule the operator's own paper gives (Liu & Li's crossover
//             and mutation); meaningless for an operator without one
//
// The counts (A2 of the 2026-09-23 task; the measure of Kononova, Caraffini &
// Bäck, Inf. Sci. 581, 2021): every repair site reports how many variables it
// found outside the box, and whether the offspring had any. The counters are
// thread-local, like sbx_var_prob(): a run is one thread (the Python binding
// runs the core on the caller's thread), and the binding reads and resets them
// between trajectory records. They never touch a random stream.
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstring>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mootation::ops {

enum class BoundRepair { Clip, Reflect, Random, Midpoint, Resample, Wrap, Native };

inline const char* bound_repair_name(BoundRepair b)
{
    switch (b) {
        case BoundRepair::Clip:     return "clip";
        case BoundRepair::Reflect:  return "reflect";
        case BoundRepair::Random:   return "random";
        case BoundRepair::Midpoint: return "midpoint";
        case BoundRepair::Resample: return "resample";
        case BoundRepair::Wrap:     return "wrap";
        case BoundRepair::Native:   return "native";
    }
    return "?";
}

inline std::optional<BoundRepair> parse_bound_repair(const std::string& s)
{
    for (BoundRepair b : {BoundRepair::Clip, BoundRepair::Reflect, BoundRepair::Random,
                          BoundRepair::Midpoint, BoundRepair::Resample, BoundRepair::Wrap,
                          BoundRepair::Native})
        if (s == bound_repair_name(b)) return b;
    return std::nullopt;
}

// Draws allowed before a resampling operator gives up and clips.
inline int& resample_tries() { thread_local int k = 10; return k; }

// ── the counters ─────────────────────────────────────────────────────────────
struct RepairCounts {
    long long checked      = 0;   // offspring passed through a repair site
    long long out_children = 0;   // ... with at least one variable outside
    long long out_vars     = 0;   // variables found outside
};
inline RepairCounts& repair_counts() { thread_local RepairCounts c; return c; }

// The operators a run used, each with its repair ("none" for one that cannot
// leave the box), in order of first use.
struct OperatorUse { const char* op; const char* repair; };
inline std::vector<OperatorUse>& operators_used()
{
    thread_local std::vector<OperatorUse> v;
    return v;
}
inline void note_operator(const char* op, const char* repair)
{
    auto& v = operators_used();
    for (const auto& u : v)
        if (u.op == op || std::strcmp(u.op, op) == 0) {
            if (u.repair == repair || std::strcmp(u.repair, repair) == 0) return;
        }
    v.push_back({op, repair});
}
inline void reset_operator_records()
{
    repair_counts() = RepairCounts{};
    operators_used().clear();
}

// One offspring's worth of counting: construct, report each variable found
// outside with out(), and the destructor books it.
class RepairTally {
public:
    RepairTally() = default;
    RepairTally(const RepairTally&) = delete;
    RepairTally& operator=(const RepairTally&) = delete;
    ~RepairTally()
    {
        auto& c = repair_counts();
        ++c.checked;
        if (n_ > 0) { ++c.out_children; c.out_vars += n_; }
    }
    void out() { ++n_; }
private:
    long long n_ = 0;
};

// An operator that cannot redraw one variable on its own refuses resample,
// and one whose paper gives no repair refuses native.
inline void require_repair(BoundRepair b, const char* op, bool resample_ok, bool native_ok)
{
    if ((b == BoundRepair::Resample && !resample_ok) || (b == BoundRepair::Native && !native_ok))
        throw std::invalid_argument(std::string(op) + ": bound_repair '" + bound_repair_name(b) +
                                    "' is not available here");
}

// ── one variable ─────────────────────────────────────────────────────────────
// v is outside [lo, hi]; parent is the value the offspring came from (for
// midpoint). Resample and Native are the operator's business: here they clip.
// Random constructs the distribution exactly as the DE code always has, so an
// algorithm that used DERepair::RandomReset draws the same numbers.
template <typename RNG>
inline double repair_value(double v, double lo, double hi, double parent,
                           BoundRepair how, RNG& rng)
{
    switch (how) {
        case BoundRepair::Random: {
            std::uniform_real_distribution<double> dom(lo, hi);
            return dom(rng);
        }
        case BoundRepair::Reflect: {
            const double d = hi - lo;
            if (!(d > 0.0)) return lo;
            double t = std::fmod(v - lo, 2.0 * d);
            if (t < 0.0) t += 2.0 * d;
            if (t > d) t = 2.0 * d - t;
            return std::clamp(lo + t, lo, hi);
        }
        case BoundRepair::Midpoint: {
            const double p = std::clamp(parent, lo, hi);
            return 0.5 * (p + (v < lo ? lo : hi));
        }
        case BoundRepair::Wrap: {
            const double d = hi - lo;
            if (!(d > 0.0)) return lo;
            double t = std::fmod(v - lo, d);
            if (t < 0.0) t += d;
            return std::clamp(lo + t, lo, hi);
        }
        case BoundRepair::Clip:
        case BoundRepair::Resample:
        case BoundRepair::Native:
            break;
    }
    return std::clamp(v, lo, hi);
}

}   // namespace mootation::ops

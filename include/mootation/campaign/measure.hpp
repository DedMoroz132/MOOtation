#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// run/metrics.py's compute() for the campaign engine in C++ — task 17, stage 5,
// step 4c: what metrics.compute adds around run_compute.hpp (_compute_args and
// after) — the hypervolume's defaults and the caller's options, a constrained
// problem's feasibility and overall violations from its constraints, the
// rounding of range_cover_each to six decimals (Python's round) — so that a
// trajectory record or a run's final has the same values, in the same order,
// as the Python engine's.
//
// The problem is a suite problem (registry.hpp); its distance to the Pareto
// set (g_min, g_median) is Python's yet: those come back None here, and the
// engine leaves a run that asks for them to Python.
// ============================================================================

#include <charconv>
#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <system_error>
#include <vector>

#include "../json.hpp"
#include "../run_compute.hpp"
#include "../suite/suite.hpp"
#include "references.hpp"

namespace mootation::campaign {

// metrics.HV_EXACT_MAX_M, HV_MC_SAMPLES: compute()'s hypervolume defaults
constexpr int kHvExactMaxM = 5;
constexpr std::size_t kHvMcSamples = 100000;

struct HvOptions {
    int exact_max_m = kHvExactMaxM;
    std::size_t mc_samples = kHvMcSamples;
    std::uint64_t seed = 0;
    std::string algo = "auto";
};

// What compute() measures against, of the problem and the run.
struct MeasureInputs {
    std::optional<Matrix> ref;                    // the reference front
    std::optional<std::vector<double>> ideal, nadir;
    std::size_t pop = 0;                          // the problem's population size; 0: None
    std::optional<Matrix> pset;                   // the Pareto-set sample
    std::vector<std::pair<double, double>> bounds;
    std::vector<std::size_t> cyclic;
    std::shared_ptr<suite::Native> constraints;   // set for a constrained problem
};

// Python's round(x, ndigits) of a float: the decimal string correctly rounded
// to ndigits places (half to even on the exact binary value, as dtoa's mode 3)
// read back
inline double py_round(double x, int ndigits)
{
    if (!std::isfinite(x)) return x;
    char buf[400];
    const auto r = std::to_chars(buf, buf + sizeof buf, x, std::chars_format::fixed, ndigits);
    if (r.ec != std::errc()) return x;
    double y = 0.0;
    std::from_chars(buf, r.ptr, y);
    return y;
}

// a run_compute value as Python's dict holds it
inline json::Value to_json(const run_compute::Value& v)
{
    using K = run_compute::Value;
    switch (v.kind) {
    case K::None: return json::Value::null();
    case K::Real: return json::Value::number(v.real);
    case K::Int: return json::Value::integer(v.integer);
    case K::Bool: return json::Value::boolean(v.flag);
    case K::Text: return json::Value::string(v.text);
    case K::List: {
        json::Value a = json::Value::array();
        for (double x : v.list) a.items.push_back(json::Value::number(x));
        return a;
    }
    case K::Dict: {
        json::Value o = json::Value::object();
        for (const auto& [k, x] : v.dict) o.set(k, json::Value::number(x));
        return o;
    }
    }
    return json::Value::null();
}

// metrics.compute(F, ref_front, ideal, nadir, which, pop, hv_options, X,
// pareto_set, bounds, cyclic, constraints): the indicators of the rows F (n x
// m) with variables X (n x d, or null) as a dict in Python's order
inline json::Value measure(const double* F, std::size_t n, std::size_t m, const double* X, std::size_t d,
                           const std::vector<std::string>& which, const MeasureInputs& in, const HvOptions& hv)
{
    namespace rc = run_compute;
    rc::Context c;
    c.m = m;
    if (in.ref) {
        c.ref = in.ref->values.data();
        c.ref_n = in.ref->rows();
    }
    c.has_frame = in.ideal && in.nadir;
    if (c.has_frame) {
        c.ideal = *in.ideal;
        c.nadir = *in.nadir;
    }
    c.pop = in.pop;
    c.exact_max_m = hv.exact_max_m;
    c.mc_samples = hv.mc_samples;
    c.seed = hv.seed;
    c.hv_algo = hv.algo;
    rc::Rows rows;
    rows.F = F;
    rows.n = n;
    if (X) {
        rows.X = X;
        rows.d = d;
        if (in.pset && in.pset->rows() && in.pset->values.size()) {
            c.pset = in.pset->values.data();
            c.pset_n = in.pset->rows();
        }
        if (!in.bounds.empty()) {
            for (const auto& [lo, hi] : in.bounds) {
                c.lo.push_back(lo);
                c.hi.push_back(hi);
            }
            c.has_bounds = true;
        }
        c.cyclic = in.cyclic;
    }
    // a constrained problem: every row's feasibility and, for i_cmop with a
    // frame, its overall violation (metrics.feasible_mask, violations)
    std::vector<char> ok;
    std::vector<double> V;
    if (in.constraints) {
        if (!X) throw std::invalid_argument("measure: a constrained problem's indicators need the variables");
        const suite::Native& p = *in.constraints;
        std::vector<double> f(static_cast<std::size_t>(p.n_obj)), g(static_cast<std::size_t>(p.n_con));
        bool want_v = false;
        for (const auto& w : which) want_v = want_v || w == "i_cmop";
        want_v = want_v && c.has_frame && n;
        for (std::size_t i = 0; i < n; ++i) {
            p.evaluate(X + i * d, f.data(), g.empty() ? nullptr : g.data());
            bool feasible = true;
            double v = 0.0;
            for (double x : g) {
                feasible = feasible && x <= 0.0;
                v += x < 0.0 ? 0.0 : x;                  // Python's max(c, 0.0): a NaN stays
            }
            ok.push_back(feasible ? 1 : 0);
            if (want_v) V.push_back(v);
        }
        rows.constrained = true;
        rows.feasible = ok.data();
        if (want_v) rows.V = V.data();
    }
    const rc::Out out = rc::compute(rows, c, which);
    json::Value o = json::Value::object();
    for (const auto& [name, v] : out.items) {
        json::Value j = to_json(v);
        if (name == "range_cover_each")
            for (auto& x : j.items) x.f = py_round(x.f, 6);
        o.set(name, std::move(j));
    }
    return o;
}

}   // namespace mootation::campaign

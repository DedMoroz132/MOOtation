#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// python/mootation/run/metrics.py's compute() in C++ (task 17, stage 2b, step
// 1): every indicator a trajectory record or a run's final asks for, by name,
// from the kernels the run layer already takes from C++ (indicators.hpp,
// decision_indicators.hpp, run_metrics.hpp), in the same order and with the
// same values, bit for bit — a step towards run_job in C++ (stage 5 runs a
// campaign without Python). metrics.compute calls it where the extension is
// built and keeps its Python as the reference the tests compare with.
//
// What the caller computes. A constrained problem's indicators are measured
// on the feasible part of the set: the caller gives every row's feasibility
// and, for i_cmop, its overall violation (metrics.feasible_mask,
// metrics.violations; a Python problem's constraints are Python). g_min and
// g_median read the problem's distance to its Pareto set, distance_fn, one
// value per row, also given. range_cover_each comes back unrounded: metrics.py
// rounds it to six decimals with Python's round().
// ============================================================================

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "decision_indicators.hpp"
#include "indicators.hpp"
#include "numpy_sum.hpp"
#include "run_metrics.hpp"

namespace mootation::run_compute {

using Vec = std::vector<double>;

// One indicator's value, as metrics.compute puts it into its dict: None, a
// float, an int, a bool, a string, a list of floats or a dict of floats.
struct Value {
    enum Kind { None, Real, Int, Bool, Text, List, Dict } kind = None;
    double real = 0.0;
    long long integer = 0;
    bool flag = false;
    std::string text;
    Vec list;
    std::vector<std::pair<std::string, double>> dict;

    static Value none() { return Value{}; }
    static Value of(double v) { Value o; o.kind = Real; o.real = v; return o; }
    static Value of_int(long long v) { Value o; o.kind = Int; o.integer = v; return o; }
    static Value of_bool(bool v) { Value o; o.kind = Bool; o.flag = v; return o; }
    static Value of_text(std::string v) { Value o; o.kind = Text; o.text = std::move(v); return o; }
};

// The dict, in Python's order: a name set again keeps its place.
struct Out {
    std::vector<std::pair<std::string, Value>> items;

    void set(const std::string& name, Value v)
    {
        for (auto& kv : items)
            if (kv.first == name) { kv.second = std::move(v); return; }
        items.emplace_back(name, std::move(v));
    }
    bool has(const std::string& name) const
    {
        for (const auto& kv : items)
            if (kv.first == name) return true;
        return false;
    }
};

// What the indicators are measured against: compute()'s keyword arguments.
struct Context {
    std::size_t m = 0;                  // objectives
    const double* ref = nullptr;        // the reference front, ref_n x m; null: None
    std::size_t ref_n = 0;
    bool has_frame = false;             // ideal and nadir given
    Vec ideal, nadir;
    std::size_t pop = 0;                // the problem's population size; 0: None
    int exact_max_m = 5;                // hv_options
    std::size_t mc_samples = 100000;
    std::uint64_t seed = 0;
    std::string hv_algo = "auto";
    const double* pset = nullptr;       // the Pareto-set sample, pset_n x d; null: None
    std::size_t pset_n = 0;
    bool has_bounds = false;
    Vec lo, hi;
    std::vector<std::size_t> cyclic;
    // reference_step of the normalised reference (metrics.py caches it per
    // process); NaN: computed here
    double ref_step = std::numeric_limits<double>::quiet_NaN();
};

// The rows given: objectives F (n x m), variables X (n x d, or null), and
// the caller's per-row values (each null where not given): feasible (1/0;
// constrained says it is given, an empty set's included), the overall
// violation V, the distance g.
struct Rows {
    const double* F = nullptr;
    std::size_t n = 0;
    const double* X = nullptr;
    std::size_t d = 0;
    bool constrained = false;
    const char* feasible = nullptr;
    const double* V = nullptr;
    const double* g = nullptr;
};

namespace detail {

inline double np_min(const double* a, std::size_t n)        // ndarray.min(): a NaN wins
{
    double v = a[0];
    for (std::size_t i = 1; i < n; ++i) v = run_metrics::detail::nan_min(v, a[i]);
    return v;
}

inline double np_max(const double* a, std::size_t n)
{
    double v = a[0];
    for (std::size_t i = 1; i < n; ++i) v = run_metrics::detail::nan_max(v, a[i]);
    return v;
}

// metrics.lattice_h: the largest H whose Das-Dennis lattice has at most pop points.
inline long long lattice_h(std::size_t m, std::size_t pop)
{
    if (m < 2) return 1;
    auto comb = [](unsigned long long n, unsigned long long k, unsigned long long cap) {
        unsigned long long c = 1;                         // exact while it stays <= cap
        for (unsigned long long i = 1; i <= k; ++i) {
            c = c * (n - k + i) / i;
            if (c > cap) return cap + 1;
        }
        return c;
    };
    long long h = 1;
    while (comb(static_cast<unsigned long long>(h) + m, m - 1, pop) <= pop) ++h;
    return h;
}

// metrics.das_dennis: every weight vector of multiples of 1/h summing to 1,
// the first component from h down.
inline Vec das_dennis(std::size_t m, long long h)
{
    Vec rows;
    std::vector<long long> prefix;
    auto rec = [&](auto&& self, long long left, std::size_t k) -> void {
        if (k == 1) {
            for (long long v : prefix) rows.push_back(static_cast<double>(v));
            rows.push_back(static_cast<double>(left));
            return;
        }
        for (long long v = left; v >= 0; --v) {
            prefix.push_back(v);
            self(self, left - v, k - 1);
            prefix.pop_back();
        }
    };
    rec(rec, h, m);
    for (double& v : rows) v /= static_cast<double>(h);
    return rows;
}

}   // namespace detail

// metrics.compute(F, ..., which=which): the indicators named, in their order,
// with the extra entries metrics.compute adds (hv_method, coverage_curve, ...).
// An unknown name is the caller's to refuse (metrics.compute raises first).
inline Out compute(const Rows& rows, const Context& c, const std::vector<std::string>& which)
{
    using run_metrics::Hv;
    const double inf = std::numeric_limits<double>::infinity();
    const std::size_t m = c.m;
    Out out;
    auto wants = [&](const char* name) {
        for (const auto& w : which)
            if (w == name) return true;
        return false;
    };

    // the set measured: the feasible rows where feasibility is given
    std::vector<double> Fs(rows.F, rows.F + rows.n * m), Xs, gs;
    if (rows.X) Xs.assign(rows.X, rows.X + rows.n * rows.d);
    if (rows.g) gs.assign(rows.g, rows.g + rows.n);
    std::size_t n = rows.n;

    if (wants("i_cmop") && c.has_frame && rows.n) {
        // the whole set, infeasible points included (Vodopija et al., Eq. 12)
        Vec V(rows.n, 0.0);
        if (rows.V) V.assign(rows.V, rows.V + rows.n);
        Vec ok_rows, inside_rows;
        bool any_ok = false;
        for (std::size_t i = 0; i < rows.n; ++i) {
            if (V[i] <= 0.0) {
                any_ok = true;
                const double* f = rows.F + i * m;
                ok_rows.insert(ok_rows.end(), f, f + m);
                bool all_le = true, any_lt = false;
                for (std::size_t j = 0; j < m; ++j) {
                    const double gj = (f[j] - c.ideal[j]) / (c.nadir[j] - c.ideal[j]);
                    all_le = all_le && gj <= 1.0;
                    any_lt = any_lt || gj < 1.0;
                }
                if (all_le && any_lt) inside_rows.insert(inside_rows.end(), f, f + m);
            }
        }
        if (!any_ok) {
            out.set("i_cmop", Value::of(detail::np_min(V.data(), V.size()) + 1.0));
        } else {
            double I;
            if (!inside_rows.empty()) {
                const Hv h = run_metrics::hypervolume(inside_rows.data(), inside_rows.size() / m, m, c.ideal,
                                                      c.nadir, 1.0, c.mc_samples, c.seed, c.exact_max_m, c.hv_algo);
                I = -h.value;
            } else {
                I = run_metrics::roi_dist(ok_rows.data(), ok_rows.size() / m, m, c.ideal, c.nadir);
            }
            out.set("i_cmop", Value::of(1.0 < I ? 1.0 : I));        // Python's min(I, tau): NaN stays
        }
    }
    if (rows.constrained) {
        std::size_t k = 0;
        for (std::size_t i = 0; i < rows.n; ++i) k += rows.feasible[i] ? 1 : 0;
        out.set("feas_share", Value::of(rows.n ? static_cast<double>(k) / static_cast<double>(rows.n) : 0.0));
        if (k == 0) {
            for (const auto& name : which)
                if (name != "i_cmop") out.set(name, name == "n_final" ? Value::of_int(0) : Value::none());
            return out;
        }
        Vec F2, X2, g2;
        for (std::size_t i = 0; i < rows.n; ++i) {
            if (!rows.feasible[i]) continue;
            F2.insert(F2.end(), rows.F + i * m, rows.F + (i + 1) * m);
            if (rows.X) X2.insert(X2.end(), rows.X + i * rows.d, rows.X + (i + 1) * rows.d);
            if (rows.g) g2.push_back(rows.g[i]);
        }
        Fs.swap(F2); Xs.swap(X2); gs.swap(g2);
        n = k;
    }
    const double* F = Fs.data();
    const bool have_box = c.has_frame;
    const bool have_x = rows.X != nullptr && n > 0 && rows.d > 0;
    const std::size_t d = rows.d;
    const bool have_pset = c.pset != nullptr && c.pset_n > 0;

    // igdp_norm, gdp_norm, eps_norm, tau90, gap_max and gdp_min read the same
    // normalised differences: one pass for all (metrics.compute's dplus_norm)
    bool shared_done = false;
    Vec to_set, to_ref, eps_v, Rn;
    auto dplus_norm = [&]() {
        if (shared_done) return;
        const Vec span = run_metrics::span_of(c.ideal, c.nadir);
        const Vec G = run_metrics::normalised(F, n, m, c.ideal, span);
        Rn = run_metrics::normalised(c.ref, c.ref_n, m, c.ideal, span);
        to_set.assign(c.ref_n, 0.0); to_ref.assign(n, 0.0); eps_v.assign(c.ref_n, 0.0);
        indicators::front_distances(G.data(), n, Rn.data(), c.ref_n, m, to_set.data(), to_ref.data(),
                                    eps_v.data());
        shared_done = true;
    };
    // the raw distances of igdp, gdp and eps (front_distances of F itself)
    auto raw_front = [&](int which_part) {
        Vec a(c.ref_n), b(n), e(c.ref_n);
        indicators::front_distances(F, n, c.ref, c.ref_n, m, a.data(), b.data(), e.data());
        if (which_part == 0) return np::mean(a.data(), a.size());
        if (which_part == 1) return np::mean(b.data(), b.size());
        return detail::np_max(e.data(), e.size());
    };
    // the decision-space rows normalised by the bounds, and the sample's
    std::vector<char> cyc;
    if (!c.cyclic.empty()) {
        cyc.assign(d, 0);
        for (std::size_t j : c.cyclic) cyc[j] = 1;
    }
    auto unit = [&](const double* A, std::size_t k) {
        return indicators::unit_rows(A, k, d, c.lo.data(), c.hi.data());
    };

    for (const auto& name : which) {
        if (name == "igd" || name == "igdp" || name == "gdp" || name == "eps") {
            if (c.ref == nullptr) { out.set(name, Value::none()); continue; }
            if (n == 0) { out.set(name, Value::of(inf)); continue; }
            if (name == "igd") {
                Vec near(c.ref_n);
                indicators::igd_distances(F, n, c.ref, c.ref_n, m, near.data());
                out.set(name, Value::of(np::mean(near.data(), near.size())));
            } else {
                out.set(name, Value::of(raw_front(name == "igdp" ? 0 : name == "gdp" ? 1 : 2)));
            }
        } else if (name == "roi_dist") {
            out.set(name, have_box && n ? Value::of(run_metrics::roi_dist(F, n, m, c.ideal, c.nadir))
                                        : Value::none());
        } else if (name == "range_cover") {
            if (!have_box || n == 0) { out.set(name, Value::none()); continue; }
            const Vec each = run_metrics::range_cover_each(F, n, m, c.ideal, c.nadir);
            out.set(name, Value::of(detail::np_min(each.data(), each.size())));
            Value list;
            list.kind = Value::List;
            list.list = each;                     // metrics.py rounds them to six decimals
            out.set("range_cover_each", list);
        } else if (name == "nd_share") {
            out.set(name, n ? Value::of(run_metrics::nd_share(F, n, m)) : Value::none());
        } else if (name == "dup_share") {
            out.set(name, n ? Value::of(run_metrics::dup_share(F, n, m)) : Value::none());
        } else if (name == "igdp_norm" || name == "gdp_norm" || name == "eps_norm") {
            if (c.ref == nullptr || !have_box) { out.set(name, Value::none()); continue; }
            if (n == 0) { out.set(name, Value::of(inf)); continue; }
            dplus_norm();
            out.set(name, Value::of(name == "igdp_norm" ? np::mean(to_set.data(), to_set.size())
                                  : name == "gdp_norm" ? np::mean(to_ref.data(), to_ref.size())
                                                       : detail::np_max(eps_v.data(), eps_v.size())));
        } else if (name == "tau90") {
            if (c.ref == nullptr || !have_box || n == 0) { out.set(name, Value::none()); continue; }
            dplus_norm();
            out.set(name, Value::of(run_metrics::quantile(to_set, 0.9)));
            Value curve;
            curve.kind = Value::Dict;
            const std::pair<const char*, double> taus[] = {
                {"0.01", 0.01}, {"0.02", 0.02}, {"0.05", 0.05}, {"0.1", 0.1}, {"0.2", 0.2}};
            for (const auto& t : taus) {
                std::size_t k = 0;
                for (double v : to_set) k += v <= t.second ? 1 : 0;
                curve.dict.emplace_back(t.first, static_cast<double>(k) / static_cast<double>(to_set.size()));
            }
            out.set("coverage_curve", curve);
        } else if (name == "gap_max") {
            if (c.ref == nullptr || !have_box || n == 0) { out.set(name, Value::none()); continue; }
            dplus_norm();
            out.set(name, Value::of(detail::np_max(to_set.data(), to_set.size())));
        } else if (name == "nn_cv") {
            if (!have_box || n == 0) { out.set(name, Value::none()); continue; }
            const double v = run_metrics::nn_cv(F, n, m, c.ideal, c.nadir);
            out.set(name, std::isnan(v) ? Value::none() : Value::of(v));
        } else if (name == "n_final") {
            out.set(name, Value::of_int(static_cast<long long>(n)));
        } else if (name == "hv") {
            if (!have_box) { out.set("hv", Value::none()); continue; }
            const Hv h = run_metrics::hypervolume(F, n, m, c.ideal, c.nadir, 1.1, c.mc_samples, c.seed,
                                                  c.exact_max_m, c.hv_algo);
            out.set("hv", Value::of(h.value));
            out.set("hv_method", Value::of_text(h.method));
            if (!std::isnan(h.se)) out.set("hv_se", Value::of(h.se));
        } else if (name == "hv_h") {
            if (!have_box || c.pop == 0) { out.set("hv_h", Value::none()); continue; }
            const double scale = 1.0 + 1.0 / static_cast<double>(detail::lattice_h(m, c.pop));
            const Hv h = run_metrics::hypervolume(F, n, m, c.ideal, c.nadir, scale, c.mc_samples, c.seed,
                                                  c.exact_max_m, c.hv_algo);
            out.set("hv_h", Value::of(h.value));
            out.set("hv_h_ref", Value::of(scale));
            out.set("hv_method", Value::of_text(h.method));
            if (!std::isnan(h.se)) out.set("hv_h_se", Value::of(h.se));
        } else if (name == "r2") {
            if (!have_box || c.pop == 0 || n == 0) { out.set("r2", Value::none()); continue; }
            const long long h = detail::lattice_h(m, c.pop);
            const Vec G = run_metrics::normalised(F, n, m, c.ideal, run_metrics::span_of(c.ideal, c.nadir));
            const Vec W = detail::das_dennis(m, h);
            out.set("r2", Value::of(indicators::r2_mean(G.data(), n, W.data(), W.size() / m, m)));
            out.set("r2_h", Value::of_int(h));
        } else if (name == "igdx" || name == "cr" || name == "psp") {
            if (!have_x || !have_pset || !c.has_bounds) { out.set(name, Value::none()); continue; }
            const double* X = Xs.data();
            auto igdx_v = [&]() {
                const Vec U = unit(X, n), R = unit(c.pset, c.pset_n);
                return indicators::igdx(U.data(), n, R.data(), c.pset_n, d, cyc);
            };
            if (name == "igdx") {
                out.set(name, Value::of(igdx_v()));
            } else if (name == "cr") {
                out.set(name, Value::of(indicators::cover_rate(X, n, c.pset, c.pset_n, d)));
            } else {
                const double dx = igdx_v();
                const double cr = indicators::cover_rate(X, n, c.pset, c.pset_n, d);
                out.set(name, Value::of(dx == 0.0 ? inf : cr / dx));
            }
        } else if (name == "pdist") {
            if (!have_x || !c.has_bounds) { out.set(name, Value::none()); continue; }
            const Vec U = unit(Xs.data(), n);
            out.set(name, Value::of(indicators::pairwise_distance(U.data(), n, d, cyc)));
        } else if (name == "gdp_min") {
            if (c.ref == nullptr || !have_box) { out.set(name, Value::none()); continue; }
            if (n == 0) { out.set(name, Value::of(inf)); continue; }
            dplus_norm();
            const double v = detail::np_min(to_ref.data(), to_ref.size());
            double step = c.ref_step;
            if (std::isnan(step))           // metrics.reference_step of the normalised reference
                step = c.ref_n > 1 ? run_metrics::median(indicators::nearest_other(Rn.data(), c.ref_n, m))
                                   : c.ref_n == 1 ? inf : std::numeric_limits<double>::quiet_NaN();
            out.set(name, Value::of(v));
            out.set("gdp_min_ref_step", Value::of(step));
            out.set("gdp_min_below_step", Value::of_bool(v < step));
        } else if (name == "g_min" || name == "g_median") {
            if (!have_x || rows.g == nullptr) { out.set(name, Value::none()); continue; }
            out.set(name, Value::of(name == "g_min" ? detail::np_min(gs.data(), gs.size())
                                                    : run_metrics::median(gs)));
        } else if (name == "ps_dmin") {
            if (!have_x || !have_pset || !c.has_bounds) { out.set(name, Value::none()); continue; }
            const Vec U = unit(Xs.data(), n), R = unit(c.pset, c.pset_n);
            out.set(name, Value::of(indicators::ps_dmin(U.data(), n, R.data(), c.pset_n, d, cyc)));
        } else if (name == "i_cmop") {
            if (!out.has("i_cmop")) out.set("i_cmop", Value::none());
        }
    }
    return out;
}

}   // namespace mootation::run_compute

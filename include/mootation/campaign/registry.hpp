#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// The problem registry read in C++ — task 17, stage 5, step 4: the file
// run/registry_export.py writes (<refs root>/registry/<code stamp>.json): every
// problem's sizes and box, default population and generations, version, what
// it modifies, its references, and — for a problem of the C++ suite — the spec
// its evaluation is built from (suite::Spec, as python/suite_module.cpp's
// spec_of reads it from Python). The engine runs a problem of the suite whose
// run needs no Python (engine_refusal says why not, where not).
// ============================================================================

#include <algorithm>
#include <filesystem>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "../io/npz.hpp"
#include "../json.hpp"
#include "../numpy_sum.hpp"
#include "../suite/suite.hpp"

namespace mootation::campaign {

// A failing copy's failure region (benchmarks/failing.py), in the box's
// normalised units, (x - lb) / (ub - lb): "b" the band within t of the
// faces; "r" (random balls) and "n" (a tube about part of the Pareto-set
// sample) the balls of one radius about the centres. With the base's spec.
struct FailRegion {
    std::string kind;
    double t = 0.0, radius = 0.0;
    std::vector<std::vector<double>> centres;
    json::Value spec;
};

struct RegistryProblem {
    std::string name;
    std::string family;                           // families.family: the store's pack it goes to
    int n_var = 0, n_obj = 0, n_eq = 0;
    std::optional<int> n_con;                     // the suite's, for a problem of the suite
    bool has_cons = false;
    std::vector<std::pair<double, double>> bounds;
    std::int64_t pop_size = 0, n_gen = 0;
    int problem_version = 1;
    json::Value modification;                     // properties.modification_of: null or a dict
    bool multimodal_ps = false, has_pareto_front = false, has_pareto_set = false, front_exact = true;
    bool shared_reference = false, dense_reference = false, stochastic = false;
    std::vector<int> cyclic_vars;
    bool own_evaluator = false, typed = false, failing = false, distance = false;
    std::optional<json::Value> spec;              // null: not a problem of the suite
    std::optional<FailRegion> failing_region;     // a failing copy of a problem of the suite
};

// The lists of the Python modules a config is checked against and a
// campaign's jobs are made from (registry_export.names).
struct RegistryNames {
    std::vector<std::string> baselines, scalar_names, scalar_others, design_kinds, knob_names, list_knobs,
        metric_names, budget_dependent;
    std::vector<std::pair<std::string, std::vector<std::string>>> text_knobs;
};

struct ProblemRegistry {
    std::string code_stamp;
    int reference_version = 0;
    RegistryNames names;
    std::map<std::string, RegistryProblem> problems;
};

namespace detail {

inline std::vector<double> floats_of(const json::Value* v)
{
    std::vector<double> out;
    if (!v || v->kind != json::Value::Kind::Array) return out;
    for (const auto& x : v->items) out.push_back(x.kind == json::Value::Kind::Float ? x.f : static_cast<double>(x.i));
    return out;
}

inline bool flag_of(const json::Value& o, const char* key)
{
    const json::Value* v = o.find(key);
    return v && v->kind == json::Value::Kind::Bool && v->b;
}

inline std::int64_t int_of(const json::Value& o, const char* key, std::int64_t fallback = 0)
{
    const json::Value* v = o.find(key);
    return v && v->kind == json::Value::Kind::Int ? v->i : fallback;
}

}   // namespace detail

// a spec as python/suite_module.cpp's spec_of reads one
inline suite::Spec spec_of(const json::Value& d)
{
    using detail::floats_of;
    using detail::int_of;
    using Kind = json::Value::Kind;
    suite::Spec s;
    s.family = d.find("family")->s;
    s.n_var = static_cast<int>(int_of(d, "n_var"));
    s.n_obj = static_cast<int>(int_of(d, "n_obj"));
    if (const json::Value* v = d.find("map")) s.map = v->s;
    s.first = static_cast<int>(int_of(d, "first"));
    s.centre = floats_of(d.find("centre"));
    s.spread = floats_of(d.find("spread"));
    if (const json::Value* v = d.find("matrices"); v && v->kind == Kind::Array)
        for (const auto& m : v->items) s.matrices.push_back(floats_of(&m));
    s.scale = floats_of(d.find("scale"));
    if (const json::Value* v = d.find("r2")) s.r2 = v->kind == Kind::Float ? v->f : static_cast<double>(v->i);
    if (const json::Value* v = d.find("centre_m")) s.centre_m = v->kind == Kind::Float ? v->f : static_cast<double>(v->i);
    s.lines = floats_of(d.find("lines"));
    if (const json::Value* v = d.find("polygons"); v && v->kind == Kind::Array)
        for (const auto& m : v->items) s.polygons.push_back(floats_of(&m));
    s.vertices = floats_of(d.find("vertices"));
    s.link = floats_of(d.find("link"));
    s.parts = floats_of(d.find("parts"));
    s.k = static_cast<int>(int_of(d, "k"));
    s.constants = floats_of(d.find("constants"));
    s.bbob_fnum = static_cast<int>(int_of(d, "bbob_fnum"));
    s.bbob_iid = static_cast<int>(int_of(d, "bbob_iid", 1));
    if (const json::Value* r = d.find("rotation"); r && r->kind == Kind::Object) {
        s.rotation.family = r->find("family")->s;
        s.rotation.M = static_cast<int>(int_of(*r, "M"));
        s.rotation.instance = static_cast<int>(int_of(*r, "instance", 1));
        s.rotation.rotate = detail::flag_of(*r, "rotate");
        s.rotation.condition = detail::flag_of(*r, "condition");
        for (const auto& x : r->find("sizes")->items) s.rotation.sizes.push_back(static_cast<int>(x.i));
    }
    return s;
}

inline RegistryProblem registry_problem_of(const json::Value& o)
{
    using detail::flag_of;
    using detail::int_of;
    RegistryProblem p;
    p.name = o.find("name")->s;
    if (const json::Value* f = o.find("family")) p.family = f->s;
    p.n_var = static_cast<int>(int_of(o, "n_var"));
    p.n_obj = static_cast<int>(int_of(o, "n_obj"));
    p.n_eq = static_cast<int>(int_of(o, "n_eq"));
    if (const json::Value* v = o.find("n_con"); v && v->kind == json::Value::Kind::Int) p.n_con = static_cast<int>(v->i);
    p.has_cons = flag_of(o, "has_cons");
    for (const auto& b : o.find("bounds")->items) {
        const auto lo = detail::floats_of(&b);
        p.bounds.emplace_back(lo.at(0), lo.at(1));
    }
    p.pop_size = int_of(o, "pop_size");
    p.n_gen = int_of(o, "n_gen");
    p.problem_version = static_cast<int>(int_of(o, "problem_version", 1));
    if (const json::Value* v = o.find("modification")) p.modification = *v;
    p.multimodal_ps = flag_of(o, "multimodal_ps");
    p.has_pareto_front = flag_of(o, "has_pareto_front");
    p.has_pareto_set = flag_of(o, "has_pareto_set");
    p.front_exact = flag_of(o, "front_exact");
    p.shared_reference = flag_of(o, "shared_reference");
    p.dense_reference = flag_of(o, "dense_reference");
    p.stochastic = flag_of(o, "stochastic");
    for (const auto& i : o.find("cyclic_vars")->items) p.cyclic_vars.push_back(static_cast<int>(i.i));
    p.own_evaluator = flag_of(o, "own_evaluator");
    p.typed = flag_of(o, "typed");
    p.failing = flag_of(o, "failing");
    p.distance = flag_of(o, "distance");
    if (const json::Value* v = o.find("spec"); v && v->kind == json::Value::Kind::Object) p.spec = *v;
    if (const json::Value* v = o.find("failing_region"); v && v->kind == json::Value::Kind::Object) {
        FailRegion r;
        r.kind = v->find("kind")->s;
        r.spec = *v->find("spec");
        if (const json::Value* t = v->find("t")) r.t = t->kind == json::Value::Kind::Int ? double(t->i) : t->f;
        if (const json::Value* c = v->find("radius")) r.radius = c->kind == json::Value::Kind::Int ? double(c->i) : c->f;
        if (const json::Value* c = v->find("centres"))
            for (const auto& row : c->items) r.centres.push_back(detail::floats_of(&row));
        p.failing_region = std::move(r);
    }
    return p;
}

inline ProblemRegistry load_registry(const std::filesystem::path& path)
{
    const auto bytes = npz::read_file(path);
    const json::Value doc = json::parse(std::string(bytes.begin(), bytes.end()));
    if (detail::int_of(doc, "format") != 1)
        throw std::runtime_error("registry " + path.string() + ": format " +
                                 std::to_string(detail::int_of(doc, "format")) + " is not read");
    ProblemRegistry r;
    r.code_stamp = doc.find("code_stamp")->s;
    r.reference_version = static_cast<int>(detail::int_of(doc, "reference_version"));
    if (const json::Value* n = doc.find("names"); n && n->kind == json::Value::Kind::Object) {
        auto words = [n](const char* k) {
            std::vector<std::string> out;
            if (const json::Value* v = n->find(k))
                for (const auto& x : v->items) out.push_back(x.s);
            return out;
        };
        r.names.baselines = words("baselines");
        r.names.scalar_names = words("scalar_names");
        r.names.scalar_others = words("scalar_others");
        r.names.design_kinds = words("design_kinds");
        r.names.knob_names = words("knob_names");
        r.names.list_knobs = words("list_knobs");
        r.names.metric_names = words("metric_names");
        r.names.budget_dependent = words("budget_dependent");
        if (const json::Value* v = n->find("text_knobs"))
            for (const auto& pair : v->items) {
                std::vector<std::string> w;
                for (const auto& x : pair.items.at(1).items) w.push_back(x.s);
                r.names.text_knobs.emplace_back(pair.items.at(0).s, std::move(w));
            }
    }
    for (const auto& item : doc.find("problems")->items) {
        RegistryProblem p = registry_problem_of(item);
        r.problems.emplace(p.name, std::move(p));
    }
    return r;
}

// why the C++ engine does not run this problem (the Python engine does), or nothing
inline std::optional<std::string> engine_refusal(const RegistryProblem& p)
{
    if (!p.spec && !p.failing_region) return "not a problem of the C++ suite";
    if (p.own_evaluator) return "an evaluator of the run's own (noise, an uninformative problem)";
    if (p.typed) return "typed variables";
    if (p.failing && !p.failing_region) return "a failing copy";
    if (p.n_eq) return "equality constraints";
    return std::nullopt;
}

// failing.make's evaluate: the base's, but NaN in every objective at a point
// of the failure region (Region.inside). The squared distance to a centre is
// taken as Region.inside takes it, |u|^2 + |c|^2 - 2 u.c, the squares summed in
// NumPy's order; u.c, NumPy's matrix product, in order — the two agree on
// which points fail but within rounding of the region's edge.
class FailingNative : public suite::Native {
public:
    FailingNative(std::shared_ptr<suite::Native> base, FailRegion region,
                  std::vector<std::pair<double, double>> bounds)
        : base_(std::move(base)), region_(std::move(region)), bounds_(std::move(bounds))
    {
        n_var = base_->n_var;
        n_obj = base_->n_obj;
        n_con = base_->n_con;
        for (const auto& c : region_.centres) {
            std::vector<double> sq(c.size());
            for (std::size_t j = 0; j < c.size(); ++j) sq[j] = c[j] * c[j];
            cc_.push_back(np::pairwise_sum(sq.data(), sq.size()));
        }
    }

    void evaluate(const double* x, double* f, double* g) const override
    {
        if (!inside(x)) {
            base_->evaluate(x, f, g);
            return;
        }
        for (int j = 0; j < n_obj; ++j) f[j] = std::numeric_limits<double>::quiet_NaN();
        for (int j = 0; j < n_con; ++j) g[j] = std::numeric_limits<double>::quiet_NaN();
    }

    bool inside(const double* x) const
    {
        const std::size_t n = static_cast<std::size_t>(n_var);
        std::vector<double> u(n), sq(n);
        for (std::size_t j = 0; j < n; ++j) {
            u[j] = (x[j] - bounds_[j].first) / (bounds_[j].second - bounds_[j].first);
            sq[j] = u[j] * u[j];
        }
        if (region_.kind == "b") {
            for (double v : u)
                if (v < region_.t || v > 1.0 - region_.t) return true;
            return false;
        }
        if (region_.centres.empty()) return false;
        const double uu = np::pairwise_sum(sq.data(), n);
        double best = std::numeric_limits<double>::infinity();
        for (std::size_t k = 0; k < region_.centres.size(); ++k) {
            double dot = 0.0;
            for (std::size_t j = 0; j < n; ++j) dot += u[j] * region_.centres[k][j];
            best = std::min(best, uu + cc_[k] - 2.0 * dot);
        }
        return best < region_.radius * region_.radius;
    }

private:
    std::shared_ptr<suite::Native> base_;
    FailRegion region_;
    std::vector<std::pair<double, double>> bounds_;
    std::vector<double> cc_;                     // |c|^2 of each centre
};

// the problem's evaluation in C++, built from its spec (a failing copy: its
// base's, failing in the region)
inline std::shared_ptr<suite::Native> make_native(const RegistryProblem& p)
{
    if (p.failing_region) {
        std::shared_ptr<suite::Native> base(suite::make(spec_of(p.failing_region->spec)));
        return std::make_shared<FailingNative>(std::move(base), *p.failing_region, p.bounds);
    }
    if (!p.spec) throw std::invalid_argument(p.name + ": not a problem of the C++ suite");
    return std::shared_ptr<suite::Native>(suite::make(spec_of(*p.spec)));
}

}   // namespace mootation::campaign

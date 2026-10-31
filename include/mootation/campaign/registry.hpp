#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// The problem registry read in C++ — task 17, stage 5, step 4: the file
// run/registry_export.py writes (the repository's data/registry.json, or
// <refs root>/registry/<code stamp>-<hash>.json):
// every problem's sizes and box, default population and generations, version, what
// it modifies, its references, and — for a problem of the C++ suite — the spec
// its evaluation is built from (suite::Spec, as python/suite_module.cpp's
// spec_of reads it from Python). The engine runs a problem of the suite whose
// run needs no Python (engine_refusal says why not, where not).
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "../detmath.hpp"
#include "../io/npz.hpp"
#include "../json.hpp"
#include "../numpy_sum.hpp"
#include "../random.hpp"
#include "../store/crc32c.hpp"
#include "../suite/instances.hpp"
#include "../suite/suite.hpp"

namespace mootation::campaign {

// A failing copy's failure region (benchmarks/failing.py; task 17, addendum
// 7, section 2.3), in the box's normalised units, (x - lb) / (ub - lb): "b"
// the band within t of the faces; "r" the union of kFailBalls balls of one
// radius about the centres; "n" the points within kFailTube of W = [lo, hi],
// the part of the base's Pareto set (a box) whose first position variable
// lies in a window about 1/2. Made from the name and the base (detail::
// derive_numbers) where the file holds none. With the base's spec.
struct FailRegion {
    std::string of, kind;                         // the base's name; the region's kind
    double t = 0.0, radius = 0.0;
    std::vector<std::vector<double>> centres;
    std::vector<double> lo, hi;                   // "n": W
    json::Value spec;
};

constexpr int kFailBalls = 4;                     // failing.FAIL_BALLS
constexpr double kFailTube = 0.05;                // failing.TUBE
constexpr std::size_t kFailProbe = std::size_t{1} << 14;

// "r": the centres uniform in the unit box and the radius whose balls take
// `share` of a probe of 2^14 uniform points (bisection on the radius), all
// drawn by the library's generator seeded with instances::seed of 20261011
// and the CRC-32 of the copy's name — the centres first, then the probe
inline void failing_balls(const std::string& name, std::size_t n, double share,
                          std::vector<std::vector<double>>& centres, double& radius)
{
    std::mt19937 g(suite::instances::seed(
        {20261011u, store::crc32_zlib(reinterpret_cast<const std::uint8_t*>(name.data()), name.size())}));
    centres.assign(kFailBalls, std::vector<double>(n));
    for (auto& c : centres)
        for (double& v : c) v = rnd::uniform01(g);
    std::vector<double> d(kFailProbe), u(n);
    for (std::size_t i = 0; i < kFailProbe; ++i) {
        for (double& v : u) v = rnd::uniform01(g);
        double best = std::numeric_limits<double>::infinity();
        for (const auto& c : centres) {
            double s = 0.0;
            for (std::size_t j = 0; j < n; ++j) s += (u[j] - c[j]) * (u[j] - c[j]);
            best = std::min(best, s);
        }
        d[i] = std::sqrt(best);
    }
    double a = 0.0, b = std::sqrt(static_cast<double>(n));
    for (int it = 0; it < 100; ++it) {
        const double r = (a + b) / 2.0;
        std::size_t in = 0;
        for (double v : d) in += v < r;
        if (static_cast<double>(in) > share * static_cast<double>(kFailProbe)) b = r;
        else a = r;
    }
    radius = (a + b) / 2.0;
}

// "n": W, the base's Pareto set — ZDT's x1 in [0, 1], the others 0; DTLZ's
// first M - 1 in [0, 1], the others 1/2; WFG's k positions in [0, 1], the
// distance variables 0.35 (of 2i, normalised) — its first variable in the
// window [1/2 - share/2 + TUBE, 1/2 + share/2 - TUBE], narrowed by the tube
// so that the region takes `share` of the set (under share = 2 TUBE, none)
inline void failing_window(const suite::Spec& base, double share, std::vector<double>& lo, std::vector<double>& hi)
{
    if (share < 2.0 * kFailTube)
        throw std::invalid_argument("failing copy: a near-front share under 2 TUBE (" + std::to_string(share) + ")");
    const std::size_t n = static_cast<std::size_t>(base.n_var), M = static_cast<std::size_t>(base.n_obj);
    std::size_t free = 0;
    double rest = 0.0;
    if (base.family.rfind("ZDT", 0) == 0) {
        free = 1;
    } else if (base.family.rfind("DTLZ", 0) == 0) {
        free = M - 1;
        rest = 0.5;
    } else if (base.family.rfind("WFG", 0) == 0) {
        free = base.k > 0 ? static_cast<std::size_t>(base.k) : 2 * (M - 1);
        rest = 0.35;
    } else {
        throw std::invalid_argument("failing copy: no near-front region for " + base.family);
    }
    lo.assign(n, rest);
    hi.assign(n, rest);
    for (std::size_t j = 0; j < free; ++j) {
        lo[j] = 0.0;
        hi[j] = 1.0;
    }
    lo[0] = 0.5 - share / 2.0 + kFailTube;
    hi[0] = 0.5 + share / 2.0 - kFailTube;
}

// A noisy copy's noise (benchmarks/noisy.py): its kind (n, d, s, h), sigma,
// the scale sigma (nadir - ideal) per objective and the base's ideal point,
// as the run's evaluator holds them; with the base's spec.
struct NoiseSpec {
    std::string kind;
    double sigma = 0.0;
    std::vector<double> scale, ideal;
    json::Value spec;
};

struct RegistryProblem {
    std::string name;
    std::string family;                           // families.family: the store's pack it goes to
    int n_var = 0, n_obj = 0, n_eq = 0;
    std::optional<int> n_con;                     // the suite's, for a problem of the suite
    bool has_cons = false;
    std::vector<std::pair<double, double>> bounds;       // from the definition (definition_bounds)
    std::vector<std::pair<double, double>> file_bounds;  // the file's, for comparison only
    std::int64_t pop_size = 0, n_gen = 0;
    int problem_version = 1;
    json::Value modification;                     // properties.modification_of: null or a dict
    bool multimodal_ps = false, has_pareto_front = false, has_pareto_set = false, front_exact = true;
    bool shared_reference = false, dense_reference = false, stochastic = false;
    std::vector<int> cyclic_vars;
    bool own_evaluator = false, typed = false, failing = false, distance = false;
    std::optional<json::Value> spec;              // null: not a problem of the suite
    std::optional<FailRegion> failing_region;     // a failing copy of a problem of the suite
    std::optional<NoiseSpec> noise;               // a noisy copy of a problem of the suite
    std::optional<std::string> uninformative;     // an uninformative problem's kind: f1, f2b, f3b, f4b, f5
    json::Value frame_files;                      // RE, CRE: where its authors' ideal and nadir points are
    std::optional<std::string> seeded_stream;     // RE91: its run's draws from the seed (re.RE91Evaluator)
    // integer and discrete variables beside real ones (var_types.Layout): the
    // core's kinds (0 real, 1 integer, 2 discrete) and box; empty for real
    // variables alone
    std::vector<int> var_kinds;
    std::vector<std::pair<double, double>> core_bounds;
    std::vector<std::vector<double>> discrete_values;   // a discrete variable's values, sorted; [] elsewhere
    // every variable's kind (real, int, discrete, bin, cat; empty for real
    // variables alone), a categorical one's labels, the integer genome's
    // levels (gene_levels), and the front of bits cover_gen looks for
    std::vector<std::string> user_kinds;
    std::vector<int> cat_levels, gene_levels;
    std::vector<std::vector<double>> cover_front;
    // the frame (BenchProblem.ideal, nadir), and the samplers of the reference
    // front and the Pareto set (benchmarks/front_specs.py; refs/fronts.hpp,
    // refs/sets.hpp): null where C++ does not build them
    std::vector<double> ideal, nadir;
    json::Value front, pset;
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

inline suite::Spec spec_of(const json::Value& d);

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

// The numbers of a noisy and of a failing copy that come from its name and
// its base (task 17, addendum 7, sections 2.2 and 2.3), where the file holds
// none: noisy.py's level (<base>_<kind><1|5>: sigma 0.01 or 0.05), the scale
// sigma (nadir_i - ideal_i) and the ideal point of the base's frame;
// failing.py's band (<base>_fail<share>b): t from (1 - 2t)^n = 1 - share.
inline void derive_numbers(ProblemRegistry& r)
{
    for (auto& [name, p] : r.problems) {
        if (p.noise && p.noise->scale.empty()) {
            const auto cut = name.rfind('_');
            const std::string level = cut == std::string::npos ? std::string() : name.substr(cut + 1);
            if (level.size() != 2 || (level[1] != '1' && level[1] != '5'))
                throw std::runtime_error("registry: " + name + " is not a noisy copy's name");
            const auto base = r.problems.find(name.substr(0, cut));
            if (base == r.problems.end() || base->second.ideal.empty()) continue;   // its run refuses (make_run_native)
            NoiseSpec& z = *p.noise;
            z.sigma = level[1] == '1' ? 0.01 : 0.05;
            z.ideal = base->second.ideal;
            for (std::size_t i = 0; i < z.ideal.size(); ++i)
                z.scale.push_back(z.sigma * (base->second.nadir.at(i) - z.ideal[i]));
        }
        if (p.failing_region) {
            FailRegion& f = *p.failing_region;
            const auto at = name.rfind("_fail");
            const double share = std::stod(name.substr(at + 5, name.size() - at - 6)) / 100.0;
            if (f.kind == "b" && f.t == 0.0)
                f.t = (1.0 - dm::pow(1.0 - share, 1.0 / static_cast<double>(p.n_var))) / 2.0;
            if (f.kind == "r" && f.centres.empty())
                failing_balls(name, static_cast<std::size_t>(p.n_var), share, f.centres, f.radius);
            if (f.kind == "n" && f.lo.empty()) failing_window(spec_of(f.spec), share, f.lo, f.hi);
        }
    }
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
    s.wide = detail::flag_of(d, "wide");
    s.constants = floats_of(d.find("constants"));
    s.triplet = floats_of(d.find("triplet"));
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

// The box of a problem from its definition in C++ (suite::bounds; task 17,
// supplement 7, section 2.4): a noisy or failing copy its base's, an
// uninformative probe [0, 1]^n (uninformative.py). data/registry.json holds
// the same numbers for comparison only (file_bounds; the test
// tests/test_registry_bounds.cpp). A problem the C++ suite does not have,
// which the C++ engine does not run (engine_refusal), keeps the file's.
inline std::vector<std::pair<double, double>> definition_bounds(const RegistryProblem& p)
{
    const json::Value* spec = p.spec              ? &*p.spec
                            : p.noise             ? &p.noise->spec
                            : p.failing_region    ? &p.failing_region->spec
                                                  : nullptr;
    if (spec != nullptr) return suite::bounds(spec_of(*spec));
    if (p.uninformative) return suite::box(p.n_var, 0.0, 1.0);
    return p.file_bounds;
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
        p.file_bounds.emplace_back(lo.at(0), lo.at(1));
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
    p.ideal = detail::floats_of(o.find("ideal"));
    p.nadir = detail::floats_of(o.find("nadir"));
    if (const json::Value* v = o.find("front")) p.front = *v;
    if (const json::Value* v = o.find("set")) p.pset = *v;
    if (const json::Value* v = o.find("spec"); v && v->kind == json::Value::Kind::Object) p.spec = *v;
    if (const json::Value* v = o.find("failing_region"); v && v->kind == json::Value::Kind::Object) {
        FailRegion r;
        r.kind = v->find("kind")->s;
        if (const json::Value* of = v->find("of")) r.of = of->s;
        r.spec = *v->find("spec");
        if (const json::Value* t = v->find("t")) r.t = t->kind == json::Value::Kind::Int ? double(t->i) : t->f;
        if (const json::Value* c = v->find("radius")) r.radius = c->kind == json::Value::Kind::Int ? double(c->i) : c->f;
        if (const json::Value* c = v->find("centres"))
            for (const auto& row : c->items) r.centres.push_back(detail::floats_of(&row));
        p.failing_region = std::move(r);
    }
    if (const json::Value* v = o.find("noise"); v && v->kind == json::Value::Kind::Object) {
        NoiseSpec z;
        z.kind = v->find("kind")->s;
        // sigma, the scale and the ideal from the name and the base's frame
        // (derive_numbers) where the file holds none (addendum 7, section 2.2)
        if (const json::Value* sg = v->find("sigma"))
            z.sigma = sg->kind == json::Value::Kind::Int ? double(sg->i) : sg->f;
        z.scale = detail::floats_of(v->find("scale"));
        if (const json::Value* i = v->find("ideal"); i && i->kind == json::Value::Kind::Array)
            z.ideal = detail::floats_of(i);
        z.spec = *v->find("spec");
        p.noise = std::move(z);
    }
    if (const json::Value* v = o.find("uninformative"); v && v->kind == json::Value::Kind::String)
        p.uninformative = v->s;
    if (const json::Value* v = o.find("frame_files")) p.frame_files = *v;
    if (const json::Value* v = o.find("seeded_stream"); v && v->kind == json::Value::Kind::String)
        p.seeded_stream = v->s;
    if (const json::Value* v = o.find("var_kinds"); v && v->kind == json::Value::Kind::Array)
        for (const auto& k : v->items) p.var_kinds.push_back(static_cast<int>(k.i));
    if (const json::Value* v = o.find("core_bounds"); v && v->kind == json::Value::Kind::Array)
        for (const auto& b : v->items) {
            const auto lh = detail::floats_of(&b);
            p.core_bounds.emplace_back(lh.at(0), lh.at(1));
        }
    if (const json::Value* v = o.find("discrete_values"); v && v->kind == json::Value::Kind::Array)
        for (const auto& vs : v->items) p.discrete_values.push_back(detail::floats_of(&vs));
    if (const json::Value* v = o.find("user_kinds"); v && v->kind == json::Value::Kind::Array)
        for (const auto& k : v->items) p.user_kinds.push_back(k.s);
    if (const json::Value* v = o.find("cat_levels"); v && v->kind == json::Value::Kind::Array)
        for (const auto& k : v->items) p.cat_levels.push_back(static_cast<int>(k.i));
    if (const json::Value* v = o.find("gene_levels"); v && v->kind == json::Value::Kind::Array)
        for (const auto& k : v->items) p.gene_levels.push_back(static_cast<int>(k.i));
    if (const json::Value* v = o.find("cover_front"); v && v->kind == json::Value::Kind::Array)
        for (const auto& row : v->items) p.cover_front.push_back(detail::floats_of(&row));
    p.bounds = definition_bounds(p);
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
    detail::derive_numbers(r);
    return r;
}

// why the C++ engine does not run this problem (the Python engine does), or nothing
inline std::optional<std::string> engine_refusal(const RegistryProblem& p)
{
    if (!p.spec && !p.failing_region && !p.noise && !p.uninformative) return "not a problem of the C++ suite";
    if (p.own_evaluator && !p.noise && !p.uninformative && !p.seeded_stream) return "an evaluator of the run's own";
    if (p.typed && p.user_kinds.empty()) return "typed variables";
    if (p.has_cons && std::any_of(p.user_kinds.begin(), p.user_kinds.end(),
                                  [](const std::string& k) { return k == "bin" || k == "cat"; }))
        return "constraints with bits or categories";
    if (p.failing && !p.failing_region) return "a failing copy";
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
        if (region_.kind == "n") {                // |u - clamp(u, W)| <= TUBE
            double d2 = 0.0;
            for (std::size_t j = 0; j < n; ++j) {
                const double e = u[j] - std::min(std::max(u[j], region_.lo[j]), region_.hi[j]);
                d2 += e * e;
            }
            return d2 <= kFailTube * kFailTube;
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

// noisy.Evaluator: one run's objectives, the base's values plus s_i z_i, z_i
// a standard normal by Box-Muller from two uniforms of the splitmix64 hash —
// kinds n and h from (the run's seed, the evaluation's number), d from the
// doubles of x, s from the cell floor((x - lb) / (cell (ub - lb))); s_i the
// scale, or sigma (f_i - ideal_i) for kind h. The evaluations are numbered
// in the order they are made: one object per run, on one thread. log and cos
// are the C library's, called through a pointer: MSVC's inline versions
// differ from it (and from CPython's math) in the last bit now and then.
class NoisyNative : public suite::Native {
public:
    NoisyNative(std::shared_ptr<suite::Native> base, NoiseSpec noise, std::vector<std::pair<double, double>> bounds,
                std::uint64_t seed, double cell)
        : base_(std::move(base)), noise_(std::move(noise)), bounds_(std::move(bounds)), seed_(seed)
    {
        n_var = base_->n_var;
        n_obj = base_->n_obj;
        n_con = base_->n_con;
        for (const auto& b : bounds_) width_.push_back(cell * (b.second - b.first));
    }

    void evaluate(const double* x, double* f, double* g) const override
    {
        base_->evaluate(x, f, g);
        std::uint64_t h = 0;
        if (noise_.kind == "d") {
            for (int j = 0; j < n_var; ++j) {
                std::uint64_t bits;
                std::memcpy(&bits, &x[j], sizeof bits);
                h = splitmix64(h ^ bits);
            }
        } else if (noise_.kind == "s") {
            h = kDomainX;
            for (std::size_t j = 0; j < static_cast<std::size_t>(n_var); ++j) {
                const double cell = std::floor((x[j] - bounds_[j].first) / width_[j]);
                h = splitmix64(h ^ static_cast<std::uint64_t>(static_cast<std::int64_t>(cell)));
            }
        }
        for (int i = 0; i < n_obj; ++i) {
            const std::uint64_t two_i = 2 * static_cast<std::uint64_t>(i);
            double u1, u2;
            if (noise_.kind == "n" || noise_.kind == "h") {
                u1 = unit(splitmix64(seed_ ^ splitmix64(kDomain + 4 * k_ + two_i)));
                u2 = unit(splitmix64(seed_ ^ splitmix64(kDomain + 4 * k_ + two_i + 1)));
            } else {
                u1 = unit(splitmix64(h + two_i + 1));
                u2 = unit(splitmix64(h + two_i + 2));
            }
            const double z = std::sqrt(-2.0 * crt_log_(1.0 - u1)) * crt_cos_(2.0 * kPi * u2);
            const double si = noise_.kind == "h" ? noise_.sigma * (f[i] - noise_.ideal[static_cast<std::size_t>(i)])
                                                 : noise_.scale[static_cast<std::size_t>(i)];
            f[i] = f[i] + si * z;
        }
        ++k_;
    }

    static std::uint64_t splitmix64(std::uint64_t z)
    {
        z += 0x9E3779B97F4A7C15ULL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }

    // 53 high bits of a 64-bit hash as a double in [0, 1)
    static double unit(std::uint64_t h) { return static_cast<double>(h >> 11) * (1.0 / 9007199254740992.0); }

private:
    static constexpr std::uint64_t kDomain = 3ULL << 61;      // noisy._DOMAIN
    static constexpr std::uint64_t kDomainX = 5ULL << 60;     // noisy._DOMAIN_X
    static constexpr double kPi = 3.141592653589793;           // math.pi
    std::shared_ptr<suite::Native> base_;
    NoiseSpec noise_;
    std::vector<std::pair<double, double>> bounds_;
    std::vector<double> width_;
    std::uint64_t seed_;
    mutable std::uint64_t k_ = 0;                              // the evaluations so far
    double (*volatile crt_log_)(double) = ::log;
    double (*volatile crt_cos_)(double) = ::cos;
};

// uninformative.Evaluator: one run's objectives, drawn at random whatever x
// is — the k-th evaluation the k-th draw of the splitmix64 stream of the
// run's seed (f1: two uniforms; f2b-f5: Kudela et al.'s [a, g2(a) + R z^2]
// of a uniform a and a normal z by Box-Muller) — and where the evaluated
// points lie: 20 bins of [0, 1] per coordinate and the points exactly at 0,
// 1 and 1/2, over every point and over the run's non-dominated ones
// (summary(), meta.json's structural_bias). The C library's log, cos, sin
// and pow through pointers, as CPython's math and ** call them.
class UninformativeNative : public suite::Native {
public:
    UninformativeNative(std::string problem, int n_vars, std::uint64_t seed) : problem_(std::move(problem)), seed_(seed)
    {
        n_var = n_vars;
        n_obj = 2;
        n_con = 0;
    }

    void evaluate(const double* x, double* f, double*) const override
    {
        if (problem_ == "f1") {
            for (std::uint64_t i = 0; i < 2; ++i)
                f[i] = NoisyNative::unit(NoisyNative::splitmix64(seed_ ^ NoisyNative::splitmix64(k_ * 2 + i)));
        } else {
            const double a = draw(0), u1 = draw(1), u2 = draw(2);
            const double z = std::sqrt(-2.0 * log_(1.0 - u1)) * cos_(2.0 * kPi * u2);
            objectives(a, z, f);
        }
        ++k_;
        std::vector<double> v(x, x + n_var);
        all_.add(v);
        front(f, v);
    }

    json::Value summary() const
    {
        Counts nd;
        for (const auto& x : front_x_) nd.add(x);
        json::Value s = json::Value::object();
        s.set("bins", json::Value::integer(kBins));
        s.set("all", all_.as_value());
        s.set("nondominated", nd.as_value());
        s.set("evaluations", json::Value::integer(static_cast<std::int64_t>(k_)));
        return s;
    }

    // the frame (registry._register_uninformative; task 17, addendum 7,
    // section 2.1): f1's the unit square; f2b-f5's the noise-free curve's (z =
    // 0) over a = numpy.linspace(0, 1, 1001), g1 = a in [0, 1]
    void frame(double ideal[2], double nadir[2]) const
    {
        ideal[0] = 0.0;
        nadir[0] = 1.0;
        if (problem_ == "f1") {
            ideal[1] = 0.0;
            nadir[1] = 1.0;
            return;
        }
        ideal[1] = std::numeric_limits<double>::infinity();
        nadir[1] = -std::numeric_limits<double>::infinity();
        for (int i = 0; i <= 1000; ++i) {
            const double a = i == 1000 ? 1.0 : i * (1.0 / 1000.0);
            double f[2];
            objectives(a, 0.0, f);
            ideal[1] = std::min(ideal[1], f[1]);
            nadir[1] = std::max(nadir[1], f[1]);
        }
    }

private:
    static constexpr int kBins = 20;
    static constexpr double kPi = 3.141592653589793;           // math.pi

    struct Counts {
        std::vector<std::int64_t> hist = std::vector<std::int64_t>(kBins, 0);
        std::int64_t points = 0, lower = 0, upper = 0, half = 0;
        void add(const std::vector<double>& x)
        {
            ++points;
            for (double v : x) {
                // min(BINS - 1, max(0, int(v * BINS)))
                const double b = std::trunc(v * kBins);
                hist[static_cast<std::size_t>(b < 0.0 ? 0.0 : b > kBins - 1 ? kBins - 1 : b)] += 1;
                lower += v == 0.0;
                upper += v == 1.0;
                half += v == 0.5;
            }
        }
        json::Value as_value() const
        {
            json::Value d = json::Value::object(), h = json::Value::array();
            for (auto c : hist) h.items.push_back(json::Value::integer(c));
            d.set("points", json::Value::integer(points));
            d.set("hist", h);
            d.set("at_lower", json::Value::integer(lower));
            d.set("at_upper", json::Value::integer(upper));
            d.set("at_half", json::Value::integer(half));
            return d;
        }
    };

    // uniform i (0..3) of the evaluation: a domain apart from f1's
    double draw(std::uint64_t i) const
    {
        return NoisyNative::unit(NoisyNative::splitmix64(seed_ ^ NoisyNative::splitmix64((1ULL << 62) + 4 * k_ + i)));
    }

    // uninformative.objectives: [g1, g2] for a = U(0, 1) and z = N(0, 1)
    void objectives(double a, double z, double* f) const
    {
        f[0] = a;
        if (problem_ == "f5") {
            f[1] = -a;
            return;
        }
        double g2 = 0.0, r = 0.0;
        if (problem_ == "f2b") {
            g2 = 1.0 - sin_(12.0 * kPi * pow_(a, 1.5)) / 4.0 - std::sqrt(a / 2.0);
            r = 0.08;
        } else if (problem_ == "f3b") {
            int b = 0;
            for (int i = 1; i < 10; ++i) b += a > i / 10.0;
            g2 = 0.1 / pow_(10.0 * (a - b / 10.0) + 1.0, 10.0) + (1.0 - b / 10.0);
            r = 0.9;
        } else {                                                   // f4b
            g2 = a < 0.5 ? -2.0 * a * a : -2.0 * pow_(a - 0.5, 2.0) - 0.5;
            r = 2.0;
        }
        f[1] = g2 + r * z * z;
    }

    // the run's non-dominated points so far, by f1 ascending (f2 falls along)
    void front(const double* f, const std::vector<double>& x) const
    {
        const auto i = static_cast<std::size_t>(std::lower_bound(front_f1_.begin(), front_f1_.end(), f[0]) -
                                                front_f1_.begin());
        if (i > 0 && front_f2_[i - 1] <= f[1]) return;
        if (i < front_f1_.size() && front_f1_[i] == f[0] && front_f2_[i] <= f[1]) return;
        std::size_t j = i;
        while (j < front_f1_.size() && front_f2_[j] >= f[1]) ++j;
        const auto at = static_cast<std::ptrdiff_t>(i), to = static_cast<std::ptrdiff_t>(j);
        front_f1_.erase(front_f1_.begin() + at, front_f1_.begin() + to);
        front_f2_.erase(front_f2_.begin() + at, front_f2_.begin() + to);
        front_x_.erase(front_x_.begin() + at, front_x_.begin() + to);
        front_f1_.insert(front_f1_.begin() + at, f[0]);
        front_f2_.insert(front_f2_.begin() + at, f[1]);
        front_x_.insert(front_x_.begin() + at, x);
    }

    std::string problem_;
    std::uint64_t seed_;
    mutable std::uint64_t k_ = 0;
    mutable Counts all_;
    mutable std::vector<double> front_f1_, front_f2_;
    mutable std::vector<std::vector<double>> front_x_;
    double (*volatile log_)(double) = ::log;
    double (*volatile cos_)(double) = ::cos;
    double (*volatile sin_)(double) = ::sin;
    double (*volatile pow_)(double, double) = ::pow;
};

// re.RE91Evaluator: one run's RE91, the k-th evaluation the k-th four
// normals of the stream of the run's seed (splitmix64 and Box-Muller; log and
// cos the C library's, as CPython's math, through a pointer)
class RE91Native : public suite::Native {
public:
    explicit RE91Native(std::uint64_t seed) : seed_(seed)
    {
        n_var = 7;
        n_obj = 9;
        n_con = 0;
    }

    void evaluate(const double* x, double* f, double*) const override
    {
        double z[4];
        for (std::uint64_t i = 0; i < 4; ++i) {
            const double u1 = NoisyNative::unit(NoisyNative::splitmix64(seed_ ^ NoisyNative::splitmix64(kDomain + 8 * k_ + 2 * i)));
            const double u2 =
                NoisyNative::unit(NoisyNative::splitmix64(seed_ ^ NoisyNative::splitmix64(kDomain + 8 * k_ + 2 * i + 1)));
            z[i] = std::sqrt(-2.0 * log_(1.0 - u1)) * cos_(2.0 * 3.141592653589793 * u2);
        }
        ++k_;
        suite::re::re91(x, z, f);
    }

private:
    static constexpr std::uint64_t kDomain = 5ULL << 60;         // re._DOMAIN
    std::uint64_t seed_;
    mutable std::uint64_t k_ = 0;
    double (*volatile log_)(double) = ::log;
    double (*volatile cos_)(double) = ::cos;
};

// the problem's evaluation in C++, built from its spec (a failing copy: its
// base's, failing in the region; a noisy copy: its base's, noise-free — the
// run's noisy evaluation is make_run_native's)
inline std::shared_ptr<suite::Native> make_native(const RegistryProblem& p)
{
    if (p.noise) return std::shared_ptr<suite::Native>(suite::make(spec_of(p.noise->spec)));
    if (p.failing_region) {
        std::shared_ptr<suite::Native> base(suite::make(spec_of(p.failing_region->spec)));
        return std::make_shared<FailingNative>(std::move(base), *p.failing_region, p.bounds);
    }
    if (!p.spec) throw std::invalid_argument(p.name + ": not a problem of the C++ suite");
    return std::shared_ptr<suite::Native>(suite::make(spec_of(*p.spec)));
}

// what the core of one run evaluates: the problem (make_native), a noisy
// copy's with the run's noise (seed: the run's; cell: [campaign] noise_cell)
inline std::shared_ptr<suite::Native> make_run_native(const RegistryProblem& p, std::uint64_t seed, double cell)
{
    if (p.noise) {
        if (p.noise->scale.empty())
            throw std::runtime_error("registry: " + p.name + ": its base's frame is not in the registry");
        return std::make_shared<NoisyNative>(make_native(p), *p.noise, p.bounds, seed, cell);
    }
    if (p.uninformative) return std::make_shared<UninformativeNative>(*p.uninformative, p.n_var, seed);
    if (p.seeded_stream && *p.seeded_stream == "RE91") return std::make_shared<RE91Native>(seed);
    return make_native(p);
}

}   // namespace mootation::campaign

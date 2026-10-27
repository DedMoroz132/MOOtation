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
#include <cmath>
#include <cstdint>
#include <cstring>
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
    std::optional<NoiseSpec> noise;               // a noisy copy of a problem of the suite
    std::optional<std::string> uninformative;     // an uninformative problem's kind: f1, f2b, f3b, f4b, f5
    // integer variables beside real ones (var_types.Layout): the core's kinds
    // (0 real, 1 integer) and box; empty for real variables alone
    std::vector<int> var_kinds;
    std::vector<std::pair<double, double>> core_bounds;
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
    if (const json::Value* v = o.find("noise"); v && v->kind == json::Value::Kind::Object) {
        NoiseSpec z;
        z.kind = v->find("kind")->s;
        const json::Value* sg = v->find("sigma");
        z.sigma = sg->kind == json::Value::Kind::Int ? double(sg->i) : sg->f;
        z.scale = detail::floats_of(v->find("scale"));
        if (const json::Value* i = v->find("ideal"); i && i->kind == json::Value::Kind::Array)
            z.ideal = detail::floats_of(i);
        z.spec = *v->find("spec");
        p.noise = std::move(z);
    }
    if (const json::Value* v = o.find("uninformative"); v && v->kind == json::Value::Kind::String)
        p.uninformative = v->s;
    if (const json::Value* v = o.find("var_kinds"); v && v->kind == json::Value::Kind::Array)
        for (const auto& k : v->items) p.var_kinds.push_back(static_cast<int>(k.i));
    if (const json::Value* v = o.find("core_bounds"); v && v->kind == json::Value::Kind::Array)
        for (const auto& b : v->items) {
            const auto lh = detail::floats_of(&b);
            p.core_bounds.emplace_back(lh.at(0), lh.at(1));
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
    if (!p.spec && !p.failing_region && !p.noise && !p.uninformative) return "not a problem of the C++ suite";
    if (p.own_evaluator && !p.noise && !p.uninformative) return "an evaluator of the run's own";
    if (p.typed && p.var_kinds.empty()) return "typed variables (bits or categories)";
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
    if (p.noise) return std::make_shared<NoisyNative>(make_native(p), *p.noise, p.bounds, seed, cell);
    if (p.uninformative) return std::make_shared<UninformativeNative>(*p.uninformative, p.n_var, seed);
    return make_native(p);
}

}   // namespace mootation::campaign

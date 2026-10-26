#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// A campaign's settings and its jobs — task 17, stage 5: run/campaign.py's
// campaign_spec(), fit_pop(), expand_jobs() and shard_of(), with the same
// messages and the same job list, job for job (index, problem, algorithm,
// seed, population, generations, budget). The Python stays as the
// cross-check.
//
// What the library knows and the config does not — the metric names, each
// problem's objectives, population and generations, the algorithms whose run
// depends on the budget — the caller gives (JobsRegistry), as config.hpp's
// Registry.
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "../json.hpp"
#include "../store/pack.hpp"
#include "../store/sha256.hpp"
#include "config.hpp"

namespace mootation::campaign {

struct JobsRegistry {
    std::vector<std::string> metric_names;                // metric_names.METRIC_NAMES, in order
    struct Problem {
        int n_obj = 0;
        std::int64_t pop_size = 0, n_gen = 0;
    };
    std::map<std::string, Problem> problems;              // the benchmarks
    std::vector<std::string> budget_dependent;            // budget.BUDGET_DEPENDENT
    std::vector<std::string> scalar_names, scalar_others; // scalar.NAMES, scalar.OTHERS
};

// run/campaign.py's CampaignSpec: the fields, their defaults and order
struct CampaignSpec {
    std::string out = "results/{name}";
    std::int64_t budget_fe = 0, record_every = 1;
    std::vector<std::string> metrics = {"igd", "igdp", "hv"}, final_metrics;
    std::int64_t n_ref = 1000;
    bool dense_reference = true, set_metrics = true;
    std::string set_answer = "evaluated";
    std::vector<std::int64_t> seeds;
    std::string record_grid = "budgets";
    std::int64_t record_per_decade = 10;
    std::vector<std::int64_t> record_at;
    std::vector<std::string> record_at_metrics;
    std::vector<std::int64_t> ladder;
    std::int64_t ladder_min_gens = 0;
    std::int64_t trajectory_hv_max_m = 0, trajectory_hv_mc_samples = 0;
    std::int64_t hv_exact_max_m = 10, hv_grid_exact_max_m = 8, hv_archive_grid_max_m = 6,
                 hv_archive_final_max_m = 8, hv_mc_samples = 100000;
    bool archive_scenario = true;
    std::string answer_select = "dss";
    bool operator_stats = false, coords_diag = false, archive = true;
    double archive_delta = 0.0;
    std::string archive_mode = "online", native = "auto";
    Value noise_cell = Value::number(0.01);               // as the config gave it: an int stays one
    std::string archive_variables = "all";
    std::int64_t archive_cap = 0;
    bool all_evaluated = true;
    std::vector<double> all_levels = {1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001};
    Value snapshots = Value::boolean(false);              // true/false, or the problems' names
    std::string snapshot_variables = "none";
    bool archive_checkpoints = false;
    std::vector<std::int64_t> hard_budgets;
    std::int64_t answer_k = 0;
    std::string pop_fit = "below", storage = "files", keep = "standard";
    std::int64_t segment_mb = 1024;
    double run_timeout = 0.0;
};

struct Job {
    std::int64_t index = 0;
    std::string problem, algorithm;
    std::int64_t seed = 0, pop = 0, gens = 0;
    Value params = Value::object();
    int n_objs = 0;
    std::string pop_note, label;
    std::int64_t evaluations = 0;
    std::string ladder_of, init = "uniform";
    double init_share = 0.0;

    std::string key() const { return label.empty() ? algorithm : label; }
    std::int64_t budget() const { return evaluations ? evaluations : pop * gens; }
    // problem/key/run_seed, as Path.as_posix() writes it
    std::string rel_dir() const { return problem + "/" + key() + "/run_" + std::to_string(seed); }
};

// the part of k a job belongs to (--shard i/k): the first 8 bytes of the
// SHA-256 of its rel_dir, big-endian, mod k
inline std::int64_t shard_of(const Job& job, std::int64_t k)
{
    const std::string dir = job.rel_dir();
    const auto h = store::sha256(reinterpret_cast<const std::uint8_t*>(dir.data()), dir.size());
    std::uint64_t v = 0;
    for (int j = 0; j < 8; ++j) v = (v << 8) | h[static_cast<std::size_t>(j)];
    return static_cast<std::int64_t>(v % static_cast<std::uint64_t>(k));
}

namespace detail {

inline bool is_count(const Value& v) { return v.kind == Kind::Int; }

// [campaign] key: an integer (true/false, a float or a word is an error)
inline std::int64_t integer(const Value& raw, const std::string& key, std::int64_t fallback)
{
    const Value* v = raw.find(key);
    if (!v) return fallback;
    if (!is_count(*v)) throw ConfigError("campaign." + key, "an integer, not " + py_repr(*v));
    return v->i;
}

inline bool flag(const Value& raw, const std::string& key, bool fallback)
{
    const Value* v = raw.find(key);
    if (!v) return fallback;
    if (v->kind != Kind::Bool) throw ConfigError("campaign." + key, "true or false");
    return v->b;
}

// a word out of `words`; the message lists them quoted as Python does
inline std::string word(const Value& raw, const std::string& key, const std::string& fallback,
                        const std::vector<std::string>& words, const std::string& message)
{
    const Value* v = raw.find(key);
    if (!v) return fallback;
    if (v->kind == Kind::String && !v->datetime)
        for (const auto& w : words)
            if (v->s == w) return w;
    throw ConfigError("campaign." + key, message + py_repr(*v));
}

// a list of positive evaluation counts, sorted, without repeats
inline std::vector<std::int64_t> counts(const Value& raw, const std::string& key, const std::string& message)
{
    const Value* v = raw.find(key);
    if (!v) return {};
    bool ok = v->kind == Kind::Array;
    for (const auto& x : v->items) ok = ok && is_count(x) && x.i > 0;
    if (!ok) throw ConfigError("campaign." + key, message + py_repr(*v));
    std::set<std::int64_t> s;
    for (const auto& x : v->items) s.insert(x.i);
    return std::vector<std::int64_t>(s.begin(), s.end());
}

inline bool truthy(const Value* x)
{
    if (!x) return false;
    switch (x->kind) {
    case Kind::Null: return false;
    case Kind::Bool: return x->b;
    case Kind::Int: return x->i != 0;
    case Kind::Float: return x->f != 0.0;
    case Kind::String: return !x->s.empty();
    case Kind::Array: return !x->items.empty();
    case Kind::Object: return !x->keys.empty();
    }
    return false;
}

inline bool contains(const std::vector<std::string>& words, const Value& v)
{
    if (v.kind != Kind::String || v.datetime) return false;
    for (const auto& w : words)
        if (w == v.s) return true;
    return false;
}

}   // namespace detail

// spec.out with {name} put in (str.format with the one field), or nothing
// when it holds another field
inline std::optional<std::string> format_out(const std::string& out, const std::string& name)
{
    std::string r;
    for (std::size_t k = 0; k < out.size(); ++k) {
        const char c = out[k];
        if ((c == '{' || c == '}') && k + 1 < out.size() && out[k + 1] == c) {
            r += c;
            ++k;
        } else if (c == '{') {
            const auto end = out.find('}', k);
            if (end == std::string::npos || out.substr(k + 1, end - k - 1) != "name") return std::nullopt;
            r += name;
            k = end;
        } else if (c == '}') {
            return std::nullopt;
        } else {
            r += c;
        }
    }
    return r;
}

// out_root(): spec.out under the config's folder (the working one without a file)
inline std::optional<std::filesystem::path> out_root(const Config& cfg, const CampaignSpec& spec)
{
    const auto out = format_out(spec.out, cfg.name);
    if (!out) return std::nullopt;
    const std::filesystem::path p = store::file::path_of(*out);
    if (p.is_absolute()) return p;
    const std::filesystem::path base = cfg.source_path ? cfg.source_path->parent_path() : std::filesystem::current_path();
    return base / p;
}

// the record_grid a results folder was started with (its campaign.json), if it was
inline std::optional<std::string> started_grid(const Config& cfg, const CampaignSpec& spec)
{
    const auto root = out_root(cfg, spec);
    if (!root) return std::nullopt;
    const std::filesystem::path path = *root / "campaign.json";
    std::error_code ec;
    if (!std::filesystem::is_regular_file(path, ec)) return std::nullopt;
    std::ifstream in(path, std::ios::binary);
    std::stringstream text;
    text << in.rdbuf();
    try {
        const Value got = json::parse(text.str());
        const Value* s = got.kind == Kind::Object ? got.find("spec") : nullptr;
        const Value* g = s && s->kind == Kind::Object ? s->find("record_grid") : nullptr;
        return detail::truthy(g) ? detail::py_str(*g) : std::string("generations");
    } catch (const std::exception&) {
        return std::nullopt;
    }
}

// run/campaign.py's campaign_spec()
inline CampaignSpec campaign_spec(const Config& cfg, const JobsRegistry& reg)
{
    using namespace detail;
    const Value raw = cfg.campaign.kind == Kind::Object ? cfg.campaign : Value::object();
    CampaignSpec spec;
    const std::vector<std::string> known = {
        "out", "budget_fe", "record_every", "metrics", "final_metrics", "n_ref", "seeds",
        "record_grid", "record_per_decade", "record_at", "record_at_metrics", "ladder",
        "trajectory_hv_max_m", "trajectory_hv_mc_samples", "hv_exact_max_m", "hv_mc_samples",
        "hv_grid_exact_max_m", "hv_archive_grid_max_m", "hv_archive_final_max_m",
        "archive", "archive_delta", "archive_variables", "archive_scenario", "snapshots",
        "snapshot_variables", "archive_checkpoints", "operator_stats", "hard_budgets",
        "answer_k", "pop_fit", "ladder_min_gens", "dense_reference", "answer_select",
        "noise_cell", "coords_diag", "set_metrics", "set_answer", "all_evaluated", "all_levels",
        "archive_mode", "native", "storage", "keep", "segment_mb", "run_timeout", "archive_cap"};
    const std::set<std::string> known_set(known.begin(), known.end());
    std::set<std::string> unknown;
    for (const auto& k : raw.keys)
        if (!known_set.count(k)) unknown.insert(k);
    if (!unknown.empty())
        throw ConfigError("campaign", "unknown key(s): " + join({unknown.begin(), unknown.end()}, ", ") +
                                          ". Known: " + join({known_set.begin(), known_set.end()}, ", "));
    if (const Value* v = raw.find("out")) {
        if (v->kind != Kind::String || v->datetime) throw ConfigError("campaign.out", "a path, not " + py_repr(*v));
        spec.out = v->s;
    }
    spec.budget_fe = integer(raw, "budget_fe", 0);
    spec.record_every = integer(raw, "record_every", 1);
    for (const std::string key : {"metrics", "final_metrics"}) {
        const Value* names = raw.find(key);
        if (!names) continue;
        if (names->kind != Kind::Array)
            throw ConfigError("campaign." + key, "a list of metric names, not " + py_repr(*names));
        std::vector<std::string> got;
        for (const auto& m : names->items) {
            if (!contains(reg.metric_names, m))
                throw ConfigError("campaign." + key,
                                  "unknown metric '" + py_str(m) + "'; known: " + join(reg.metric_names, ", "));
            got.push_back(m.s);
        }
        (key == "metrics" ? spec.metrics : spec.final_metrics) = got;
    }
    spec.n_ref = integer(raw, "n_ref", 1000);
    spec.dense_reference = flag(raw, "dense_reference", true);
    spec.set_metrics = flag(raw, "set_metrics", true);
    spec.set_answer = word(raw, "set_answer", "evaluated", {"off", "population", "evaluated"},
                           "\"off\", \"population\" or \"evaluated\", not ");
    if (const Value* v = raw.find("seeds")) {
        bool ok = v->kind == Kind::Array;
        for (const auto& s : v->items) ok = ok && is_count(s);
        if (!ok) throw ConfigError("campaign.seeds", "a list of integers, not " + py_repr(*v));
        for (const auto& s : v->items) spec.seeds.push_back(s.i);
    }
    // the default grid: "budgets"; "generations" with hard budgets; the grid
    // the results folder was started with, so that a resumed campaign keeps it
    std::string default_grid = truthy(raw.find("hard_budgets")) ? "generations" : "budgets";
    if (!raw.find("record_grid"))
        if (const auto g = started_grid(cfg, spec)) default_grid = *g;
    const Value* grid = raw.find("record_grid");
    spec.record_per_decade = integer(raw, "record_per_decade", 10);
    spec.trajectory_hv_max_m = integer(raw, "trajectory_hv_max_m", 0);
    spec.record_grid = grid ? py_str(*grid) : default_grid;
    if (!(grid ? contains({"budgets", "generations", "log"}, *grid)
               : (default_grid == "budgets" || default_grid == "generations" || default_grid == "log")))
        throw ConfigError("campaign.record_grid", "'budgets', 'generations' or 'log', not '" + spec.record_grid + "'");
    if (spec.record_per_decade < 1) throw ConfigError("campaign.record_per_decade", "must be >= 1");
    spec.record_at = counts(raw, "record_at", "a list of positive evaluation counts, not ");
    if (!spec.record_at.empty() && spec.record_grid != "log" && spec.record_grid != "budgets")
        throw ConfigError("campaign.record_at", "needs record_grid = \"budgets\" or \"log\"");
    if (const Value* sparse = raw.find("record_at_metrics")) {
        bool ok = sparse->kind == Kind::Array;
        for (const auto& m : sparse->items) ok = ok && contains(spec.metrics, m);
        if (!ok) throw ConfigError("campaign.record_at_metrics", "a list of names from `metrics`, not " + py_repr(*sparse));
        if (!sparse->items.empty() && spec.record_at.empty())
            throw ConfigError("campaign.record_at_metrics", "needs record_at");
        for (const auto& m : sparse->items) spec.record_at_metrics.push_back(m.s);
    }
    spec.ladder = counts(raw, "ladder", "a list of positive evaluation counts, not ");
    if (spec.trajectory_hv_max_m < 0)
        throw ConfigError("campaign.trajectory_hv_max_m", "must be >= 0 (0 = no limit)");
    const std::vector<std::pair<std::string, std::int64_t*>> bounded = {
        {"trajectory_hv_mc_samples", &spec.trajectory_hv_mc_samples}, {"hv_exact_max_m", &spec.hv_exact_max_m},
        {"hv_mc_samples", &spec.hv_mc_samples}, {"hv_grid_exact_max_m", &spec.hv_grid_exact_max_m},
        {"hv_archive_grid_max_m", &spec.hv_archive_grid_max_m},
        {"hv_archive_final_max_m", &spec.hv_archive_final_max_m}};
    for (const auto& [key, field] : bounded) {
        const std::int64_t lo = key == "hv_mc_samples" ? 1 : 0;
        if (const Value* v = raw.find(key)) {
            if (!is_count(*v) || v->i < lo)
                throw ConfigError("campaign." + key, "an integer >= " + std::to_string(lo) + ", not " + py_repr(*v));
            *field = v->i;
        }
    }
    spec.operator_stats = flag(raw, "operator_stats", false);
    spec.coords_diag = flag(raw, "coords_diag", false);
    spec.archive_scenario = flag(raw, "archive_scenario", true);
    spec.archive = flag(raw, "archive", true);
    if (const Value* v = raw.find("archive_delta")) {
        if (v->kind != Kind::Int && v->kind != Kind::Float)
            throw ConfigError("campaign.archive_delta", "a number, not " + py_repr(*v));
        spec.archive_delta = as_float(*v);
    }
    spec.archive_mode = word(raw, "archive_mode", "online", {"online", "posthoc"}, "\"online\" or \"posthoc\", not ");
    spec.native = word(raw, "native", "auto", {"auto", "off"}, "\"auto\" or \"off\", not ");
    if (spec.archive_delta < 0) throw ConfigError("campaign.archive_delta", "must be >= 0 (0 = the default step)");
    if (const Value* v = raw.find("noise_cell")) {
        const bool number = v->kind == Kind::Int || v->kind == Kind::Float;
        const double x = number ? as_float(*v) : 0.0;
        if (!number || !(0 < x && x <= 1))
            throw ConfigError("campaign.noise_cell", "a share in (0, 1], not " + py_repr(*v));
        spec.noise_cell = *v;
    }
    if (const Value* v = raw.find("archive_cap")) {
        if (!is_count(*v) || v->i < 0)
            throw ConfigError("campaign.archive_cap", "an integer >= 0 (x N; 0: the whole archive), not " + py_repr(*v));
        spec.archive_cap = v->i;
    }
    spec.archive_variables = word(raw, "archive_variables", "all", {"all", "selected"}, "\"all\" or \"selected\", not ");
    if (const Value* v = raw.find("snapshots")) {
        if (v->kind == Kind::Bool) {
            spec.snapshots = *v;
        } else if (v->kind == Kind::Array) {
            spec.snapshots = Value::array();
            for (const auto& s : v->items) spec.snapshots.items.push_back(Value::string(py_str(s)));
        } else {
            throw ConfigError("campaign.snapshots", "true, false, or a list of problem names");
        }
    }
    spec.snapshot_variables = word(raw, "snapshot_variables", "none", {"none", "pareto_set", "all"},
                                   "\"none\", \"pareto_set\" or \"all\", not ");
    spec.archive_checkpoints = flag(raw, "archive_checkpoints", false);
    spec.all_evaluated = flag(raw, "all_evaluated", true);
    if (const Value* v = raw.find("all_levels")) {
        bool ok = v->kind == Kind::Array && !v->items.empty();
        for (const auto& x : v->items) ok = ok && (x.kind == Kind::Int || x.kind == Kind::Float) && as_float(x) >= 0;
        if (!ok) throw ConfigError("campaign.all_levels", "a list of levels of igdp_norm >= 0, not " + py_repr(*v));
        spec.all_levels.clear();
        for (const auto& x : v->items) spec.all_levels.push_back(as_float(x));
    }
    if (spec.record_every < 1) throw ConfigError("campaign.record_every", "must be >= 1");
    spec.hard_budgets = counts(raw, "hard_budgets", "a list of positive evaluation counts, not ");
    if (!spec.hard_budgets.empty() && (spec.record_grid != "generations" || spec.record_every != 1))
        throw ConfigError("campaign.hard_budgets", "reads the last record at or below each budget: needs a record "
                                                   "every generation (record_grid = \"generations\", record_every = 1)");
    if (const Value* v = raw.find("answer_k")) {
        if (!is_count(*v) || v->i < 0) throw ConfigError("campaign.answer_k", "an integer >= 0, not " + py_repr(*v));
        spec.answer_k = v->i;
    }
    spec.answer_select = word(raw, "answer_select", "dss", {"dss", "igdp_self"}, "\"dss\" or \"igdp_self\", not ");
    spec.pop_fit = word(raw, "pop_fit", "below", {"below", "nearest"}, "\"below\" or \"nearest\", not ");
    if (const Value* v = raw.find("ladder_min_gens")) {
        if (!is_count(*v) || v->i < 0)
            throw ConfigError("campaign.ladder_min_gens", "an integer >= 0, not " + py_repr(*v));
        spec.ladder_min_gens = v->i;
    }
    if (spec.budget_fe < 0) throw ConfigError("campaign.budget_fe", "must be >= 0");
    spec.storage = word(raw, "storage", "files", {"files", "store"}, "\"files\" or \"store\", not ");
    spec.keep = word(raw, "keep", "standard", {"standard", "full"}, "\"standard\" or \"full\", not ");
    if (const Value* v = raw.find("segment_mb")) {
        if (!is_count(*v) || v->i < 1) throw ConfigError("campaign.segment_mb", "an integer >= 1, not " + py_repr(*v));
        spec.segment_mb = v->i;
    }
    if (const Value* v = raw.find("run_timeout")) {
        if ((v->kind != Kind::Int && v->kind != Kind::Float) || as_float(*v) < 0)
            throw ConfigError("campaign.run_timeout", "seconds >= 0 (0: no limit), not " + py_repr(*v));
        spec.run_timeout = as_float(*v);
    }
    return spec;
}

// fit_pop(): the population this algorithm accepts, and a note if it changed;
// rule "below" — the largest accepted size up to pop (the smallest above when
// none), "nearest" — the nearest, the smaller of two as near
inline std::pair<std::int64_t, std::string> fit_pop(const std::string& name, std::int64_t pop, std::int64_t m,
                                                     const Value& params, const std::string& rule)
{
    if (pop::free_h().count(name)) {
        const Value* r = params.find("n_ref_points");
        if (detail::truthy(r)) return {pop, ""};
    }
    const std::string M = std::to_string(m);
    if (pop::exact_lattice().count(name)) {
        // the single- and two-layer lattice sizes up to 2 pop
        std::set<std::int64_t> ok;
        const std::int64_t limit = 2 * pop;
        if (m < 2) {
            if (limit >= 1) ok.insert(1);
            if (limit >= 2) ok.insert(2);
        } else {
            for (std::int64_t h = 1; pop::das_dennis_count(m, h) <= limit; ++h) ok.insert(pop::das_dennis_count(m, h));
            for (std::int64_t h1 = 1; pop::das_dennis_count(m, h1) <= limit; ++h1)
                for (std::int64_t h2 = 1; h2 < h1; ++h2) {
                    const std::int64_t s = pop::das_dennis_count(m, h1) + pop::das_dennis_count(m, h2);
                    if (s <= limit) ok.insert(s);
                }
        }
        if (ok.count(pop)) return {pop, ""};
        if (rule == "nearest") {
            std::set<std::int64_t> all = ok;
            if (const auto above = pop::nearest(m, pop).second) all.insert(*above);
            std::int64_t near = *all.begin();
            for (const std::int64_t s : all)                  // ascending: the smaller of two as near kept
                if (std::llabs(s - pop) < std::llabs(near - pop)) near = s;
            return {near, "pop " + std::to_string(pop) + " -> " + std::to_string(near) +
                              " (nearest lattice size for M=" + M + ")"};
        }
        std::optional<std::int64_t> below;
        for (const std::int64_t s : ok)
            if (s <= pop) below = s;
        if (!below) {
            const auto above = pop::nearest(m, pop).second;
            return {above.value_or(0), "pop " + std::to_string(pop) + " -> " +
                                           (above ? std::to_string(*above) : "None") +
                                           " (smallest Das-Dennis lattice for M=" + M + ")"};
        }
        return {*below, "pop " + std::to_string(pop) + " -> " + std::to_string(*below) +
                            " (largest lattice size <= " + std::to_string(pop) + " for M=" + M + ")"};
    }
    const auto kd = pop::k_divisible().find(name);
    if (kd != pop::k_divisible().end()) {
        const Value* kv = params.find("K");
        if (!kv) kv = params.find("n_clusters");
        const std::int64_t k = !kv ? kd->second
                               : kv->kind == Kind::Float ? static_cast<std::int64_t>(kv->f)
                               : kv->kind == Kind::Bool  ? (kv->b ? 1 : 0)
                                                         : kv->i;
        if (pop % k == 0) return {pop, ""};
        std::int64_t fitted = std::max(k, (pop / k) * k);
        if (rule == "nearest" && pop - fitted > fitted + k - pop) fitted += k;
        return {fitted, "pop " + std::to_string(pop) + " -> " + std::to_string(fitted) + " (multiple of K=" +
                            std::to_string(k) + ")"};
    }
    return {pop, ""};
}

// budget.depends_on_budget(): a run that changes with its budget
inline bool depends_on_budget(const JobsRegistry& reg, const std::string& name, const Value& params)
{
    auto in = [&name](const std::vector<std::string>& v) { return std::find(v.begin(), v.end(), name) != v.end(); };
    if (in(reg.budget_dependent)) return true;
    if (in(reg.scalar_names) && !in(reg.scalar_others)) return true;
    const Value* polish = params.find("polish");
    if (polish && detail::py_str(*polish) != "off") return true;
    if (name == "smpso") {
        const Value* t = params.find("turbulence");
        if (detail::truthy(t)) return detail::py_str(*t) == "thirds";
        const Value* preset = params.find("smpso_preset");
        return preset && preset->kind == Kind::String && preset->s == "omopso";
    }
    return false;
}

// expand_jobs(): every problem x algorithm x seed, then the budget ladder's rungs
inline std::vector<Job> expand_jobs(const Config& cfg, const CampaignSpec& spec, const JobsRegistry& reg)
{
    if (cfg.benchmark_problems.empty()) throw ConfigError("benchmarks", "the campaign selects no problems");
    std::vector<std::int64_t> seeds = spec.seeds;
    if (seeds.empty()) {
        const Value* runs = cfg.benchmarks.kind == Kind::Object ? cfg.benchmarks.find("runs") : nullptr;
        const std::int64_t n = runs ? detail::as_int(*runs) : 31;
        for (std::int64_t s = 1; s <= n; ++s) seeds.push_back(s);
    }
    auto ceil_div = [](std::int64_t a, std::int64_t b) { return (a + b - 1) / b; };
    auto takes = [](const Algorithm& a, const std::string& problem) {
        if (a.problems.empty()) return true;
        for (const auto& p : a.problems)
            if (p.kind == Kind::String && p.s == problem) return true;
        return false;
    };
    std::vector<Job> jobs;
    std::int64_t idx = 0;
    for (const auto& pname : cfg.benchmark_problems) {
        const auto& p = reg.problems.at(pname);
        for (const auto& a : cfg.algorithms) {
            if (!takes(a, pname)) continue;
            const auto [pop, note] = fit_pop(a.name, a.pop > 0 ? a.pop : p.pop_size, p.n_obj, a.params, spec.pop_fit);
            std::int64_t gens;
            if (a.evaluations > 0) gens = std::max<std::int64_t>(1, ceil_div(a.evaluations, pop));
            else if (spec.budget_fe > 0) gens = std::max<std::int64_t>(1, ceil_div(spec.budget_fe, pop));
            else gens = a.gens > 0 ? a.gens : p.n_gen;
            for (const std::int64_t s : seeds) {
                Job j;
                j.index = idx++;
                j.problem = pname;
                j.algorithm = a.name;
                j.seed = s;
                j.pop = pop;
                j.gens = gens;
                j.params = a.params;
                j.n_objs = p.n_obj;
                j.pop_note = note;
                j.label = a.key();
                j.evaluations = a.evaluations;
                j.init = a.init;
                j.init_share = a.init_share;
                jobs.push_back(std::move(j));
            }
        }
    }
    if (!spec.ladder.empty()) {
        // the rungs, after every full-budget job
        std::map<std::pair<std::string, std::string>, std::size_t> full;
        for (std::size_t k = 0; k < jobs.size(); ++k) full[{jobs[k].problem, jobs[k].key()}] = k;
        for (const auto& pname : cfg.benchmark_problems) {
            for (const auto& a : cfg.algorithms) {
                if (!takes(a, pname)) continue;
                if ((!depends_on_budget(reg, a.name, a.params) && a.init_share == 0.0) || a.evaluations > 0) continue;
                const Job base = jobs[full.at({pname, a.key()})];
                for (const std::int64_t b : spec.ladder) {
                    if (b >= base.budget() || b < spec.ladder_min_gens * base.pop) continue;
                    const std::int64_t gens = std::max<std::int64_t>(1, ceil_div(b, base.pop));
                    for (const std::int64_t s : seeds) {
                        Job j;
                        j.index = idx++;
                        j.problem = pname;
                        j.algorithm = a.name;
                        j.seed = s;
                        j.pop = base.pop;
                        j.gens = gens;
                        j.params = a.params;
                        j.n_objs = base.n_objs;
                        j.pop_note = base.pop_note;
                        j.label = a.key() + "@" + std::to_string(b);
                        j.evaluations = b;
                        j.ladder_of = a.key();
                        j.init = a.init;
                        j.init_share = a.init_share;
                        jobs.push_back(std::move(j));
                    }
                }
            }
        }
    }
    return jobs;
}

// the spec as dataclasses.asdict gives it, for the cross-check and campaign.json
inline Value to_value(const CampaignSpec& s)
{
    auto ints = [](const std::vector<std::int64_t>& v) {
        Value a = Value::array();
        for (auto x : v) a.items.push_back(Value::integer(x));
        return a;
    };
    auto words = [](const std::vector<std::string>& v) {
        Value a = Value::array();
        for (const auto& x : v) a.items.push_back(Value::string(x));
        return a;
    };
    Value o = Value::object();
    o.set("out", Value::string(s.out));
    o.set("budget_fe", Value::integer(s.budget_fe));
    o.set("record_every", Value::integer(s.record_every));
    o.set("metrics", words(s.metrics));
    o.set("final_metrics", words(s.final_metrics));
    o.set("n_ref", Value::integer(s.n_ref));
    o.set("dense_reference", Value::boolean(s.dense_reference));
    o.set("set_metrics", Value::boolean(s.set_metrics));
    o.set("set_answer", Value::string(s.set_answer));
    o.set("seeds", ints(s.seeds));
    o.set("record_grid", Value::string(s.record_grid));
    o.set("record_per_decade", Value::integer(s.record_per_decade));
    o.set("record_at", ints(s.record_at));
    o.set("record_at_metrics", words(s.record_at_metrics));
    o.set("ladder", ints(s.ladder));
    o.set("ladder_min_gens", Value::integer(s.ladder_min_gens));
    o.set("trajectory_hv_max_m", Value::integer(s.trajectory_hv_max_m));
    o.set("trajectory_hv_mc_samples", Value::integer(s.trajectory_hv_mc_samples));
    o.set("hv_exact_max_m", Value::integer(s.hv_exact_max_m));
    o.set("hv_grid_exact_max_m", Value::integer(s.hv_grid_exact_max_m));
    o.set("hv_archive_grid_max_m", Value::integer(s.hv_archive_grid_max_m));
    o.set("hv_archive_final_max_m", Value::integer(s.hv_archive_final_max_m));
    o.set("hv_mc_samples", Value::integer(s.hv_mc_samples));
    o.set("archive_scenario", Value::boolean(s.archive_scenario));
    o.set("answer_select", Value::string(s.answer_select));
    o.set("operator_stats", Value::boolean(s.operator_stats));
    o.set("coords_diag", Value::boolean(s.coords_diag));
    o.set("archive", Value::boolean(s.archive));
    o.set("archive_delta", Value::number(s.archive_delta));
    o.set("archive_mode", Value::string(s.archive_mode));
    o.set("native", Value::string(s.native));
    o.set("noise_cell", s.noise_cell);
    o.set("archive_variables", Value::string(s.archive_variables));
    o.set("archive_cap", Value::integer(s.archive_cap));
    o.set("all_evaluated", Value::boolean(s.all_evaluated));
    Value levels = Value::array();
    for (double x : s.all_levels) levels.items.push_back(Value::number(x));
    o.set("all_levels", levels);
    o.set("snapshots", s.snapshots);
    o.set("snapshot_variables", Value::string(s.snapshot_variables));
    o.set("archive_checkpoints", Value::boolean(s.archive_checkpoints));
    o.set("hard_budgets", ints(s.hard_budgets));
    o.set("answer_k", Value::integer(s.answer_k));
    o.set("pop_fit", Value::string(s.pop_fit));
    o.set("storage", Value::string(s.storage));
    o.set("keep", Value::string(s.keep));
    o.set("segment_mb", Value::integer(s.segment_mb));
    o.set("run_timeout", Value::number(s.run_timeout));
    return o;
}

// a job as dataclasses.asdict gives it
inline Value to_value(const Job& j)
{
    Value o = Value::object();
    o.set("index", Value::integer(j.index));
    o.set("problem", Value::string(j.problem));
    o.set("algorithm", Value::string(j.algorithm));
    o.set("seed", Value::integer(j.seed));
    o.set("pop", Value::integer(j.pop));
    o.set("gens", Value::integer(j.gens));
    o.set("params", j.params);
    o.set("n_objs", Value::integer(j.n_objs));
    o.set("pop_note", Value::string(j.pop_note));
    o.set("label", Value::string(j.label));
    o.set("evaluations", Value::integer(j.evaluations));
    o.set("ladder_of", Value::string(j.ladder_of));
    o.set("init", Value::string(j.init));
    o.set("init_share", Value::number(j.init_share));
    return o;
}

}   // namespace mootation::campaign

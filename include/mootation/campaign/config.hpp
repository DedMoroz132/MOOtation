#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// A run description (a campaign's TOML), read and checked — task 17, stage 5:
// the campaign engine in C++. run/config.py's loads() and validate(), with
// the same messages, word for word, so that a user sees one text whichever
// side reads the file; the Python stays as its cross-check.
//
// What the checks look things up in — the algorithm registry, the baselines,
// the scalarization presets, the designs, the knobs, the problems and their
// objective counts — the caller gives (Registry): the Python package from its
// modules, the engine from the library's registries. So do the two checks
// that are the platform's or Python's own: whether a program is on PATH, and
// whether a pattern is a regular expression (and its groups).
//
// The population rules (check_pop: an exact Das-Dennis lattice, K equal
// subregions, a free population with n_ref_points) are written here, from
// the cores' headers they mirror (run/algorithms.py names each).
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "../json.hpp"
#include "../toml.hpp"

namespace mootation::campaign {

using json::Value;
using Kind = Value::Kind;

// ── Python's ways of printing a value, for the messages ─────────────────────

// str(x) of a float
inline std::string py_float(double x)
{
    if (std::isnan(x)) return "nan";
    if (std::isinf(x)) return x > 0 ? "inf" : "-inf";
    return json::float_repr(x);
}

// repr(s) of a str: single quotes unless it holds one and no double quote
inline std::string py_repr_str(const std::string& s)
{
    const bool sq = s.find('\'') != std::string::npos, dq = s.find('"') != std::string::npos;
    const char q = sq && !dq ? '"' : '\'';
    std::string out(1, q);
    static const char* hex = "0123456789abcdef";
    for (unsigned char c : s) {
        if (c == '\\') out += "\\\\";
        else if (c == static_cast<unsigned char>(q)) out += std::string("\\") + q;
        else if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else if (c == '\t') out += "\\t";
        else if (c < 0x20 || c == 0x7F) {
            out += "\\x";
            out += hex[c >> 4];
            out += hex[c & 0xF];
        } else {
            out += static_cast<char>(c);                  // UTF-8 kept: printable non-ASCII stays as it is
        }
    }
    return out + q;
}

// repr(v) of a TOML value
inline std::string py_repr(const Value& v)
{
    switch (v.kind) {
    case Kind::Null: return "None";
    case Kind::Bool: return v.b ? "True" : "False";
    case Kind::Int: return std::to_string(v.i);
    case Kind::Float: return py_float(v.f);
    case Kind::String: return v.datetime ? v.s : py_repr_str(v.s);
    case Kind::Array: {
        std::string out = "[";
        for (std::size_t k = 0; k < v.items.size(); ++k) out += (k ? ", " : "") + py_repr(v.items[k]);
        return out + "]";
    }
    case Kind::Object: {
        std::string out = "{";
        for (std::size_t k = 0; k < v.keys.size(); ++k)
            out += (k ? ", " : "") + py_repr_str(v.keys[k]) + ": " + py_repr(v.items[k]);
        return out + "}";
    }
    }
    return "";
}

// type(v).__name__
inline std::string py_type(const Value& v)
{
    switch (v.kind) {
    case Kind::Null: return "NoneType";
    case Kind::Bool: return "bool";
    case Kind::Int: return "int";
    case Kind::Float: return "float";
    case Kind::String: return v.datetime ? "datetime" : "str";
    case Kind::Array: return "list";
    case Kind::Object: return "dict";
    }
    return "";
}

inline std::string join(const std::vector<std::string>& parts, const std::string& sep)
{
    std::string out;
    for (std::size_t k = 0; k < parts.size(); ++k) out += (k ? sep : "") + parts[k];
    return out;
}

// ── errors ──────────────────────────────────────────────────────────────────

class ConfigError : public std::invalid_argument {
public:
    ConfigError(const std::string& path, const std::string& message)
        : std::invalid_argument(path.empty() ? message : path + ": " + message), path(path), message(message) {}
    std::string path, message;
};

// ── the model (run/config.py's dataclasses) ─────────────────────────────────

struct Step {
    std::string name;
    std::vector<std::string> argv;
    std::optional<double> timeout;
    bool shell = false;
};

struct Output {
    std::string parser;
    std::optional<std::string> source;
    std::vector<Value> objectives, constraints;
    std::string take = "last";
    std::optional<std::int64_t> nth;
    std::vector<std::pair<std::string, Value>> patterns;
    std::int64_t line = -1;
    std::vector<std::int64_t> fields;
};

struct Algorithm {
    std::string name;
    std::int64_t pop = 0, gens = 0;
    Value params = Value::object();                       // a knob's value as TOML gave it
    std::optional<std::string> label;
    std::int64_t evaluations = 0;
    std::string init = "uniform";
    double init_share = 0.0;
    std::vector<Value> problems;

    std::string key() const { return label && !label->empty() ? *label : name; }
};

struct WarmStart {
    std::string source;
    std::string on_size_mismatch = "error";
};

struct Config {
    std::string name, scratch, ledger;
    std::int64_t workers = 1;
    std::string on_fail = "penalty";
    double penalty = 1e8;
    bool resume = true;
    std::optional<WarmStart> warm_start;

    std::string kind = "external";
    std::int64_t n_vars = 0, n_objs = 0;
    std::vector<std::pair<double, double>> bounds;
    std::optional<std::string> input_template, input_write_to;
    std::vector<Step> steps;
    std::optional<Output> output;

    std::vector<Algorithm> algorithms;
    Value benchmarks = Value::object(), campaign = Value::object();
    std::vector<std::string> benchmark_problems;          // resolved by validate()
    std::optional<std::filesystem::path> source_path;
};

// ── what the checks look things up in ───────────────────────────────────────

struct Registry {
    std::vector<std::string> algorithms;                  // algorithms.def's names
    std::vector<std::string> baselines, scalar_names, design_kinds;
    std::vector<std::string> knob_names;                  // in their order (knobs.knob_names)
    std::vector<std::pair<std::string, std::vector<std::string>>> text_knobs;   // knob, its words
    std::vector<std::string> list_knobs;
    std::vector<std::pair<std::string, int>> problems;    // the benchmarks: name, objectives
    std::string platform = "windows";                     // windows, darwin, linux
    std::string algorithms_error;                         // a registry that could not be read: its message
    // whether a program is on PATH or an existing file
    std::function<bool(const std::string&)> executable_findable;
    // a pattern's capturing groups, or its error as Python's re says it
    std::function<int(const std::string& pattern, std::string& error)> regex_groups;

    const std::vector<std::string>* words_of(const std::string& knob) const
    {
        for (const auto& [k, w] : text_knobs)
            if (k == knob) return &w;
        return nullptr;
    }
};

// ── the population rules (run/algorithms.py's check_pop) ────────────────────

namespace pop {

inline const std::set<std::string>& exact_lattice()
{
    static const std::set<std::string> s = {"a_nsga3", "adaw", "crea", "ctaea", "edv", "irea", "mbra",
                                            "moead", "moead_awa", "moead_dd", "moead_de", "moead_dra",
                                            "mombi2", "nsga3", "rvea", "rvea_star", "srv", "srv_nsga3",
                                            "theta_dea"};
    return s;
}
inline const std::map<std::string, std::int64_t>& k_divisible()
{
    static const std::map<std::string, std::int64_t> m = {{"moead_m2m", 10}, {"sms_m2m", 10}};
    return m;
}
inline const std::set<std::string>& free_h()
{
    static const std::set<std::string> s = {"nsga3", "a_nsga3", "theta_dea", "srv_nsga3"};
    return s;
}

// C(h + m - 1, m - 1), saturating at 2^62
inline std::int64_t das_dennis_count(std::int64_t m, std::int64_t h)
{
    const std::int64_t cap = std::int64_t{1} << 62;
    std::int64_t r = 1;
    const std::int64_t k = m - 1, n = h + m - 1;
    for (std::int64_t j = 1; j <= k; ++j) {
        // r = r * (n - k + j) / j, exact at every step
        const std::int64_t f = n - k + j;
        if (r > cap / f) return cap;
        r = r * f / j;
    }
    return r;
}

// (largest <= p, smallest > p); at one objective there is no larger one
inline std::pair<std::optional<std::int64_t>, std::optional<std::int64_t>> nearest(std::int64_t m, std::int64_t p)
{
    if (m < 2) {                  // one objective: the lattice is one point for every h
        if (p == 1) return {1, 1};
        if (p > 1) return {1, std::nullopt};
        return {std::nullopt, 1};
    }
    std::optional<std::int64_t> below;
    for (std::int64_t h = 1;; ++h) {
        const std::int64_t n = das_dennis_count(m, h);
        if (n == p) return {n, n};
        if (n > p) return {below, n};
        below = n;
    }
}

inline bool two_layer(std::int64_t m, std::int64_t p)
{
    if (m < 2) return p == 2;     // one objective: two one-point layers
    for (std::int64_t h1 = 1; das_dennis_count(m, h1) <= p; ++h1)
        for (std::int64_t h2 = 1; h2 < h1; ++h2)
            if (das_dennis_count(m, h1) + das_dennis_count(m, h2) == p) return true;
    return false;
}

}   // namespace pop

// A reason pop is unusable for the core at m objectives, or nothing.
inline std::optional<std::string> check_pop(const std::string& name, std::int64_t p, std::int64_t m, const Value& params)
{
    // task 18, item 2.3: the single-objective solvers (scalar.hpp, single_names()) at one objective only
    if ((name == "slsqp" || name == "nelder_mead" || name == "cmaes" || name == "lm" || name == "lbfgsb") && m != 1)
        return name + " is a single-objective algorithm: n_objs must be 1, got " + std::to_string(m);
    if (pop::free_h().count(name)) {
        const Value* r = params.find("n_ref_points");
        const bool set = r && ((r->kind == Kind::Int && r->i != 0) || (r->kind == Kind::Float && r->f != 0.0) ||
                               (r->kind == Kind::Bool && r->b) || (r->kind == Kind::String && !r->s.empty()) ||
                               (r->kind == Kind::Array && !r->items.empty()));
        if (set) return std::nullopt;
    }
    if (pop::exact_lattice().count(name)) {
        const auto [below, above] = pop::nearest(m, p);
        if (below && *below == p) return std::nullopt;
        if (pop::two_layer(m, p)) return std::nullopt;
        std::vector<std::string> hint;
        if (below) hint.push_back(std::to_string(*below));
        if (above) hint.push_back(std::to_string(*above));
        return "pop = " + std::to_string(p) + " is not a Das-Dennis lattice size for n_objs = " + std::to_string(m) +
               "; " + name + " requires an exact lattice. Nearest attainable: " + join(hint, ", ");
    }
    const auto kd = pop::k_divisible().find(name);
    if (kd != pop::k_divisible().end()) {
        std::int64_t k = kd->second;
        const Value* kv = params.find("K");
        if (!kv) kv = params.find("n_clusters");
        if (kv) k = kv->kind == Kind::Float ? static_cast<std::int64_t>(kv->f) : kv->kind == Kind::Bool ? kv->b : kv->i;
        if (k < 1) return "K = " + std::to_string(k) + " must be >= 1";
        if (p % k != 0) {
            std::vector<std::string> divisors;
            for (std::int64_t d = 2; d <= std::min<std::int64_t>(p, 64); ++d)
                if (p % d == 0) divisors.push_back(std::to_string(d));
            if (divisors.size() > 8) divisors.resize(8);
            const std::string hint = divisors.empty() ? "none in 2..64" : join(divisors, ", ");
            return "pop = " + std::to_string(p) + " is not divisible by K = " + std::to_string(k) + "; " + name +
                   " partitions the population into K equal subregions (pop = K*S). Divisors of " +
                   std::to_string(p) + ": " + hint;
        }
    }
    return std::nullopt;
}

// ── loading (run/config.py's loads) ─────────────────────────────────────────

namespace detail {

enum class Want { Int, Number, Str, Bool, List, Dict };

inline bool is(const Value& v, Want w)
{
    switch (w) {
    case Want::Int: return v.kind == Kind::Int;          // true/false given for a count: refused
    case Want::Number: return v.kind == Kind::Int || v.kind == Kind::Float;
    case Want::Str: return v.kind == Kind::String && !v.datetime;
    case Want::Bool: return v.kind == Kind::Bool;
    case Want::List: return v.kind == Kind::Array;
    case Want::Dict: return v.kind == Kind::Object;
    }
    return false;
}

inline const char* name_of(Want w)
{
    switch (w) {
    case Want::Int: return "int";
    case Want::Number: return "int/float";
    case Want::Str: return "str";
    case Want::Bool: return "bool";
    case Want::List: return "list";
    case Want::Dict: return "dict";
    }
    return "";
}

// key and where by value: a reference to the table's member comes back, and
// GCC's -Wdangling-reference takes a temporary string argument for its source
inline const Value& req(const Value& table, std::string_view key_, std::string_view where_, Want w)
{
    const std::string key(key_), where(where_);
    const Value* v = table.find(key);
    if (!v) throw ConfigError(where, "missing required key '" + key + "'");
    if (!is(*v, w))
        throw ConfigError(where + "." + key, std::string("expected ") + name_of(w) + ", got " + py_type(*v));
    return *v;
}

inline const Value* opt(const Value& table, std::string_view key, std::string_view where, Want w)
{
    if (!table.find(std::string(key))) return nullptr;
    return &req(table, key, where, w);
}

inline std::int64_t as_int(const Value& v) { return v.kind == Kind::Bool ? (v.b ? 1 : 0) : v.i; }
inline double as_float(const Value& v)
{
    return v.kind == Kind::Float ? v.f : static_cast<double>(as_int(v));
}

inline std::string one_of(const std::vector<std::string>& words) { return join(words, ", "); }

inline std::vector<std::string> resolve_argv(const Value& step, const std::string& where, const std::string& platform)
{
    const std::string key = "run_" + platform;
    const Value* argv = step.find(key);
    if (!argv) argv = step.find("run");
    if (!argv) throw ConfigError(where, "no command for this platform: neither '" + key + "' nor 'run' is set");
    if (argv->kind != Kind::Array || argv->items.empty())
        throw ConfigError(where + "." + (step.find(key) ? key : std::string("run")),
                          "a command must be a non-empty array of arguments, e.g. "
                          "[\"solver\", \"--mesh\", \"{scratch}/mesh.msh\"]");
    std::vector<std::string> out;
    for (std::size_t i = 0; i < argv->items.size(); ++i) {
        const Value& a = argv->items[i];
        if (!is(a, Want::Str))
            throw ConfigError(where + ".run[" + std::to_string(i) + "]",
                              "every argument must be a string, got " + py_type(a));
        out.push_back(a.s);
    }
    return out;
}

inline Output load_output(const Value& out)
{
    const std::string where = "problem.output";
    const std::string parser = req(out, "parser", where, Want::Str).s;
    const std::vector<std::string> parsers = {"csv", "json", "regex", "columns"};
    if (std::find(parsers.begin(), parsers.end(), parser) == parsers.end() && parser.rfind("python:", 0) != 0)
        throw ConfigError(where + ".parser", "must be one of " + one_of(parsers) +
                                                 " or 'python:module.function'; got '" + parser + "'");
    Output o;
    o.parser = parser;
    if (const Value* v = opt(out, "from", where, Want::Str)) o.source = v->s;
    if (const Value* v = opt(out, "objectives", where, Want::List)) o.objectives = v->items;
    if (const Value* v = opt(out, "constraints", where, Want::List)) o.constraints = v->items;
    if (const Value* v = opt(out, "take", where, Want::Str)) o.take = v->s;
    if (const Value* v = opt(out, "nth", where, Want::Int)) o.nth = as_int(*v);
    if (const Value* v = opt(out, "patterns", where, Want::Dict))
        for (std::size_t k = 0; k < v->keys.size(); ++k) o.patterns.emplace_back(v->keys[k], v->items[k]);
    if (const Value* v = opt(out, "line", where, Want::Int)) o.line = as_int(*v);
    if (const Value* v = opt(out, "fields", where, Want::List)) {
        for (const auto& f : v->items) {
            if (!is(f, Want::Int))
                throw ConfigError(where + ".fields", "expected a list of integers, got " + py_repr(*v));
            o.fields.push_back(f.i);
        }
    }
    if (o.take != "first" && o.take != "last" && o.take != "nth")
        throw ConfigError(where + ".take", "must be first, last or nth; got '" + o.take + "'");
    if (o.take == "nth" && !o.nth) throw ConfigError(where + ".nth", "take = \"nth\" requires an 'nth' index");
    return o;
}

}   // namespace detail

// A run description from TOML text; ConfigError where it does not hold one.
inline Config loads(const std::string& text, const Registry& reg,
                    const std::optional<std::filesystem::path>& source_path = std::nullopt)
{
    using namespace detail;
    Value raw;
    try {
        raw = toml::parse(text);
    } catch (const toml::Error& e) {
        throw ConfigError("", std::string("TOML does not parse: ") + e.what());
    }
    const Value& run = req(raw, "run", "", Want::Dict);
    Config cfg;
    cfg.name = req(run, "name", "run", Want::Str).s;
    const Value* v;
    cfg.scratch = (v = opt(run, "scratch", "run", Want::Str)) ? v->s : "scratch/worker_{worker:02d}";
    cfg.ledger = (v = opt(run, "ledger", "run", Want::Str)) ? v->s : "results/{name}/evaluations.jsonl";
    cfg.workers = (v = opt(run, "workers", "run", Want::Int)) ? as_int(*v) : 1;
    cfg.on_fail = (v = opt(run, "on_fail", "run", Want::Str)) ? v->s : "penalty";
    cfg.penalty = (v = opt(run, "penalty", "run", Want::Number)) ? as_float(*v) : 1e8;
    cfg.resume = (v = opt(run, "resume", "run", Want::Bool)) ? v->b : true;
    cfg.source_path = source_path;

    const std::vector<std::string> on_fail = {"penalty", "skip", "abort"};
    if (std::find(on_fail.begin(), on_fail.end(), cfg.on_fail) == on_fail.end())
        throw ConfigError("run.on_fail", "must be one of " + one_of(on_fail) + "; got '" + cfg.on_fail + "'");
    if (cfg.workers < 1) throw ConfigError("run.workers", "must be >= 1, got " + std::to_string(cfg.workers));

    if (run.find("warm_start")) {
        const Value& ws = req(run, "warm_start", "run", Want::Dict);
        const std::string mode = (v = opt(ws, "on_size_mismatch", "run.warm_start", Want::Str)) ? v->s : "error";
        const std::vector<std::string> modes = {"truncate", "pad", "error"};
        if (std::find(modes.begin(), modes.end(), mode) == modes.end())
            throw ConfigError("run.warm_start.on_size_mismatch",
                              "must be one of " + one_of(modes) + "; got '" + mode + "'");
        cfg.warm_start = WarmStart{req(ws, "from", "run.warm_start", Want::Str).s, mode};
    }

    const Value& prob = req(raw, "problem", "", Want::Dict);
    cfg.kind = (v = opt(prob, "kind", "problem", Want::Str)) ? v->s : "external";
    const std::vector<std::string> kinds = {"external", "builtin"};
    if (std::find(kinds.begin(), kinds.end(), cfg.kind) == kinds.end())
        throw ConfigError("problem.kind", "must be one of " + one_of(kinds) + "; got '" + cfg.kind + "'");

    if (cfg.kind == "external") {
        cfg.n_vars = as_int(req(prob, "n_vars", "problem", Want::Int));
        cfg.n_objs = as_int(req(prob, "n_objs", "problem", Want::Int));
        const Value& raw_bounds = req(prob, "bounds", "problem", Want::List);
        for (std::size_t i = 0; i < raw_bounds.items.size(); ++i) {
            const Value& b = raw_bounds.items[i];
            if (b.kind != Kind::Array || b.items.size() != 2)
                throw ConfigError("problem.bounds[" + std::to_string(i) + "]", "expected a [lower, upper] pair");
            if (!is(b.items[0], Want::Number) || !is(b.items[1], Want::Number))
                throw ConfigError("problem.bounds[" + std::to_string(i) + "]", "bounds must be numbers");
            cfg.bounds.emplace_back(as_float(b.items[0]), as_float(b.items[1]));
        }
        if (prob.find("input")) {
            const Value& pin = req(prob, "input", "problem", Want::Dict);
            cfg.input_template = req(pin, "template", "problem.input", Want::Str).s;
            cfg.input_write_to = req(pin, "write_to", "problem.input", Want::Str).s;
        }
        if (const Value* steps = opt(prob, "steps", "problem", Want::List)) {
            for (std::size_t i = 0; i < steps->items.size(); ++i) {
                const std::string where = "problem.steps[" + std::to_string(i) + "]";
                const Value& st = steps->items[i];
                if (st.kind != Kind::Object) throw ConfigError(where, "each step must be a table");
                Step s;
                s.name = (v = opt(st, "name", where, Want::Str)) ? v->s : "step" + std::to_string(i);
                s.argv = resolve_argv(st, where, reg.platform);
                if (const Value* t = opt(st, "timeout", where, Want::Number)) s.timeout = as_float(*t);
                s.shell = (v = opt(st, "shell", where, Want::Bool)) ? v->b : false;
                cfg.steps.push_back(std::move(s));
            }
        }
        if (prob.find("output")) cfg.output = load_output(req(prob, "output", "problem", Want::Dict));
    }

    if (const Value* algs = opt(raw, "algorithms", "", Want::List)) {
        const std::set<std::string> allowed = {"name", "pop", "gens", "label", "evaluations", "init",
                                               "init_share", "params", "problems"};
        for (std::size_t i = 0; i < algs->items.size(); ++i) {
            const std::string where = "algorithms[" + std::to_string(i) + "]";
            const Value& a = algs->items[i];
            if (a.kind != Kind::Object) throw ConfigError(where, "each algorithm must be a table");
            std::vector<std::string> unknown;
            for (const auto& k : a.keys)
                if (!allowed.count(k)) unknown.push_back(k);
            std::sort(unknown.begin(), unknown.end());
            if (!unknown.empty())
                throw ConfigError(where, "unknown key(s): " + join(unknown, ", ") +
                                             "; an algorithm's knobs go in params = { ... }");
            Algorithm al;
            if (const Value* params = opt(a, "params", where, Want::Dict)) {
                for (std::size_t k = 0; k < params->keys.size(); ++k) {
                    const std::string& name = params->keys[k];
                    const Value& pv = params->items[k];
                    const auto* words = reg.words_of(name);
                    const bool listed = std::find(reg.list_knobs.begin(), reg.list_knobs.end(), name) != reg.list_knobs.end();
                    if (listed) {
                        bool ok = is(pv, Want::Str);
                        if (!ok && pv.kind == Kind::Array) {
                            ok = true;
                            for (const auto& w : pv.items) ok = ok && is(w, Want::Str);
                        }
                        if (!ok)
                            throw ConfigError(where + ".params." + name, "expected a list of " +
                                                                             (words ? join(*words, ", ") : "") + ", got " + py_repr(pv));
                    } else if (words) {
                        if (!is(pv, Want::Str))
                            throw ConfigError(where + ".params." + name,
                                              "expected one of " + join(*words, ", ") + ", got " + py_repr(pv));
                    } else if (!is(pv, Want::Number) && !is(pv, Want::Bool)) {
                        throw ConfigError(where + ".params." + name, "expected a number or true/false, got " + py_type(pv));
                    }
                    al.params.set(name, pv);
                }
            }
            al.evaluations = (v = opt(a, "evaluations", where, Want::Int)) ? as_int(*v) : 0;
            al.name = req(a, "name", where, Want::Str).s;
            al.pop = as_int(req(a, "pop", where, Want::Int));
            if (al.evaluations) al.gens = (v = opt(a, "gens", where, Want::Int)) ? as_int(*v) : 0;
            else al.gens = as_int(req(a, "gens", where, Want::Int));
            if ((v = opt(a, "label", where, Want::Str))) al.label = v->s;
            al.init = (v = opt(a, "init", where, Want::Str)) ? v->s : "uniform";
            al.init_share = (v = opt(a, "init_share", where, Want::Number)) ? as_float(*v) : 0.0;
            if ((v = opt(a, "problems", where, Want::List))) al.problems = v->items;
            cfg.algorithms.push_back(std::move(al));
        }
    }
    if ((v = opt(raw, "benchmarks", "", Want::Dict))) cfg.benchmarks = *v;
    if ((v = opt(raw, "campaign", "", Want::Dict))) cfg.campaign = *v;
    return cfg;
}

// ── checking (run/config.py's validate) ─────────────────────────────────────

namespace detail {

inline bool label_ok(const std::string& s)
{
    if (s.empty()) return false;
    for (char c : s)
        if (!((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_' || c == '.' ||
              c == '+' || c == '-'))
            return false;
    return true;
}

inline std::string upper(std::string s)
{
    for (auto& c : s) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    return s;
}

// str(v) of a TOML value, as f"{v}" prints it
inline std::string py_str(const Value& v) { return v.kind == Kind::String ? v.s : py_repr(v); }

inline std::vector<std::string> benchmark_selection(const Value& spec, const Registry& reg,
                                                    const std::function<void(const std::string&, const std::string&)>& bad)
{
    const std::set<std::string> known_keys = {"families", "objectives", "runs", "problems"};
    std::vector<std::string> unknown;
    for (const auto& k : spec.keys)
        if (!known_keys.count(k)) unknown.push_back(k);
    std::sort(unknown.begin(), unknown.end());
    if (!unknown.empty())
        bad("benchmarks", "unknown key(s): " + join(unknown, ", ") + ". Known: families, objectives, problems, runs");
    if (const Value* runs = spec.find("runs")) {
        if (!is(*runs, Want::Int) || as_int(*runs) < 1)
            bad("benchmarks.runs", "must be an integer >= 1, got " + py_repr(*runs));
    }
    std::vector<std::string> names;
    std::set<std::string> registry;
    for (const auto& [n, m] : reg.problems) {
        names.push_back(n);
        registry.insert(n);
    }
    std::sort(names.begin(), names.end());
    if (const Value* problems = spec.find("problems")) {
        bool names_only = problems->kind == Kind::Array;
        for (const auto& p : problems->items) names_only = names_only && is(p, Want::Str);
        if (!names_only) {
            bad("benchmarks.problems", "expected a list of problem names, got " + py_repr(*problems));
            return {};
        }
        std::vector<std::string> wanted;
        for (const auto& p : problems->items) wanted.push_back(p.s);
        std::vector<std::string> out;
        for (const auto& p : wanted) {
            if (registry.count(p)) {
                out.push_back(p);
                continue;
            }
            const std::string head = p.substr(0, p.find('_'));
            std::vector<std::string> near;
            for (const auto& n : names)
                if (upper(n).rfind(upper(head), 0) == 0) near.push_back(n);
            if (near.empty()) {
                std::string fam;
                for (char c : head)
                    if (std::isalpha(static_cast<unsigned char>(c))) fam += c;
                for (const auto& n : names)
                    if (upper(n).rfind(upper(fam), 0) == 0) near.push_back(n);
            }
            if (near.size() > 5) near.resize(5);
            bad("benchmarks.problems",
                "unknown problem '" + p + "'" + (near.empty() ? "" : "; did you mean: " + join(near, ", ")));
        }
        return out;
    }
    const Value* fams = spec.find("families");
    const Value* objs = spec.find("objectives");
    auto truthy = [](const Value* x) {
        return x && !((x->kind == Kind::Array && x->items.empty()) || (x->kind == Kind::String && x->s.empty()) ||
                      (x->kind == Kind::Int && x->i == 0) || (x->kind == Kind::Bool && !x->b) ||
                      (x->kind == Kind::Float && x->f == 0.0) || (x->kind == Kind::Object && x->keys.empty()));
    };
    if (!truthy(fams) || !truthy(objs)) {
        bad("benchmarks", "give either 'problems = [...]' or both 'families = [...]' and 'objectives = [...]'");
        return {};
    }
    if (fams->kind != Kind::Array || objs->kind != Kind::Array) {
        bad("benchmarks", "'families' and 'objectives' must be lists");
        return {};
    }
    std::set<std::string> selected;
    for (const auto& fam : fams->items) {
        const std::string f = upper(py_str(fam));
        for (const auto& m : objs->items) {
            const std::string tail = "_" + py_str(m) + "D";
            bool hit = false;
            for (const auto& n : names) {
                if (upper(n).rfind(f, 0) == 0 && n.size() >= tail.size() &&
                    n.compare(n.size() - tail.size(), tail.size(), tail) == 0) {
                    selected.insert(n);
                    hit = true;
                }
            }
            if (!hit) {
                std::set<std::string> avail;
                for (const auto& n : names)
                    if (upper(n).rfind(f, 0) == 0) avail.insert(n.substr(n.rfind('_') == std::string::npos ? 0 : n.rfind('_') + 1));
                bad("benchmarks", "no problems for family '" + py_str(fam) + "' at " + py_str(m) + " objectives" +
                                      (avail.empty() ? "; no family '" + py_str(fam) + "' in the registry"
                                                     : "; available sizes: " +
                                                           join(std::vector<std::string>(avail.begin(), avail.end()), ", ")));
            }
        }
    }
    return std::vector<std::string>(selected.begin(), selected.end());
}

// the objective count of a JSONL population's first record, or nothing
inline std::optional<std::size_t> warm_start_objs(const std::filesystem::path& path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in) return std::nullopt;
    std::string line;
    while (std::getline(in, line)) {
        const auto a = line.find_first_not_of(" \t\r\n\f\v");
        if (a == std::string::npos) continue;
        try {
            const Value rec = json::parse(line);
            const Value* f = rec.kind == Kind::Object ? rec.find("f") : nullptr;
            if (f && f->kind == Kind::Array) return f->items.size();
        } catch (const std::invalid_argument&) {
        }
        return std::nullopt;
    }
    return std::nullopt;
}

}   // namespace detail

// The problems with a run description; empty: it can start. Relative paths
// resolve against `base` (the file's own folder by default).
inline std::vector<std::string> validate(Config& cfg, const Registry& reg,
                                         std::optional<std::filesystem::path> base = std::nullopt)
{
    using namespace detail;
    namespace fs = std::filesystem;
    if (!base) base = cfg.source_path ? cfg.source_path->parent_path() : fs::current_path();
    std::vector<std::string> problems;
    auto bad = [&](const std::string& where, const std::string& msg) { problems.push_back(where + ": " + msg); };

    if (cfg.kind == "external") {
        if (cfg.n_vars < 1) bad("problem.n_vars", "must be >= 1, got " + std::to_string(cfg.n_vars));
        if (cfg.n_objs < 1) bad("problem.n_objs", "must be >= 1, got " + std::to_string(cfg.n_objs));
        if (static_cast<std::int64_t>(cfg.bounds.size()) != cfg.n_vars)
            bad("problem.bounds", "has " + std::to_string(cfg.bounds.size()) + " entries but n_vars = " +
                                      std::to_string(cfg.n_vars) +
                                      "; there must be exactly one [lower, upper] pair per variable");
        for (std::size_t i = 0; i < cfg.bounds.size(); ++i)
            if (!(cfg.bounds[i].first < cfg.bounds[i].second))
                bad("problem.bounds[" + std::to_string(i) + "]", "lower (" + py_float(cfg.bounds[i].first) +
                                                                     ") must be strictly less than upper (" +
                                                                     py_float(cfg.bounds[i].second) + ")");
        if (cfg.input_template && !cfg.input_template->empty()) {
            const fs::path tpl = (*base / *cfg.input_template).make_preferred();
            if (!fs::is_regular_file(tpl)) {
                bad("problem.input.template", "file not found: " + tpl.string());
            } else {
                std::ifstream in(tpl, std::ios::binary);
                std::stringstream ss;
                ss << in.rdbuf();
                const std::string text = ss.str();
                for (std::size_t at = text.find("{x["); at != std::string::npos; at = text.find("{x[", at + 1)) {
                    std::size_t k = at + 3;
                    std::string digits;
                    while (k < text.size() && std::isdigit(static_cast<unsigned char>(text[k]))) digits += text[k++];
                    if (digits.empty() || k + 1 >= text.size() || text[k] != ']' || text[k + 1] != '}') continue;
                    const long long idx = std::stoll(digits);
                    if (idx >= cfg.n_vars)
                        bad("problem.input.template", "uses {x[" + std::to_string(idx) + "]} but n_vars = " +
                                                          std::to_string(cfg.n_vars) + " (valid indices are 0.." +
                                                          std::to_string(cfg.n_vars - 1) + ")");
                }
            }
            if (!cfg.input_write_to || cfg.input_write_to->empty())
                bad("problem.input.write_to", "required when a template is given");
        }
        if (cfg.steps.empty()) bad("problem.steps", "an external problem needs at least one step");
        for (const auto& st : cfg.steps) {
            if (reg.executable_findable && !reg.executable_findable(st.argv[0]))
                bad("problem.steps." + st.name, "executable not found on PATH and not an existing file: " +
                                                    py_repr_str(st.argv[0]) + " (platform: " + reg.platform + ")");
            if (st.timeout && *st.timeout <= 0)
                bad("problem.steps." + st.name, "timeout must be positive, got " + py_float(*st.timeout));
        }
        if (!cfg.output) {
            bad("problem.output", "an external problem needs an output section");
        } else {
            const Output& o = *cfg.output;
            if (static_cast<std::int64_t>(o.objectives.size()) != cfg.n_objs)
                bad("problem.output.objectives", "lists " + std::to_string(o.objectives.size()) +
                                                     " names but n_objs = " + std::to_string(cfg.n_objs));
            if ((o.parser == "csv" || o.parser == "regex" || o.parser == "columns") && (!o.source || o.source->empty()))
                bad("problem.output.from", "parser '" + o.parser + "' needs a 'from' path");
            if (o.parser == "regex") {
                std::set<std::string> wanted, have;
                for (const auto& n : o.objectives) wanted.insert(py_str(n));
                for (const auto& n : o.constraints) wanted.insert(py_str(n));
                for (const auto& [k, _] : o.patterns) have.insert(k);
                std::vector<std::string> missing;
                for (const auto& w : wanted)
                    if (!have.count(w)) missing.push_back(w);
                if (!missing.empty()) bad("problem.output.patterns", "no pattern for: " + join(missing, ", "));
                for (const auto& [k, pat] : o.patterns) {
                    if (!reg.regex_groups) continue;
                    std::string error;
                    const int groups = reg.regex_groups(py_str(pat), error);
                    if (!error.empty()) {
                        bad("problem.output.patterns." + k, "not a valid regular expression: " + error);
                        continue;
                    }
                    if (groups < 1)
                        bad("problem.output.patterns." + k,
                            "must contain a capturing group for the value, e.g. 'MASS\\s*=\\s*([0-9.eE+-]+)'");
                }
            }
            if (o.parser == "columns") {
                const std::size_t need = o.objectives.size() + o.constraints.size();
                if (o.fields.size() != need)
                    bad("problem.output.fields", "lists " + std::to_string(o.fields.size()) + " indices but " +
                                                     std::to_string(need) + " values are expected (" +
                                                     std::to_string(o.objectives.size()) + " objectives + " +
                                                     std::to_string(o.constraints.size()) + " constraints)");
                for (auto f : o.fields)
                    if (f < 0)
                        bad("problem.output.fields",
                            "column indices are zero-based and non-negative, got " + std::to_string(f));
            }
        }
    }

    // algorithms
    std::set<std::string> known;
    if (!reg.algorithms_error.empty()) {
        bad("algorithms", "cannot read the algorithm registry: " + reg.algorithms_error);
    } else {
        known.insert(reg.algorithms.begin(), reg.algorithms.end());
        known.insert(reg.baselines.begin(), reg.baselines.end());
        known.insert(reg.scalar_names.begin(), reg.scalar_names.end());
    }
    auto benchmarks_given = !cfg.benchmarks.keys.empty();
    if (cfg.algorithms.empty() && !benchmarks_given) bad("algorithms", "no algorithms and no benchmarks — nothing to run");
    std::map<std::string, std::size_t> seen;
    for (std::size_t i = 0; i < cfg.algorithms.size(); ++i) {
        const auto& a = cfg.algorithms[i];
        const auto it = seen.find(a.key());
        if (it != seen.end())
            bad("algorithms[" + std::to_string(i) + "] (" + a.name + ")",
                py_repr_str(a.key()) + " is already algorithms[" + std::to_string(it->second) +
                    "]: results would land in one directory; give one of them a label");
        seen.emplace(a.key(), i);
    }
    std::vector<std::string> sorted_knobs(reg.knob_names.begin(), reg.knob_names.end());
    std::sort(sorted_knobs.begin(), sorted_knobs.end());
    sorted_knobs.erase(std::unique(sorted_knobs.begin(), sorted_knobs.end()), sorted_knobs.end());
    const std::set<std::string> valid_knobs(reg.knob_names.begin(), reg.knob_names.end());
    const auto is_baseline = [&](const std::string& n) {
        return std::find(reg.baselines.begin(), reg.baselines.end(), n) != reg.baselines.end();
    };
    for (std::size_t i = 0; i < cfg.algorithms.size(); ++i) {
        const auto& a = cfg.algorithms[i];
        const std::string where = "algorithms[" + std::to_string(i) + "] (" + a.name + ")";
        if (!known.empty() && !known.count(a.name)) {
            std::vector<std::string> near;
            for (const auto& n : known)
                if (n.rfind(a.name.substr(0, 3), 0) == 0) near.push_back(n);
            if (near.size() > 5) near.resize(5);
            bad(where, "unknown algorithm '" + a.name + "'" + (near.empty() ? "" : "; did you mean: " + join(near, ", ")));
        }
        if (a.pop < 2 && !(cfg.kind == "builtin" && a.pop == 0)) bad(where, "pop must be >= 2, got " + std::to_string(a.pop));
        if (a.evaluations < 0) bad(where, "evaluations must be >= 0, got " + std::to_string(a.evaluations));
        if (a.gens < 1 && !(cfg.kind == "builtin" && a.gens == 0) && a.evaluations <= 0)
            bad(where, "gens must be >= 1, got " + std::to_string(a.gens));
        if (a.label && !label_ok(*a.label))
            bad(where, "label " + py_repr_str(*a.label) +
                           " must be letters, digits and _ . + - only: it names a results directory");
        if (is_baseline(a.name) && !a.params.keys.empty()) {
            std::vector<std::string> ks = a.params.keys;
            std::sort(ks.begin(), ks.end());
            bad(where, a.name + " is a baseline and takes no parameters, got: " + join(ks, ", "));
        }
        if (a.init != "uniform" || a.init_share != 0.0) {
            const bool design = std::find(reg.design_kinds.begin(), reg.design_kinds.end(), a.init) != reg.design_kinds.end();
            if (is_baseline(a.name))
                bad(where, a.name + " is a baseline: it draws every point itself, no init");
            else if (a.init != "uniform" && !design)
                bad(where, "init = " + py_repr_str(a.init) + " is not one of: uniform, " + join(reg.design_kinds, ", "));
            else if (a.init_share != 0.0 && a.init == "uniform")
                bad(where, "init_share is the share of the budget a design takes: give init");
            else if (!(0.0 <= a.init_share && a.init_share < 1.0))
                bad(where, "init_share must be in [0, 1), got " + py_float(a.init_share));
        }
        std::vector<std::string> unknown;
        for (const auto& k : a.params.keys)
            if (!valid_knobs.count(k)) unknown.push_back(k);
        std::sort(unknown.begin(), unknown.end());
        if (!unknown.empty())
            bad(where, "unknown parameter(s): " + join(unknown, ", ") + ". Known: " + join(sorted_knobs, ", "));
        for (std::size_t k = 0; k < a.params.keys.size(); ++k) {
            const auto* words = reg.words_of(a.params.keys[k]);
            if (!words) continue;
            const Value& pv = a.params.items[k];
            const bool listed =
                std::find(reg.list_knobs.begin(), reg.list_knobs.end(), a.params.keys[k]) != reg.list_knobs.end();
            bool ok;
            if (listed) {
                std::vector<Value> items;
                if (pv.kind == Kind::Array) {
                    items = pv.items;
                } else {
                    const std::string s = py_str(pv);
                    std::size_t at = 0;
                    while (true) {
                        const auto c = s.find(',', at);
                        items.push_back(Value::string(s.substr(at, c == std::string::npos ? std::string::npos : c - at)));
                        if (c == std::string::npos) break;
                        at = c + 1;
                    }
                }
                ok = !items.empty();
                for (const auto& w : items)
                    ok = ok && is(w, Want::Str) && std::find(words->begin(), words->end(), w.s) != words->end();
            } else {
                ok = is(pv, Want::Str) && std::find(words->begin(), words->end(), pv.s) != words->end();
            }
            if (!ok) bad(where, a.params.keys[k] + " = " + py_repr(pv) + " is not one of: " + join(*words, ", "));
        }
        if (cfg.n_objs >= 1 && a.pop >= 2)
            if (const auto reason = check_pop(a.name, a.pop, cfg.n_objs, a.params)) bad(where, *reason);
    }

    // benchmarks
    if (benchmarks_given) {
        cfg.benchmark_problems = benchmark_selection(cfg.benchmarks, reg, bad);
        const std::set<std::string> chosen(cfg.benchmark_problems.begin(), cfg.benchmark_problems.end());
        for (std::size_t i = 0; i < cfg.algorithms.size(); ++i) {
            const auto& a = cfg.algorithms[i];
            std::vector<std::string> outside;
            for (const auto& p : a.problems)
                if (!chosen.count(py_str(p))) outside.push_back(py_str(p));
            if (outside.size() > 5) outside.resize(5);
            if (!outside.empty())
                bad("algorithms[" + std::to_string(i) + "] (" + a.key() + ").problems",
                    "not among the campaign's problems: " + join(outside, ", "));
        }
        if (cfg.kind == "builtin" && !cfg.benchmark_problems.empty() && !cfg.campaign.find("pop_fit")) {
            std::map<int, std::vector<std::string>> by_m;
            std::map<std::string, int> m_of(reg.problems.begin(), reg.problems.end());
            for (const auto& n : cfg.benchmark_problems) {
                const auto it = m_of.find(n);
                if (it != m_of.end()) by_m[it->second].push_back(n);
            }
            for (std::size_t i = 0; i < cfg.algorithms.size(); ++i) {
                const auto& a = cfg.algorithms[i];
                if (a.pop < 2) continue;
                for (const auto& [m, names] : by_m) {
                    if (!a.problems.empty()) {
                        bool any = false;
                        for (const auto& p : a.problems)
                            any = any || std::find(names.begin(), names.end(), py_str(p)) != names.end();
                        if (!any) continue;
                    }
                    if (const auto reason = check_pop(a.name, a.pop, m, a.params)) {
                        std::vector<std::string> head(names.begin(), names.begin() + std::min<std::size_t>(3, names.size()));
                        bad("algorithms[" + std::to_string(i) + "] (" + a.name + ")",
                            "for the " + std::to_string(m) + "-objective problems (" + join(head, ", ") +
                                (names.size() > 3 ? ", ..." : "") + "): " + *reason);
                    }
                }
            }
        }
    }

    if (cfg.warm_start) {
        const fs::path src = (*base / cfg.warm_start->source).make_preferred();
        if (!fs::is_regular_file(src)) {
            bad("run.warm_start.from", "file not found: " + src.string());
        } else {
            const auto n = warm_start_objs(src);
            if (n && cfg.n_objs && static_cast<std::int64_t>(*n) != cfg.n_objs)
                bad("run.warm_start.from", "seed population has " + std::to_string(*n) +
                                               " objectives but the run declares n_objs = " + std::to_string(cfg.n_objs) +
                                               "; a warm start cannot change the number of objectives");
        }
    }
    return problems;
}

}   // namespace mootation::campaign

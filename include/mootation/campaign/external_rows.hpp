#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// The rows of external problems (task 24, §24.7.1; external.hpp): the
// folder's problems.toml read into the problem registry beside the
// repository's rows, as the engine sees them (registry.hpp). An array of
// tables, one a problem:
//
//   [[problem]]
//   name = "ext_rosenbrock_n04"        # letters, digits, _ - . ; not a name of the registry
//   family = "extended_rosenbrock"     # a family the registration added
//   n_var = 4
//   lo = -5.0                          # the box: lo and hi (a number, or n_var numbers),
//   hi = 5.0                           #   or bounds = [[lo, hi], ...]; none: no box (-inf, inf)
//   x0 = [-1.2, 1.0, -1.2, 1.0]        # the start of lm, lbfgsb and eki (as a row of MGH)
//   f_opt = 0.0                        # the minimum where it is known, as written (EXT-2) ...
//   f_opt_kind = "exact"               # ... with its kind: exact, computed, certified, best_known, infimum
//   targets = "absolute"               # the report's targets: absolute (default) or relative (needs x0)
//   pop_size = 100                     # the budget pop_size * n_gen where the campaign gives none
//   n_gen = 250                        #   (minimize's defaults)
//   problem_version = 1                # raised by hand when the problem changes
//   n_obj = 1                          # one objective only (EXT-3)
//   [problem.params]                   # anything else: handed to the family's make (EXT-1)
//   scale = 10.0
//
// A key the reader does not know is an error (a misspelt key would otherwise
// fall back to its default unnoticed). Each row's problem is made once while
// it is read, so a wrong size, a second objective or constraints are refused
// at the start of a campaign, not in its jobs.
//
// READINGS
//   EXT-1  the family's parameters are the row's [problem.params] table as
//          toml.hpp reads it (a JSON tree; integers stay integers);
//   EXT-2  f_opt is taken as written, never recomputed; a row with f_opt
//          says its kind (the targets of so_metrics.hpp depend on it);
//   EXT-3  one objective: the multi-objective path needs reference fronts,
//          which an external problem does not give (refused);
//   EXT-4  no constraints (n_con = 0): the campaign's single-objective path
//          (job_so.hpp) hands the methods none (refused);
//   EXT-5  no reference of the campaign is waited for (engine.hpp): a run
//          at one objective reads none, and the Python engine, which builds
//          the registry's problems' ones, does not know the problem; a job
//          the C++ engine would leave to the Python engine refuses the
//          campaign (external_refusal, external.hpp).
// ============================================================================

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../io/npz.hpp"
#include "../json.hpp"
#include "../store/crc32c.hpp"
#include "../toml.hpp"
#include "external.hpp"
#include "registry.hpp"

namespace mootation::campaign {

namespace detail {

inline bool number_of(const json::Value& v, double& out)
{
    if (v.kind == json::Value::Kind::Float) out = v.f;
    else if (v.kind == json::Value::Kind::Int) out = static_cast<double>(v.i);
    else return false;
    return true;
}

// lo or hi: a number for every variable, or n numbers
inline std::vector<double> edge_of(const json::Value& v, int n, const std::string& where)
{
    double x = 0.0;
    if (number_of(v, x)) return std::vector<double>(static_cast<std::size_t>(n), x);
    std::vector<double> out;
    if (v.kind == json::Value::Kind::Array)
        for (const auto& e : v.items) {
            if (!number_of(e, x)) throw std::runtime_error(where + ": a number, or a list of n_var numbers");
            out.push_back(x);
        }
    if (out.size() != static_cast<std::size_t>(n))
        throw std::runtime_error(where + ": a number, or a list of n_var = " + std::to_string(n) + " numbers");
    return out;
}

inline std::string path_text(const std::filesystem::path& p)
{
    const auto s = p.u8string();
    return std::string(s.begin(), s.end());
}

inline bool external_name_ok(const std::string& s)
{
    if (s.empty()) return false;
    for (char c : s)
        if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '-' || c == '.')) return false;
    return true;
}

}   // namespace detail

// The rows of `text` (problems.toml; `origin` names it in messages) added to
// the registry, each with its family's make from `reg`; `sources_sha256`
// the folder's sources' checksum (engine/external_stamp.cmake; "": none)
inline void add_external_rows(ProblemRegistry& preg, const std::string& text, const std::string& origin,
                              const ExternalProblems& reg, const std::string& sources_sha256 = "")
{
    using Kind = json::Value::Kind;
    json::Value doc;
    try {
        doc = toml::parse(text);
    } catch (const std::exception& e) {
        throw std::runtime_error(origin + ": " + e.what());
    }
    for (const auto& key : doc.keys)
        if (key != "problem") throw std::runtime_error(origin + ": '" + key + "': only [[problem]] tables are read");
    const json::Value* rows = doc.find("problem");
    if (rows == nullptr) return;
    if (rows->kind != Kind::Array) throw std::runtime_error(origin + ": problem: an array of tables, [[problem]]");
    std::vector<std::string> registry_families;
    for (const auto& [name, p] : preg.problems)
        if (!p.external) registry_families.push_back(p.family);
    static const std::vector<std::string> known = {"name",     "family",     "n_var",  "n_obj",  "lo",
                                                   "hi",       "bounds",     "x0",     "f_opt",  "f_opt_kind",
                                                   "targets",  "pop_size",   "n_gen",  "problem_version", "params"};
    std::size_t row = 0;
    for (const auto& o : rows->items) {
        ++row;
        std::string where = origin + ": [[problem]] " + std::to_string(row);
        if (o.kind != Kind::Object) throw std::runtime_error(where + ": a table");
        for (const auto& key : o.keys)
            if (std::find(known.begin(), known.end(), key) == known.end())
                throw std::runtime_error(where + ": unknown key '" + key + "'");
        auto text_of = [&](const char* key, bool required) -> std::string {
            const json::Value* v = o.find(key);
            if (v == nullptr) {
                if (required) throw std::runtime_error(where + ": " + key + " is required");
                return {};
            }
            if (v->kind != Kind::String) throw std::runtime_error(where + ": " + key + ": a string");
            return v->s;
        };
        auto int_of = [&](const char* key, std::int64_t fallback, std::int64_t least) {
            const json::Value* v = o.find(key);
            if (v == nullptr) return fallback;
            if (v->kind != Kind::Int || v->i < least)
                throw std::runtime_error(where + ": " + key + ": an integer >= " + std::to_string(least));
            return v->i;
        };

        RegistryProblem p;
        p.name = text_of("name", true);
        where += " (" + p.name + ")";
        if (!detail::external_name_ok(p.name))
            throw std::runtime_error(where + ": a name of letters, digits, '_', '-' and '.'");
        if (preg.problems.count(p.name))
            throw std::runtime_error(where + ": the name is taken (" +
                                     (preg.problems.at(p.name).external ? "by another external row"
                                                                        : "by the problem registry") + ")");
        p.family = text_of("family", true);
        if (std::find(registry_families.begin(), registry_families.end(), p.family) != registry_families.end())
            throw std::runtime_error(where + ": the family '" + p.family + "' is one of the problem registry's");
        const ExternalFamily* fam = reg.find(p.family);
        if (fam == nullptr) {
            std::string have;
            for (const auto& [f, x] : reg.families()) have += (have.empty() ? "" : ", ") + f;
            throw std::runtime_error(where + ": the family '" + p.family + "' is not registered (registered: " +
                                     (have.empty() ? "none" : have) + ")");
        }
        p.n_var = static_cast<int>(int_of("n_var", -1, 1));
        if (p.n_var < 1) throw std::runtime_error(where + ": n_var is required");
        if (int_of("n_obj", 1, 1) != 1)
            throw std::runtime_error(where + ": n_obj: one objective only (EXT-3; incompatible by construction)");
        p.n_obj = 1;
        p.n_con = 0;
        // the box
        const double inf = std::numeric_limits<double>::infinity();
        std::vector<double> lo(static_cast<std::size_t>(p.n_var), -inf), hi(static_cast<std::size_t>(p.n_var), inf);
        const json::Value* b = o.find("bounds");
        if (b != nullptr && (o.find("lo") || o.find("hi")))
            throw std::runtime_error(where + ": the box by bounds or by lo and hi, not both");
        if (b != nullptr) {
            if (b->kind != Kind::Array || b->items.size() != static_cast<std::size_t>(p.n_var))
                throw std::runtime_error(where + ": bounds: n_var pairs [lo, hi]");
            for (std::size_t j = 0; j < b->items.size(); ++j) {
                const json::Value& pair = b->items[j];
                if (pair.kind != Kind::Array || pair.items.size() != 2 || !detail::number_of(pair.items[0], lo[j]) ||
                    !detail::number_of(pair.items[1], hi[j]))
                    throw std::runtime_error(where + ": bounds: n_var pairs [lo, hi]");
            }
        }
        if (const json::Value* v = o.find("lo")) lo = detail::edge_of(*v, p.n_var, where + ": lo");
        if (const json::Value* v = o.find("hi")) hi = detail::edge_of(*v, p.n_var, where + ": hi");
        for (std::size_t j = 0; j < lo.size(); ++j) {
            if (std::isnan(lo[j]) || std::isnan(hi[j]) || !(lo[j] < hi[j]))
                throw std::runtime_error(where + ": the box: lo < hi in every variable (variable " +
                                         std::to_string(j) + ")");
            p.file_bounds.emplace_back(lo[j], hi[j]);
        }
        p.bounds = p.file_bounds;
        // the start, the minimum, the targets
        if (const json::Value* v = o.find("x0")) {
            p.x0 = detail::edge_of(*v, p.n_var, where + ": x0");
            if (v->kind != Kind::Array) throw std::runtime_error(where + ": x0: a list of n_var numbers");
        }
        if (const json::Value* v = o.find("f_opt")) {
            double f = 0.0;
            if (!detail::number_of(*v, f) || std::isnan(f)) throw std::runtime_error(where + ": f_opt: a number");
            p.f_opt = f;
            p.f_opt_kind = text_of("f_opt_kind", true);
            static const std::vector<std::string> kinds = {"exact", "computed", "certified", "best_known", "infimum"};
            if (std::find(kinds.begin(), kinds.end(), p.f_opt_kind) == kinds.end())
                throw std::runtime_error(where + ": f_opt_kind: exact, computed, certified, best_known or infimum");
        } else if (o.find("f_opt_kind")) {
            throw std::runtime_error(where + ": f_opt_kind without f_opt");
        }
        p.targets = text_of("targets", false);
        if (p.targets.empty()) p.targets = "absolute";
        if (p.targets != "absolute" && p.targets != "relative")
            throw std::runtime_error(where + ": targets: absolute or relative");
        if (p.targets == "relative" && p.x0.empty())
            throw std::runtime_error(where + ": targets = relative needs x0 (the targets are fractions of f(x0) - f_opt)");
        p.pop_size = int_of("pop_size", 100, 1);
        p.n_gen = int_of("n_gen", 250, 1);
        p.problem_version = static_cast<int>(int_of("problem_version", 1, 1));
        json::Value params = json::Value::object();
        if (const json::Value* v = o.find("params")) {
            if (v->kind != Kind::Object) throw std::runtime_error(where + ": params: a table, [problem.params]");
            params = *v;
        }
        // made once now: its sizes checked against the row
        const ExternalMake make = fam->make;
        const int n_var = p.n_var;
        std::shared_ptr<suite::Native> made;
        try {
            made = make(params, n_var);
        } catch (const std::exception& e) {
            throw std::runtime_error(where + ": the family's make refused the row: " + e.what());
        }
        if (!made) throw std::runtime_error(where + ": the family's make gave no problem");
        if (made->n_var != p.n_var)
            throw std::runtime_error(where + ": the problem made has " + std::to_string(made->n_var) +
                                     " variables, the row says n_var = " + std::to_string(p.n_var));
        if (made->n_obj != 1)
            throw std::runtime_error(where + ": the problem made has " + std::to_string(made->n_obj) +
                                     " objectives: one only (EXT-3; incompatible by construction)");
        if (made->n_con != 0)
            throw std::runtime_error(where + ": the problem made has constraints: none at one objective (EXT-4)");
        p.external = [make, params, n_var]() { return make(params, n_var); };
        // meta.json's external: what the run was made of
        const std::string dumped = json::dump(o);
        const std::uint32_t crc =
            store::crc32c(reinterpret_cast<const std::uint8_t*>(dumped.data()), dumped.size());
        char hex[9];
        std::snprintf(hex, sizeof hex, "%08x", static_cast<unsigned>(crc));
        json::Value meta = json::Value::object();
        meta.set("family", json::Value::string(p.family));
        meta.set("version", json::Value::string(fam->version));
        meta.set("row_crc32c", json::Value::string(hex));
        meta.set("sources_sha256",
                 sources_sha256.empty() ? json::Value::null() : json::Value::string(sources_sha256));
        p.external_meta = std::move(meta);
        preg.problems.emplace(p.name, std::move(p));
    }
}

// the rows of the file at `path` (problems.toml)
inline void add_external(ProblemRegistry& preg, const std::filesystem::path& path, const ExternalProblems& reg,
                         const std::string& sources_sha256 = "")
{
    const auto bytes = npz::read_file(path);
    add_external_rows(preg, std::string(bytes.begin(), bytes.end()), detail::path_text(path), reg, sources_sha256);
}

}   // namespace mootation::campaign

#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// Settings — a plain struct describing one optimization run, plus a plain
// `key = value` text format for it.
//
// There is no builder, no registry and no inheritance here on purpose: the
// struct IS the configuration. Fill it in code, or load it from a file, or
// load it from a file and then override two fields in code. All three are the
// same struct.
//
// FILE FORMAT
//   One `key = value` per line. `#` and `;` start a comment. Blank lines are
//   ignored. Keys are case-sensitive. Lists are comma-separated.
//   Unknown keys are an error, not a silent no-op — a typo in a config file
//   must not quietly give you a different run than you asked for.
//
//   # ---- the run --------------------------------------------------------
//   algorithm   = nsga3        # any name from algorithm_names()
//   pop_size    = 92
//   max_gen     = 300
//   seed        = 42
//
//   # ---- the problem ----------------------------------------------------
//   n_vars      = 7            # only needed when lower/upper are scalars
//   n_objs      = 3
//   n_cons      = 0            # constraint values per individual (0 = none)
//   lower       = 0            # one value broadcasts to all n_vars ...
//   upper       = 1
//   # lower     = 0, 0, -5     # ... or one value per variable
//   constraints = none         # none | feasibility | cdp | eps_constraint
//
//   # ---- optional algorithm knobs ---------------------------------------
//   # Anything not listed keeps the algorithm's own paper default.
//   # eta_c eta_m pc pm T delta nr kappa K n_clusters theta alpha F CR div
//   eta_c       = 30
//
// A knob an algorithm does not have is REPORTED (Result::ignored), not
// dropped: a run configured with an ignored parameter is not the run the
// caller asked for.
// ============================================================================

#include <algorithm>
#include <cctype>
#include <fstream>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "constraint_mode.hpp"

namespace mootation {

// The optional per-algorithm knobs, in one place. Anything here may be set;
// whether a given algorithm HAS it is discovered at compile time and reported
// at run time. Keep this list in sync with embed.hpp's setter table.
inline const std::vector<std::string>& knob_names() {
    static const std::vector<std::string> v = {
        "eta_c", "eta_m", "pc", "pm", "T", "delta", "nr", "kappa",
        "K", "n_clusters", "theta", "alpha", "F", "CR", "div"
    };
    return v;
}

struct Settings {
    // ── the run ──────────────────────────────────────────────────────────
    std::string algorithm = "nsga2";
    int         pop_size  = 100;
    int         max_gen   = 250;
    unsigned    seed      = 0;

    // ── the problem ──────────────────────────────────────────────────────
    std::vector<double> lower;          // one entry per variable
    std::vector<double> upper;
    int                 n_objs = 2;
    int                 n_cons = 0;     // constraint values per individual
    ConstraintMode      constraints = ConstraintMode::NONE;  // see constraint_mode.hpp

    // ── optional algorithm knobs (empty = keep the paper default) ────────
    std::map<std::string, double> params;

    int n_vars() const { return static_cast<int>(lower.size()); }

    // Convenience for the common "same box for every variable" case.
    void set_box(int n_vars_, double lo, double hi) {
        lower.assign(static_cast<std::size_t>(n_vars_), lo);
        upper.assign(static_cast<std::size_t>(n_vars_), hi);
    }

    // Throws std::invalid_argument with a message naming the offending field.
    void validate() const {
        auto bad = [](const std::string& m) {
            throw std::invalid_argument("mootation::Settings: " + m);
        };
        if (algorithm.empty())        bad("algorithm is empty");
        if (pop_size < 2)             bad("pop_size must be >= 2, got " + std::to_string(pop_size));
        if (max_gen < 0)              bad("max_gen must be >= 0, got " + std::to_string(max_gen));
        if (lower.empty())            bad("lower/upper are empty — no decision variables");
        if (lower.size() != upper.size())
            bad("lower.size()=" + std::to_string(lower.size()) +
                " != upper.size()=" + std::to_string(upper.size()));
        for (std::size_t i = 0; i < lower.size(); ++i)
            if (!(lower[i] < upper[i]))
                bad("lower[" + std::to_string(i) + "]=" + std::to_string(lower[i]) +
                    " must be < upper[" + std::to_string(i) + "]=" + std::to_string(upper[i]));
        if (n_objs < 1)               bad("n_objs must be >= 1, got " + std::to_string(n_objs));
        if (n_cons < 0)               bad("n_cons must be >= 0, got " + std::to_string(n_cons));
        if (constraints != ConstraintMode::NONE && n_cons == 0)
            bad("constraints is enabled but n_cons = 0 — there is nothing to constrain");
        for (const auto& kv : params) {
            const auto& kn = knob_names();
            if (std::find(kn.begin(), kn.end(), kv.first) == kn.end())
                bad("unknown knob '" + kv.first + "' in params");
        }
    }

    static Settings from_file(const std::string& path);
    static Settings from_string(const std::string& text,
                                const std::string& origin = "<string>");
    std::string     to_string() const;
    void            to_file(const std::string& path) const;
};

// ── parsing helpers (internal) ───────────────────────────────────────────────
namespace detail {

inline std::string trim(const std::string& s) {
    std::size_t a = 0, b = s.size();
    while (a < b && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
    while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
    return s.substr(a, b - a);
}

inline std::vector<double> parse_list(const std::string& v, const std::string& where) {
    std::vector<double> out;
    std::stringstream ss(v);
    std::string item;
    while (std::getline(ss, item, ',')) {
        std::string t = trim(item);
        if (t.empty()) continue;
        try {
            std::size_t used = 0;
            double d = std::stod(t, &used);
            if (used != t.size())
                throw std::invalid_argument("trailing characters");
            out.push_back(d);
        } catch (const std::exception&) {
            throw std::invalid_argument(where + ": '" + t + "' is not a number");
        }
    }
    if (out.empty())
        throw std::invalid_argument(where + ": empty list");
    return out;
}

inline double parse_num(const std::string& v, const std::string& where) {
    auto l = parse_list(v, where);
    if (l.size() != 1)
        throw std::invalid_argument(where + ": expected one number, got " +
                                    std::to_string(l.size()));
    return l[0];
}

inline int parse_int(const std::string& v, const std::string& where) {
    double d = parse_num(v, where);
    if (d != static_cast<double>(static_cast<long long>(d)))
        throw std::invalid_argument(where + ": expected an integer, got " + v);
    return static_cast<int>(d);
}

inline ConstraintMode parse_constraints(const std::string& v, const std::string& where) {
    std::string t;
    for (char c : trim(v)) t += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (t == "none")           return ConstraintMode::NONE;
    if (t == "feasibility")    return ConstraintMode::FEASIBILITY;
    if (t == "cdp")            return ConstraintMode::CDP;
    if (t == "eps_constraint") return ConstraintMode::EPS_CONSTRAINT;
    throw std::invalid_argument(where + ": constraints must be "
                                "none|feasibility|cdp|eps_constraint, got '" + v + "'");
}

inline const char* constraints_name(ConstraintMode m) {
    switch (m) {
        case ConstraintMode::FEASIBILITY: return "feasibility";
        case ConstraintMode::CDP:         return "cdp";
        case ConstraintMode::EPS_CONSTRAINT: return "eps_constraint";
        default:                          return "none";
    }
}

} // namespace detail

inline Settings Settings::from_string(const std::string& text, const std::string& origin) {
    Settings s;
    std::optional<int> n_vars_decl;
    std::istringstream in(text);
    std::string line;
    int lineno = 0;

    while (std::getline(in, line)) {
        ++lineno;
        // Strip comments. Nothing in this format quotes '#' or ';', so a plain
        // scan is exact rather than merely convenient.
        std::size_t cut = line.find_first_of("#;");
        if (cut != std::string::npos) line = line.substr(0, cut);
        line = detail::trim(line);
        if (line.empty()) continue;

        std::size_t eq = line.find('=');
        const std::string where =
            origin + ":" + std::to_string(lineno);
        if (eq == std::string::npos)
            throw std::invalid_argument(where + ": expected 'key = value', got '" + line + "'");

        std::string key = detail::trim(line.substr(0, eq));
        std::string val = detail::trim(line.substr(eq + 1));
        if (key.empty())
            throw std::invalid_argument(where + ": empty key");
        if (val.empty())
            throw std::invalid_argument(where + ": empty value for '" + key + "'");

        const std::string w = where + " (" + key + ")";
        if      (key == "algorithm")   s.algorithm = val;
        else if (key == "pop_size")    s.pop_size  = detail::parse_int(val, w);
        else if (key == "max_gen")     s.max_gen   = detail::parse_int(val, w);
        else if (key == "seed")        s.seed      = static_cast<unsigned>(detail::parse_int(val, w));
        else if (key == "n_objs")      s.n_objs    = detail::parse_int(val, w);
        else if (key == "n_cons")      s.n_cons    = detail::parse_int(val, w);
        else if (key == "n_vars")      n_vars_decl = detail::parse_int(val, w);
        else if (key == "lower")       s.lower     = detail::parse_list(val, w);
        else if (key == "upper")       s.upper     = detail::parse_list(val, w);
        else if (key == "constraints") s.constraints = detail::parse_constraints(val, w);
        else {
            const auto& kn = knob_names();
            if (std::find(kn.begin(), kn.end(), key) == kn.end())
                throw std::invalid_argument(
                    where + ": unknown key '" + key + "'. Known keys: algorithm, "
                    "pop_size, max_gen, seed, n_vars, n_objs, n_cons, lower, upper, "
                    "constraints, and the knobs listed in knob_names()");
            s.params[key] = detail::parse_num(val, w);
        }
    }

    // A scalar bound broadcasts to n_vars. This is the only implicit step in
    // the format, and it needs n_vars to be known.
    auto broadcast = [&](std::vector<double>& v, const char* name) {
        if (v.size() == 1 && n_vars_decl && *n_vars_decl > 1)
            v.assign(static_cast<std::size_t>(*n_vars_decl), v[0]);
        else if (n_vars_decl && static_cast<int>(v.size()) != *n_vars_decl && v.size() != 1)
            throw std::invalid_argument(
                std::string(origin) + ": " + name + " has " + std::to_string(v.size()) +
                " entries but n_vars = " + std::to_string(*n_vars_decl));
    };
    broadcast(s.lower, "lower");
    broadcast(s.upper, "upper");

    if (n_vars_decl && s.lower.empty() && s.upper.empty())
        throw std::invalid_argument(
            std::string(origin) + ": n_vars given but neither lower nor upper");

    s.validate();
    return s;
}

inline Settings Settings::from_file(const std::string& path) {
    std::ifstream f(path);
    if (!f)
        throw std::runtime_error("mootation::Settings::from_file: cannot open '" + path + "'");
    std::stringstream buf;
    buf << f.rdbuf();
    return from_string(buf.str(), path);
}

inline std::string Settings::to_string() const {
    std::ostringstream o;
    auto list = [&](const std::vector<double>& v) {
        for (std::size_t i = 0; i < v.size(); ++i) {
            if (i) o << ", ";
            o << v[i];
        }
    };
    o << "# mootation settings\n";
    o << "algorithm   = " << algorithm << "\n";
    o << "pop_size    = " << pop_size  << "\n";
    o << "max_gen     = " << max_gen   << "\n";
    o << "seed        = " << seed      << "\n";
    o << "n_objs      = " << n_objs    << "\n";
    o << "n_cons      = " << n_cons    << "\n";
    o << "constraints = " << detail::constraints_name(constraints) << "\n";
    o << "lower       = "; list(lower); o << "\n";
    o << "upper       = "; list(upper); o << "\n";
    if (!params.empty()) {
        o << "\n# algorithm knobs\n";
        for (const auto& kv : params)
            o << kv.first << " = " << kv.second << "\n";
    }
    return o.str();
}

inline void Settings::to_file(const std::string& path) const {
    std::ofstream f(path);
    if (!f)
        throw std::runtime_error("mootation::Settings::to_file: cannot open '" + path + "'");
    f << to_string();
    if (!f)
        throw std::runtime_error("mootation::Settings::to_file: write failed for '" + path + "'");
}

} // namespace mootation

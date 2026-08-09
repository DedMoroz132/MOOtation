#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// Saving a population, and loading one back.
//
// The point of the pair is restarting: finish a run, keep the population, and
// start the next one from it — with the same settings, or with different ones.
// That is a warm start, not a checkpoint. Nothing about the algorithm's
// internal state is written: no RNG position, no neighbourhood tables, no
// per-core bookkeeping. Only what every algorithm shares — the decision
// variables, the objectives, the constraint violation — because that is
// exactly what `Optimizer::setup_with_seed` can plant, and it is what makes a
// population saved by NSGA-II loadable by MOEA/D.
//
// A run resumed this way is NOT a continuation of the original: the random
// stream restarts, so it will not reproduce what the uninterrupted run would
// have done. It starts from the same place, which for a long optimization is
// usually what was wanted anyway.
//
// FORMAT — CSV with a `#` preamble, so pandas and numpy read it directly:
//
//     # mootation population v1
//     # algorithm=nsga2 pop_size=100 n_gen=250
//     # n_vars=10 n_bin=0 n_objs=2 n_lims=0
//     x1,x2,...,f1,f2,cv
//     5.0e-01,2.0e-01,...,5.0e-01,7.1e-01,0.0
//
//     pandas.read_csv(path, comment="#")
//
// The preamble carries whatever settings the caller passed in, so one file
// answers both "what is the population" and "what produced it".
// ============================================================================

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iomanip>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../data_vault.hpp"
#include "../size_mismatch.hpp"

namespace mootation::io {

struct Population {
    std::vector<std::vector<double>> variables;
    std::vector<std::vector<int>>    binary_variables;
    std::vector<std::vector<double>> objectives;
    std::vector<double>              cv;
    // Per-constraint values, one row per individual, `<= 0` meaning satisfied.
    // Empty for an unconstrained run, and empty for any file written before
    // these columns existed. Carried because cv is a SUM: it cannot be taken
    // apart again, and a warm start that reported every constraint satisfied
    // because it only had the total would be worse than refusing to start.
    std::vector<std::vector<double>> limits;
    // Everything the `#` preamble carried: algorithm, pop_size, n_gen, and
    // whatever else the writer chose to record.
    std::map<std::string, std::string> meta;

    std::size_t size() const { return variables.size(); }
    int n_vars() const { return variables.empty() ? 0 : (int)variables[0].size(); }
    int n_objs() const { return objectives.empty() ? 0 : (int)objectives[0].size(); }
    int n_lims() const { return limits.empty()    ? 0 : (int)limits[0].size(); }
};

// ── Saving ──────────────────────────────────────────────────────────────────

template <typename Ind_t>
void save_population(DataVault<Ind_t>& vault,
                     const std::string& filename,
                     const std::map<std::string, std::string>& meta = {},
                     const std::string& note = "")
{
    // NOT active_n(): steady-state cores keep a persistent scratch slot at
    // active index pop_size(), and writing it out would put an unselected
    // offspring into the saved population — which would then be loaded back as
    // if it had been chosen. See data_vault.hpp, "SCRATCH SLOTS AND CONSUMERS".
    const int n    = std::min<int>(static_cast<int>(vault.active_n()),
                                   vault.pop_size());
    const int nvar = vault.vars_n();
    const int nbin = vault.bin_vars_n();
    const int nobj = vault.objs_n();
    const int nlim = vault.lims_n();

    std::vector<int> order(static_cast<std::size_t>(n));
    std::iota(order.begin(), order.end(), 0);
    if (nobj > 0) {
        // Sorted by the first objective so a two-objective front reads as a
        // front. With no objectives there is nothing to sort by, and indexing
        // objectives_of(a)[0] would be an out-of-bounds read.
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            return vault.objectives_of(a)[0] < vault.objectives_of(b)[0];
        });
    }

    std::ofstream f(filename);
    if (!f) throw std::runtime_error("save_population: cannot open " + filename);

    f << "# mootation population v1\n";
    if (!note.empty()) {
        // On its own line and deliberately WITHOUT '=': load_population scans
        // comment lines for key=value pairs split on spaces, so a multi-word
        // note stored as `note=...` would lose everything after the first
        // space. Free text and metadata therefore live on separate lines.
        f << "# " << note << "\n";
    }
    if (!meta.empty()) {
        f << "#";
        for (const auto& kv : meta) f << " " << kv.first << "=" << kv.second;
        f << "\n";
    }
    f << "# n_vars=" << nvar << " n_bin=" << nbin << " n_objs=" << nobj
      << " n_lims=" << nlim << "\n";

    // Header row, without a leading '#': it is the CSV header, and pandas
    // needs to see it after comment lines are dropped.
    bool first = true;
    auto col = [&](const std::string& name) {
        if (!first) f << ",";
        f << name;
        first = false;
    };
    for (int j = 0; j < nvar; ++j) col("x" + std::to_string(j + 1));
    for (int j = 0; j < nbin; ++j) col("b" + std::to_string(j + 1));
    for (int k = 0; k < nobj; ++k) col("f" + std::to_string(k + 1));
    col("cv");
    // The per-constraint columns go AFTER cv, so a file written before they
    // existed still parses: the loader simply finds nothing past cv. Anything
    // reading by column name is unaffected either way.
    for (int k = 0; k < nlim; ++k) col("g" + std::to_string(k + 1));
    f << "\n";

    f << std::scientific << std::setprecision(12);
    for (int idx : order) {
        bool sep = false;
        auto put = [&](double v) {
            if (sep) f << ",";
            f << v;
            sep = true;
        };
        for (int j = 0; j < nvar; ++j) put(vault.get_variable(idx, j));
        for (int j = 0; j < nbin; ++j) put(vault.get_bin_variable(idx, j));
        for (int k = 0; k < nobj; ++k) put(vault.objectives_of(idx)[k]);
        put(vault.get_cv(idx));
        for (int k = 0; k < nlim; ++k) put(vault.limits_of(idx)[k]);
        f << "\n";
    }
    if (!f) throw std::runtime_error("save_population: write failed for " + filename);
}

// A free-text note and nothing else — the common case in the examples.
template <typename Ind_t>
void save_population(DataVault<Ind_t>& vault,
                     const std::string& filename,
                     const std::string& note)
{
    save_population(vault, filename, std::map<std::string, std::string>{}, note);
}

// ── Loading ─────────────────────────────────────────────────────────────────

inline Population load_population(const std::string& filename)
{
    std::ifstream f(filename);
    if (!f) throw std::runtime_error("load_population: cannot open " + filename);

    Population pop;
    int nvar = -1, nbin = 0, nobj = -1, nlim = 0;
    std::vector<std::string> header;
    std::string line;

    auto trim = [](std::string s) {
        std::size_t a = 0, b = s.size();
        while (a < b && std::isspace((unsigned char)s[a])) ++a;
        while (b > a && std::isspace((unsigned char)s[b - 1])) --b;
        return s.substr(a, b - a);
    };
    auto split = [](const std::string& s, char sep) {
        std::vector<std::string> out;
        std::stringstream ss(s);
        std::string item;
        while (std::getline(ss, item, sep)) out.push_back(item);
        return out;
    };

    while (std::getline(f, line)) {
        std::string t = trim(line);
        if (t.empty()) continue;

        if (t[0] == '#') {
            // key=value pairs anywhere in a comment line; the shape line
            // (n_vars=... n_bin=... n_objs=...) is read out of the same map.
            for (const auto& tok : split(t.substr(1), ' ')) {
                auto kv = trim(tok);
                auto eq = kv.find('=');
                if (eq == std::string::npos) continue;
                std::string k = trim(kv.substr(0, eq));
                std::string v = trim(kv.substr(eq + 1));
                if      (k == "n_vars") nvar = std::stoi(v);
                else if (k == "n_bin")  nbin = std::stoi(v);
                else if (k == "n_objs") nobj = std::stoi(v);
                else if (k == "n_lims") nlim = std::stoi(v);
                else pop.meta[k] = v;
            }
            continue;
        }

        if (header.empty()) {           // the CSV header row
            for (auto& h : split(t, ',')) header.push_back(trim(h));
            if (nvar < 0 || nobj < 0) {
                // No shape preamble: infer it from the column names, so a file
                // trimmed by hand still loads.
                nvar = nbin = nobj = nlim = 0;
                for (const auto& h : header) {
                    if (h.empty()) continue;
                    if (h[0] == 'x') ++nvar;
                    else if (h[0] == 'b') ++nbin;
                    else if (h[0] == 'f') ++nobj;
                    else if (h[0] == 'g') ++nlim;   // 'cv' starts with 'c'
                }
            }
            continue;
        }

        auto cells = split(t, ',');
        const std::size_t want = static_cast<std::size_t>(nvar + nbin + nobj);
        if (cells.size() < want)
            throw std::runtime_error(
                "load_population: " + filename + ": a row has " +
                std::to_string(cells.size()) + " values, expected at least " +
                std::to_string(want));

        std::vector<double> x, fo;
        std::vector<int> b;
        std::size_t c = 0;
        try {
            for (int j = 0; j < nvar; ++j) x.push_back(std::stod(cells[c++]));
            for (int j = 0; j < nbin; ++j) b.push_back((int)std::lround(std::stod(cells[c++])));
            for (int k = 0; k < nobj; ++k) fo.push_back(std::stod(cells[c++]));
        } catch (const std::exception&) {
            throw std::runtime_error(
                "load_population: " + filename + ": a value is not a number");
        }
        double cv = 0.0;
        if (c < cells.size()) {
            try { cv = std::stod(cells[c]); } catch (const std::exception&) { cv = 0.0; }
            ++c;
        }

        // Per-constraint values, if this file carries them. A file written
        // before these columns existed simply runs out of cells here, and
        // `limits` stays empty — which is what tells a warm start that it
        // cannot restore a constrained run from it.
        std::vector<double> lm;
        for (int k = 0; k < nlim && c < cells.size(); ++k, ++c) {
            try { lm.push_back(std::stod(cells[c])); }
            catch (const std::exception&) {
                throw std::runtime_error(
                    "load_population: " + filename + ": constraint value '" +
                    cells[c] + "' is not a number");
            }
        }

        pop.variables.push_back(std::move(x));
        pop.binary_variables.push_back(std::move(b));
        pop.objectives.push_back(std::move(fo));
        pop.cv.push_back(cv);
        if (!lm.empty()) pop.limits.push_back(std::move(lm));
    }

    if (pop.variables.empty())
        throw std::runtime_error("load_population: " + filename + " has no rows");
    return pop;
}

// ── Resizing a loaded population to the run that will use it ────────────────

// The enum itself lives in <mootation/size_mismatch.hpp> so that Settings can
// name it without pulling in the vault. Re-exported here under the name it has
// always had in this namespace.
using SizeMismatch = mootation::SizeMismatch;

// A population saved by one run rarely matches the next run's pop_size — and
// for M2M or the NSGA-III family, pop_size is constrained by the algorithm, so
// the mismatch is routine rather than exceptional. The policy is the caller's
// because the right answer depends on why they are restarting.
inline void fit_population(Population& pop, int pop_size, SizeMismatch policy)
{
    const int have = static_cast<int>(pop.size());
    if (have == pop_size) return;

    if (policy == SizeMismatch::Error)
        throw std::invalid_argument(
            "the seed population has " + std::to_string(have) +
            " individuals but pop_size is " + std::to_string(pop_size) +
            "; pass truncate or pad to resize it");

    if (have > pop_size) {
        // Truncate: the file is sorted by the first objective, so this keeps a
        // contiguous slice of the front rather than a random sample. Not a
        // nondominated-sort truncation — that would need the whole machinery
        // here for a warm start that the algorithm is about to reshape anyway.
        pop.variables.resize(pop_size);
        pop.binary_variables.resize(pop_size);
        pop.objectives.resize(pop_size);
        pop.cv.resize(pop_size);
        // Guarded: resizing an EMPTY limits vector would manufacture pop_size
        // rows of no constraints, and an unconstrained population would start
        // claiming to carry constraint values.
        if (!pop.limits.empty()) pop.limits.resize(pop_size);
        return;
    }

    if (policy == SizeMismatch::Pad) {
        // Pad by cycling. Duplicates in a seed population are harmless: the
        // first generation's variation separates them, and any alternative
        // (random fill, mutation) would need an RNG this function does not
        // have and should not own.
        const int base = have;
        for (int i = have; i < pop_size; ++i) {
            const std::size_t src = static_cast<std::size_t>(i % base);
            pop.variables.push_back(pop.variables[src]);
            pop.binary_variables.push_back(pop.binary_variables[src]);
            pop.objectives.push_back(pop.objectives[src]);
            pop.cv.push_back(pop.cv[src]);
            if (!pop.limits.empty()) pop.limits.push_back(pop.limits[src]);
        }
        return;
    }

    throw std::invalid_argument(
        "the seed population has " + std::to_string(have) +
        " individuals but pop_size is " + std::to_string(pop_size));
}

} // namespace mootation::io

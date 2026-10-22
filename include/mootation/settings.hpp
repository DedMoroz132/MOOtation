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
//   # max_evaluations = 30000  # instead of max_gen: stop on evaluations spent
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
//   # ---- warm start (optional) -----------------------------------------
//   # seed_population  = pop.csv   # a file written by io::save_population
//   # on_size_mismatch = error     # error | truncate | pad
//
//   # ---- optional algorithm knobs ---------------------------------------
//   # Anything not listed keeps the algorithm's own paper default.
//   # eta_c eta_m pc pm T delta nr kappa K n_clusters theta alpha F CR div
//   # normalize (0 or 1: the objective normalization a few cores switch)
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
#include <utility>
#include <vector>

#include "constraint_mode.hpp"
#include "size_mismatch.hpp"

namespace mootation {

// The optional per-algorithm knobs, in one place. Anything here may be set;
// whether a given algorithm HAS it is discovered at compile time and reported
// at run time. Keep this list in sync with embed.hpp's setter table.
inline const std::vector<std::string>& knob_names() {
    static const std::vector<std::string> v = {
        "eta_c", "eta_m", "pc", "pm", "T", "delta", "nr", "kappa",
        "K", "n_clusters", "theta", "alpha", "F", "CR", "div", "normalize",
        "mutation_scale", "mixture_q", "blx_alpha", "sbx_var_prob",
        "step_share", "F_spread",
        // task 7 of 2026-10-10: handed to a core's set_param (below)
        "ref_delta", "de_p", "de_memory",
        "p_de", "sa_chains", "sa_reseed", "sa_t_ratio", "sa_s0", "sa_s_end",
        "sa_archive_cap", "sa_psa_alpha", "sa_adapt_window",
        "coords_warmup", "coords_update",
        "spu_frac", "inner_archive_cap", "archive_reuse_p",
        "dn_cf",
        // task 9 of 2026-10-18: smpso (algorithms/smpso.hpp)
        "vmax_frac", "vel_on_bound_low", "vel_on_bound_up", "w_min", "w_max", "c_min", "c_max",
        "turbulence_k", "turbulence_p", "archive_size", "eps_archive_eta",
        // task 9: r2_emoa (algorithms/r2_emoa.hpp)
        "r2_weights", "r2_offset",
        // task 9: bc_spx, bc_rex (algorithms/bc_spx.hpp, bc_rex.hpp); paes (algorithms/paes.hpp)
        "bc_lambda", "paes_depth",
        // task 9: dmulti_mads (algorithms/dmulti_mads.hpp)
        "mads_tau", "mads_delta0",
        // task 9: multiglods (algorithms/multiglods.hpp)
        "glods_search_every", "glods_local_tol",
        // task 9: rm_meda (algorithms/rm_meda.hpp)
        "rm_extension",
        // task 9: the scalarization methods (python/mootation/scalar) and the
        // polishing every algorithm takes (minimize.py); no core has them
        "n_sub", "sub_budget", "fd_step", "local_tol", "cma_sigma0", "eps_objective",
        "polish_share", "polish_points", "mosqp_tau", "mosqp_mu", "mosqp_beta", "mosqp_spread",
        // task 10 of 2026-10-18, §7
        "n_samples",
        // task 13 of 2026-10-19, §3
        "hv_exact_max_m",
        // task 14 of 2026-10-20, §11
        "race_eps", "race_delta", "race_max_samples", "race_abs_tol",
        // task 14, supplement 3, Д8.3
        "hv_incremental",
        // task 15, §2 and §6
        "n_offspring", "n_ref_points", "ada"
    };
    return v;
}

// The knobs that came after the per-knob setters (task 7 of 2026-10-10): a
// core takes them through bool set_param(name, value) and bool set_word(name,
// word), and one without the method, or answering false, reports them
// ignored, as any knob it lacks.
//   ref_delta      sms_emoa, hype: the reference point nad + δ (nad − ideal)
//                  instead of their papers' absolute offsets
//   de_p, de_memory, de_adapt, de_adapt_weight, de_archive
//                  crossover = de / de_pbest (operators/de_crossover.hpp)
//   portfolio, portfolio_credit
//                  crossover = portfolio (operators/real_crossover.hpp)
//   p_de, sa_*     mosa (algorithms/mosa.hpp)
//   coords, coords_warmup, coords_update
//                  the switchable algorithms and mosa's sbx step: the variation
//                  in learned coordinates (operators/learned_coords.hpp)
//   survival, spu_frac
//                  sms_emoa, nsga2, nsga3: the stochastic population update
//   inner_archive, inner_archive_cap, archive_reuse_p
//                  sms_emoa, nsga2, nsga3, ibea_eplus: an archive the
//                  algorithm sees and draws parents from
//                  (operators/inner_archive.hpp)
//   bin_crossover  sms_emoa, nsga2: uniform or none for the binary genome
//   mating         nsga2: tournament, or each member a parent once
//   dn_cf          dn_nsga2: the crowding factor of its niching mating pool
//   vmax_frac, vel_on_bound_low, vel_on_bound_up, w_min, w_max, c_min, c_max,
//   turbulence_k, turbulence_p, archive_size, eps_archive_eta
//                  smpso: the velocity bound and the velocity at the box, the
//                  inertia and acceleration ranges, the turbulence, the
//                  leaders' bound and omopso's ε (algorithms/smpso.hpp)
//   r2_weights, r2_offset
//                  r2_emoa: the number of weights and the utopian point's
//                  offset (algorithms/r2_emoa.hpp)
//   bc_lambda      bc_spx, bc_rex: the children of one step, 0: 10n (bc_spx.hpp)
//   paes_depth     paes: the grid's recursive halvings per objective (paes.hpp)
//   mads_tau, mads_delta0
//                  dmulti_mads: the frame size's factor and its start
//   glods_search_every, glods_local_tol
//                  multiglods: the failures before a search, and α of a
//                  local point
//   rm_extension   rm_meda: how far each model extends past its points
//   n_samples      sms_emoa, mo_cma_es, hype: the Monte-Carlo sample points a
//                  step from four objectives (10 000); sms_emoa and mo_cma_es:
//                  given, it decides from M = 4 (0 exact at every M, N > 0
//                  Monte-Carlo with N), not given hv_exact_max_m does
//   hv_exact_max_m sms_emoa: the exact contributions up to that many
//                  objectives, the race (or Monte-Carlo) above (5;
//                  hv_contribution.hpp, hv::ContribChoice); mo_cma_es is
//                  exact at every M and does not take it (nor race_*)
//   ada            nsga3, theta_dea, rvea: 1 runs ADA in place of the
//                  generation (operators/ada.hpp; task 15, §6), 0 (default) not
//   n_offspring, n_ref_points
//                  nsga3, a_nsga3, theta_dea, srv_nsga3: the children a
//                  generation (λ) and the reference points (H) apart from
//                  pop_size (μ; task 15, §2); not set, λ = μ, H = μ's lattice
//   race_eps, race_delta, race_max_samples, race_abs_tol
//                  sms_emoa: the race for the least contributor
//                  above hv_exact_max_m (Bringmann & Friedrich 2012): the
//                  answer within (1 + eps) of the least with probability
//                  1 - delta, or within abs_tol * prod(r - ideal) of it;
//                  0.01, 1e-6, 10^6 samples a step at most, 1e-12
inline bool is_core_param(const std::string& name)
{
    for (const char* k : {"ref_delta", "de_p", "de_memory", "p_de", "sa_chains", "sa_reseed",
                          "sa_t_ratio", "sa_s0", "sa_s_end", "sa_archive_cap", "sa_psa_alpha",
                          "sa_adapt_window", "coords_warmup", "coords_update", "spu_frac",
                          "inner_archive_cap", "archive_reuse_p", "dn_cf", "vmax_frac",
                          "vel_on_bound_low", "vel_on_bound_up", "w_min", "w_max", "c_min",
                          "c_max", "turbulence_k", "turbulence_p", "archive_size",
                          "eps_archive_eta", "r2_weights", "r2_offset", "bc_lambda", "paes_depth",
                          "mads_tau", "mads_delta0", "glods_search_every",
                          "glods_local_tol", "rm_extension", "n_sub",
                          "sub_budget", "fd_step", "local_tol",
                          "cma_sigma0", "eps_objective", "polish_share", "polish_points",
                          "mosqp_tau", "mosqp_mu", "mosqp_beta", "mosqp_spread",
                          "n_samples", "hv_exact_max_m", "race_eps", "race_delta",
                          "race_max_samples", "race_abs_tol", "hv_incremental",
                          "n_offspring", "n_ref_points", "ada"})
        if (name == k) return true;
    return false;
}

// Knobs whose value is a word, each with the words it takes. Same rule as
// knob_names(): the list is the authority (python/mootation/run/knobs.py reads
// it out of this file), an algorithm without the knob reports it as ignored.
//   bound_repair  what an operator that can leave the box does with the
//                 variables it put outside (operators/bound_repair.hpp)
//   de_base, de_index
//                 moead_de, moead_dra: the DE mutation's strategy and index
//                 selection (operators/de_mutation.hpp; task 15, §1)
//   crossover     nsga2, nsga2_sdr, ibea_eplus, spea2_sde, agemoea, sms_emoa
//                 (operators/real_crossover.hpp, multi_parent.hpp)
//   mutation      the same six and moead_de (operators/real_mutation.hpp)
//   crowding_space nsga2: its crowding distance over the objectives or the
//                 decision variables
//   dms_init      dms: the initial list, n points on the diagonal, one, or n
//                 drawn from the seed
//   dms_poll      dms: the next poll centre, the first still worth polling or
//                 one of them at random
//   dms_cache, paes_cache
//                 dms, paes: a point evaluated before taken from the run's
//                 cache (on) or evaluated again (off, the default, as the
//                 papers; task 16, supplement 1)
//   smpso_preset, chi_mode, turbulence, r_per_dim, archive_prune, leader_select,
//   leader_offer  smpso: the preset and the readings it fixes (smpso.hpp)
//   r2_mode, r2_space
//                 r2_emoa: the exact or the discrete R2, weights uniform in
//                 weight or in angle space (r2_emoa.hpp)
//   bc_selection  bc_spx, bc_rex: BC (default), or the paper's BF and BA
//   mads_select, mads_poll, mads_search, mads_cache
//                 dmulti_mads: the incumbent by spread or first, the poll
//                 complete or opportunistic, the speculative search or none,
//                 the cache of evaluated points on (the default) or off
//                 (task 15, §8)
//   constraint_mode
//                 minimize()'s, for a campaign's algorithm entry: none lets the
//                 core rank by the objectives only (task 15, §5); no core has it
//   scalarization, local, sub_plan, tch_weights, boost_scalarization, polish
//                 the Python-level methods of task 9 (python/mootation/scalar):
//                 no core has them, every core reports them ignored
//   hv_algo       sms_emoa, mo_cma_es: the exact contributions' hypervolumes by
//                 WFG (auto, the default, and wfg) or by box decomposition
//                 (box; hv_box.hpp)
//   hv_contrib    sms_emoa, mo_cma_es: the exact contributions at 3 and 4
//                 objectives by HVC3D / HVC4D (hvc, the default; hvc.hpp) or by
//                 WFG (wfg); above hv_exact_max_m the race (race, the default)
//                 or Monte-Carlo shared out equally (mc)
// Numeric knobs of one algorithm each: step_share and F_spread (emyo_c: the
// clamp of the difference vector as a share of the range, and the spread of
// its factor F), the knobs that test its mechanism (emyo_c.hpp).
inline const std::vector<std::pair<std::string, std::vector<std::string>>>& text_knobs() {
    static const std::vector<std::pair<std::string, std::vector<std::string>>> v = {
        {"bound_repair", {"clip", "reflect", "random", "midpoint", "resample", "wrap", "native", "r_reflect"}},
        {"crossover",    {"sbx", "uniform", "blx_alpha", "spx", "rex", "undx", "pcx", "de",
                          "de_pbest", "portfolio", "none"}},
        {"mutation",     {"polynomial", "gaussian", "cauchy", "uniform_reset", "mixture",
                          "mixture_cauchy", "none"}},
        {"crowding_space", {"objectives", "decision"}},
        {"dms_init",     {"line", "single", "random"}},
        {"dms_poll",     {"first", "random"}},
        {"dms_cache",    {"off", "on"}},
        {"paes_cache",   {"off", "on"}},
        {"de_adapt",     {"none", "shade"}},
        {"de_base",      {"rand1", "current1", "current_to_pbest1"}},
        {"de_index",     {"wor", "wr", "wpr"}},
        {"de_adapt_weight", {"equal", "contribution"}},
        {"de_archive",   {"yes", "no"}},
        {"portfolio",    {"sbx", "uniform", "blx_alpha", "spx", "rex", "undx", "pcx", "de",
                          "de_pbest"}},
        {"portfolio_credit", {"uniform", "borg"}},
        {"sa_energy",    {"cmb", "eps", "dom", "tch"}},
        {"sa_step",      {"mix", "laplace", "de", "sbx", "adaptive"}},
        {"sa_weights",   {"psa", "fixed"}},
        {"sa_tch_accept", {"weighted_sum", "chebyshev"}},
        {"coords",       {"axis", "pca", "pca_nd", "randrot"}},
        {"survival",     {"greedy", "stochastic"}},
        {"inner_archive", {"off", "nd"}},
        {"bin_crossover", {"uniform", "none"}},
        {"mating",       {"tournament", "each"}},
        {"smpso_preset", {"jmetal", "platemo", "fair", "omopso"}},
        {"chi_mode", {"signed", "abs", "none"}},
        {"turbulence", {"every_k", "prob", "thirds", "off"}},
        {"r_per_dim", {"no", "yes"}},
        {"archive_prune", {"one_by_one", "one_shot"}},
        {"leader_select", {"crowding_tournament", "random"}},
        {"leader_offer", {"all", "improved"}},
        {"r2_mode",      {"exact", "discrete"}},
        {"r2_space",     {"weight", "angle"}},
        {"bc_selection", {"bc", "bf", "ba"}},
        {"mads_select",  {"spread", "first"}},
        {"mads_poll",    {"complete", "opportunistic"}},
        {"mads_search",  {"speculative", "none"}},
        {"mads_cache",   {"off", "on"}},
        {"boost_scalarization", {"eps", "minmax"}},
        {"scalarization", {"eps", "tch", "nbi", "nnc", "ps"}},
        {"constraint_mode", {"feasibility", "eps_level", "none"}},
        {"local",        {"sqp", "nm", "cmaes"}},
        {"sub_plan",     {"grid", "laumanns", "gap"}},
        {"tch_weights",  {"inverse", "lattice"}},
        {"hv_algo",      {"auto", "wfg", "box"}},
        {"hv_contrib",   {"iwfg", "hvc", "wfg", "race", "mc"}},
        {"polish",       {"off", "sqp", "nm"}},
    };
    return v;
}

// The words a text knob takes, or nullptr for a name that is not one.
inline const std::vector<std::string>* text_knob_words(const std::string& name) {
    for (const auto& kv : text_knobs())
        if (kv.first == name) return &kv.second;
    return nullptr;
}

// Text knobs that take a comma-separated LIST of their words.
inline bool is_list_knob(const std::string& name) { return name == "portfolio"; }

// The text knobs a core takes through set_word (above, is_core_param).
inline bool is_core_word(const std::string& name)
{
    for (const char* k : {"de_adapt", "de_adapt_weight", "de_archive", "de_base", "de_index", "portfolio",
                          "portfolio_credit", "sa_energy", "sa_step", "sa_weights",
                          "sa_tch_accept", "coords", "survival", "inner_archive",
                          "bin_crossover", "mating", "smpso_preset", "chi_mode", "turbulence",
                          "r_per_dim", "archive_prune", "leader_select", "leader_offer", "r2_mode", "r2_space",
                          "bc_selection", "mads_select", "mads_poll", "mads_search", "mads_cache",
                          "dms_cache", "paes_cache", "boost_scalarization",
                          "scalarization",
                          "local", "sub_plan", "tch_weights", "polish", "hv_algo",
                          "hv_contrib"})
        if (name == k) return true;
    return false;
}

// Is `value` a valid value of the text knob `name` (one word, or for a list
// knob a comma-separated list of them)?
inline bool text_knob_accepts(const std::string& name, const std::string& value)
{
    const auto* words = text_knob_words(name);
    if (!words) return false;
    auto one = [&](const std::string& w) {
        return std::find(words->begin(), words->end(), w) != words->end();
    };
    if (!is_list_knob(name)) return one(value);
    std::size_t a = 0;
    while (a <= value.size()) {
        const std::size_t b = std::min(value.find(',', a), value.size());
        std::size_t x = a, y = b;                       // the item without spaces
        while (x < y && std::isspace(static_cast<unsigned char>(value[x]))) ++x;
        while (y > x && std::isspace(static_cast<unsigned char>(value[y - 1]))) --y;
        if (!one(value.substr(x, y - x))) return false;
        a = b + 1;
    }
    return true;
}

struct Settings {
    // ── the run ──────────────────────────────────────────────────────────
    std::string algorithm = "nsga2";
    int         pop_size  = 100;
    int         max_gen   = 250;
    // Evaluation budget. 0 = run max_gen steps; > 0 = step until this many
    // evaluations have been spent (setup included), and max_gen is ignored.
    // A step is not a generation of pop_size evaluations for every core —
    // NIMMO evaluates one offspring per step, MOEA/D-DRA and -AWA a fifth of
    // the population, LIS/LCS N − K_old + K_new — so a budget in steps is not
    // a budget. Checked between steps: a generational core overshoots by less
    // than one step.
    int         max_evaluations = 0;
    unsigned    seed      = 0;

    // ── the problem ──────────────────────────────────────────────────────
    std::vector<double> lower;          // one entry per variable
    std::vector<double> upper;
    int                 n_objs = 2;
    int                 n_cons = 0;     // constraint values per individual
    ConstraintMode      constraints = ConstraintMode::NONE;  // see constraint_mode.hpp

    // ── warm start ───────────────────────────────────────────────────────
    // Path to a population written by save_population (or by the Python or
    // TOML layers — one format). Empty means a fresh random start.
    //
    // Seeding costs ZERO function evaluations: the objectives come out of the
    // file, not out of the evaluator. What it is not is a checkpoint — no RNG
    // position and no per-algorithm state are stored, so a resumed run does not
    // reproduce what the uninterrupted one would have done. It starts from the
    // same place, which is the part that costs money to recompute.
    std::string  seed_population;
    SizeMismatch on_size_mismatch = SizeMismatch::Error;

    // ── optional algorithm knobs (empty = keep the paper default) ────────
    std::map<std::string, double> params;
    std::map<std::string, std::string> text_params;   // text_knobs()

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
        if (max_evaluations < 0)      bad("max_evaluations must be >= 0, got " + std::to_string(max_evaluations));
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
        for (const auto& kv : text_params) {
            const auto* words = text_knob_words(kv.first);
            if (!words) bad("unknown text knob '" + kv.first + "'");
            if (!text_knob_accepts(kv.first, kv.second))
                bad(kv.first + " = '" + kv.second + "' is not one of its values");
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

inline SizeMismatch parse_size_mismatch(const std::string& v, const std::string& where) {
    std::string t;
    for (char c : trim(v)) t += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (t == "error")    return SizeMismatch::Error;
    if (t == "truncate") return SizeMismatch::Truncate;
    if (t == "pad")      return SizeMismatch::Pad;
    throw std::invalid_argument(where + ": on_size_mismatch must be "
                                "error|truncate|pad, got '" + v + "'");
}

inline const char* size_mismatch_name(SizeMismatch m) {
    switch (m) {
        case SizeMismatch::Truncate: return "truncate";
        case SizeMismatch::Pad:      return "pad";
        default:                     return "error";
    }
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
        else if (key == "max_evaluations") s.max_evaluations = detail::parse_int(val, w);
        else if (key == "seed")        s.seed      = static_cast<unsigned>(detail::parse_int(val, w));
        else if (key == "n_objs")      s.n_objs    = detail::parse_int(val, w);
        else if (key == "n_cons")      s.n_cons    = detail::parse_int(val, w);
        else if (key == "n_vars")      n_vars_decl = detail::parse_int(val, w);
        else if (key == "lower")       s.lower     = detail::parse_list(val, w);
        else if (key == "upper")       s.upper     = detail::parse_list(val, w);
        else if (key == "constraints") s.constraints = detail::parse_constraints(val, w);
        // Taken verbatim, not parsed as a number or a list: it is a path, and
        // on Windows it will contain backslashes, spaces and a drive letter.
        else if (key == "seed_population")   s.seed_population   = val;
        else if (key == "on_size_mismatch")  s.on_size_mismatch  =
                                                 detail::parse_size_mismatch(val, w);
        else if (const auto* words = text_knob_words(key)) {
            if (!text_knob_accepts(key, val)) {
                std::string all;
                for (const auto& w2 : *words) all += (all.empty() ? "" : ", ") + w2;
                throw std::invalid_argument(where + ": " + key + " = '" + val +
                                            "' is not one of: " + all);
            }
            s.text_params[key] = val;
        }
        else {
            const auto& kn = knob_names();
            if (std::find(kn.begin(), kn.end(), key) == kn.end())
                throw std::invalid_argument(
                    where + ": unknown key '" + key + "'. Known keys: algorithm, "
                    "pop_size, max_gen, max_evaluations, seed, n_vars, n_objs, n_cons, lower, upper, "
                    "constraints, seed_population, on_size_mismatch, and the knobs "
                    "listed in knob_names()");
            s.params[key] = detail::parse_num(val, w);
        }
    }

    // A scalar bound broadcasts to n_vars. This is the only implicit step in
    // the format, and it needs n_vars to be known.
    auto broadcast = [&](std::vector<double>& v, const char* name) {
        if (v.size() == 1 && n_vars_decl && *n_vars_decl > 1) {
            // The scalar is copied out FIRST. v.assign(n, v[0]) passes a
            // reference into the vector being assigned, and assign() is not
            // required to tolerate an argument that aliases its own storage:
            // libstdc++ and MSVC happen to survive it, libc++ destroys the
            // element before reading it and every bound comes back 0. That is
            // how `lower = 0 / upper = 1` turned into `0 must be < 0` on arm64
            // macOS while passing everywhere else.
            const double scalar = v[0];
            v.assign(static_cast<std::size_t>(*n_vars_decl), scalar);
        }
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
    if (max_evaluations > 0)
        o << "max_evaluations = " << max_evaluations << "\n";
    o << "seed        = " << seed      << "\n";
    o << "n_objs      = " << n_objs    << "\n";
    o << "n_cons      = " << n_cons    << "\n";
    o << "constraints = " << detail::constraints_name(constraints) << "\n";
    o << "lower       = "; list(lower); o << "\n";
    o << "upper       = "; list(upper); o << "\n";
    // Written only when set: a `seed_population =` with nothing after it is not
    // a valid line to read back, and an empty path is the default anyway.
    if (!seed_population.empty()) {
        o << "seed_population  = " << seed_population << "\n";
        o << "on_size_mismatch = " << detail::size_mismatch_name(on_size_mismatch) << "\n";
    }
    if (!params.empty() || !text_params.empty()) {
        o << "\n# algorithm knobs\n";
        for (const auto& kv : params)
            o << kv.first << " = " << kv.second << "\n";
        for (const auto& kv : text_params)
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

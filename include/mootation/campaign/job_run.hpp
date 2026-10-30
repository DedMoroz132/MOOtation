#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// One job of a campaign run in C++ — task 17, stage 5, step 4c: run/campaign.py's
// _run_job up to the end of the run: the reference data, the run archive, the
// second answer's archive, everything evaluated, the accounting of every row
// the core evaluates (RunLedger), the core's run (run_core.hpp) and the
// trajectory records (the "budgets", "log" and "generations" grids; the
// indicators, the dense reference's, hv_archive, igdp_norm_all, fail_share),
// each record the line Python writes to trajectory.jsonl but its "t" (the
// seconds since the start). What comes after the run — final.csv, the
// archive's files, `final`, meta.json — is step 4d.
//
// The jobs it runs (refusal() says why not the others, which the Python
// engine runs): a problem of the C++ suite whose run needs no Python
// (registry.hpp), a core of algorithms.def (not a baseline or a
// scalarization method), the first population uniform, no polishing, no
// learned coordinates' diagnostics ([campaign] coords_diag); g_min and
// g_median only where the problem has no distance of its own.
// ============================================================================

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "../ablations.hpp"
#include "../archive_select.hpp"
#include "../designs.hpp"
#include "../grid_archive.hpp"
#include "../indicators.hpp"
#include "../json.hpp"
#include "../run_core.hpp"
#include "../run_ledger.hpp"
#include "../run_metrics.hpp"
#include "../scalar/polish.hpp"
#include "../scalar/scalar.hpp"
#include "../set_archive.hpp"
#include "../settings.hpp"
#include "jobs.hpp"
#include "measure.hpp"
#include "references.hpp"
#include "registry.hpp"

namespace mootation::campaign {

// metric_names.py
inline const std::set<std::string>& needs_front()
{
    static const std::set<std::string> s = {"igd", "igdp", "igdp_norm", "gdp", "eps", "eps_norm", "tau90",
                                            "gap_max", "gdp_norm", "gdp_min"};
    return s;
}
inline const std::set<std::string>& needs_x()
{
    static const std::set<std::string> s = {"igdx", "cr", "psp", "pdist", "g_min", "g_median", "ps_dmin"};
    return s;
}
inline const std::set<std::string>& needs_set()
{
    static const std::set<std::string> s = {"igdx", "cr", "psp", "ps_dmin"};
    return s;
}
inline const std::vector<std::string>& set_metrics_names()
{
    static const std::vector<std::string> v = {"igdx", "cr", "psp"};
    return v;
}
inline const std::set<std::string>& hv_names()
{
    static const std::set<std::string> s = {"hv", "hv_h"};
    return s;
}
constexpr const char* kDenseSuffix = "_ref_v3";

// campaign.budget_grid
inline std::vector<std::int64_t> budget_grid(std::int64_t limit, const std::vector<std::int64_t>& extra)
{
    static const double steps[] = {1.0, 1.5, 2.0, 3.0, 5.0, 7.0};
    std::set<std::int64_t> out = {25000};
    for (auto v : extra) out.insert(v);
    std::int64_t p = 100;
    while (p <= limit) {
        for (double s : steps) out.insert(static_cast<std::int64_t>(std::nearbyint(s * static_cast<double>(p))));
        p *= 10;
    }
    std::vector<std::int64_t> v;
    for (auto x : out)
        if (x > 0 && x <= limit) v.push_back(x);
    return v;
}

// campaign.log_grid: round(10 ** (j / per_decade)) up to limit, without repeats
// (the library's pow: the same integers as the C library's on every grid up
// to 10^9 and 100 a decade, checked)
inline std::vector<std::int64_t> log_grid(std::int64_t limit, std::int64_t per_decade)
{
    std::vector<std::int64_t> out;
    for (std::int64_t j = 0;; ++j) {
        const std::int64_t v = static_cast<std::int64_t>(
            std::nearbyint(dm::pow(10.0, static_cast<double>(j) / static_cast<double>(per_decade))));
        if (v > limit) return out;
        if (out.empty() || v > out.back()) out.push_back(v);
    }
}

// What a job is measured against: reference_data's arrays.
struct JobReferences {
    std::optional<Matrix> ref, ref3, pset;
    json::Value info3;                            // the dense reference's info (null without)
    std::optional<std::vector<double>> ideal, nadir;
    std::string hash;                             // reference_hash
};

// campaign._shared_reference: the cache's entry, and of it what the run asks for
inline JobReferences shared_references(const ReferenceEntry& e, bool front, bool dense, bool pset)
{
    JobReferences r;
    if (e.ideal) r.ideal = e.ideal->values;
    if (e.nadir) r.nadir = e.nadir->values;
    if (front) r.ref = e.ref;
    if (dense && r.ref && e.ref3) {
        r.ref3 = e.ref3;
        if (const json::Value* info = e.meta.find("info3")) r.info3 = *info;
    }
    if (pset) r.pset = e.pset;
    r.hash = reference_hash(e.ideal, e.nadir, r.ref, r.ref3, r.pset);
    return r;
}

// campaign.reference_data for a problem whose reference is not in the cache
// of references (refs.shared): the campaign's own cache, <results>/_references/
// <name>.npz, which the Python engine builds (np.savez of stamp, want, ideal,
// nadir and what the run asks for), read when it is there for this code
// (stamp: the registry's code stamp) and this ask (want); nothing otherwise.
inline std::optional<JobReferences> campaign_references(const std::filesystem::path& path, const std::string& stamp,
                                                        std::int64_t n_ref, bool front, bool dense, bool pset)
{
    std::error_code ec;
    if (!std::filesystem::is_regular_file(path, ec)) return std::nullopt;
    const std::string want = "n" + std::to_string(n_ref) + "-f" + (front ? "1" : "0") + "-d" + (dense ? "1" : "0") +
                             "-s" + (pset ? "1" : "0");
    std::map<std::string, npz::Array> z;
    try {
        z = npz::load(path);
        if (!z.count("stamp") || z.at("stamp").text() != stamp || !z.count("want") || z.at("want").text() != want ||
            !z.count("ideal") || !z.count("nadir"))
            return std::nullopt;
    } catch (const std::exception&) {
        return std::nullopt;                          // unreadable: the Python engine builds it again
    }
    std::optional<Matrix> ideal = matrix_of(z.at("ideal")), nadir = matrix_of(z.at("nadir"));
    JobReferences r;
    r.ideal = ideal->values;
    r.nadir = nadir->values;
    if (z.count("ref")) r.ref = matrix_of(z.at("ref"));
    if (z.count("ref3")) r.ref3 = matrix_of(z.at("ref3"));
    if (z.count("pset")) r.pset = matrix_of(z.at("pset"));
    r.info3 = z.count("info3") ? json::parse(z.at("info3").text()) : json::Value::null();
    r.hash = reference_hash(ideal, nadir, r.ref, r.ref3, r.pset);
    return r;
}

// designs.start_points: k rows of F to start from, the nondominated ones
// first (DSS among them when there are more than k), the rest by DSS among
// the others; a failed row (a non-finite value) last
inline std::vector<std::size_t> start_points(const std::vector<std::vector<double>>& F, std::size_t k)
{
    const std::size_t n = F.size();
    std::vector<std::size_t> out;
    if (k >= n) {
        for (std::size_t i = 0; i < n; ++i) out.push_back(i);
        return out;
    }
    std::vector<std::size_t> ok, bad;
    for (std::size_t i = 0; i < n; ++i) {
        bool finite = true;
        for (double v : F[i]) finite = finite && std::isfinite(v);
        (finite ? ok : bad).push_back(i);
    }
    if (!bad.empty()) {
        if (ok.size() <= k) {
            out = ok;
            for (std::size_t i = 0; out.size() < k; ++i) out.push_back(bad[i]);
            return out;
        }
        std::vector<std::vector<double>> FO;
        for (auto i : ok) FO.push_back(F[i]);
        for (auto j : start_points(FO, k)) out.push_back(ok[j]);
        return out;
    }
    auto rows_of = [&F](const std::vector<std::size_t>& idx) {
        select::Rows R;
        R.n = idx.size();
        R.m = F.empty() ? 0 : F[0].size();
        for (auto i : idx) R.v.insert(R.v.end(), F[i].begin(), F[i].end());
        return R;
    };
    std::vector<std::size_t> nd, rest;                   // designs.nondominated
    for (std::size_t i = 0; i < n; ++i) {
        bool dominated = false;
        for (std::size_t j = 0; j < n && !dominated; ++j) {
            bool all_le = true, any_lt = false;
            for (std::size_t c = 0; c < F[i].size(); ++c) {
                all_le = all_le && F[j][c] <= F[i][c];
                any_lt = any_lt || F[j][c] < F[i][c];
            }
            dominated = all_le && any_lt;
        }
        (dominated ? rest : nd).push_back(i);
    }
    if (nd.size() >= k) {
        for (auto j : select::dss_order(rows_of(nd), k)) out.push_back(nd[j]);
        return out;
    }
    out = nd;
    for (auto j : select::dss_order(rows_of(rest), k - nd.size())) out.push_back(rest[j]);
    return out;
}

// polish.check: (how, share, points) from the params, or nothing where
// minimize refuses them (left to the Python engine, which records why); how
// "off" is no polishing
struct PolishKnobs {
    std::string how = "off";
    double share = 0.2;
    long long points = 0;
};

inline std::optional<PolishKnobs> polish_knobs(const json::Value& params)
{
    using Kind = json::Value::Kind;
    PolishKnobs k;
    auto number = [](const json::Value& v, double& out) {
        if (v.kind == Kind::Float) out = v.f;
        else if (v.kind == Kind::Int) out = static_cast<double>(v.i);
        else if (v.kind == Kind::Bool) out = v.b ? 1.0 : 0.0;
        else return false;
        return true;
    };
    if (const json::Value* v = params.find("polish")) k.how = detail::py_str(*v);
    if (k.how != "off" && k.how != "sqp" && k.how != "nm") return std::nullopt;
    if (const json::Value* v = params.find("polish_share"); v && !number(*v, k.share)) return std::nullopt;
    double pts = 0.0;
    if (const json::Value* v = params.find("polish_points"); v && !number(*v, pts)) return std::nullopt;
    if (k.how == "off") return k;
    if (!(0.0 < k.share && k.share < 1.0) || pts < 0.0 || pts != std::floor(pts)) return std::nullopt;
    k.points = static_cast<long long>(pts);
    return k;
}

// why the C++ engine leaves this job to Python, or nothing
inline std::optional<std::string> refusal(const Job& job, const CampaignSpec& spec, const RegistryProblem& p,
                                          const std::vector<std::string>& algorithms,
                                          const std::vector<std::string>& baselines = {})
{
    if (auto why = engine_refusal(p)) return why;
    const bool baseline = std::find(baselines.begin(), baselines.end(), job.algorithm) != baselines.end();
    if (baseline && !p.user_kinds.empty()) return "a baseline on typed variables";
    const auto& sn = scalar::names();
    const bool scalar = std::find(sn.begin(), sn.end(), job.algorithm) != sn.end();
    if (scalar && !p.user_kinds.empty()) return "a scalarization method on typed variables";
    if (scalar)
        if (const json::Value* v = job.params.find("constraint_mode"); v && detail::py_str(*v) != "feasibility")
            return "a scalarization method under another constraint_mode";
    if (!baseline && !scalar && std::find(algorithms.begin(), algorithms.end(), job.algorithm) == algorithms.end())
        return "not a core of algorithms.def";
    if (job.init != "uniform") {
        const auto& kinds = designs::kinds();
        if (std::find(kinds.begin(), kinds.end(), job.init) == kinds.end()) return "an init minimize refuses";
        for (const auto& k : p.user_kinds)
            if (k == "bin" || k == "cat") return "a designed first population on bits or categories";
    }
    if (const json::Value* v = job.params.find("polish"); v && detail::py_str(*v) != "off") {
        if (!p.user_kinds.empty()) return "polish on typed variables";
        if (!polish_knobs(job.params)) return "polish knobs minimize refuses";
    }
    if (spec.coords_diag) return "the learned coordinates' diagnostics";
    if (p.distance) {
        for (const auto& lists : {spec.metrics, spec.final_metrics})
            for (const auto& m : lists)
                if (m == "g_min" || m == "g_median") return "g_min and g_median: the problem's distance is Python's";
    }
    return std::nullopt;
}

// minimize()'s knobs on the run's configuration: the per-knob ones, the
// params (is_core_param, as floats) and the words (is_core_word; a list as
// comma-separated words)
inline void apply_knobs(runner::RunConfig& cfg, const json::Value& params)
{
    using Kind = json::Value::Kind;
    auto number = [](const json::Value& v) { return v.kind == Kind::Float ? v.f : v.kind == Kind::Bool ? (v.b ? 1.0 : 0.0) : static_cast<double>(v.i); };
    auto integer = [&](const std::string& k, const json::Value& v) {
        if (v.kind == Kind::Float) {
            if (v.f != std::floor(v.f)) throw std::invalid_argument("parameter " + k + " takes an integer, got " + py_repr(v));
            return static_cast<int>(v.f);
        }
        return static_cast<int>(v.kind == Kind::Bool ? (v.b ? 1 : 0) : v.i);
    };
    for (std::size_t i = 0; i < params.keys.size(); ++i) {
        const std::string& k = params.keys[i];
        const json::Value& v = params.items[i];
        if (k == "polish" || k == "polish_share" || k == "polish_points") continue;
        if (is_core_param(k)) {
            cfg.params[k] = number(v);
        } else if (is_core_word(k)) {
            if (v.kind == Kind::Array) {
                std::string w;
                for (std::size_t j = 0; j < v.items.size(); ++j) w += (j ? "," : "") + v.items[j].s;
                cfg.words[k] = w;
            } else {
                cfg.words[k] = detail::py_str(v);
            }
        } else if (k == "eta_c") cfg.eta_c = number(v);
        else if (k == "eta_m") cfg.eta_m = number(v);
        else if (k == "pc") cfg.pc = number(v);
        else if (k == "pm") cfg.pm = number(v);
        else if (k == "delta") cfg.delta = number(v);
        else if (k == "kappa") cfg.kappa = number(v);
        else if (k == "theta") cfg.theta = number(v);
        else if (k == "alpha") cfg.alpha = number(v);
        else if (k == "F") cfg.F = number(v);
        else if (k == "CR") cfg.CR = number(v);
        else if (k == "T") cfg.T = integer(k, v);
        else if (k == "nr") cfg.nr = integer(k, v);
        else if (k == "K") cfg.K = integer(k, v);
        else if (k == "n_clusters") cfg.n_clusters = integer(k, v);
        else if (k == "div") cfg.div = integer(k, v);
        else if (k == "normalize") cfg.normalize = v.kind == Kind::Bool ? v.b : number(v) != 0.0;
        else if (k == "bound_repair") cfg.bound_repair = detail::py_str(v);
        else if (k == "crossover") cfg.crossover = detail::py_str(v);
        else if (k == "mutation") cfg.mutation = detail::py_str(v);
        else if (k == "crowding_space") cfg.crowding_space = detail::py_str(v);
        else if (k == "dms_init") cfg.dms_init = detail::py_str(v);
        else if (k == "dms_poll") cfg.dms_poll = detail::py_str(v);
        else if (k == "mutation_scale") cfg.mutation_scale = number(v);
        else if (k == "mixture_q") cfg.mixture_q = number(v);
        else if (k == "blx_alpha") cfg.blx_alpha = number(v);
        else if (k == "sbx_var_prob") cfg.sbx_var_prob = number(v);
        else if (k == "step_share") cfg.step_share = number(v);
        else if (k == "F_spread") cfg.F_spread = number(v);
        else if (k == "eta_int" || k == "int_repair") continue;     // minimize's, for integer variables (run())
        else throw std::invalid_argument("unknown parameter(s): " + k);
    }
}

// algorithms.seed_independent: a run of `name` with these knobs that is the
// same run for every seed
inline bool seed_independent(const std::string& name, const json::Value& params, int n_vars, const std::string& init)
{
    if (!init.empty() && init != "uniform") return false;
    auto word = [&params](const char* k, const char* fallback) {
        const json::Value* v = params.find(k);
        return v ? detail::py_str(*v) : std::string(fallback);
    };
    if (name == "dms") return word("dms_init", "line") != "random" && word("dms_poll", "first") != "random";
    if (name == "multiglods") return n_vars <= 100;
    return name == "boostdms" || name == "mosqp";
}

// The fields of meta.json that are the process's, not the run's: what the
// caller (the engine) says of itself. A run's other fields are the same
// whichever engine made it.
struct Process {
    json::Value mootation = json::Value::null();        // the version
    json::Value revision = json::Value::null();         // provenance.revision()
    json::Value host = json::Value::null(), platform = json::Value::null();
    json::Value python = json::Value::null();           // the Python version; null from C++
    json::Value pid = json::Value::null(), fp_env = json::Value::null();
    std::function<std::string()> now;                   // time.strftime("%Y-%m-%dT%H:%M:%S")
    // the seconds a run has taken (a record's "t", run_timeout); unset: the
    // steady clock (a test sets one that stands still)
    std::function<double()> seconds;
};

// A table of a run (store.table_blocks): head {comments, columns, ints,
// newline} and the numbers, rows x columns.
// An .npz of a run instead (store.array_blocks: snapshots.npz,
// archive_at.npz): its arrays in order, each with a NumPy dtype's text, a
// shape and its bytes (little-endian, as the machines it runs on).
struct NamedArray {
    std::string key, dtype;
    std::vector<std::uint64_t> shape;
    std::vector<std::uint8_t> data;
};

struct Table {
    std::string name;
    json::Value head;
    std::size_t rows = 0, cols = 0;
    std::vector<double> values;
    std::vector<NamedArray> arrays;                // an .npz: these, not head and values
};

struct JobFiles {
    json::Value meta;
    std::vector<Table> tables;
};

// What the run leaves for its end (step 4d).
struct JobOutcome {
    std::string status;                            // "done", "failed" or "timeout"
    std::string error;
    std::vector<std::string> trajectory;          // the records, json.dumps each
    runner::RunResult result;
    long long fe = 0, n_failed = 0, first_feasible = -1;
    std::size_t records = 0;
};

// a run stopped at run_timeout ([campaign]): thrown from the observer, it
// ends the core's run; the record so far is kept, "timeout"
struct TimeoutStop {};

class JobRun {
public:
    JobRun(const Job& job, const CampaignSpec& spec, const RegistryProblem& problem, JobReferences refs)
        : job_(job), spec_(spec), p_(problem), refs_(std::move(refs))
    {
    }

    JobOutcome run(const Process& process = {})
    {
        setup();
        process_ = process;
        begin_meta();
        start_ = std::chrono::steady_clock::now();
        JobOutcome out;
        runner::RunProblem rp;
        // typed variables: the core's box, each integer's widened by just under
        // a half (var_types.Layout.core_bounds); no bounds for the bits and
        // the categories, the integer genome after the reals
        const bool layout = !p_.user_kinds.empty();
        for (const auto& b : layout ? p_.core_bounds : p_.bounds) rp.bounds.push_back(b);
        rp.n_objectives = p_.n_obj;
        native_ = make_run_native(p_, static_cast<std::uint64_t>(job_.seed), noise_cell());
        rp.stochastic = p_.stochastic;
        if (n_genes_ > 0) {
            // bits or categories: the core hands its genome, which minimize's
            // layout decodes for the problem (direct_path refuses them)
            rp.n_bin = static_cast<int>(n_genes_);
            rp.evaluate_batch = [this](const std::vector<std::vector<double>>& rows) { return evaluate_rows(rows); };
        } else {
            rp.native = native_;
            rp.n_limits = native_->n_con;
            rp.on_rows = [this](const std::vector<double>& X, const std::vector<double>& F,
                                const std::vector<double>& G, std::size_t n) { account(X, F, G, n); };
        }
        runner::RunConfig cfg;
        cfg.pop_size = static_cast<int>(job_.pop);
        cfg.n_gen = static_cast<int>(job_.gens);
        cfg.seed = static_cast<unsigned>(job_.seed);
        // polishing: the algorithm on the budget without the polishing's
        // share (Python's round: half to even), every evaluation recorded
        polish_ = *polish_knobs(job_.params);
        algo_budget_ = budget_;
        recording_ = polish_.how != "off";
        if (polish_.how != "off")
            algo_budget_ = budget_ - static_cast<std::int64_t>(std::nearbyint(polish_.share * static_cast<double>(budget_)));
        cfg.max_evaluations = algo_budget_;
        if (native_->n_con > 0) cfg.constraint_mode = ConstraintMode::FEASIBILITY;
        apply_knobs(cfg, job_.params);
        cfg.operator_stats = op_stats();
        if (layout) {
            // minimize's layout: the kinds, rounded into the genome (lamarck,
            // int_repair's default), polynomial mutation's index 3 for them;
            // the genes' levels
            cfg.var_kinds = p_.var_kinds;
            cfg.discrete_values = p_.discrete_values;
            cfg.discrete_values.resize(p_.var_kinds.size());
            cfg.gene_levels = p_.gene_levels;
            const json::Value* repair = job_.params.find("int_repair");
            cfg.lamarck = !repair || detail::py_str(*repair) == "lamarck";
            if (std::any_of(p_.var_kinds.begin(), p_.var_kinds.end(), [](int k) { return k != 0; })) {
                const json::Value* eta = job_.params.find("eta_int");
                cfg.eta_int = !eta ? 3.0 : eta->kind == json::Value::Kind::Float ? eta->f : static_cast<double>(eta->i);
            }
        }
        if (job_.init != "uniform" && !baseline_ && !scalar_ && !seed_design(cfg, rp)) {
            out.status = "left";         // the design takes the whole budget: minimize refuses it
            out.error = "a design that takes the whole budget";
            return out;
        }
        cfg.on_generation = [this](int gen, std::vector<std::vector<double>> F) { on_gen(gen, std::move(F)); };
        cfg.record_every = grid_kind_ != Grid::Generations ? 1 : static_cast<int>(spec_.record_every);
        try {
            out.result = baseline_ ? run_baseline(cfg.record_every)
                         : scalar_   ? run_scalar(cfg.record_every)
                                     : runner::run(job_.algorithm, rp, cfg);
            if (polish_.how != "off") out.result = run_polish(out.result, cfg.record_every);
        } catch (const TimeoutStop&) {
            // stopped at run_timeout: the trajectory up to its last record
            out.status = "timeout";
            out.trajectory = std::move(lines_);
            out.fe = fe_;
            return out;
        } catch (const std::exception& e) {
            const std::string why = e.what();
            if (n_genes_ > 0 && (why.find("binary") != std::string::npos ||
                                 why.find("continuous") != std::string::npos)) {
                // a core without the integer genome: left to the Python
                // engine, which records the pair as incompatible
                out.status = "left";
                return out;
            }
            out.status = "failed";
            out.error = e.what();
            out.trajectory = std::move(lines_);
            out.fe = fe_;
            return out;
        }
        if (grid_kind_ == Grid::Budgets && held_ && held_->fe > last_fe_)
            record(held_->gen, held_->fe, held_->F, held_->X, std::nullopt);
        out.status = "done";
        out.trajectory = std::move(lines_);
        out.fe = fe_;
        out.n_failed = ledger_->n_failed();
        out.first_feasible = ledger_->first_feasible();
        out.records = records_;
        return out;
    }

    // After run(): meta.json and the tables, as _run_job writes them after the
    // core's run (the run "done"; a failed run: its meta only)
    JobFiles finish(const JobOutcome& out)
    {
        using V = json::Value;
        const std::size_t m = static_cast<std::size_t>(p_.n_obj), d = static_cast<std::size_t>(p_.n_var);
        JobFiles files;
        V& meta = meta_;
        if (out.status == "timeout") {
            // campaign._record_timeout: the meta.json the run wrote at its
            // start, "timeout", the evaluations and the values of its last record
            V t = meta_at_start_;
            V last = V::null();
            std::int64_t fe = 0;
            if (!out.trajectory.empty()) {
                last = json::parse(out.trajectory.back());
                if (const V* f = last.find("fe")) fe = f->i;
            }
            t.set("status", V::string("timeout"));
            t.set("timeout_seconds", V::number(spec_.run_timeout));
            t.set("fe", V::integer(fe));
            t.set("last_record", last);
            t.set("finished", now());
            files.meta = t;
            return files;
        }
        if (out.status != "done") {
            meta.set("status", V::string("failed"));
            meta.set("error", V::string(out.error));
            meta.set("fe", V::integer(out.fe));
            meta.set("finished", now());
            files.meta = meta;
            return files;
        }
        const runner::RunResult& res = out.result;
        // the final population: objectives, then variables
        std::vector<double> F, X;
        for (const auto& r : res.objectives) F.insert(F.end(), r.begin(), r.end());
        const std::size_t n = res.objectives.size();
        // the variables: the real genome, or, with a categorical variable, the
        // caller's variables (the genes after the reals, decoded: user_rows)
        std::size_t dx = d;
        if (has_cat_) {
            for (std::size_t i = 0; i < n; ++i) {
                std::vector<double> g = res.variables[i];
                for (int b : res.binary_variables.at(i)) g.push_back(static_cast<double>(b));
                const auto u = user_row(g.data());
                X.insert(X.end(), u.begin(), u.end());
            }
        } else {
            for (const auto& r : res.variables) X.insert(X.end(), r.begin(), r.end());
            if (n_genes_ > 0) dx = p_.core_bounds.size();
        }
        // a noisy copy: f noise-free, f_seen what the algorithm saw
        std::vector<double> seen;
        const bool noisy = truth_ && n;
        if (noisy) {
            seen = F;
            F = true_rows(X.data(), n);
        }
        {
            Table t;
            t.name = "final.csv";
            std::vector<std::string> cols;
            if (n) {
                for (std::size_t j = 0; j < m; ++j) cols.push_back("f" + std::to_string(j + 1));
                for (std::size_t j = 0; j < dx; ++j) cols.push_back("x" + std::to_string(j + 1));
            }
            if (noisy)
                for (std::size_t j = 0; j < m; ++j) cols.push_back("f" + std::to_string(j + 1) + "_seen");
            t.head = table_head(t.name, {std::string("# mootation campaign final population v1") +
                                         (noisy ? "; f noise-free, f_seen as the algorithm saw it" : "")},
                                cols, {});
            t.cols = cols.size();
            t.rows = t.cols ? n : 0;
            for (std::size_t i = 0; i < t.rows; ++i) {
                t.values.insert(t.values.end(), F.begin() + static_cast<std::ptrdiff_t>(i * m),
                                F.begin() + static_cast<std::ptrdiff_t>((i + 1) * m));
                t.values.insert(t.values.end(), X.begin() + static_cast<std::ptrdiff_t>(i * dx),
                                X.begin() + static_cast<std::ptrdiff_t>((i + 1) * dx));
                if (noisy)
                    t.values.insert(t.values.end(), seen.begin() + static_cast<std::ptrdiff_t>(i * m),
                                    seen.begin() + static_cast<std::ptrdiff_t>((i + 1) * m));
            }
            files.tables.push_back(std::move(t));
        }
        if (constrained_ && res.cv.size() == n) {
            Table t;
            t.name = "final_cv.csv";
            t.head = table_head(t.name, {"# mootation campaign final population's constraint violation v1: the sum of "
                                         "the violations, the rows of final.csv"},
                                {"cv"}, {});
            t.cols = 1;
            t.rows = n;
            t.values = res.cv;
            files.tables.push_back(std::move(t));
        }
        std::vector<std::size_t> arc_sel;
        bool have_sel = false;
        select::Rows AF;
        std::vector<double> AX;
        std::vector<char> AE;
        if (arc_) {
            AF = archive_rows(AX, AE);
            const bool whole = spec_.archive_variables == "all";
            if ((AF.n || constrained_) && (spec_.archive_scenario || !whole)) {
                arc_sel = archive_select(AF, static_cast<std::size_t>(p_.pop_size));
                have_sel = true;
            }
            const std::int64_t cap = spec_.archive_cap * p_.pop_size;
            const auto kept = archive_kept(AF, have_sel ? &arc_sel : nullptr);
            const std::string name = whole ? "archive.csv" : "archive.csv.gz";
            char delta[32];
            std::snprintf(delta, sizeof delta, "%g", arc_->delta());
            const std::string normalization = arc_->problem_frame() ? "problem" : "archive";
            std::string comment = std::string("# mootation campaign archive v1: every nondominated point evaluated") +
                                  (p_.has_cons ? " (feasible only)" : "") + ", at most one per cell of a " + delta +
                                  " grid in " + normalization +
                                  "-normalized objectives, each objective's best point kept whatever its cell holds";
            if (!whole)
                comment += "; objectives only, the variables of the points the archive scenario selects are in "
                           "final_archive.csv.gz";
            if (kept.size() < AF.n) comment += "; cut to " + std::to_string(cap) + " points by DSS";
            // a noisy copy's archive is written noise-free, and beside it the
            // values the archive scenario selected by (f*_seen)
            if (truth_) comment += "; f noise-free, f_seen as the algorithm saw it";
            const std::vector<double> AT = truth_ ? true_rows(AX.data(), AF.n) : AF.v;
            Table t;
            t.name = name;
            std::vector<std::string> cols;
            for (std::size_t j = 0; j < m; ++j) cols.push_back("f" + std::to_string(j + 1));
            if (whole)
                for (std::size_t j = 0; j < d; ++j) cols.push_back("x" + std::to_string(j + 1));
            cols.push_back("extreme");
            if (truth_)
                for (std::size_t j = 0; j < m; ++j) cols.push_back("f" + std::to_string(j + 1) + "_seen");
            t.head = table_head(name, {comment}, cols, {static_cast<std::int64_t>(m + (whole ? d : 0))});
            t.cols = cols.size();
            t.rows = kept.size();
            for (auto i : kept) {
                t.values.insert(t.values.end(), AT.begin() + static_cast<std::ptrdiff_t>(i * m),
                                AT.begin() + static_cast<std::ptrdiff_t>((i + 1) * m));
                if (whole) t.values.insert(t.values.end(), AX.begin() + static_cast<std::ptrdiff_t>(i * d),
                                           AX.begin() + static_cast<std::ptrdiff_t>((i + 1) * d));
                t.values.push_back(AE[i] ? 1.0 : 0.0);
                if (truth_) t.values.insert(t.values.end(), AF.row(i), AF.row(i) + m);
            }
            files.tables.push_back(std::move(t));
            if (!whole && have_sel) {
                Table s;
                s.name = "final_archive.csv.gz";
                const std::string how = spec_.answer_select == "dss" ? "DSS" : spec_.answer_select;
                std::vector<std::string> c2;
                for (std::size_t j = 0; j < m; ++j) c2.push_back("f" + std::to_string(j + 1));
                for (std::size_t j = 0; j < d; ++j) c2.push_back("x" + std::to_string(j + 1));
                s.head = table_head(s.name, {"# mootation campaign archive selection v1: the points " + how +
                                             " selects from the archive for the archive scenario"},
                                    c2, {});
                s.cols = c2.size();
                s.rows = arc_sel.size();
                for (auto i : arc_sel) {
                    s.values.insert(s.values.end(), AT.begin() + static_cast<std::ptrdiff_t>(i * m),
                                    AT.begin() + static_cast<std::ptrdiff_t>((i + 1) * m));
                    s.values.insert(s.values.end(), AX.begin() + static_cast<std::ptrdiff_t>(i * d),
                                    AX.begin() + static_cast<std::ptrdiff_t>((i + 1) * d));
                }
                files.tables.push_back(std::move(s));
            }
            std::vector<double> lo, hi;
            archive_frame(AF, lo, hi);
            long long extremes = 0;
            for (char e : AE) extremes += e != 0;
            V a = V::object();
            a.set("size", V::integer(static_cast<std::int64_t>(arc_->size())));
            a.set("delta", V::number(arc_->delta()));
            a.set("normalization", V::string(normalization));
            a.set("offered", V::integer(arc_->offered()));
            a.set("accepted", V::integer(arc_->accepted()));
            a.set("extremes", V::integer(extremes));
            a.set("file", V::string(name));
            a.set("written", V::integer(static_cast<std::int64_t>(kept.size())));
            a.set("cap", cap ? V::integer(cap) : V::null());
            a.set("answer_select", V::string(spec_.answer_select));
            a.set("variables", V::string(whole ? "all, in archive.csv" : "the selected points', in final_archive.csv.gz"));
            a.set("feasible_only", V::boolean(p_.has_cons));
            if (lo.empty()) {
                a.set("frame", V::null());
            } else {
                V fr = V::array();
                fr.items.push_back(numbers(lo));
                fr.items.push_back(numbers(hi));
                a.set("frame", fr);
            }
            meta.set("archive", a);
        }
        auto rows_of = [](std::size_t size, std::size_t width) {
            return static_cast<std::uint64_t>(width ? size / width : 0);
        };
        if (snap_ && !snap_n_.empty()) {
            Table s;
            s.name = "snapshots.npz";
            s.arrays.push_back(named_array("fe", "<i8", {snap_fe_.size()}, snap_fe_));
            s.arrays.push_back(named_array("gen", "<i8", {snap_gen_.size()}, snap_gen_));
            s.arrays.push_back(named_array("n", "<i8", {snap_n_.size()}, snap_n_));
            s.arrays.push_back(named_array("F", "<f4", {rows_of(snap_F_.size(), m), m}, snap_F_));
            if (keep_x_) s.arrays.push_back(named_array("X", "<f4", {rows_of(snap_X_.size(), d), d}, snap_X_));
            files.tables.push_back(std::move(s));
            V sm = V::object();
            sm.set("file", V::string("snapshots.npz"));
            sm.set("records", V::integer(static_cast<std::int64_t>(snap_n_.size())));
            sm.set("dtype", V::string("float32"));
            sm.set("variables", V::boolean(keep_x_));
            sm.set("layout", V::string("F (and X) rows of every record, stacked in order; n gives each record's "
                                       "row count"));
            meta.set("snapshots", sm);
        }
        if (!hard_.empty()) {
            // the hard budgets: the archive at exactly each (taken at the end
            // where the run stopped before it) and the population's last
            // record at or below it
            for (auto b : hard_)
                if (!hard_arc_.count(b) && arc_) take_hard_archive(b);
            V h = V::object();
            for (auto b : hard_) {
                V e = V::object();
                e.set("population", hard_pop_.count(b) ? hard_pop_.at(b) : V::null());
                e.set("archive", hard_arc_.count(b) ? hard_arc_.at(b) : V::null());
                h.set(std::to_string(b), e);
            }
            hard_at_ = h;
        }
        if (!checkpoints_.empty()) {
            // the archive scenario at the record_at counts: indicators and points
            // (the points only where [campaign] keep = "full": store.FULL_ONLY)
            meta.set("archive_at", archive_at_);
            if (!ck_budget_.empty() && spec_.keep == "full") {
                Table c;
                c.name = "archive_at.npz";
                c.arrays.push_back(named_array("budget", "<i8", {ck_budget_.size()}, ck_budget_));
                c.arrays.push_back(named_array("fe", "<i8", {ck_budget_.size()}, ck_budget_));
                c.arrays.push_back(named_array("n", "<i8", {ck_n_.size()}, ck_n_));
                c.arrays.push_back(named_array("F", "<f4", {rows_of(ck_F_.size(), m), m}, ck_F_));
                if (keep_x_) c.arrays.push_back(named_array("X", "<f4", {rows_of(ck_X_.size(), d), d}, ck_X_));
                files.tables.push_back(std::move(c));
            }
        }
        if (!hard_.empty()) {
            meta.set("hard_at", hard_at_);
            meta.set("answer_k", V::integer(spec_.answer_k ? spec_.answer_k : p_.pop_size));
        }
        // the final indicators
        V final = V::object();
        bool all_finite = !F.empty();
        for (double v : F) all_finite = all_finite && std::isfinite(v);
        MeasureInputs fin = measure_in_;
        HvOptions final_hv;
        final_hv.exact_max_m = static_cast<int>(spec_.hv_exact_max_m);
        final_hv.mc_samples = static_cast<std::size_t>(spec_.hv_mc_samples);
        if (all_finite) {
            final = measure(F.data(), n, m, dx ? X.data() : nullptr, dx, final_metrics_, fin, final_hv);
            add_dense(final, F.data(), n, m, dx ? X.data() : nullptr, dx, final_metrics_);
        }
        if (inc_ && !final.keys.empty()) {
            final.set("igdp_norm_all", inc_->seen() ? V::number(inc_->igd_plus()) : V::null());
            final.set("eps_norm_all", inc_->seen() ? V::number(inc_->eps_plus()) : V::null());
        }
        if (inc_) {
            V ae = V::object();
            ae.set("points", V::integer(static_cast<std::int64_t>(inc_->seen())));
            ae.set("levels", numbers(spec_.all_levels));
            V reached = V::array();
            for (long long v : inc_->reached()) reached.items.push_back(v < 0 ? V::null() : V::integer(v));
            ae.set("fe_to_level", reached);
            ae.set("measured", V::string("IGD+ and the additive epsilon of every feasible, finite point evaluated "
                                         "(noise-free on a noisy copy) against the reference, in igdp_norm's frame; "
                                         "fe_to_level: the evaluation at which igdp_norm_all first reached each "
                                         "level"));
            meta.set("all_evaluated", ae);
        }
        if (constrained_)
            final.set("fe_first_feas", out.first_feasible < 0 ? V::null() : V::integer(out.first_feasible));
        if (p_.failing || out.n_failed) final.set("n_failed", V::integer(out.n_failed));
        if (p_.typed && !final.keys.empty())
            final.set("dup_x_share", V::number(static_cast<double>(dup_x_) / static_cast<double>(std::max(out.fe, 1LL))));
        if (!p_.cover_front.empty()) {
            // the bits' front covered: the first generation and evaluation at
            // which the population, and the inner archive, held every vector
            // of it; NaN where it did not within the budget
            const V nan = V::number(std::numeric_limits<double>::quiet_NaN());
            final.set("cover_gen", cover_gen_ < 0 ? nan : V::integer(cover_gen_));
            final.set("cover_evals", cover_gen_ < 0 ? nan : V::integer(cover_fe_));
            final.set("cover_reached", V::boolean(cover_gen_ >= 0));
            final.set("cover_archive_gen", cover_archive_gen_ < 0 ? nan : V::integer(cover_archive_gen_));
            final.set("cover_archive_evals", cover_archive_gen_ < 0 ? nan : V::integer(cover_archive_fe_));
        }
        if (truth_) {
            if (!final.keys.empty()) {
                final.set("true_dominated_share", dominated_share(F.data(), n, m));
                // the observed answer: its noisy values, as the algorithm saw them
                const V obs = measure(rows_ptr(seen), n, m, d ? X.data() : nullptr, d, final_metrics_, fin, final_hv);
                for (std::size_t k = 0; k < obs.keys.size(); ++k) final.set(obs.keys[k] + "_observed", obs.items[k]);
            }
            std::vector<double> OX;
            std::vector<char> OE;
            select::Rows OF;
            oracle_->points(OF.v, OX, OE);
            OF.m = m;
            OF.n = m ? OF.v.size() / m : 0;
            if (OF.n) {
                // the oracle: the noise-free non-dominated set of every
                // evaluation, reduced to the answer's size by the archive
                // scenario's rule
                const auto sel = archive_select(OF, static_cast<std::size_t>(p_.pop_size), *oracle_);
                std::vector<double> SF, SX;
                for (auto i : sel) {
                    SF.insert(SF.end(), OF.row(i), OF.row(i) + m);
                    SX.insert(SX.end(), OX.begin() + static_cast<std::ptrdiff_t>(i * d),
                              OX.begin() + static_cast<std::ptrdiff_t>((i + 1) * d));
                }
                const V orc = measure(rows_ptr(SF), sel.size(), m, d ? rows_ptr(SX) : nullptr, d, final_metrics_, fin,
                                      final_hv);
                for (std::size_t k = 0; k < orc.keys.size(); ++k) final.set(orc.keys[k] + "_oracle", orc.items[k]);
            }
            const NoiseSpec& z = *p_.noise;
            V nz = V::object();
            nz.set("kind", V::string(z.kind));
            nz.set("scale", numbers(z.scale));
            nz.set("cell", z.kind == "s" ? V::number(noise_cell()) : V::null());
            nz.set("measured", V::string("every indicator on the noise-free objectives of the answer; _oracle on the "
                                         "noise-free non-dominated set of every evaluation; _observed on the noisy "
                                         "values of the answer"));
            nz.set("oracle_n", V::integer(static_cast<std::int64_t>(OF.n)));
            meta.set("noise", nz);
        }
        if (have_sel && spec_.archive_scenario) {
            std::vector<double> T, TX;
            for (auto i : arc_sel) {
                T.insert(T.end(), AF.row(i), AF.row(i) + m);
                TX.insert(TX.end(), AX.begin() + static_cast<std::ptrdiff_t>(i * d),
                          AX.begin() + static_cast<std::ptrdiff_t>((i + 1) * d));
            }
            if (truth_) T = true_rows(TX.data(), arc_sel.size());
            V fa = measure(rows_ptr(T), arc_sel.size(), m, d ? rows_ptr(TX) : nullptr, d, final_metrics_, fin, final_hv);
            add_dense(fa, rows_ptr(T), arc_sel.size(), m, d ? rows_ptr(TX) : nullptr, d, final_metrics_);
            fa.set("n", V::integer(static_cast<std::int64_t>(arc_sel.size())));
            if (truth_ && !arc_sel.empty()) fa.set("true_dominated_share", dominated_share(T.data(), arc_sel.size(), m));
            if (std::find(final_metrics_.begin(), final_metrics_.end(), "hv") != final_metrics_.end() && refs_.ideal &&
                refs_.nadir && static_cast<std::int64_t>(m) <= spec_.hv_archive_final_max_m)
                fa.set("hv_archive", V::number(archive_hv()));
            meta.set("final_archive", fa);
        }
        if (spec_.set_answer != "off" && p_.multimodal_ps && refs_.pset && refs_.pset->rows() &&
            !refs_.pset->values.empty() && d) {
            std::unique_ptr<SetArchive> own;
            SetArchive* sa = sarc_.get();
            if (!sa) {                                   // "population": the final population's non-dominated
                own = std::make_unique<SetArchive>(m, d);
                for (std::size_t i = 0; i < n; ++i) {
                    bool keep = true;
                    for (std::size_t j = 0; j < m; ++j) keep = keep && std::isfinite(F[i * m + j]);
                    if (constrained_ && res.cv.size() == n) keep = keep && res.cv[i] <= 0.0;
                    if (keep) own->add(&F[i * m], &X[i * d]);
                }
                sa = own.get();
            }
            V sv = V::object();
            sv.set("candidates", V::string(spec_.set_answer));
            if (sa->size()) {
                std::vector<double> SF, SX;
                sa->points(SF, SX);
                const std::size_t rows = sa->size();
                std::vector<double> lo, hi;
                for (const auto& [l, h] : p_.bounds) {
                    lo.push_back(l);
                    hi.push_back(h);
                }
                std::vector<std::size_t> sel;
                if (rows <= static_cast<std::size_t>(p_.pop_size)) {
                    for (std::size_t i = 0; i < rows; ++i) sel.push_back(i);
                } else {
                    const auto Z = indicators::unit_rows(SX.data(), rows, d, lo.data(), hi.data());
                    sel = indicators::spread_select(Z.data(), rows, d, static_cast<std::size_t>(p_.pop_size),
                                                    indicators::spread_first(static_cast<std::uint64_t>(job_.seed), rows));
                }
                std::vector<double> T, TX;
                for (auto i : sel) {
                    T.insert(T.end(), SF.begin() + static_cast<std::ptrdiff_t>(i * m),
                             SF.begin() + static_cast<std::ptrdiff_t>((i + 1) * m));
                    TX.insert(TX.end(), SX.begin() + static_cast<std::ptrdiff_t>(i * d),
                              SX.begin() + static_cast<std::ptrdiff_t>((i + 1) * d));
                }
                if (truth_) T = true_rows(TX.data(), sel.size());
                const V got = measure(T.data(), sel.size(), m, TX.data(), d, set_metrics_names(), fin, final_hv);
                Table t;
                t.name = "set_answer.csv";
                std::vector<std::string> cols;
                for (std::size_t j = 0; j < m; ++j) cols.push_back("f" + std::to_string(j + 1));
                for (std::size_t j = 0; j < d; ++j) cols.push_back("x" + std::to_string(j + 1));
                t.head = table_head(t.name, {"# mootation campaign second answer v1: the points spread_select "
                                             "takes for the decision-space indicators"},
                                    cols, {});
                t.cols = cols.size();
                t.rows = sel.size();
                for (std::size_t k = 0; k < sel.size(); ++k) {
                    t.values.insert(t.values.end(), T.begin() + static_cast<std::ptrdiff_t>(k * m),
                                    T.begin() + static_cast<std::ptrdiff_t>((k + 1) * m));
                    t.values.insert(t.values.end(), TX.begin() + static_cast<std::ptrdiff_t>(k * d),
                                    TX.begin() + static_cast<std::ptrdiff_t>((k + 1) * d));
                }
                files.tables.push_back(std::move(t));
                sv.set("n_candidates", V::integer(static_cast<std::int64_t>(rows)));
                sv.set("n", V::integer(static_cast<std::int64_t>(sel.size())));
                sv.set("version", V::integer(2));
                for (const auto& w : set_metrics_names()) {
                    const V* x = got.find(w);
                    sv.set(w, x ? *x : V::null());
                }
            } else {
                sv.set("n_candidates", V::integer(0));
                sv.set("n", V::integer(0));
                sv.set("version", V::integer(2));
            }
            meta.set("set_answer", sv);
        }
        if (!baseline_) {
            V ops = V::array();
            for (const auto& [o, r] : res.operators) {
                V e = V::object();
                e.set("operator", V::string(o));
                e.set("bound_repair", V::string(r));
                ops.items.push_back(e);
            }
            meta.set("operators", ops);
            if (!res.settings.empty()) {
                V s = V::object();
                for (const auto& [k, v] : res.settings) s.set(k, V::string(v));
                meta.set("settings", s);
            }
        }
        if (op_stats() && !final.keys.empty())
            for (const auto& [k, v] : res.operator_totals) final.set(k, V::number(v));
        meta.set("operator_stats", V::boolean(op_stats()));
        // the uninformative problems count where every evaluated point and
        // every non-dominated one fell
        if (const auto u = std::dynamic_pointer_cast<const UninformativeNative>(native_))
            meta.set("structural_bias", u->summary());
        V timing = V::object();
        for (const char* k : {"evaluate", "constraints", "feasible", "archive", "evaluator", "checkpoints", "records",
                              "minimize", "final", "write", "set_answer", "setup"})
            timing.set(k, V::integer(0));
        timing.set("callbacks", V::integer(0));
        timing.set("calls", V::integer(res.callbacks));
        timing.set("native", V::integer(static_cast<std::int64_t>(std::llround(1e6 * res.native_seconds))));
        meta.set("timing_us", timing);
        meta.set("status", V::string("done"));
        meta.set("fe", V::integer(out.fe));
        meta.set("records", V::integer(static_cast<std::int64_t>(out.records)));
        meta.set("final", final);
        V ignored = V::array();
        for (const auto& k : res.ignored) ignored.items.push_back(V::string(k));
        meta.set("ignored_knobs", ignored);
        meta.set("active_n", V::integer(static_cast<std::int64_t>(res.active_n)));
        meta.set("finished", now());
        // the run's wall time since it started, and of it what the records
        // took (indicators, statistics, snapshots, archive checkpoints), as
        // campaign.run_job: round(..., 3)
        meta.set("seconds", V::number(std::nearbyint(seconds() * 1000.0) / 1000.0));
        meta.set("record_seconds", V::number(std::nearbyint(record_seconds_ * 1000.0) / 1000.0));
        if (res.finished && out.fe < budget_) {
            V cs = V::object();
            cs.set("evaluations", V::integer(out.fe));
            cs.set("budget", V::integer(budget_));
            meta.set("core_stopped", cs);
        }
        if (!baseline_) meta.set("unique_fe", V::integer(static_cast<std::int64_t>(seen_rows_.size())));
        files.meta = meta;
        return files;
    }

    // what step 4d reads
    GridArchive* archive() { return arc_.get(); }
    SetArchive* second_archive() { return sarc_.get(); }
    indicators::IncrementalFront* everything() { return inc_.get(); }
    const std::vector<std::string>& final_metrics() const { return final_metrics_; }
    const std::vector<std::string>& trajectory_metrics() const { return traj_metrics_; }
    const std::vector<std::string>& hv_skipped() const { return hv_skipped_; }
    const HvOptions& trajectory_hv() const { return traj_hv_; }

private:
    enum class Grid { Budgets, Log, Generations };
    struct Held {
        int gen = 0;
        long long fe = 0;
        std::vector<double> F, X;
    };

    void setup()
    {
        const std::size_t m = static_cast<std::size_t>(p_.n_obj), d = static_cast<std::size_t>(p_.n_var);
        final_metrics_ = spec_.final_metrics.empty() ? spec_.metrics : spec_.final_metrics;
        if (spec_.set_metrics && p_.has_pareto_set) {
            for (const auto& s : set_metrics_names())
                if (std::find(spec_.metrics.begin(), spec_.metrics.end(), s) == spec_.metrics.end()) set_extra_.push_back(s);
            for (const auto& s : set_metrics_names())
                if (std::find(final_metrics_.begin(), final_metrics_.end(), s) == final_metrics_.end())
                    final_metrics_.push_back(s);
        }
        budget_ = job_.budget();
        if (spec_.archive || baseline_) {
            std::vector<double> lo, hi;
            if (refs_.ideal && refs_.nadir) {
                lo = *refs_.ideal;
                hi = *refs_.nadir;
            }
            arc_ = std::make_unique<GridArchive>(p_.n_obj, p_.n_var, lo, hi, spec_.archive_delta);
        }
        // A noisy copy (benchmarks/noisy.py): the algorithm sees the noisy
        // values, every indicator is measured on the noise-free ones at the
        // same x (truth_, the base), and its oracle keeps every non-dominated
        // point by the noise-free values of all the points evaluated
        if (p_.noise) {
            truth_ = make_native(p_);
            std::vector<double> lo, hi;
            if (refs_.ideal && refs_.nadir) {
                lo = *refs_.ideal;
                hi = *refs_.nadir;
            }
            oracle_ = std::make_unique<GridArchive>(p_.n_obj, p_.n_var, lo, hi, spec_.archive_delta);
        }
        if (spec_.set_answer == "evaluated" && p_.multimodal_ps && refs_.pset && refs_.pset->rows() &&
            !refs_.pset->values.empty() && d)
            sarc_ = std::make_unique<SetArchive>(m, d);
        constrained_ = p_.has_cons;
        if (spec_.all_evaluated && refs_.ref && refs_.ideal && refs_.nadir)
            inc_ = std::make_unique<indicators::IncrementalFront>(
                refs_.ref->values.data(), refs_.ref->rows(), m, *refs_.ideal, *refs_.nadir, spec_.all_levels,
                indicators::IncrementalFront::skip_pays(m));
        arc_hv_ = arc_ && refs_.ideal && refs_.nadir &&
                  std::find(spec_.metrics.begin(), spec_.metrics.end(), "hv") != spec_.metrics.end() &&
                  static_cast<std::int64_t>(p_.n_obj) <= spec_.hv_archive_grid_max_m;
        traj_metrics_ = spec_.metrics;
        traj_metrics_.insert(traj_metrics_.end(), set_extra_.begin(), set_extra_.end());
        traj_hv_.exact_max_m = static_cast<int>(spec_.hv_grid_exact_max_m);
        traj_hv_.mc_samples = static_cast<std::size_t>(spec_.hv_mc_samples);
        if (spec_.trajectory_hv_max_m && p_.n_obj > spec_.trajectory_hv_max_m) {
            if (spec_.trajectory_hv_mc_samples > 0) {
                traj_hv_.exact_max_m = 0;
                traj_hv_.mc_samples = static_cast<std::size_t>(spec_.trajectory_hv_mc_samples);
            } else {
                std::vector<std::string> kept;
                for (const auto& w : traj_metrics_) (hv_names().count(w) ? hv_skipped_ : kept).push_back(w);
                traj_metrics_ = kept;
            }
        }
        if (spec_.record_grid == "log") {
            grid_kind_ = Grid::Log;
            std::set<std::int64_t> g;
            for (auto v : log_grid(budget_, spec_.record_per_decade)) g.insert(v);
            for (auto v : spec_.record_at)
                if (v <= budget_) g.insert(v);
            grid_.assign(g.begin(), g.end());
        } else if (spec_.record_grid == "budgets") {
            grid_kind_ = Grid::Budgets;
            grid_ = budget_grid(budget_, spec_.record_at);
            grid_set_.insert(grid_.begin(), grid_.end());
        } else {
            grid_kind_ = Grid::Generations;
        }
        // snapshots.npz: the populations of the records; their variables where
        // snapshot_variables says so, and always on a constrained problem
        if (spec_.snapshots.kind == json::Value::Kind::Bool) {
            snap_ = spec_.snapshots.b;
        } else {
            for (const auto& s : spec_.snapshots.items) snap_ = snap_ || s.s == job_.problem;
        }
        keep_x_ = spec_.snapshot_variables == "all" || constrained_ ||
                  (spec_.snapshot_variables == "pareto_set" && p_.has_pareto_set);
        // the archive scenario at the record_at counts below the budget, nominal
        // and actual, taken after exactly that many evaluations
        if (spec_.archive_checkpoints && arc_ && grid_kind_ != Grid::Generations) {
            const std::int64_t nominal = job_.evaluations ? job_.evaluations
                                         : spec_.budget_fe ? spec_.budget_fe : budget_;
            for (auto b : spec_.record_at)
                if (b < std::min(budget_, nominal)) checkpoints_.insert(b);
        }
        for (const auto& k : p_.user_kinds) {
            n_genes_ += k == "bin" || k == "cat";
            has_cat_ = has_cat_ || k == "cat";
        }
        need_x_ = (snap_ && keep_x_) || constrained_ || truth_ != nullptr;
        for (const auto& w : traj_metrics_) need_x_ = need_x_ || needs_x().count(w);
        for (const auto& w : traj_metrics_)
            if (std::find(spec_.record_at_metrics.begin(), spec_.record_at_metrics.end(), w) ==
                spec_.record_at_metrics.end())
                everywhere_.push_back(w);
        ledger_ = std::make_unique<RunLedger>(m, d, constrained_, arc_.get(), oracle_.get(), inc_.get(), sarc_.get());
        // the hard budgets (CampaignSpec.hard_budgets): the archive after
        // exactly b evaluations, the population's last record at or below b
        for (auto b : spec_.hard_budgets)
            if (b <= budget_) hard_.push_back(b);
        std::set<long long> read_at(checkpoints_.begin(), checkpoints_.end());
        if (arc_) read_at.insert(hard_.begin(), hard_.end());
        if (inc_ || arc_hv_) read_at.insert(grid_set_.begin(), grid_set_.end());
        const std::vector<long long> reads(read_at.begin(), read_at.end());
        ledger_->set_reads(reads, inc_ ? std::vector<long long>(grid_set_.begin(), grid_set_.end())
                                       : std::vector<long long>{});
        measure_in_.ref = refs_.ref;
        measure_in_.ideal = refs_.ideal;
        measure_in_.nadir = refs_.nadir;
        measure_in_.pop = static_cast<std::size_t>(p_.pop_size);
        measure_in_.pset = refs_.pset;
        measure_in_.bounds = p_.bounds;
        for (int c : p_.cyclic_vars) measure_in_.cyclic.push_back(static_cast<std::size_t>(c));
        if (constrained_) measure_in_.constraints = make_native(p_);
        dense_in_.ref = refs_.ref3;
        dense_in_.ideal = refs_.ideal;
        dense_in_.nadir = refs_.nadir;
        dense_in_.constraints = measure_in_.constraints;
    }

    // the rows' variables, an empty set's included (Python hands an n x 0 array,
    // not None)
    static const double* rows_ptr(const std::vector<double>& v)
    {
        static const double none = 0.0;
        return v.empty() ? &none : v.data();
    }

    static json::Value numbers(const std::vector<double>& v)
    {
        json::Value a = json::Value::array();
        for (double x : v) a.items.push_back(json::Value::number(x));
        return a;
    }

    json::Value now() const { return process_.now ? json::Value::string(process_.now()) : json::Value::null(); }

    // put_table's head: a CSV's line end is the platform's (os.linesep), a
    // gzipped one's "\n"
    static json::Value table_head(const std::string& name, const std::vector<std::string>& comments,
                                  const std::vector<std::string>& columns, const std::vector<std::int64_t>& ints)
    {
        json::Value h = json::Value::object(), c = json::Value::array(), k = json::Value::array(),
                    i = json::Value::array();
        for (const auto& s : comments) c.items.push_back(json::Value::string(s));
        for (const auto& s : columns) k.items.push_back(json::Value::string(s));
        for (auto x : ints) i.items.push_back(json::Value::integer(x));
        h.set("comments", c);
        h.set("columns", k);
        h.set("ints", i);
#if defined(_WIN32)
        const char* linesep = "\r\n";
#else
        const char* linesep = "\n";
#endif
        const bool gz = name.size() > 3 && name.compare(name.size() - 3, 3, ".gz") == 0;
        h.set("newline", json::Value::string(gz ? "\n" : linesep));
        return h;
    }

    // dense(): the front indicators among `which` against the dense reference,
    // those alone (not what compute adds beside them: coverage_curve, ...)
    void add_dense(json::Value& into, const double* F, std::size_t n, std::size_t m, const double* X, std::size_t d,
                   const std::vector<std::string>& which) const
    {
        std::vector<std::string> front;
        for (const auto& w : which)
            if (needs_front().count(w)) front.push_back(w);
        if (!dense_in_.ref || front.empty() || !n) return;
        const json::Value dv = measure(F, n, m, X, d, front, dense_in_, HvOptions{});
        for (const auto& w : front) {
            const json::Value* v = dv.find(w);
            into.set(w + kDenseSuffix, v ? *v : json::Value::null());
        }
    }

    // meta.json as _run_job writes it before the run
    void begin_meta()
    {
        using V = json::Value;
        const std::size_t m = static_cast<std::size_t>(p_.n_obj);
        V& meta = meta_;
        meta = V::object();
        meta.set("status", V::string("running"));
        meta.set("problem", V::string(job_.problem));
        meta.set("algorithm", V::string(job_.key()));
        meta.set("core", V::string(job_.algorithm));
        meta.set("budget_dependent", V::boolean(budget_dependent_ || job_.init_share != 0.0));
        meta.set("ladder_of", job_.ladder_of.empty() ? V::null() : V::string(job_.ladder_of));
        meta.set("init", V::string(job_.init));
        meta.set("init_share", V::number(job_.init_share));
        meta.set("seed", V::integer(job_.seed));
        meta.set("pop", V::integer(job_.pop));
        meta.set("gens", V::integer(job_.gens));
        meta.set("budget_fe", V::integer(job_.budget()));
        meta.set("seed_independent", V::boolean(seed_independent(job_.algorithm, job_.params, p_.n_var, job_.init)));
        meta.set("budget_nominal", V::integer(job_.evaluations ? job_.evaluations
                                              : spec_.budget_fe ? spec_.budget_fe : job_.budget()));
        meta.set("params", job_.params);
        meta.set("n_objs", V::integer(p_.n_obj));
        meta.set("n_vars", V::integer(p_.n_var));
        meta.set("pop_note", V::string(job_.pop_note));
        V metrics = V::array(), fin = V::array();
        for (const auto& w : spec_.metrics) metrics.items.push_back(V::string(w));
        for (const auto& w : set_extra_) metrics.items.push_back(V::string(w));
        for (const auto& w : final_metrics_) fin.items.push_back(V::string(w));
        meta.set("metrics", metrics);
        meta.set("final_metrics", fin);
        meta.set("record_every", V::integer(spec_.record_every));
        meta.set("n_ref", V::integer(spec_.n_ref));
        meta.set("has_reference_front", V::boolean(refs_.ref.has_value()));
        meta.set("reference_version", V::integer(reference_version_));
        meta.set("problem_version", V::integer(p_.problem_version));
        meta.set("modification", p_.modification);
        meta.set("reference_hash", V::string(refs_.hash));
        meta.set("code_stamp", V::string(code_stamp_));
        meta.set("trajectory_version", V::integer(2));
        meta.set("reference_rows", refs_.ref ? V::integer(static_cast<std::int64_t>(refs_.ref->rows())) : V::null());
        meta.set("reference_v3", refs_.info3);
        meta.set("has_pareto_set", V::boolean(p_.has_pareto_set));
        meta.set("mootation", process_.mootation);
        meta.set("revision", process_.revision);
        meta.set("host", process_.host);
        meta.set("platform", process_.platform);
        meta.set("python", process_.python);
        meta.set("pid", process_.pid);
        meta.set("fp_env", process_.fp_env);
        meta.set("started", now());
        if (refs_.pset && refs_.pset->rows() == 0)
            meta.set("pareto_set_empty", V::string("the Pareto-set sample is empty (a failing copy whose failed region "
                                                   "takes all of it): no decision-space indicators"));
        if (job_.init != "uniform" || baseline_)
            meta.set("design_version", V::integer(designs::kVersion));   // the library's generators
        meta_at_start_ = meta;             // what meta.json holds while the run runs
        meta.set("record_grid", V::string(spec_.record_grid));
        meta.set("record_per_decade", spec_.record_grid == "log" ? V::integer(spec_.record_per_decade) : V::null());
        V at = V::array();
        for (auto b : spec_.record_at) at.items.push_back(V::integer(b));
        meta.set("record_at", at);
        meta.set("trajectory_hv_max_m", V::integer(spec_.trajectory_hv_max_m));
        meta.set("trajectory_hv", V::string(!hv_skipped_.empty() ? "not recorded"
                                            : traj_hv_.exact_max_m >= p_.n_obj
                                                ? "exact"
                                                : "monte-carlo, " + std::to_string(traj_hv_.mc_samples) + " samples"));
        meta.set("indicator_version", V::integer(2));
        meta.set("hv_mc_version", std::min<std::int64_t>(traj_hv_.exact_max_m, spec_.hv_exact_max_m) <
                                          static_cast<std::int64_t>(m)
                                      ? V::integer(2)
                                      : V::null());
        V tm = V::array(), sk = V::array(), ra = V::array();
        for (const auto& w : traj_metrics_) tm.items.push_back(V::string(w));
        for (const auto& w : hv_skipped_) sk.items.push_back(V::string(w));
        for (const auto& w : traj_metrics_)
            if (std::find(spec_.record_at_metrics.begin(), spec_.record_at_metrics.end(), w) !=
                spec_.record_at_metrics.end())
                ra.items.push_back(V::string(w));
        meta.set("trajectory_metrics", tm);
        meta.set("trajectory_hv_skipped", sk);
        meta.set("trajectory_metrics_at_record_at", ra);
    }

    // THE ACCOUNTING: the rows the core evaluated, a batch at a time
    void account(const std::vector<double>& X, const std::vector<double>& F, const std::vector<double>& G,
                 std::size_t n)
    {
        // the distinct points evaluated (unique.row_digests: the bytes of each
        // row, -0.0 as 0.0), meta's unique_fe
        const std::size_t width = static_cast<std::size_t>(p_.n_var);
        for (std::size_t i = 0; i < n; ++i) {
            std::string key(width * sizeof(double), '\0');
            for (std::size_t j = 0; j < width; ++j) {
                const double v = X[i * width + j] + 0.0;
                std::memcpy(&key[j * sizeof(double)], &v, sizeof v);
            }
            // a point asked for again (after the rounding of its integers)
            if (!seen_rows_.insert(std::move(key)).second) ++dup_x_;
        }
        const std::size_t c = static_cast<std::size_t>(native_->n_con);
        if (recording_) record_rows(X, F, G, n);
        const std::vector<double> T = truth_ ? true_rows(X.data(), n) : std::vector<double>{};
        ledger_->begin(X.data(), F.data(), n, constrained_ ? G.data() : nullptr, constrained_ ? c : 0,
                       truth_ ? T.data() : nullptr);
        for (;;) {
            const long long b = ledger_->next();
            if (b < 0) break;
            fe_ = b;
            read(b);
        }
        fe_ = ledger_->fe();
    }

    // polish.Recorder: every point the algorithm evaluated, in order of
    // first evaluation, with its latest values; every row a call
    void record_rows(const std::vector<double>& X, const std::vector<double>& F, const std::vector<double>& G,
                     std::size_t n)
    {
        const std::size_t d = static_cast<std::size_t>(p_.n_var), m = static_cast<std::size_t>(p_.n_obj);
        const std::size_t c = constrained_ ? static_cast<std::size_t>(native_->n_con) : 0;
        for (std::size_t i = 0; i < n; ++i) {
            std::vector<double> x(X.begin() + static_cast<std::ptrdiff_t>(i * d),
                                  X.begin() + static_cast<std::ptrdiff_t>((i + 1) * d));
            auto [it, fresh] = recorded_.try_emplace(std::move(x));
            if (fresh) record_order_.push_back(&it->first);
            it->second.first.assign(F.begin() + static_cast<std::ptrdiff_t>(i * m),
                                    F.begin() + static_cast<std::ptrdiff_t>((i + 1) * m));
            if (c)
                it->second.second.assign(G.begin() + static_cast<std::ptrdiff_t>(i * c),
                                         G.begin() + static_cast<std::ptrdiff_t>((i + 1) * c));
            ++recorded_calls_;
        }
    }

    void read(long long b)
    {
        if (arc_hv_ && grid_set_.count(b)) arc_hv_at_[b] = archive_hv();
        if (arc_ && std::find(hard_.begin(), hard_.end(), b) != hard_.end()) take_hard_archive(b);
        if (checkpoints_.count(b) && (arc_->size() || constrained_)) take_checkpoint(b);
    }

    // the archive scenario now, k points: its indicators as a record's
    // (noise-free, the dense reference's, true_dominated_share), and the
    // points (T, TX) and how many (n)
    json::Value archive_scenario(std::size_t k, std::vector<double>& T, std::vector<double>& TX, std::size_t& n) const
    {
        const std::size_t m = static_cast<std::size_t>(p_.n_obj), d = static_cast<std::size_t>(p_.n_var);
        std::vector<double> AX;
        std::vector<char> AE;
        const select::Rows AF = archive_rows(AX, AE);
        const auto idx = archive_select(AF, k);
        T.clear();
        TX.clear();
        for (auto i : idx) {
            T.insert(T.end(), AF.row(i), AF.row(i) + m);
            TX.insert(TX.end(), AX.begin() + static_cast<std::ptrdiff_t>(i * d),
                      AX.begin() + static_cast<std::ptrdiff_t>((i + 1) * d));
        }
        n = idx.size();
        if (truth_) T = true_rows(TX.data(), n);
        json::Value v = measure(rows_ptr(T), n, m, d ? rows_ptr(TX) : nullptr, d, traj_metrics_, measure_in_, traj_hv_);
        add_dense(v, rows_ptr(T), n, m, d ? rows_ptr(TX) : nullptr, d, traj_metrics_);
        if (truth_ && n) v.set("true_dominated_share", dominated_share(T.data(), n, m));
        for (const auto& h : hv_skipped_) v.set(h, json::Value::null());
        return v;
    }

    // take_hard_archive: the archive scenario of answer_k points (the
    // population's size by default) after exactly b evaluations, or at the
    // end of a run that stopped before b
    void take_hard_archive(long long b)
    {
        if (!arc_->size() && !constrained_) return;     // a constrained problem's empty archive: feas_share 0
        std::vector<double> T, TX;
        std::size_t n = 0;
        const std::size_t k = static_cast<std::size_t>(spec_.answer_k ? spec_.answer_k : p_.pop_size);
        json::Value v = archive_scenario(k, T, TX, n);
        v.set("fe", json::Value::integer(fe_));
        v.set("n", json::Value::integer(static_cast<std::int64_t>(n)));
        hard_arc_[b] = v;
    }

    // var_types.Layout.to_user, then numeric: the core's [reals..., genes...]
    // as the caller's variables in their order — an integer rounded half to
    // even into its bounds, a discrete one the nearest value (ties to the
    // lower), a bit 0 or 1, a categorical one its label's index
    std::vector<double> user_row(const double* core) const
    {
        const std::size_t nv = p_.user_kinds.size(), reals = p_.core_bounds.size();
        std::vector<double> u(nv);
        std::size_t r = 0, g = 0;
        for (std::size_t i = 0; i < nv; ++i) {
            const std::string& k = p_.user_kinds[i];
            if (k == "bin" || k == "cat") continue;
            double v = core[r++];
            if (k == "int") {
                v = std::min(std::max(std::nearbyint(v), std::ceil(p_.bounds[i].first)), std::floor(p_.bounds[i].second)) + 0.0;
            } else if (k == "discrete") {
                const auto& vals = p_.discrete_values[r - 1];
                double best = vals[0];
                for (double a : vals)
                    if (std::fabs(a - v) < std::fabs(best - v)) best = a;
                v = best;
            }
            u[i] = v;
        }
        for (std::size_t i = 0; i < nv; ++i) {
            const std::string& k = p_.user_kinds[i];
            if (k != "bin" && k != "cat") continue;
            const double gene = std::trunc(core[reals + g++]);
            if (k == "bin") u[i] = gene != 0.0 ? 1.0 : 0.0;
            else u[i] = std::min(std::max(gene, 0.0), static_cast<double>(p_.cat_levels[i] - 1));
        }
        return u;
    }

    // the core's batch with bits or categories: each row decoded, evaluated,
    // accounted for (campaign's evaluate_rows)
    std::vector<std::vector<double>> evaluate_rows(const std::vector<std::vector<double>>& rows)
    {
        const std::size_t m = static_cast<std::size_t>(p_.n_obj);
        std::vector<double> X, F;
        std::vector<std::vector<double>> out;
        for (const auto& row : rows) {
            const auto u = user_row(row.data());
            std::vector<double> f(m);
            native_->evaluate(u.data(), f.data(), nullptr);
            X.insert(X.end(), u.begin(), u.end());
            F.insert(F.end(), f.begin(), f.end());
            out.push_back(std::move(f));
        }
        account(X, F, {}, rows.size());
        return out;
    }

    // run_baselines.run_baseline: the sampling baselines' points from the
    // designs (the whole budget's plan at once), the ablations' from
    // ablations.hpp; every point evaluated and accounted for alone, the
    // answer the archive's DSS selection of N (the sampling baselines, gsemo)
    // or the last population (random_selection_ea)
    runner::RunResult run_baseline(int record_every)
    {
        const std::size_t m = static_cast<std::size_t>(p_.n_obj), d = static_cast<std::size_t>(p_.n_var);
        const std::size_t pop = static_cast<std::size_t>(job_.pop), budget = static_cast<std::size_t>(budget_);
        std::vector<double> lo, hi;
        for (const auto& [l, h] : p_.bounds) {
            lo.push_back(l);
            hi.push_back(h);
        }
        auto evaluate = [this, m](const std::vector<double>& x) {
            const std::size_t c = static_cast<std::size_t>(native_->n_con);
            std::vector<double> f(m), g(c);
            native_->evaluate(x.data(), f.data(), c ? g.data() : nullptr);
            account(x, f, constrained_ ? g : std::vector<double>{}, 1);
            return f;
        };
        auto observe = [this](int gen, const ablations::Rows& F, const ablations::Rows& X) {
            handed_x_.clear();
            for (const auto& x : X) handed_x_.insert(handed_x_.end(), x.begin(), x.end());
            on_gen(gen, F);
        };
        // the archive's answer: its DSS selection of N (GridArchive.select's default)
        auto answer = [this, d](ablations::Rows& F, ablations::Rows& X) {
            std::vector<double> AX;
            std::vector<char> AE;
            const select::Rows AF = archive_rows(AX, AE);
            F.clear();
            X.clear();
            for (auto i : archive_select(AF, static_cast<std::size_t>(job_.pop), *arc_, "dss")) {
                F.emplace_back(AF.row(i), AF.row(i) + AF.m);
                X.emplace_back(AX.begin() + static_cast<std::ptrdiff_t>(i * d),
                               AX.begin() + static_cast<std::ptrdiff_t>((i + 1) * d));
            }
        };
        ablations::Answer a;
        if (job_.algorithm == "random_selection_ea") {
            a = ablations::random_selection_ea(evaluate, lo, hi, pop, budget, job_.seed, observe, record_every);
        } else if (job_.algorithm == "gsemo") {
            std::vector<double> flo, fhi;                // the frame DSS normalises by: the problem's
            if (arc_->problem_frame()) arc_->frame(flo, fhi);
            auto choose = [flo, fhi, m](const ablations::Rows& F, std::size_t k) {
                std::vector<std::size_t> idx;
                if (F.size() <= k) {
                    for (std::size_t i = 0; i < F.size(); ++i) idx.push_back(i);
                    return idx;
                }
                select::Rows R;
                for (const auto& f : F) R.v.insert(R.v.end(), f.begin(), f.end());
                R.m = m;
                R.n = F.size();
                return select::dss_order(R, k, flo, fhi);
            };
            a = ablations::gsemo(evaluate, lo, hi, pop, budget, job_.seed, choose, observe, record_every);
        } else {
            const std::string kind = job_.algorithm == "random_search"   ? "uniform"
                                     : job_.algorithm == "sobol_search"  ? "sobol"
                                     : job_.algorithm == "lhs_search"    ? "lhs"
                                     : job_.algorithm == "halton_search" ? "halton_rand"
                                                                         : "sobol_owen";
            const auto plan = designs::design(kind, budget, static_cast<int>(d), job_.seed);
            std::size_t spent = 0;
            int gen = 0;
            ablations::Rows Fa, Xa;
            while (spent < budget) {
                const std::size_t k = std::min(pop, budget - spent);
                for (std::size_t i = spent; i < spent + k; ++i) {
                    std::vector<double> x(d);
                    for (std::size_t j = 0; j < d; ++j) x[j] = lo[j] + plan[i * d + j] * (hi[j] - lo[j]);
                    evaluate(x);
                }
                spent += k;
                if (gen % std::max(1, record_every) == 0 || spent >= budget) {
                    answer(Fa, Xa);
                    observe(gen, Fa, Xa);
                }
                ++gen;
            }
            answer(a.F, a.X);
            a.steps = gen;
        }
        runner::RunResult res;
        res.objectives = a.F;
        res.variables = a.X;
        res.active_n = a.F.size();
        return res;
    }

    // minimize's scalarization path (scalar.run, _core.scalar.run): the
    // method's own evaluator on the problem, every new point evaluated and
    // accounted for alone, the observer's rows handed over; the knobs are
    // the params but minimize's own arguments and the polishing's
    scalar::Problem scalar_problem()
    {
        scalar::Problem P;
        for (const auto& [l, h] : p_.bounds) {
            P.lo.push_back(l);
            P.hi.push_back(h);
        }
        P.m = p_.n_obj;
        P.n_cons = native_->n_con;
        P.evaluate = [this](const std::vector<double>& x, std::vector<double>& f, std::vector<double>& g) {
            native_->evaluate(x.data(), f.data(), g.empty() ? nullptr : g.data());
            account(x, f, constrained_ ? g : std::vector<double>{}, 1);
        };
        return P;
    }

    // scalar.run's first points: the design's rows (but mosqp's, which
    // starts from the box's centre), max(pop, init_share of the budget)
    scalar::Start scalar_start() const
    {
        scalar::Start st;
        if (job_.init == "uniform" || job_.algorithm == "mosqp") return st;
        std::int64_t n0 = job_.pop;
        if (job_.init_share != 0.0)
            n0 = std::max<std::int64_t>(n0, static_cast<std::int64_t>(
                                                std::nearbyint(job_.init_share * static_cast<double>(algo_budget_))));
        st.has_design = true;
        st.design = design_rows(p_.bounds, static_cast<std::size_t>(n0));
        return st;
    }

    // designs.design: n points of the box, lo + u (hi - lo)
    std::vector<std::vector<double>> design_rows(const std::vector<std::pair<double, double>>& box,
                                                 std::size_t n) const
    {
        const std::size_t d = box.size();
        const auto U = designs::design(job_.init, n, static_cast<int>(d), job_.seed);
        std::vector<std::vector<double>> rows(n, std::vector<double>(d));
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < d; ++j)
                rows[i][j] = box[j].first + U[i * d + j] * (box[j].second - box[j].first);
        return rows;
    }

    // minimize's init: the design evaluated here (decoded, accounted for as
    // one batch), the start points of it planted as the seed population, the
    // core on the budget the design leaves; false where it leaves none
    bool seed_design(runner::RunConfig& cfg, const runner::RunProblem& rp)
    {
        std::int64_t n_design = job_.pop;
        if (job_.init_share != 0.0)
            n_design = std::max<std::int64_t>(n_design, static_cast<std::int64_t>(std::nearbyint(
                                                            job_.init_share * static_cast<double>(algo_budget_))));
        if (algo_budget_ - n_design <= 0) return false;
        const auto rows = design_rows(rp.bounds, static_cast<std::size_t>(n_design));
        const std::size_t m = static_cast<std::size_t>(p_.n_obj), c = static_cast<std::size_t>(native_->n_con);
        const bool layout = !p_.user_kinds.empty();
        std::vector<double> X, F, G;
        std::vector<std::vector<double>> FR, GR;
        for (const auto& row : rows) {
            const std::vector<double> u = layout ? user_row(row.data()) : row;
            std::vector<double> f(m), g(c);
            native_->evaluate(u.data(), f.data(), c ? g.data() : nullptr);
            X.insert(X.end(), u.begin(), u.end());
            F.insert(F.end(), f.begin(), f.end());
            G.insert(G.end(), g.begin(), g.end());
            FR.push_back(std::move(f));
            GR.push_back(std::move(g));
        }
        account(X, F, constrained_ ? G : std::vector<double>{}, rows.size());
        for (auto i : start_points(FR, static_cast<std::size_t>(job_.pop))) {
            cfg.seed_variables.push_back(rows[i]);
            cfg.seed_objectives.push_back(FR[i]);
            if (c) cfg.seed_limits.push_back(GR[i]);
        }
        cfg.max_evaluations = algo_budget_ - n_design;
        return true;
    }

    runner::RunResult run_scalar(int record_every)
    {
        const scalar::Problem P = scalar_problem();
        scalar::Knobs knobs;
        for (std::size_t i = 0; i < job_.params.keys.size(); ++i) {
            const std::string& k = job_.params.keys[i];
            const json::Value& v = job_.params.items[i];
            if (k == "polish" || k == "polish_share" || k == "polish_points" || k == "eta_int" ||
                k == "constraint_mode" || k == "int_repair")
                continue;
            scalar::Knob kn;
            if (v.kind == json::Value::Kind::String) {
                kn.is_word = true;
                kn.word = v.s;
            } else if (v.kind == json::Value::Kind::Float) {
                kn.number = v.f;
            } else if (v.kind == json::Value::Kind::Int) {
                kn.number = static_cast<double>(v.i);
            } else if (v.kind == json::Value::Kind::Bool) {
                kn.number = v.b ? 1.0 : 0.0;
            } else {
                throw std::invalid_argument("parameter " + k + " takes a number or a word, got " + py_repr(v));
            }
            knobs.emplace_back(k, kn);
        }
        scalar::Observer observe = [this](long long gen, const scalar::Rows& F, const scalar::Rows& X) {
            handed_x_.clear();
            for (const auto& x : X) handed_x_.insert(handed_x_.end(), x.begin(), x.end());
            on_gen(static_cast<int>(gen), F);
        };
        const scalar::Result r = scalar::run(job_.algorithm, P, kMinimizeEqTol, false, 0.0, static_cast<int>(job_.pop),
                                             algo_budget_, static_cast<unsigned>(job_.seed), knobs, observe,
                                             record_every, scalar_start());
        runner::RunResult res;
        res.objectives = r.objectives;
        res.variables = r.variables;
        res.cv = r.cv;
        res.active_n = r.objectives.size();
        res.ignored = r.ignored;
        res.evaluations = r.evaluations;
        res.settings = r.settings;
        return res;
    }

    // scalar/polish.py's run: the local solver on some of the points the
    // algorithm evaluated (recorded_, in order of first evaluation), on the
    // budget it left; the answer the polished result, the settings the
    // algorithm's and the polishing's
    runner::RunResult run_polish(const runner::RunResult& base, int record_every)
    {
        recording_ = false;
        handing_ = true;                 // the observer's rows are the polishing's
        std::vector<scalar::Point> points;
        for (const auto* x : record_order_) {
            const auto& [f, g] = recorded_.at(*x);
            points.push_back({*x, f, g});
        }
        const scalar::Problem P = scalar_problem();
        double fd_step = 1.4901161193847656e-08, local_tol = 1e-14;
        for (const auto& [k, out] : {std::pair<const char*, double*>{"fd_step", &fd_step}, {"local_tol", &local_tol}})
            if (const json::Value* v = job_.params.find(k))
                *out = v->kind == json::Value::Kind::Float ? v->f : static_cast<double>(v->i);
        scalar::Observer observe = [this](long long gen, const scalar::Rows& F, const scalar::Rows& X) {
            handed_x_.clear();
            for (const auto& x : X) handed_x_.insert(handed_x_.end(), x.begin(), x.end());
            on_gen(static_cast<int>(gen), F);
        };
        const long long left = std::max<long long>(0, budget_ - recorded_calls_);
        const scalar::PolishResult r =
            scalar::polish(P, points, base.variables, base.evaluations, polish_.how, polish_.points, kMinimizeEqTol,
                           static_cast<int>(job_.pop), left, observe, record_every, fd_step, local_tol);
        runner::RunResult res = base;
        res.objectives = r.objectives;
        res.variables = r.variables;
        res.cv = r.cv;
        res.active_n = r.objectives.size();
        res.evaluations = recorded_calls_;
        res.settings.emplace_back("polish", polish_.how);
        res.settings.emplace_back("polish_points", std::to_string(r.chosen));
        res.settings.emplace_back("polish_evaluations", std::to_string(r.spent));
        res.settings.emplace_back("polish_version", std::to_string(scalar::METHOD_VERSION));
        res.inner_objectives.clear();
        res.inner_variables.clear();
        return res;
    }

    // every vector of the bits' front among the rows (tuples, exactly)
    bool covers(const std::vector<std::vector<double>>& rows) const
    {
        for (const auto& want : p_.cover_front) {
            bool hit = false;
            for (const auto& r : rows) hit = hit || r == want;
            if (!hit) return false;
        }
        return true;
    }

    // true_values: a noisy copy's noise-free objectives at the rows X (n x d)
    std::vector<double> true_rows(const double* X, std::size_t n) const
    {
        const std::size_t m = static_cast<std::size_t>(p_.n_obj), d = static_cast<std::size_t>(p_.n_var);
        std::vector<double> T(n * m), g(static_cast<std::size_t>(truth_->n_con));
        for (std::size_t i = 0; i < n; ++i) truth_->evaluate(X + i * d, &T[i * m], g.empty() ? nullptr : g.data());
        return T;
    }

    // the share of the rows another of them dominates (a noisy copy's
    // true_dominated_share)
    static json::Value dominated_share(const double* F, std::size_t n, std::size_t m)
    {
        return json::Value::number(1.0 - run_metrics::nd_share(F, n, m));
    }

    double noise_cell() const
    {
        const json::Value& c = spec_.noise_cell;
        return c.kind == json::Value::Kind::Int ? static_cast<double>(c.i) : c.f;
    }

    // take_checkpoint: the archive scenario after exactly b evaluations
    // (archive_checkpoints): its indicators in meta's archive_at, its points in
    // archive_at.npz
    void take_checkpoint(long long b)
    {
        const Clock clock(*this);
        std::vector<double> T, TX;
        std::size_t n = 0;
        json::Value v = archive_scenario(static_cast<std::size_t>(p_.pop_size), T, TX, n);
        v.set("fe", json::Value::integer(b));
        v.set("n", json::Value::integer(static_cast<std::int64_t>(n)));
        archive_at_.set(std::to_string(b), v);
        ck_budget_.push_back(b);
        ck_n_.push_back(static_cast<std::int64_t>(n));
        for (double x : T) ck_F_.push_back(static_cast<float>(x));
        if (keep_x_)
            for (double x : TX) ck_X_.push_back(static_cast<float>(x));
    }

    template <class T>
    static NamedArray named_array(std::string key, const char* dtype, std::vector<std::uint64_t> shape,
                                  const std::vector<T>& v)
    {
        NamedArray a{std::move(key), dtype, std::move(shape), {}};
        const auto* p = reinterpret_cast<const std::uint8_t*>(v.data());
        a.data.assign(p, p + v.size() * sizeof(T));
        return a;
    }

    // the archive's points as select() reads them
    select::Rows archive_rows(std::vector<double>& X, std::vector<char>& extreme) const
    {
        select::Rows R;
        arc_->points(R.v, X, extreme);
        R.m = static_cast<std::size_t>(p_.n_obj);
        R.n = R.m ? R.v.size() / R.m : 0;
        return R;
    }

    // archive.GridArchive.frame: the grid's, or the points' minimum and maximum
    void archive_frame(const select::Rows& F, std::vector<double>& lo, std::vector<double>& hi,
                       const GridArchive* a = nullptr) const
    {
        if ((a ? a : arc_.get())->frame(lo, hi)) return;
        lo.clear();
        hi.clear();
        if (!F.n) return;
        lo.assign(F.row(0), F.row(0) + F.m);
        hi = lo;
        for (std::size_t i = 1; i < F.n; ++i)
            for (std::size_t j = 0; j < F.m; ++j) {
                lo[j] = run_metrics::detail::nan_min(lo[j], F.row(i)[j]);
                hi[j] = run_metrics::detail::nan_max(hi[j], F.row(i)[j]);
            }
    }

    // archive.GridArchive.select(k, method)
    std::vector<std::size_t> archive_select(const select::Rows& F, std::size_t k) const
    {
        return archive_select(F, k, *arc_);
    }

    std::vector<std::size_t> archive_select(const select::Rows& F, std::size_t k, const GridArchive& a) const
    {
        return archive_select(F, k, a, spec_.answer_select);
    }

    std::vector<std::size_t> archive_select(const select::Rows& F, std::size_t k, const GridArchive& a,
                                            const std::string& method) const
    {
        std::vector<std::size_t> all;
        if (F.n <= k) {
            for (std::size_t i = 0; i < F.n; ++i) all.push_back(i);
            return all;
        }
        std::vector<double> lo, hi;
        if (a.problem_frame()) archive_frame(F, lo, hi, &a);
        if (method == "igdp_self") return select::igdp_self_order(F, k, lo, hi);
        return select::dss_order(F, k, lo, hi);
    }

    // campaign archive_kept: all, or archive_cap x N cut by DSS joined to sel
    std::vector<std::size_t> archive_kept(const select::Rows& F, const std::vector<std::size_t>* sel) const
    {
        const std::size_t cap = static_cast<std::size_t>(spec_.archive_cap * p_.pop_size);
        std::vector<std::size_t> out;
        if (!cap || F.n <= cap) {
            for (std::size_t i = 0; i < F.n; ++i) out.push_back(i);
            return out;
        }
        std::vector<double> lo, hi;
        if (arc_->problem_frame()) archive_frame(F, lo, hi);
        std::set<std::size_t> u;
        for (auto i : select::dss_order(F, cap, lo, hi)) u.insert(i);
        if (sel)
            for (auto i : *sel) u.insert(i);
        return std::vector<std::size_t>(u.begin(), u.end());
    }

    // hv_archive: the exact hypervolume of archive_kept's points
    double archive_hv() const
    {
        std::vector<double> X;
        std::vector<char> extreme;
        const select::Rows F = archive_rows(X, extreme);
        if (!F.n) return 0.0;
        std::vector<std::size_t> sel;
        const bool other = spec_.answer_select != "dss";
        if (other) sel = archive_select(F, static_cast<std::size_t>(p_.pop_size));
        const auto kept = archive_kept(F, other ? &sel : nullptr);
        std::vector<double> T;
        for (auto i : kept) T.insert(T.end(), F.row(i), F.row(i) + F.m);
        if (truth_) {
            const std::size_t d = static_cast<std::size_t>(p_.n_var);
            std::vector<double> KX;
            for (auto i : kept)
                KX.insert(KX.end(), X.begin() + static_cast<std::ptrdiff_t>(i * d),
                          X.begin() + static_cast<std::ptrdiff_t>((i + 1) * d));
            T = true_rows(KX.data(), kept.size());
        }
        return run_metrics::hypervolume(T.data(), kept.size(), F.m, *refs_.ideal, *refs_.nadir, 1.1, kHvMcSamples,
                                        0, p_.n_obj, "auto")
            .value;
    }

    std::vector<double> variables_of(std::size_t rows)
    {
        std::vector<double> X;
        if (baseline_ || scalar_ || handing_) return rows ? handed_x_ : X;   // the answer's rows, as handed over
        if (need_x_ && rows && runner::st().answer_variables)
            for (const auto& x : runner::st().answer_variables()) {
                if (!has_cat_) {
                    X.insert(X.end(), x.begin(), x.end());
                    continue;
                }
                const auto u = user_row(x.data());          // user_rows: a categorical variable's index
                X.insert(X.end(), u.begin(), u.end());
            }
        return X;
    }

    void on_gen(int gen, std::vector<std::vector<double>> P)
    {
        if (spec_.run_timeout > 0 && seconds() > spec_.run_timeout) throw TimeoutStop{};
        if (!p_.cover_front.empty()) {                   // at every call, before the grid's choice
            if (cover_gen_ < 0 && covers(P)) {
                cover_gen_ = gen;
                cover_fe_ = fe_;
            }
            if (cover_archive_gen_ < 0 && !baseline_ && !scalar_) {
                std::vector<std::vector<double>> A;
                if (const auto* live = ops::inner_archive_record().live)
                    for (const auto& mem : *live) A.push_back(mem.f);
                if (!A.empty() && covers(A)) {
                    cover_archive_gen_ = gen;
                    cover_archive_fe_ = fe_;
                }
            }
        }
        std::vector<double> F;
        for (const auto& r : P) F.insert(F.end(), r.begin(), r.end());
        if (grid_kind_ == Grid::Budgets) {
            std::vector<double> X = variables_of(P.size());
            while (next_i_ < grid_.size() && grid_[next_i_] <= fe_) {
                const long long b = grid_[next_i_++];
                if (b == fe_) record(gen, b, F, X, std::nullopt);
                else if (held_) record(held_->gen, b, held_->F, held_->X, held_->fe);
            }
            held_ = Held{gen, fe_, F, X};
            return;
        }
        if (grid_kind_ == Grid::Log) {
            if (gen > 0 && fe_ < budget_ && (next_i_ >= grid_.size() || fe_ < grid_[next_i_])) return;
            while (next_i_ < grid_.size() && grid_[next_i_] <= fe_) ++next_i_;
        }
        record(gen, fe_, F, variables_of(P.size()), std::nullopt);
    }

    void record(int gen, long long fe_rec, const std::vector<double>& F_seen, const std::vector<double>& X,
                std::optional<long long> fe_pop)
    {
        const Clock clock(*this);
        const std::size_t m = static_cast<std::size_t>(p_.n_obj), d = static_cast<std::size_t>(p_.n_var);
        const std::size_t n = m ? F_seen.size() / m : 0;
        // snapshot and indicators: noise-free
        const std::vector<double> F = truth_ && !X.empty() ? true_rows(X.data(), n) : F_seen;
        json::Value rec = json::Value::object();
        rec.set("gen", json::Value::integer(gen));
        rec.set("fe", json::Value::integer(fe_rec));
        rec.set("t", json::Value::number(py_round(seconds(), 3)));
        rec.set("n",json::Value::integer(static_cast<std::int64_t>(n)));
        if (fe_pop) rec.set("fe_pop", json::Value::integer(*fe_pop));
        const auto& failed = ledger_->failed_fe();
        if (grid_kind_ == Grid::Budgets) {
            if (p_.failing || !failed.empty()) {
                long long k = 0;
                for (long long v : failed) k += last_fe_ < v && v <= fe_rec;
                rec.set("fail_share", json::Value::number(static_cast<double>(k) /
                                                          static_cast<double>(std::max(fe_rec - last_fe_, 1LL))));
            }
        } else if (p_.failing || ledger_->n_failed()) {
            rec.set("fail_share", json::Value::number(static_cast<double>(ledger_->n_failed() - failed_at_record_) /
                                                      static_cast<double>(std::max(fe_rec - last_fe_, 1LL))));
            failed_at_record_ = ledger_->n_failed();
        }
        bool finite = true;
        for (double v : F) finite = finite && std::isfinite(v);
        rec.set("finite", json::Value::boolean(finite));
        bool mark = fe_rec >= budget_;
        for (auto b : spec_.record_at) mark = mark || (last_fe_ < b && b <= fe_rec);
        if (finite && !F.empty()) {
            const double* x = need_x_ && !X.empty() ? X.data() : nullptr;
            const auto& which = mark ? traj_metrics_ : everywhere_;
            const json::Value vals = measure(F.data(), n, m, x, d, which, measure_in_, traj_hv_);
            for (std::size_t k = 0; k < vals.keys.size(); ++k) rec.set(vals.keys[k], vals.items[k]);
            add_dense(rec, F.data(), n, m, x, d, which);
            if (truth_) rec.set("true_dominated_share", dominated_share(F.data(), n, m));
            for (const auto& h : hv_skipped_) rec.set(h, json::Value::null());
        }
        if (arc_hv_ && finite && !F.empty()) {
            if (!arc_hv_at_.count(fe_rec)) arc_hv_at_[fe_rec] = archive_hv();
            rec.set("hv_archive", json::Value::number(arc_hv_at_[fe_rec]));
        }
        if (inc_) {
            std::pair<double, double> v{inc_->igd_plus(), inc_->eps_plus()};
            if (fe_rec != fe_) {
                const auto& at = ledger_->all_at();
                const auto it = at.find(fe_rec);
                if (it != at.end()) v = it->second;
            }
            auto finite_or_null = [](double u) { return std::isfinite(u) ? json::Value::number(u) : json::Value::null(); };
            rec.set("igdp_norm_all", finite_or_null(v.first));
            rec.set("eps_norm_all", finite_or_null(v.second));
        }
        if (op_stats())
            for (const auto& s : runner::operator_stats())
                rec.set(s.name, !s.value   ? json::Value::null()
                                : s.integer ? json::Value::integer(static_cast<std::int64_t>(*s.value))
                                            : json::Value::number(*s.value));
        for (auto b : hard_) {
            if (fe_rec > b) continue;
            json::Value pop = json::Value::object();
            std::vector<std::string> keys = {"gen", "fe", "n"};
            keys.insert(keys.end(), traj_metrics_.begin(), traj_metrics_.end());
            for (const auto& k : keys) {
                const json::Value* v = rec.find(k);
                pop.set(k, v ? *v : json::Value::null());
            }
            hard_pop_[b] = pop;
        }
        if (snap_) {
            snap_fe_.push_back(fe_rec);
            snap_gen_.push_back(gen);
            snap_n_.push_back(static_cast<std::int64_t>(n));
            for (double x : F) snap_F_.push_back(static_cast<float>(x));
            if (keep_x_)
                for (double x : X) snap_X_.push_back(static_cast<float>(x));   // none: no rows
        }
        last_fe_ = fe_rec;
        lines_.push_back(json::dump(rec, -1, ",", ":"));
        ++records_;
    }

    Job job_;
    CampaignSpec spec_;
    RegistryProblem p_;
    JobReferences refs_;
    std::shared_ptr<suite::Native> native_;
    std::vector<std::string> final_metrics_, set_extra_, traj_metrics_, hv_skipped_, everywhere_;
    HvOptions traj_hv_;
    std::int64_t budget_ = 0;
    std::unique_ptr<GridArchive> arc_, oracle_;
    std::shared_ptr<suite::Native> truth_;           // a noisy copy's base: the noise-free values
    std::unique_ptr<SetArchive> sarc_;
    std::unique_ptr<indicators::IncrementalFront> inc_;
    std::unique_ptr<RunLedger> ledger_;
    bool constrained_ = false, arc_hv_ = false, need_x_ = false;
    Grid grid_kind_ = Grid::Budgets;
    std::vector<std::int64_t> grid_;
    std::set<long long> grid_set_;
    std::size_t next_i_ = 0;
    std::optional<Held> held_;
    std::map<long long, double> arc_hv_at_;
    long long fe_ = 0, last_fe_ = 0, failed_at_record_ = 0;
    std::size_t records_ = 0;
    std::vector<std::string> lines_;
    MeasureInputs measure_in_, dense_in_;
    Process process_;
    json::Value meta_, meta_at_start_;
    bool snap_ = false, keep_x_ = false;
    long long dup_x_ = 0;                            // the points evaluated again
    bool baseline_ = false;                          // run/baselines.py's, not a core
    bool scalar_ = false;                            // a scalarization method (mootation.scalar)
    PolishKnobs polish_;                             // polish, polish_share, polish_points
    std::int64_t algo_budget_ = 0;                   // the algorithm's budget: without the polishing's share
    bool recording_ = false, handing_ = false;       // polishing: the recorder on; the observer's rows handed
    std::map<std::vector<double>, std::pair<std::vector<double>, std::vector<double>>> recorded_;
    std::vector<const std::vector<double>*> record_order_;
    long long recorded_calls_ = 0;
    static constexpr double kMinimizeEqTol = 1e-4;   // minimize's eq_tol (no equalities here)

    // the operator statistics: a core's only (campaign: not in_python)
    bool op_stats() const { return spec_.operator_stats && !baseline_ && !scalar_; }
    std::vector<double> handed_x_;                 // the answer's rows the baseline hands to on_gen
    std::size_t n_genes_ = 0;                        // bits and categories: the integer genome
    bool has_cat_ = false;
    long long cover_gen_ = -1, cover_fe_ = -1, cover_archive_gen_ = -1, cover_archive_fe_ = -1;
    std::vector<std::int64_t> snap_fe_, snap_gen_, snap_n_;
    std::vector<float> snap_F_, snap_X_;
    std::set<long long> checkpoints_;
    std::vector<long long> hard_;                    // the hard budgets within the budget, in the config's order
    json::Value hard_at_;
    std::map<long long, json::Value> hard_pop_, hard_arc_;
    json::Value archive_at_ = json::Value::object();
    std::vector<std::int64_t> ck_budget_, ck_n_;
    std::vector<float> ck_F_, ck_X_;
    std::chrono::steady_clock::time_point start_;

    // the seconds since the run started (the process's clock where it has one)
    double seconds() const
    {
        if (process_.seconds) return process_.seconds();
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - start_).count();
    }
    std::set<std::string> seen_rows_;
    bool budget_dependent_ = false;
    int reference_version_ = 0;
    std::string code_stamp_;
    // of seconds(), what the records took (meta.json's record_seconds)
    double record_seconds_ = 0.0;
    // adds the run's time from here to the end of the scope to record_seconds_
    struct Clock {
        JobRun& run;
        double t;
        explicit Clock(JobRun& r) : run(r), t(r.seconds()) {}
        Clock(const Clock&) = delete;
        Clock& operator=(const Clock&) = delete;
        ~Clock() { run.record_seconds_ += run.seconds() - t; }
    };

public:
    // what meta.json says of the registry: its code stamp and reference
    // version, and whether the run depends on its budget (budget.py)
    void set_registry(const ProblemRegistry& reg, const JobsRegistry& jreg)
    {
        const auto& b = reg.names.baselines;
        baseline_ = std::find(b.begin(), b.end(), job_.algorithm) != b.end();
        const auto& sn = scalar::names();
        scalar_ = std::find(sn.begin(), sn.end(), job_.algorithm) != sn.end();
        code_stamp_ = reg.code_stamp;
        reference_version_ = reg.reference_version;
        budget_dependent_ = depends_on_budget(jreg, job_.algorithm, job_.params);
    }
};

}   // namespace mootation::campaign

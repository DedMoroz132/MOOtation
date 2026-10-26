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
// scalarization method), the first population uniform, no polishing, and of
// [campaign] neither hard budgets, operator statistics nor the learned
// coordinates' diagnostics; g_min and g_median only where the
// problem has no distance of its own.
// ============================================================================

#include <algorithm>
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

#include "../archive_select.hpp"
#include "../grid_archive.hpp"
#include "../indicators.hpp"
#include "../json.hpp"
#include "../run_core.hpp"
#include "../run_ledger.hpp"
#include "../run_metrics.hpp"
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
inline std::vector<std::int64_t> log_grid(std::int64_t limit, std::int64_t per_decade)
{
    std::vector<std::int64_t> out;
    for (std::int64_t j = 0;; ++j) {
        const std::int64_t v = static_cast<std::int64_t>(
            std::nearbyint(std::pow(10.0, static_cast<double>(j) / static_cast<double>(per_decade))));
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

// why the C++ engine leaves this job to Python, or nothing
inline std::optional<std::string> refusal(const Job& job, const CampaignSpec& spec, const RegistryProblem& p,
                                          const std::vector<std::string>& algorithms)
{
    if (auto why = engine_refusal(p)) return why;
    if (std::find(algorithms.begin(), algorithms.end(), job.algorithm) == algorithms.end())
        return "not a core of algorithms.def (a baseline or a scalarization method)";
    if (job.init != "uniform") return "a designed first population (init)";
    if (const json::Value* v = job.params.find("polish"); v && detail::py_str(*v) != "off") return "polish";
    if (!spec.hard_budgets.empty()) return "hard budgets";
    if (spec.operator_stats) return "operator statistics";
    if (spec.coords_diag) return "the learned coordinates' diagnostics";
    if (p.distance) {
        for (const auto& lists : {spec.metrics, spec.final_metrics})
            for (const auto& m : lists)
                if (m == "g_min" || m == "g_median") return "g_min and g_median: the problem's distance is Python's";
    }
    if (!p.shared_reference) return "a reference built for the campaign (not the cache of references)";
    return std::nullopt;
}

// minimize()'s knobs on the run's configuration: the per-knob ones, the
// params (is_core_param, as floats) and the words (is_core_word; a list as
// comma-separated words)
inline void apply_knobs(run::RunConfig& cfg, const json::Value& params)
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
    run::RunResult result;
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
        run::RunProblem rp;
        for (const auto& b : p_.bounds) rp.bounds.push_back(b);
        rp.n_objectives = p_.n_obj;
        native_ = make_native(p_);
        rp.native = native_;
        rp.n_limits = native_->n_con;
        rp.stochastic = p_.stochastic;
        rp.on_rows = [this](const std::vector<double>& X, const std::vector<double>& F,
                            const std::vector<double>& G, std::size_t n) { account(X, F, G, n); };
        run::RunConfig cfg;
        cfg.pop_size = static_cast<int>(job_.pop);
        cfg.n_gen = static_cast<int>(job_.gens);
        cfg.seed = static_cast<unsigned>(job_.seed);
        cfg.max_evaluations = budget_;
        if (native_->n_con > 0) cfg.constraint_mode = ConstraintMode::FEASIBILITY;
        apply_knobs(cfg, job_.params);
        cfg.on_generation = [this](int gen, std::vector<std::vector<double>> F) { on_gen(gen, std::move(F)); };
        cfg.record_every = grid_kind_ != Grid::Generations ? 1 : static_cast<int>(spec_.record_every);
        try {
            out.result = run::run(job_.algorithm, rp, cfg);
        } catch (const TimeoutStop&) {
            // stopped at run_timeout: the trajectory up to its last record
            out.status = "timeout";
            out.trajectory = std::move(lines_);
            out.fe = fe_;
            return out;
        } catch (const std::exception& e) {
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
        const run::RunResult& res = out.result;
        // the final population: objectives, then variables
        std::vector<double> F, X;
        for (const auto& r : res.objectives) F.insert(F.end(), r.begin(), r.end());
        for (const auto& r : res.variables) X.insert(X.end(), r.begin(), r.end());
        const std::size_t n = res.objectives.size();
        {
            Table t;
            t.name = "final.csv";
            std::vector<std::string> cols;
            if (n) {
                for (std::size_t j = 0; j < m; ++j) cols.push_back("f" + std::to_string(j + 1));
                for (std::size_t j = 0; j < d; ++j) cols.push_back("x" + std::to_string(j + 1));
            }
            t.head = table_head(t.name, {"# mootation campaign final population v1"}, cols, {});
            t.cols = cols.size();
            t.rows = t.cols ? n : 0;
            for (std::size_t i = 0; i < t.rows; ++i) {
                t.values.insert(t.values.end(), F.begin() + static_cast<std::ptrdiff_t>(i * m),
                                F.begin() + static_cast<std::ptrdiff_t>((i + 1) * m));
                t.values.insert(t.values.end(), X.begin() + static_cast<std::ptrdiff_t>(i * d),
                                X.begin() + static_cast<std::ptrdiff_t>((i + 1) * d));
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
            Table t;
            t.name = name;
            std::vector<std::string> cols;
            for (std::size_t j = 0; j < m; ++j) cols.push_back("f" + std::to_string(j + 1));
            if (whole)
                for (std::size_t j = 0; j < d; ++j) cols.push_back("x" + std::to_string(j + 1));
            cols.push_back("extreme");
            t.head = table_head(name, {comment}, cols, {static_cast<std::int64_t>(m + (whole ? d : 0))});
            t.cols = cols.size();
            t.rows = kept.size();
            for (auto i : kept) {
                t.values.insert(t.values.end(), AF.row(i), AF.row(i) + m);
                if (whole) t.values.insert(t.values.end(), AX.begin() + static_cast<std::ptrdiff_t>(i * d),
                                           AX.begin() + static_cast<std::ptrdiff_t>((i + 1) * d));
                t.values.push_back(AE[i] ? 1.0 : 0.0);
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
                    s.values.insert(s.values.end(), AF.row(i), AF.row(i) + m);
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
        // the final indicators
        V final = V::object();
        bool all_finite = !F.empty();
        for (double v : F) all_finite = all_finite && std::isfinite(v);
        MeasureInputs fin = measure_in_;
        HvOptions final_hv;
        final_hv.exact_max_m = static_cast<int>(spec_.hv_exact_max_m);
        final_hv.mc_samples = static_cast<std::size_t>(spec_.hv_mc_samples);
        if (all_finite) {
            final = measure(F.data(), n, m, d ? X.data() : nullptr, d, final_metrics_, fin, final_hv);
            add_dense(final, F.data(), n, m, d ? X.data() : nullptr, d, final_metrics_);
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
        if (have_sel && spec_.archive_scenario) {
            std::vector<double> T, TX;
            for (auto i : arc_sel) {
                T.insert(T.end(), AF.row(i), AF.row(i) + m);
                TX.insert(TX.end(), AX.begin() + static_cast<std::ptrdiff_t>(i * d),
                          AX.begin() + static_cast<std::ptrdiff_t>((i + 1) * d));
            }
            V fa = measure(rows_ptr(T), arc_sel.size(), m, d ? rows_ptr(TX) : nullptr, d, final_metrics_, fin, final_hv);
            add_dense(fa, rows_ptr(T), arc_sel.size(), m, d ? rows_ptr(TX) : nullptr, d, final_metrics_);
            fa.set("n", V::integer(static_cast<std::int64_t>(arc_sel.size())));
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
        meta.set("operator_stats", V::boolean(false));
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
        meta.set("seconds", V::number(0.0));
        meta.set("record_seconds", V::number(0.0));
        if (res.finished && out.fe < budget_) {
            V cs = V::object();
            cs.set("evaluations", V::integer(out.fe));
            cs.set("budget", V::integer(budget_));
            meta.set("core_stopped", cs);
        }
        meta.set("unique_fe", V::integer(static_cast<std::int64_t>(seen_rows_.size())));
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
        if (spec_.archive) {
            std::vector<double> lo, hi;
            if (refs_.ideal && refs_.nadir) {
                lo = *refs_.ideal;
                hi = *refs_.nadir;
            }
            arc_ = std::make_unique<GridArchive>(p_.n_obj, p_.n_var, lo, hi, spec_.archive_delta);
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
        need_x_ = (snap_ && keep_x_) || constrained_;
        for (const auto& w : traj_metrics_) need_x_ = need_x_ || needs_x().count(w);
        for (const auto& w : traj_metrics_)
            if (std::find(spec_.record_at_metrics.begin(), spec_.record_at_metrics.end(), w) ==
                spec_.record_at_metrics.end())
                everywhere_.push_back(w);
        ledger_ = std::make_unique<RunLedger>(m, d, constrained_, arc_.get(), nullptr, inc_.get(), sarc_.get());
        std::set<long long> read_at(checkpoints_.begin(), checkpoints_.end());
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
            seen_rows_.insert(std::move(key));
        }
        const std::size_t c = static_cast<std::size_t>(native_->n_con);
        ledger_->begin(X.data(), F.data(), n, constrained_ ? G.data() : nullptr, constrained_ ? c : 0, nullptr);
        for (;;) {
            const long long b = ledger_->next();
            if (b < 0) break;
            fe_ = b;
            read(b);
        }
        fe_ = ledger_->fe();
    }

    void read(long long b)
    {
        if (arc_hv_ && grid_set_.count(b)) arc_hv_at_[b] = archive_hv();
        if (checkpoints_.count(b) && (arc_->size() || constrained_)) take_checkpoint(b);
    }

    // take_checkpoint: the archive scenario after exactly b evaluations
    // (archive_checkpoints): its indicators in meta's archive_at, its points in
    // archive_at.npz
    void take_checkpoint(long long b)
    {
        const std::size_t m = static_cast<std::size_t>(p_.n_obj), d = static_cast<std::size_t>(p_.n_var);
        std::vector<double> AX, T, TX;
        std::vector<char> AE;
        const select::Rows AF = archive_rows(AX, AE);
        const auto idx = archive_select(AF, static_cast<std::size_t>(p_.pop_size));
        for (auto i : idx) {
            T.insert(T.end(), AF.row(i), AF.row(i) + m);
            TX.insert(TX.end(), AX.begin() + static_cast<std::ptrdiff_t>(i * d),
                      AX.begin() + static_cast<std::ptrdiff_t>((i + 1) * d));
        }
        json::Value v = measure(rows_ptr(T), idx.size(), m, d ? rows_ptr(TX) : nullptr, d, traj_metrics_, measure_in_,
                                traj_hv_);
        add_dense(v, rows_ptr(T), idx.size(), m, d ? rows_ptr(TX) : nullptr, d, traj_metrics_);
        for (const auto& h : hv_skipped_) v.set(h, json::Value::null());
        v.set("fe", json::Value::integer(b));
        v.set("n", json::Value::integer(static_cast<std::int64_t>(idx.size())));
        archive_at_.set(std::to_string(b), v);
        ck_budget_.push_back(b);
        ck_n_.push_back(static_cast<std::int64_t>(idx.size()));
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
    void archive_frame(const select::Rows& F, std::vector<double>& lo, std::vector<double>& hi) const
    {
        if (arc_->frame(lo, hi)) return;
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
        std::vector<std::size_t> all;
        if (F.n <= k) {
            for (std::size_t i = 0; i < F.n; ++i) all.push_back(i);
            return all;
        }
        std::vector<double> lo, hi;
        if (arc_->problem_frame()) archive_frame(F, lo, hi);
        if (spec_.answer_select == "igdp_self") return select::igdp_self_order(F, k, lo, hi);
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
        return run_metrics::hypervolume(T.data(), kept.size(), F.m, *refs_.ideal, *refs_.nadir, 1.1, kHvMcSamples,
                                        0, p_.n_obj, "auto")
            .value;
    }

    std::vector<double> variables_of(std::size_t rows)
    {
        std::vector<double> X;
        if (need_x_ && rows && run::st().answer_variables)
            for (const auto& x : run::st().answer_variables()) X.insert(X.end(), x.begin(), x.end());
        return X;
    }

    void on_gen(int gen, std::vector<std::vector<double>> P)
    {
        if (spec_.run_timeout > 0 && seconds() > spec_.run_timeout) throw TimeoutStop{};
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

    void record(int gen, long long fe_rec, const std::vector<double>& F, const std::vector<double>& X,
                std::optional<long long> fe_pop)
    {
        const std::size_t m = static_cast<std::size_t>(p_.n_obj), d = static_cast<std::size_t>(p_.n_var);
        const std::size_t n = m ? F.size() / m : 0;
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
    std::unique_ptr<GridArchive> arc_;
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
    std::vector<std::int64_t> snap_fe_, snap_gen_, snap_n_;
    std::vector<float> snap_F_, snap_X_;
    std::set<long long> checkpoints_;
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

public:
    // what meta.json says of the registry: its code stamp and reference
    // version, and whether the run depends on its budget (budget.py)
    void set_registry(const ProblemRegistry& reg, const JobsRegistry& jreg)
    {
        code_stamp_ = reg.code_stamp;
        reference_version_ = reg.reference_version;
        budget_dependent_ = depends_on_budget(jreg, job_.algorithm, job_.params);
    }
};

}   // namespace mootation::campaign

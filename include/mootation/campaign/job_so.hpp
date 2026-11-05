#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// One job of a campaign at one objective in C++ (task 18, stage 1, §1.4):
// run/campaign_so.py's run_job_so, its own path beside the multi-objective
// JobRun (job_run.hpp), for a problem of the registry with n_obj = 1 (bbob,
// MGH, NIST). The same run and the same files as the Python engine's
// (tools/compare_results.py); campaign_so.py's docstring says what they are.
//
// THE RUN: scalar::run (the single-objective algorithms of mootation.scalar)
// on the problem of the suite through counting functions — every value asked
// for a row of a RunLedger at one objective (the best point and its records,
// scalar/improvements.hpp), a call of the problem's Jacobian one grad_call;
// a least-squares problem's residuals (and MGH's Jacobian) to the algorithm,
// its start x0 to lm, lbfgsb and eki; a problem with symmetries (task 19)
// its gradient where it is not a sum of squares (lj_cluster) and its gauge
// basis (lm_gauge = on); a bbob problem at the instance
// [campaign] bbob_instance gives (by_run: the run's seed; fixed: 1), f_opt
// that instance's. With [campaign] wrap_shift, wrap_rotate or wrap_perm the
// problem is seen through the wrappers (suite/wrap.hpp; task 20, §20.2.1):
// the run's box, its start and its answer in the wrapped coordinates, f_opt
// the problem's; a wrapper the problem refuses (a rotation of a problem
// with a box, WRAP-3; blocks, WRAP-7) makes the job incompatible.
// ============================================================================

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "../json.hpp"
#include "../run_ledger.hpp"
#include "../scalar/scalar.hpp"
#include "../so_metrics.hpp"
#include "../suite/wrap.hpp"
#include "job_run.hpp"
#include "jobs.hpp"
#include "registry.hpp"

namespace mootation::campaign {

class SoJob {
public:
    SoJob(const Job& job, const CampaignSpec& spec, const RegistryProblem& problem, std::string code_stamp)
        : job_(job), spec_(spec), p_(problem), code_stamp_(std::move(code_stamp))
    {
    }

    // campaign_so.refusal: why the pair cannot run at all, or nothing
    static std::optional<std::string> refusal(const std::string& algorithm, const RegistryProblem& p,
                                              const suite::Native& native)
    {
        if (!single(algorithm))
            return algorithm + ": not a single-objective algorithm (mootation.scalar.SINGLE; incompatible by "
                               "construction)";
        if ((algorithm == "lm" || algorithm == "eki") && native.n_res == 0)
            return algorithm + ": the problem gives no residuals (incompatible by construction)";
        bool boxless = false;
        for (const auto& [lo, hi] : p.bounds) boxless = boxless || std::isinf(lo) || std::isinf(hi);
        if (!local(algorithm) && boxless)
            return algorithm + ": a problem without a box (bounds -inf / +inf) is for lm, lbfgsb and eki "
                               "(incompatible by construction)";
        return std::nullopt;
    }

    JobOutcome run(const Process& process = {})
    {
        process_ = process;
        start_ = std::chrono::steady_clock::now();
        make_native();
        begin_meta();
        JobOutcome out;
        if (const auto why = refusal(job_.algorithm, p_, *native_)) {
            out.status = "failed";
            out.error = *why;
            incompatible_ = true;
            return out;
        }
        if (!wrap_refusal_.empty()) {
            out.status = "failed";
            out.error = wrap_refusal_ + " (incompatible by construction)";
            incompatible_ = true;
            return out;
        }
        ledger_ = std::make_unique<RunLedger>(1, static_cast<std::size_t>(p_.n_var), false, nullptr, nullptr, nullptr,
                                              nullptr);
        ledger_->set_reads({}, {});
        grid_ = grid();
        counts0_ = ops::repair_counts();
        row_counts_ = !threaded(job_.params);
        try {
            scalar::Start st;
            if (local(job_.algorithm) && !p_.x0.empty()) {
                st.has_x0 = true;
                st.x0 = wrapped_ ? wrapped_->map().to_run(p_.x0) : p_.x0;
            }
            const scalar::Result res =
                scalar::run(job_.algorithm, problem(), kEqTol, false, 0.0, static_cast<int>(job_.pop), job_.budget(),
                            static_cast<unsigned>(job_.seed), scalar_knobs(job_.params), {}, 0, st);
            if (cec_ > 0) peaks_ = peaks(res);
            repairs_ = res.repairs;
        } catch (const std::exception& e) {
            out.status = "failed";
            out.error = e.what();
            incompatible_ = out.error.find("incompatible by construction") != std::string::npos;
            return out;
        }
        out.status = "done";
        out.fe = ledger_->fe();
        trajectory(out.trajectory);
        out.records = out.trajectory.size();
        return out;
    }

    JobFiles finish(const JobOutcome& out)
    {
        using V = json::Value;
        JobFiles files;
        V& meta = meta_;
        if (out.status != "done") {
            meta.set("status", V::string("failed"));
            meta.set("error", V::string(out.error));
            meta.set("fe", V::integer(ledger_ ? ledger_->fe() : 0));
            meta.set("finished", now());
            meta.set("seconds", V::number(rounded_seconds()));
            if (incompatible_) meta.set("incompatible", V::boolean(true));
            files.meta = meta;
            return files;
        }
        const auto& records = ledger_->improvements();
        const auto& xs = ledger_->record_x();
        {
            Table t;
            t.name = "improvements.csv";
            t.head = table_head(t.name, {"# mootation campaign improvement records v1: a record each time the best "
                                         "point improved (scalar/improvements.hpp)"},
                                {"evaluations", "grad_calls", "f", "cv"}, {0, 1});
            t.cols = 4;
            t.rows = records.size();
            for (const auto& r : records)
                t.values.insert(t.values.end(), {static_cast<double>(r.evaluations), static_cast<double>(r.grad_calls),
                                                 r.f, r.cv});
            files.tables.push_back(std::move(t));
        }
        if (!records.empty()) {
            Table t;
            t.name = "final.csv";
            std::vector<std::string> cols = {"f1"};
            for (int j = 0; j < p_.n_var; ++j) cols.push_back("x" + std::to_string(j + 1));
            t.head = table_head(t.name, {"# mootation campaign answer v1 (one objective): the best point"}, cols, {});
            t.cols = cols.size();
            t.rows = 1;
            t.values.push_back(records.back().f);
            t.values.insert(t.values.end(), xs.back().begin(), xs.back().end());
            files.tables.push_back(std::move(t));
        }
        const double grad_cost = number_param("grad_cost", 1.0);
        const std::string kind = p_.targets;
        const std::optional<double> f_x0 = records.empty() ? std::nullopt : std::optional<double>(records.front().f);
        V precisions = V::array(), runtimes = V::array();
        std::vector<double> eps;
        if (!kind.empty()) eps = so::default_precisions(so::target_kind(kind));
        for (double e : eps) precisions.items.push_back(V::number(e));
        if (f_opt_ && f_x0)
            for (double t : so::targets(so::target_kind(kind), eps, *f_opt_, *f_x0)) {
                const auto rt = so::runtime(records, t, grad_cost);
                runtimes.items.push_back(rt ? V::number(*rt) : V::null());
            }
        const long long grad_calls = grad_calls_;
        meta.set("status", V::string("done"));
        meta.set("fe", V::integer(ledger_->fe()));
        meta.set("grad_calls", V::integer(grad_calls));
        meta.set("records", V::integer(static_cast<std::int64_t>(out.records)));
        V s = V::object();
        s.set("f_opt", f_opt_ ? V::number(*f_opt_) : V::null());
        s.set("f_opt_kind", p_.f_opt_kind.empty() ? V::null() : V::string(p_.f_opt_kind));
        s.set("reference_version", p_.f_opt_kind == "best_known" ? V::integer(so::kReferenceVersion) : V::null());
        s.set("instance", instance_ ? V::integer(*instance_) : V::null());
        s.set("targets", kind.empty() ? V::null() : V::string(kind));
        s.set("precisions", precisions);
        s.set("f_x0", f_x0 ? V::number(*f_x0) : V::null());
        s.set("runtimes", runtimes);
        s.set("grad_cost", V::number(grad_cost));
        s.set("cost", V::number(static_cast<double>(ledger_->fe()) + grad_cost * static_cast<double>(grad_calls)));
        s.set("improvements", V::integer(static_cast<std::int64_t>(records.size())));
        V rep = V::object();                             // task 20, §20.2.2 (scalar::Result::repairs)
        rep.set("checked", V::integer(repairs_.checked));
        rep.set("out_children", V::integer(repairs_.out_children));
        rep.set("out_vars", V::integer(repairs_.out_vars));
        s.set("repairs", rep);
        if (cec_ > 0) s.set("peaks", peaks_);
        if (spec_.stop_at > 0.0) s.set("stop_at", V::number(spec_.stop_at));
        if (wrapped_) s.set("wrap", wrap_meta());
        meta.set("so", s);
        V fin = V::object();
        if (!records.empty()) {
            const auto& b = records.back();
            fin.set("f", V::number(b.f));
            fin.set("delta_f", delta(b.f));
            fin.set("cv", V::number(b.cv));
            fin.set("fe_best", V::integer(b.evaluations));
            fin.set("grad_calls_best", V::integer(b.grad_calls));
        }
        meta.set("final", fin);
        meta.set("finished", now());
        meta.set("seconds", V::number(rounded_seconds()));
        files.meta = meta;
        return files;
    }

private:
    static constexpr double kEqTol = 1e-4;              // minimize's eq_tol

    static bool single(const std::string& a)
    {
        const auto& s = scalar::single_names();
        const auto& p = scalar::population_names();
        const auto& t = scalar::strategy_names();
        return std::find(s.begin(), s.end(), a) != s.end() || std::find(p.begin(), p.end(), a) != p.end() ||
               std::find(t.begin(), t.end(), a) != t.end() || local(a);
    }
    static bool local(const std::string& a) { return a == "lm" || a == "lbfgsb" || a == "eki"; }

    double number_param(const char* key, double fallback) const
    {
        const json::Value* v = job_.params.find(key);
        if (!v) return fallback;
        return v->kind == json::Value::Kind::Float ? v->f : static_cast<double>(v->i);
    }

    // the problem of the suite, a bbob one at the run's instance
    void make_native()
    {
        f_opt_ = p_.f_opt;
        json::Value spec = *p_.spec;
        const json::Value* fam = spec.find("family");
        const bool bbob = fam && fam->s == "bbob";
        if (bbob) {
            instance_ = spec_.bbob_instance == "by_run" ? job_.seed : 1;
            spec.set("bbob_iid", json::Value::integer(*instance_));
            const suite::Spec s = spec_of(spec);
            f_opt_ = suite::bbob::instance(s.bbob_fid, s.bbob_iid, s.n_var).f_opt;
        }
        native_ = std::shared_ptr<suite::Native>(suite::make(spec_of(spec)));
        cec_ = spec_of(spec).cec_problem;
        suite::WrapSpec w;
        w.name = job_.problem;
        w.shift = wrap_instance(spec_.wrap_shift);
        w.rotate = wrap_instance(spec_.wrap_rotate);
        w.perm = wrap_instance(spec_.wrap_perm);
        w.shift_scale = spec_.wrap_shift_scale;
        if (w.shift == 0 && w.rotate == 0 && w.perm == 0) return;
        std::vector<double> x_opt;
        if (bbob) {                                      // WRAP-2, WRAP-3: bbob's optimum, defined on R^n
            const suite::Spec s = spec_of(spec);
            x_opt = s.bbob.empty() ? suite::bbob::instance(s.bbob_fid, s.bbob_iid, s.n_var).x_opt : s.bbob[0].x_opt;
            w.defined_everywhere = true;
        }
        try {
            suite::WrapMap map(w, p_.n_var, 1, p_.bounds, x_opt, p_.x0);
            wrapped_ = std::make_shared<suite::Wrapped>(native_, std::move(map));
            native_ = wrapped_;
        } catch (const std::invalid_argument& e) {
            wrap_refusal_ = e.what();
        }
    }

    // [campaign] wrap_*: an instance number, or the run's seed ("by_run")
    int wrap_instance(const json::Value& v) const
    {
        return v.kind == json::Value::Kind::String ? static_cast<int>(job_.seed) : static_cast<int>(v.i);
    }

    // meta.json's so.wrap: the instances and their key (WrapMap::describe)
    json::Value wrap_meta() const
    {
        using V = json::Value;
        const suite::WrapSpec& w = wrapped_->map().spec();
        V o = V::object();
        o.set("shift", V::integer(w.shift));
        o.set("rotate", V::integer(w.rotate));
        o.set("perm", V::integer(w.perm));
        o.set("shift_scale", V::number(w.shift_scale));
        o.set("instance", V::string(wrapped_->map().describe()));
        return o;
    }

    // A CEC 2013 problem's peaks found (so_metrics.hpp, peaks_found) at the
    // competition's accuracies, on the run's final set: a strategy's minima
    // log, else the answer (the owner's В3)
    json::Value peaks(const scalar::Result& res) const
    {
        using V = json::Value;
        std::vector<std::vector<double>> X;
        std::vector<double> F;
        std::string set = "answer";
        for (const auto& [k, v] : res.settings)
            if (k == "minima") {
                set = "minima";
                for (const V& row : json::parse(v).items) {
                    F.push_back(-row.items[0].f);
                    std::vector<double> x;
                    for (std::size_t j = 3; j < row.items.size(); ++j) x.push_back(row.items[j].f);
                    X.push_back(x);
                }
            }
        if (set == "answer")
            for (std::size_t i = 0; i < res.variables.size(); ++i) {
                X.push_back(res.variables[i]);
                F.push_back(-res.objectives[i][0]);
            }
        const suite::cec2013::Info info = suite::cec2013::info(cec_);
        V eps = V::array(), found = V::array();
        for (double e : {1e-1, 1e-2, 1e-3, 1e-4, 1e-5}) {
            eps.items.push_back(V::number(e));
            found.items.push_back(V::integer(so::peaks_found(X, F, info.height, e, info.r)));
        }
        V out = V::object();
        out.set("set", V::string(set));
        out.set("eps", eps);
        out.set("found", found);
        out.set("n_global", V::integer(info.n_global));
        return out;
    }

    void account(const std::vector<double>& x, double f)
    {
        ledger_->set_grad_calls(grad_calls_);
        ledger_->begin(x.data(), &f, 1, nullptr, 0, nullptr);
        while (ledger_->next() >= 0) {
        }
        // the counts of the box at each count of the grid (task 20, §20.2.2):
        // the thread's counter, which only grows, since the run began; on one
        // thread only (a strategy's threads call this from theirs: SCALAR-15)
        while (row_counts_ && grid_next_ < grid_.size() && grid_[grid_next_] <= ledger_->fe()) {
            grid_counts_.push_back(ops::repair_counts() - counts0_);
            ++grid_next_;
        }
    }

    // a strategy on more than one thread (its knob threads, or an inner
    // table's): the counter of the job's thread is not the run's between
    // rounds, and the trajectory's counts are null (the run's totals stay
    // exact: strategy.hpp, round)
    static bool threaded(const json::Value& params)
    {
        if (params.kind == json::Value::Kind::String && !params.s.empty() && params.s[0] == '{') {
            try {
                return threaded(json::parse(params.s));
            } catch (const std::exception&) {
                return false;
            }
        }
        if (params.kind != json::Value::Kind::Object) return false;
        for (std::size_t i = 0; i < params.keys.size(); ++i) {
            const json::Value& v = params.items[i];
            if (params.keys[i] == "threads" && v.kind == json::Value::Kind::Int && v.i > 1) return true;
            if (threaded(v)) return true;
        }
        return false;
    }

    // the problem as the algorithm sees it: the counting functions
    scalar::Problem problem()
    {
        scalar::Problem P;
        for (const auto& [lo, hi] : wrapped_ ? wrapped_->map().bounds() : p_.bounds) {
            P.lo.push_back(lo);
            P.hi.push_back(hi);
        }
        P.m = 1;
        if (spec_.stop_at > 0.0 && f_opt_) P.stop_at = *f_opt_ + spec_.stop_at;       // [campaign] stop_at
        P.evaluate = [this](const std::vector<double>& x, std::vector<double>& f, std::vector<double>& g) {
            native_->evaluate(x.data(), f.data(), g.empty() ? nullptr : g.data());
            account(x, f[0]);
        };
        if (native_->n_res > 0 && single(job_.algorithm)) {
            const std::size_t m = static_cast<std::size_t>(native_->n_res), n = static_cast<std::size_t>(p_.n_var);
            P.residuals = [this, m](const std::vector<double>& x, std::vector<double>& r) {
                r.assign(m, 0.0);
                native_->residuals(x.data(), r.data(), nullptr);
                double s = 0.0;
                for (double v : r) s += v * v;           // in order, as the suite and the evaluator add
                account(x, 0.5 * s);
            };
            if (native_->has_jacobian)
                P.jacobian = [this, m, n](const std::vector<double>& x, scalar::Rows& J) {
                    ++grad_calls_;
                    std::vector<double> r(m), flat(m * n);
                    native_->residuals(x.data(), r.data(), flat.data());
                    J.assign(m, std::vector<double>(n));
                    for (std::size_t i = 0; i < m; ++i)
                        std::copy(flat.begin() + static_cast<std::ptrdiff_t>(i * n),
                                  flat.begin() + static_cast<std::ptrdiff_t>((i + 1) * n), J[i].begin());
                };
        }
        const std::size_t n = static_cast<std::size_t>(p_.n_var);
        if (native_->n_res == 0 && native_->has_gradient && single(job_.algorithm))
            P.gradient = [this, n](const std::vector<double>& x, std::vector<double>& gf, scalar::Rows&) {
                ++grad_calls_;
                gf.assign(n, 0.0);
                native_->gradient(x.data(), gf.data());
            };
        if (native_->gauge_dim > 0) {
            const std::size_t k = static_cast<std::size_t>(native_->gauge_dim);
            P.gauge_basis = [this, n, k](const std::vector<double>& x, scalar::Rows& N) {
                std::vector<double> flat(n * k);
                native_->gauge_basis(x.data(), flat.data());
                N.assign(n, std::vector<double>(k));
                for (std::size_t i = 0; i < n; ++i)
                    for (std::size_t c = 0; c < k; ++c) N[i][c] = flat[i * k + c];
            };
        }
        return P;
    }

    // campaign_so.grid_of
    std::vector<std::int64_t> grid() const
    {
        const std::int64_t budget = job_.budget();
        if (spec_.record_grid == "log") {
            std::vector<std::int64_t> g = log_grid(budget, spec_.record_per_decade);
            for (auto b : spec_.record_at)
                if (b <= budget) g.push_back(b);
            std::sort(g.begin(), g.end());
            g.erase(std::unique(g.begin(), g.end()), g.end());
            return g;
        }
        if (spec_.record_grid == "budgets") return budget_grid(budget, spec_.record_at);
        const std::int64_t stride = std::max<std::int64_t>(1, spec_.record_every) * std::max<std::int64_t>(1, job_.pop);
        std::vector<std::int64_t> g;
        for (std::int64_t b = stride; b <= budget; b += stride) g.push_back(b);
        return g;
    }

    json::Value delta(double f) const { return f_opt_ ? json::Value::number(f - *f_opt_) : json::Value::null(); }

    // at each count of the grid (and the run's end): the last record at or below it
    void trajectory(std::vector<std::string>& lines) const
    {
        using V = json::Value;
        const auto& records = ledger_->improvements();
        const auto& xs = ledger_->record_x();
        const long long fe = ledger_->fe();
        std::vector<std::int64_t> counts;
        for (auto b : grid())
            if (b <= fe) counts.push_back(b);
        if (fe && (counts.empty() || counts.back() != fe)) counts.push_back(fe);
        std::ptrdiff_t k = -1;
        for (auto b : counts) {
            while (k + 1 < static_cast<std::ptrdiff_t>(records.size()) &&
                   records[static_cast<std::size_t>(k + 1)].evaluations <= b)
                ++k;
            V rec = V::object();
            rec.set("fe", V::integer(b));
            if (k >= 0) {
                const auto& r = records[static_cast<std::size_t>(k)];
                rec.set("grad_calls", V::integer(r.grad_calls));
                rec.set("f", V::number(r.f));
                rec.set("delta_f", delta(r.f));
                rec.set("cv", V::number(r.cv));
                const auto& x = xs[static_cast<std::size_t>(k)];
                for (std::size_t j = 0; j < x.size(); ++j) rec.set("x" + std::to_string(j + 1), V::number(x[j]));
            } else {
                for (const char* key : {"grad_calls", "f", "delta_f", "cv"}) rec.set(key, V::null());
                for (int j = 0; j < p_.n_var; ++j) rec.set("x" + std::to_string(j + 1), V::null());
            }
            // the counts of the box at b; the last row the run's totals (the
            // points made but not evaluated included); null on threads (above)
            const std::size_t row = lines.size();
            const bool last = row + 1 == counts.size();
            const ops::RepairCounts* c = !row_counts_ ? nullptr
                                         : last ? &repairs_
                                         : row < grid_counts_.size() ? &grid_counts_[row] : nullptr;
            rec.set("repair_checked", c ? V::integer(c->checked) : V::null());
            rec.set("repair_out_children", c ? V::integer(c->out_children) : V::null());
            rec.set("repair_out_vars", c ? V::integer(c->out_vars) : V::null());
            lines.push_back(json::dump(rec, -1, ",", ":"));
        }
    }

    // campaign_so.MO_KEYS where the spec is not their default
    json::Value ignored() const
    {
        const CampaignSpec d;
        json::Value out = json::Value::array();
        auto add = [&out](bool differs, const char* key) {
            if (differs) out.items.push_back(json::Value::string(key));
        };
        add(spec_.metrics != d.metrics, "metrics");
        add(spec_.final_metrics != d.final_metrics, "final_metrics");
        add(spec_.archive != d.archive, "archive");
        add(spec_.set_answer != d.set_answer, "set_answer");
        add(spec_.set_metrics != d.set_metrics, "set_metrics");
        add(spec_.hard_budgets != d.hard_budgets, "hard_budgets");
        add(json::dump(spec_.snapshots) != json::dump(d.snapshots), "snapshots");
        add(spec_.archive_checkpoints != d.archive_checkpoints, "archive_checkpoints");
        add(spec_.all_evaluated != d.all_evaluated, "all_evaluated");
        add(spec_.dense_reference != d.dense_reference, "dense_reference");
        add(spec_.operator_stats != d.operator_stats, "operator_stats");
        return out;
    }

    void begin_meta()
    {
        using V = json::Value;
        V& meta = meta_;
        meta = V::object();
        meta.set("status", V::string("running"));
        meta.set("problem", V::string(job_.problem));
        meta.set("algorithm", V::string(job_.key()));
        meta.set("core", V::string(job_.algorithm));
        meta.set("seed", V::integer(job_.seed));
        meta.set("pop", V::integer(job_.pop));
        meta.set("gens", V::integer(job_.gens));
        meta.set("budget_fe", V::integer(job_.budget()));
        meta.set("budget_nominal", V::integer(job_.evaluations ? job_.evaluations
                                              : spec_.budget_fe ? spec_.budget_fe : job_.budget()));
        meta.set("params", job_.params);
        meta.set("n_objs", V::integer(1));
        meta.set("n_vars", V::integer(p_.n_var));
        meta.set("pop_note", V::string(job_.pop_note));
        meta.set("record_grid", V::string(spec_.record_grid));
        meta.set("record_per_decade", spec_.record_grid == "log" ? V::integer(spec_.record_per_decade) : V::null());
        V at = V::array();
        for (auto b : spec_.record_at) at.items.push_back(V::integer(b));
        meta.set("record_at", at);
        meta.set("record_every", V::integer(spec_.record_every));
        meta.set("problem_version", V::integer(p_.problem_version));
        meta.set("modification", p_.modification);
        meta.set("code_stamp", V::string(code_stamp_));
        meta.set("trajectory_version", V::integer(2));
        meta.set("ignored_at_m1", ignored());
        meta.set("mootation", process_.mootation);
        meta.set("revision", process_.revision);
        meta.set("host", process_.host);
        meta.set("platform", process_.platform);
        meta.set("python", process_.python);
        meta.set("pid", process_.pid);
        meta.set("fp_env", process_.fp_env);
        meta.set("started", now());
    }

    json::Value now() const { return process_.now ? json::Value::string(process_.now()) : json::Value::null(); }

    double rounded_seconds() const
    {
        const double s = process_.seconds ? process_.seconds()
                                          : std::chrono::duration<double>(std::chrono::steady_clock::now() - start_).count();
        return std::nearbyint(s * 1000.0) / 1000.0;
    }

    const Job& job_;
    const CampaignSpec& spec_;
    const RegistryProblem& p_;
    std::string code_stamp_;
    Process process_;
    std::chrono::steady_clock::time_point start_;
    std::shared_ptr<suite::Native> native_;
    std::shared_ptr<suite::Wrapped> wrapped_;           // [campaign] wrap_*: native_ through the wrappers
    std::string wrap_refusal_;
    ops::RepairCounts repairs_;
    std::vector<std::int64_t> grid_;                  // the trajectory's counts (grid())
    std::size_t grid_next_ = 0;
    std::vector<ops::RepairCounts> grid_counts_;      // the counts of the box at each of them
    ops::RepairCounts counts0_;
    bool row_counts_ = true;
    std::optional<double> f_opt_;
    std::optional<std::int64_t> instance_;
    std::unique_ptr<RunLedger> ledger_;
    long long grad_calls_ = 0;
    bool incompatible_ = false;
    int cec_ = 0;                                       // a CEC 2013 problem's number
    json::Value peaks_;
    json::Value meta_;
};

}   // namespace mootation::campaign

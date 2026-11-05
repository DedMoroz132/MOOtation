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
// that instance's.
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
        ledger_ = std::make_unique<RunLedger>(1, static_cast<std::size_t>(p_.n_var), false, nullptr, nullptr, nullptr,
                                              nullptr);
        ledger_->set_reads({}, {});
        try {
            scalar::Start st;
            if (local(job_.algorithm) && !p_.x0.empty()) {
                st.has_x0 = true;
                st.x0 = p_.x0;
            }
            scalar::run(job_.algorithm, problem(), kEqTol, false, 0.0, static_cast<int>(job_.pop), job_.budget(),
                        static_cast<unsigned>(job_.seed), scalar_knobs(job_.params), {}, 0, st);
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
        return std::find(s.begin(), s.end(), a) != s.end() || std::find(p.begin(), p.end(), a) != p.end() ||
               local(a);
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
        if (fam && fam->s == "bbob") {
            instance_ = spec_.bbob_instance == "by_run" ? job_.seed : 1;
            spec.set("bbob_iid", json::Value::integer(*instance_));
            const suite::Spec s = spec_of(spec);
            f_opt_ = suite::bbob::instance(s.bbob_fid, s.bbob_iid, s.n_var).f_opt;
        }
        native_ = std::shared_ptr<suite::Native>(suite::make(spec_of(spec)));
    }

    void account(const std::vector<double>& x, double f)
    {
        ledger_->set_grad_calls(grad_calls_);
        ledger_->begin(x.data(), &f, 1, nullptr, 0, nullptr);
        while (ledger_->next() >= 0) {
        }
    }

    // the problem as the algorithm sees it: the counting functions
    scalar::Problem problem()
    {
        scalar::Problem P;
        for (const auto& [lo, hi] : p_.bounds) {
            P.lo.push_back(lo);
            P.hi.push_back(hi);
        }
        P.m = 1;
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
    std::optional<double> f_opt_;
    std::optional<std::int64_t> instance_;
    std::unique_ptr<RunLedger> ledger_;
    long long grad_calls_ = 0;
    bool incompatible_ = false;
    json::Value meta_;
};

}   // namespace mootation::campaign

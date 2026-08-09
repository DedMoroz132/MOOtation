#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// A table of every evaluation the run performed: decision variables in,
// objectives out, one row each.
//
// This is the record of what was computed, as opposed to population.hpp's
// record of what SURVIVED. An optimizer discards most of what it evaluates,
// and for an expensive evaluator that discarded work is the bulk of the cost —
// worth keeping whether or not the optimizer wanted it.
//
// OFF BY DEFAULT, and off means absent rather than disabled: you install it or
// you do not. A run with no log opens no file, allocates nothing and pays no
// per-evaluation cost, because the object that would have done those things
// was never constructed.
//
//     mootation::io::EvaluationLog log("evals.csv");
//     vault.set_batch_executor(log.wrap(my_evaluator));
//
// It wraps a BatchExecutor rather than replacing it: the evaluator still does
// the evaluating, and the log only watches. That is also why it sees exactly
// what the algorithm asked for, including re-proposals of points already
// evaluated — which is information, not noise, since it tells you how much of
// the budget went on repeats.
//
// FORMAT — the same CSV-with-a-preamble as population.hpp:
//
//     # mootation evaluations v1
//     # n_vars=10 n_objs=2 n_lims=0
//     eval,x1,...,f1,f2,cv
//     1,5.0e-01,...,5.0e-01,7.1e-01,0.0
// ============================================================================

#include <fstream>
#include <iomanip>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../batch_executor.hpp"

namespace mootation::io {

class EvaluationLog {
public:
    EvaluationLog() = default;

    explicit EvaluationLog(const std::string& filename,
                           const std::map<std::string, std::string>& meta = {})
        : path_(filename), meta_(meta) {}

    EvaluationLog(const EvaluationLog&)            = delete;
    EvaluationLog& operator=(const EvaluationLog&) = delete;

    ~EvaluationLog() { close(); }

    // Wrap an evaluator. The returned executor evaluates through `inner`, then
    // appends one row per candidate.
    BatchExecutor wrap(BatchExecutor inner) {
        return [this, inner = std::move(inner)](const BatchRequest& req,
                                                BatchResponse& resp) {
            inner(req, resp);
            this->record(req, resp);
        };
    }

    std::size_t rows() const { return rows_; }
    const std::string& path() const { return path_; }

    void close() {
        if (file_.is_open()) file_.close();
    }

private:
    void record(const BatchRequest& req, const BatchResponse& resp) {
        if (path_.empty()) return;
        const std::size_t n = req.size();
        if (n == 0 || resp.objectives.size() != n) return;

        const int nvar = req.variables.empty() ? 0 : (int)req.variables[0].size();
        const int nbin = req.binary_variables.empty()
                             ? 0 : (int)req.binary_variables[0].size();
        const int nobj = resp.objectives.empty() ? 0 : (int)resp.objectives[0].size();
        const int nlim = resp.limits.empty() ? 0 : (int)resp.limits[0].size();

        if (!opened_) open(nvar, nbin, nobj, nlim);

        file_ << std::scientific << std::setprecision(12);
        for (std::size_t i = 0; i < n; ++i) {
            file_ << (++rows_);
            for (int j = 0; j < nvar; ++j) file_ << "," << req.variables[i][(std::size_t)j];
            for (int j = 0; j < nbin; ++j)
                file_ << "," << req.binary_variables[i][(std::size_t)j];
            for (int k = 0; k < nobj; ++k) file_ << "," << resp.objectives[i][(std::size_t)k];
            // One cv column rather than the raw limits: it is the number every
            // consumer actually compares, and the limits are recoverable from
            // the evaluator if anyone wants them.
            double cv = 0.0;
            if (nlim > 0)
                for (int k = 0; k < nlim; ++k)
                    cv += resp.limits[i][(std::size_t)k] > 0.0
                              ? resp.limits[i][(std::size_t)k] : 0.0;
            file_ << "," << cv << "\n";
        }
        // Flushed per batch, not per row: a run killed with Ctrl-C keeps
        // everything up to the last generation, and the cost is one flush per
        // batch of evaluations rather than one per evaluation.
        file_.flush();
    }

    void open(int nvar, int nbin, int nobj, int nlim) {
        file_.open(path_);
        if (!file_)
            throw std::runtime_error("EvaluationLog: cannot open " + path_);
        file_ << "# mootation evaluations v1\n";
        if (!meta_.empty()) {
            file_ << "#";
            for (const auto& kv : meta_) file_ << " " << kv.first << "=" << kv.second;
            file_ << "\n";
        }
        file_ << "# n_vars=" << nvar << " n_bin=" << nbin
              << " n_objs=" << nobj << " n_lims=" << nlim << "\n";

        file_ << "eval";
        for (int j = 0; j < nvar; ++j) file_ << ",x" << (j + 1);
        for (int j = 0; j < nbin; ++j) file_ << ",b" << (j + 1);
        for (int k = 0; k < nobj; ++k) file_ << ",f" << (k + 1);
        file_ << ",cv\n";
        opened_ = true;
    }

    std::string                        path_;
    std::map<std::string, std::string> meta_;
    std::ofstream                      file_;
    bool                               opened_ = false;
    std::size_t                        rows_   = 0;
};

} // namespace mootation::io

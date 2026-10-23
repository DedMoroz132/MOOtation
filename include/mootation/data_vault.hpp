#pragma once
// SPDX-License-Identifier: Apache-2.0

// ============================================================================
// DataVault — unified storage for individuals (population + archive in one buffer).
//
// Objective evaluation model (simplified 2026-06, EAGER/ASYNC modes removed):
//   1. set_variables / set_all_variables mark a slot dirty.
//   2. vault.sync() evaluates ALL dirty slots of the active population in bulk:
//      via batch_executor_ (if set) or the built-in Problem.calc_objs.
//   3. Lazy reads (objectives_of / get_cv / limits_of) evaluate a single
//      slot on demand via ensure_ready — correct even without sync().
//
// Slot contract: init_slot fully resets an individual (including derived
// fields of Ind subclasses — fitness/rank/...; fix X1 per the 2026-06 audit).
//
// FAILED EVALUATIONS (task 7 of 2026-10-10, supplement 3, §2). An evaluation
// that gives a non-finite objective or constraint value has failed — the
// binding turns an exception of the user's function, or a row a batch
// returned as None, into NaN. With set_fail_raise(true) the vault throws.
// Otherwise (the default) the slot is marked failed and carries, for the
// algorithm, a penalty: every objective one span above the worst finite value
// evaluated so far (span = worst − best, 1 while that is 0; 1 before any
// finite value), every constraint more than twice above the worst value seen
// (and positive). So a failed point is dominated by every point that did not
// fail, infeasible beyond any infeasible one, never the ideal, and equal to
// every other failure; when a finite value moves the worst or the best, every
// failed slot takes the new penalty. A failure is an evaluation: the binding
// counts it in the budget. The answer and the observers leave failed slots
// out (python/mootation_module.cpp), and outside the core their objectives
// are NaN. The penalty reaches an algorithm's nadir point only while a failure
// is still in its population, which an elitist selection ends as soon as it
// holds pop_size points that did not fail.
//
// EQUALITY CONSTRAINTS (task 7 of 2026-10-10, supplement 3, §3). With
// set_equalities(n, delta) the last n of the problem's constraint values are
// equalities h = 0. An evaluation's h becomes the inequality value |h| - delta
// in limits, so that every reader of limits_of keeps the meaning "<= 0
// satisfied" and cv adds max(|h| - delta, 0); |h| itself stays in the slot
// (eq_abs). set_eq_tol(delta) moves the tolerance — the binding's linear
// schedule does so between steps — and rewrites the equalities' values and
// cv of every evaluated slot from |h|, exactly. Raw values enter only from an
// evaluation and from seed_individual(..., raw = true), the seed population
// the caller hands over; an algorithm writing back values it read from
// limits_of (seed_individual, archive_push_data) writes them in the vault's
// convention, and |h| is taken as that value + delta. A core that keeps its
// own copies of the values sees the tolerance of the moment it read them.
// Without equalities (n = 0) nothing here runs. The violation max(|h| - delta,
// 0) and delta = 1e-4, the binding's default, are Liu and Wang's (IEEE TEVC
// 23(5), 2019, Eq. 3, the DOC problems); the RWMOP competition's guidelines
// (Kumar et al. 2021) take 1e-4 too, but count |h| itself beyond it.
//
// EPSILON LEVEL (supplement 3, §4.1). set_cv_level(eps): a point whose
// violation is at most eps has cv 0 for the algorithm, any other its
// violation — so every core that ranks by feasibility first (FEASIBILITY,
// CDP) does Takahama and Sakai's epsilon-level comparison (2006; Fan et al.
// 2019, Eq. 6): by the objectives where both violations are at most eps, by
// the violation otherwise. The violation itself stays in the slot (cv_raw);
// raw_cv(v) and max_raw_cv() read it. The binding's EPS_LEVEL schedule moves
// eps between steps; at 0, the default, cv is the violation, as before.
//
// WRITING BACK (task 8 of 2026-10-12). A core that keeps its own copies of
// points and writes its population back from them keeps a VaultPoint for each
// — everything the slot held: the variables and the integer genome, the
// objectives, the constraint values, |h| of the equalities, whether the
// evaluation failed — taken by point_of(v) and written back whole by
// put_point(v, p), so that no core lists the fields itself. put_point brings
// the point to the vault's present: the equalities from |h| at the current
// tolerance, a failed point to the current penalty, cv at the current level;
// refresh_point(p) does the same to a copy the core still holds, for a core
// that compares its copies by their cv. seed_individual writes a point from
// outside (a seed population, a resume); with constraints it needs their
// values: without them it throws, where it used to write zeros, a violating
// point passing for a feasible one.
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <deque>
#include <functional>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "batch_executor.hpp"
#include "individuals.hpp"
#include "operators/bit_flip.hpp"
#include "operators/bound_repair.hpp"
#include "problem.hpp"
#include "fp_contract.hpp"

namespace mootation {

// A point as a slot holds it (WRITING BACK above).
struct VaultPoint {
    std::vector<double> variables, objectives, limits;
    std::vector<int>    binary_variables;
    std::vector<double> eq_abs;           // |h| of the equalities
    bool                failed = false;
    double              cv = 0.0, cv_raw = 0.0;
};

// A copy a core keeps of a point, brought to the vault's present before the
// core compares it again (WRITING BACK): its cv, where the core reads one
// (with_cv: constraint handling on), and a failed point's objectives, the
// current penalty.
template <typename Vault>
void refresh_copy(const Vault& vault, VaultPoint& pt, std::vector<double>& objs, double& cv,
                  bool with_cv) {
    const double c = vault.refresh_point(pt);
    if (with_cv) cv = c;
    if (pt.failed) objs = pt.objectives;
}

// cache_check (task 17, supplement 2, §1.2): a point a core took from its
// cache of points evaluated (DMS-7, PAES-7, DMM-8), brought to the present,
// against the same x evaluated again — the objectives, the violation and the
// failure bit for bit; std::logic_error naming the core where they differ.
inline void check_cache_hit(const char* core, const std::vector<double>& f_cached, double cv_cached,
                            bool failed_cached, const std::vector<double>& f_new, double cv_new,
                            bool failed_new) {
    auto same = [](double a, double b) { return a == b || (std::isnan(a) && std::isnan(b)); };
    bool ok = f_cached.size() == f_new.size() && same(cv_cached, cv_new) && failed_cached == failed_new;
    for (std::size_t j = 0; ok && j < f_new.size(); ++j) ok = same(f_cached[j], f_new[j]);
    if (!ok)
        throw std::logic_error(std::string(core) + " cache_check: a point taken from the cache differs "
                               "from the same x evaluated again");
}

// The same for every copy a core keeps in a container of its own (members pt,
// objs and cv, as the cores name them), or in a container of such containers.
template <typename Vault, typename Copy>
void refresh_copies(const Vault& vault, std::vector<Copy>& copies, bool with_cv) {
    for (auto& c : copies) refresh_copy(vault, c.pt, c.objs, c.cv, with_cv);
}
template <typename Vault, typename Copy>
void refresh_copies(const Vault& vault, std::vector<std::vector<Copy>>& copies, bool with_cv) {
    for (auto& c : copies) refresh_copies(vault, c, with_cv);
}

// Whether a Problem says it is stochastic (problem.hpp, is_stochastic).
template <typename P, typename = void>
struct has_is_stochastic : std::false_type {};
template <typename P>
struct has_is_stochastic<P, std::void_t<decltype(std::declval<const P&>().is_stochastic())>>
    : std::true_type {};

template <typename Ind_type>
class DataVault {
private:
    int pop_size_, num_vars_, num_bin_vars_, num_objs_, num_lims_;

    // FIX 2026-07-08: DV-1 (BUG-crit).
    // Individuals are stored in a std::deque (was std::vector). A deque does
    // not relocate existing elements when it grows, so the addresses behind
    // references previously handed out (objectives_of/variables_of/get_ind/
    // archive_get/...) remain valid after expand()/alloc_slot()/archive_push*()
    // → eliminates the "dangling reference after buffer growth" class of UB.
    // Access is index-based only, buf_[i] (deque supports operator[]/size()/
    // resize()); memory contiguity is not required anywhere (no .data()/
    // memcpy/pointer arithmetic over the buffer).
    std::deque<Ind_type>  buf_;
    std::vector<char>     dirty_;

    std::vector<std::size_t> active_;
    std::vector<std::size_t> free_;

    // ── Archive ──────────────────────────────────────────────────────────────
    // Stores the real indices of slots holding archived individuals.
    // Slots are taken from the same buf_ as the main population —
    // a single pool, no duplicate buffers.
    std::vector<std::size_t> archive_;

    Problem<Ind_type>     problem_;
    BatchExecutor         batch_executor_;
    std::function<void()> post_sync_hook_;
    std::function<void(std::vector<double>&)> snap_;   // integer and discrete variables

    // ── Failed evaluations (FAILED EVALUATIONS above) ───────────────────────
    bool                fail_raise_ = false;
    long long           failed_n_   = 0;          // evaluations that failed
    std::vector<double> fin_lo_, fin_hi_;         // per objective: best, worst finite
    std::vector<double> lim_hi_;                  // per constraint: worst finite value

    // ── Equality constraints (EQUALITY CONSTRAINTS above) ───────────────────
    std::size_t eq_n_   = 0;                      // the last eq_n_ limits are equalities
    double      eq_tol_ = 0.0;                    // their tolerance delta

    // ── Epsilon level (EPSILON LEVEL above) ─────────────────────────────────
    double      cv_level_ = 0.0;                  // a violation up to this counts as none
    double      cv_max_   = 0.0;                  // the largest violation evaluated so far

    // ── Writing back (WRITING BACK above) ───────────────────────────────────
    unsigned long long present_ = 0;              // moves with the tolerance, level, penalty

    // ── Internal helpers ─────────────────────────────────────────────────────

    std::size_t real_idx(std::size_t v) const {
        if (v >= active_.size())
            throw std::out_of_range("DataVault: v=" + std::to_string(v) +
                                    " >= active=" + std::to_string(active_.size()));
        return active_[v];
    }

    // Allocates a new slot from free_, dynamically growing buf_ if needed.
    std::size_t alloc_slot() {
        if (free_.empty()) grow_buf(1);
        std::size_t r = free_.back();
        free_.pop_back();
        init_slot(r);
        return r;
    }

    // Grows the buffer by `extra` slots, adding them to free_.
    void grow_buf(std::size_t extra) {
        std::size_t old_sz = buf_.size();
        std::size_t new_sz = old_sz + extra;
        buf_  .resize(new_sz);
        dirty_.resize(new_sz, 0);
        for (std::size_t i = old_sz; i < new_sz; ++i) {
            init_slot(i);
            free_.push_back(i);
        }
    }

    // FIX X1 (2026-06 audit): full slot reset, including derived public
    // fields of Ind subclasses (fitness/rank/strength/...). Previously a
    // reused slot inherited the fields of its previous occupant — this
    // showed up as garbage fitness on individuals moved into the archive
    // (SPEA2+SDE) and potentially on any path through alloc_slot/expand.
    void init_slot(std::size_t r) {
        buf_[r] = Ind_type{};
        buf_[r].variables       .assign(num_vars_,     0.0);
        buf_[r].binary_variables.assign(num_bin_vars_, 0);
        buf_[r].objectives      .assign(num_objs_,     0.0);
        buf_[r].limits          .assign(num_lims_,     0.0);
        dirty_[r] = 0;
    }

    void calc_cv_slot(std::size_t r) {
        double s = 0.0;
        for (double g : buf_[r].limits) if (g > 0.0) s += g;
        buf_[r].cv_raw = s;
        buf_[r].cv = s > cv_level_ ? s : 0.0;
        if (!buf_[r].failed && s > cv_max_) cv_max_ = s;
    }

    // The penalty of a failed slot (FAILED EVALUATIONS), its cv included; of a
    // slot or of a VaultPoint.
    template <typename P>
    void penalize(P& s) const {
        for (std::size_t j = 0; j < s.objectives.size(); ++j) {
            if (fin_hi_.empty() || !(fin_hi_[j] >= fin_lo_[j])) { s.objectives[j] = 1.0; continue; }
            const double span = fin_hi_[j] > fin_lo_[j] ? fin_hi_[j] - fin_lo_[j] : 1.0;
            s.objectives[j] = fin_hi_[j] + span;
        }
        double cv = 0.0;
        for (std::size_t j = 0; j < s.limits.size(); ++j) {
            const double worst = lim_hi_.empty() ? 0.0 : lim_hi_[j];
            s.limits[j] = 2.0 * std::max(worst, 0.0) + 1.0;
            cv += s.limits[j];
        }
        s.cv = s.cv_raw = cv;
    }

    // A finite slot widens the finite range; true if the penalty moves.
    bool widen(const Ind_type& s) {
        if (fin_lo_.empty()) {
            fin_lo_.assign(static_cast<std::size_t>(num_objs_), std::numeric_limits<double>::infinity());
            fin_hi_.assign(static_cast<std::size_t>(num_objs_), -std::numeric_limits<double>::infinity());
            lim_hi_.assign(static_cast<std::size_t>(num_lims_), -std::numeric_limits<double>::infinity());
        }
        bool moved = false;
        for (std::size_t j = 0; j < s.objectives.size(); ++j) {
            if (s.objectives[j] < fin_lo_[j]) { fin_lo_[j] = s.objectives[j]; moved = true; }
            if (s.objectives[j] > fin_hi_[j]) { fin_hi_[j] = s.objectives[j]; moved = true; }
        }
        for (std::size_t j = 0; j < s.limits.size(); ++j)
            if (s.limits[j] > lim_hi_[j]) { lim_hi_[j] = s.limits[j]; moved = true; }
        return moved;
    }

    // After every evaluation of slot r (and every seeded one): a finite one
    // widens the range, a failed one is marked and penalized, or throws.
    void settle(std::size_t r) {
        Ind_type& s = buf_[r];
        bool bad = false;
        for (double v : s.objectives) bad = bad || !std::isfinite(v);
        for (double v : s.limits)     bad = bad || !std::isfinite(v);
        if (!bad) {
            s.failed = false;
            if (widen(s) && failed_n_ > 0) {
                for (auto& t : buf_) if (t.failed) penalize(t);
                ++present_;
            }
            return;
        }
        if (fail_raise_)
            throw std::runtime_error("mootation: an evaluation gave a non-finite value "
                                     "(on_failure = raise)");
        s.failed = true;
        ++failed_n_;
        penalize(s);
    }

    // The equalities' raw values h of slot s: |h| kept, |h| - delta in limits.
    void equalize(Ind_type& s) const {
        if (eq_n_ == 0) return;
        const std::size_t first = s.limits.size() - eq_n_;
        s.eq_abs.resize(eq_n_);
        for (std::size_t k = 0; k < eq_n_; ++k) {
            s.eq_abs[k] = std::abs(s.limits[first + k]);
            s.limits[first + k] = s.eq_abs[k] - eq_tol_;
        }
    }

    // The equalities' values of slot s in the vault's convention (written back
    // by an algorithm): |h| = value + delta; none given (all 0), none kept.
    void adopt(Ind_type& s, bool given) const {
        if (eq_n_ == 0) return;
        if (!given) { s.eq_abs.clear(); return; }
        const std::size_t first = s.limits.size() - eq_n_;
        s.eq_abs.resize(eq_n_);
        for (std::size_t k = 0; k < eq_n_; ++k) s.eq_abs[k] = s.limits[first + k] + eq_tol_;
    }

    // A point evaluated earlier, brought to the vault's present (WRITING
    // BACK): a failed one to the current penalty, otherwise the equalities
    // from |h| at the current tolerance; cv at the current level.
    template <typename P>
    void bring_up(P& p) const {
        if (p.failed) {
            penalize(p);
        } else if (eq_n_ > 0 && p.eq_abs.size() == eq_n_) {
            const std::size_t first = p.limits.size() - eq_n_;
            for (std::size_t k = 0; k < eq_n_; ++k) p.limits[first + k] = p.eq_abs[k] - eq_tol_;
        }
        double s = 0.0;
        for (double g : p.limits) if (g > 0.0) s += g;
        p.cv_raw = s;
        p.cv = s > cv_level_ ? s : 0.0;
    }

    void check_point(const VaultPoint& p, const char* who) const {
        if (static_cast<int>(p.variables.size()) != num_vars_ ||
                static_cast<int>(p.objectives.size()) != num_objs_ ||
                static_cast<int>(p.limits.size()) != num_lims_ ||
                static_cast<int>(p.binary_variables.size()) != num_bin_vars_)
            throw std::invalid_argument(std::string(who) +
                                        ": the point is not one of this vault's (point_of)");
    }

    void write_point(std::size_t r, const VaultPoint& p) {
        Ind_type& s = buf_[r];
        s.variables        = p.variables;
        s.binary_variables = p.binary_variables;
        s.objectives       = p.objectives;
        s.limits           = p.limits;
        s.eq_abs           = p.eq_abs;
        s.failed           = p.failed;
        bring_up(s);
        if (!s.failed && s.cv_raw > cv_max_) cv_max_ = s.cv_raw;
        dirty_[r] = 0;
    }

    VaultPoint read_point(std::size_t r) const {
        const Ind_type& s = buf_[r];
        VaultPoint p;
        p.variables        = s.variables;
        p.binary_variables = s.binary_variables;
        p.objectives       = s.objectives;
        p.limits           = s.limits;
        p.eq_abs           = s.eq_abs;
        p.failed           = s.failed;
        p.cv               = s.cv;
        p.cv_raw           = s.cv_raw;
        return p;
    }

    void calc_one(std::size_t r) {
        problem_.calc_objs(buf_[r]);
        equalize(buf_[r]);
        settle(r);
        calc_cv_slot(r);
        dirty_[r] = 0;
    }

    // Single-point batch: one point through batch_executor_.
    // Used by ensure_ready and refresh_objectives.
    void run_batch_single(std::size_t r) {
        BatchRequest req;
        req.real_indices.push_back(r);
        req.variables.push_back(buf_[r].variables);
        if (num_bin_vars_ > 0)
            req.binary_variables.push_back(buf_[r].binary_variables);

        BatchResponse resp;
        batch_executor_(req, resp);

        if (resp.objectives.empty()) return;   // executor gave no answer → slot stays dirty
        if (static_cast<int>(resp.objectives[0].size()) != num_objs_)
            throw std::logic_error("BatchExecutor: objectives[0] wrong size");
        buf_[r].objectives = resp.objectives[0];
        if (!resp.limits.empty() && !resp.limits[0].empty()) {
            if (static_cast<int>(resp.limits[0].size()) != num_lims_)
                throw std::logic_error("BatchExecutor: limits[0] wrong size");
            buf_[r].limits = resp.limits[0];
        } else if (num_lims_ > 0) {
            std::fill(buf_[r].limits.begin(), buf_[r].limits.end(), 0.0);
        }
        equalize(buf_[r]);
        settle(r);
        calc_cv_slot(r);
        dirty_[r] = 0;
    }

    // Lazy evaluation of a single slot (priority: batch_executor → Problem).
    void ensure_ready(std::size_t r) {
        if (!dirty_[r]) return;
        if (batch_executor_) run_batch_single(r);
        else                 calc_one(r);
    }

    void run_batch_executor() {
        BatchRequest req;
        req.real_indices.reserve(active_.size());
        req.variables   .reserve(active_.size());
        if (num_bin_vars_ > 0) req.binary_variables.reserve(active_.size());

        for (std::size_t v = 0; v < active_.size(); ++v) {
            std::size_t r = active_[v];
            if (!dirty_[r]) continue;
            req.real_indices.push_back(r);
            req.variables   .push_back(buf_[r].variables);
            if (num_bin_vars_ > 0)
                req.binary_variables.push_back(buf_[r].binary_variables);
        }
        if (req.size() == 0) return;

        BatchResponse resp;
        batch_executor_(req, resp);

        if (resp.objectives.size() != req.size())
            throw std::logic_error(
                "BatchExecutor: objectives.size()=" +
                std::to_string(resp.objectives.size()) +
                " != request.size()=" + std::to_string(req.size()));
        bool has_lims = !resp.limits.empty();
        if (has_lims && resp.limits.size() != req.size())
            throw std::logic_error("BatchExecutor: limits.size() mismatch");

        for (std::size_t i = 0; i < req.size(); ++i) {
            std::size_t r = req.real_indices[i];
            const auto& o = resp.objectives[i];
            if (static_cast<int>(o.size()) != num_objs_)
                throw std::logic_error("BatchExecutor: objectives[i] wrong size");
            buf_[r].objectives = o;
            if (has_lims) {
                const auto& g = resp.limits[i];
                if (static_cast<int>(g.size()) != num_lims_)
                    throw std::logic_error("BatchExecutor: limits[i] wrong size");
                buf_[r].limits = g;
            } else if (num_lims_ > 0) {
                std::fill(buf_[r].limits.begin(), buf_[r].limits.end(), 0.0);
            }
            equalize(buf_[r]);
            settle(r);
            calc_cv_slot(r);
            dirty_[r] = 0;
        }
    }

public:
    // ── Constructor ──────────────────────────────────────────────────────────

    DataVault(int n, Problem<Ind_type> prob)
        : pop_size_(n), problem_(std::move(prob))
    {
        num_vars_     = problem_.get_vars_n();
        num_bin_vars_ = problem_.get_bin_vars_n();
        num_objs_     = problem_.get_objs_n();
        num_lims_     = problem_.get_lims_n();

        // Base buffer 2*N: N active + N free.
        // grow_buf adds more when slots run out.
        std::size_t total = static_cast<std::size_t>(pop_size_) * 2;
        buf_  .resize(total);
        dirty_.resize(total, 0);
        for (std::size_t i = 0; i < total; ++i) init_slot(i);

        active_.resize(static_cast<std::size_t>(pop_size_));
        for (int i = 0; i < pop_size_; ++i)
            active_[i] = static_cast<std::size_t>(i);

        free_.resize(static_cast<std::size_t>(pop_size_));
        for (int i = 0; i < pop_size_; ++i)
            free_[i] = static_cast<std::size_t>(pop_size_ + i);
    }

    // ── Configuration ────────────────────────────────────────────────────────

    void set_batch_executor(BatchExecutor e)         { batch_executor_ = std::move(e); }
    void set_post_sync_hook(std::function<void()> h) { post_sync_hook_ = std::move(h); }

    // FAILED EVALUATIONS: throw on a non-finite value instead of the penalty.
    void set_fail_raise(bool r) { fail_raise_ = r; }
    // How many evaluations failed so far; whether active individual v did.
    long long failed_n() const { return failed_n_; }
    bool is_failed(std::size_t v) const { return buf_[real_idx(v)].failed; }
    bool archive_is_failed(std::size_t i) const {
        if (i >= archive_.size()) throw std::out_of_range("DataVault::archive_is_failed");
        return buf_[archive_[i]].failed;
    }

    bool has_batch_executor() const { return static_cast<bool>(batch_executor_); }

    // EQUALITY CONSTRAINTS: the last n constraint values are equalities, with
    // the tolerance delta. Set before the first evaluation.
    void set_equalities(std::size_t n, double delta) {
        if (n > static_cast<std::size_t>(num_lims_))
            throw std::invalid_argument("DataVault::set_equalities: " + std::to_string(n) +
                                        " equalities of " + std::to_string(num_lims_) +
                                        " constraints");
        eq_n_   = n;
        eq_tol_ = delta;
    }
    // A new tolerance: every evaluated slot's equalities and cv follow it.
    void set_eq_tol(double delta) {
        if (eq_n_ == 0 || delta == eq_tol_) { eq_tol_ = delta; return; }
        eq_tol_ = delta;
        ++present_;
        const std::size_t first = static_cast<std::size_t>(num_lims_) - eq_n_;
        for (std::size_t r = 0; r < buf_.size(); ++r) {
            Ind_type& s = buf_[r];
            if (dirty_[r] || s.failed || s.eq_abs.size() != eq_n_) continue;
            for (std::size_t k = 0; k < eq_n_; ++k) s.limits[first + k] = s.eq_abs[k] - eq_tol_;
            calc_cv_slot(r);
        }
    }
    std::size_t eq_n()   const { return eq_n_; }
    double      eq_tol() const { return eq_tol_; }

    // EPSILON LEVEL: a new level; every evaluated slot's cv follows it.
    void set_cv_level(double eps) {
        if (eps == cv_level_) return;
        cv_level_ = eps;
        ++present_;
        for (std::size_t r = 0; r < buf_.size(); ++r) {
            if (dirty_[r]) continue;
            Ind_type& s = buf_[r];
            s.cv = s.cv_raw > cv_level_ ? s.cv_raw : 0.0;
        }
    }
    double cv_level()   const { return cv_level_; }
    double max_raw_cv() const { return cv_max_; }
    // Active individual v's violation, whatever the level.
    double raw_cv(std::size_t v) {
        std::size_t r = real_idx(v); ensure_ready(r); return buf_[r].cv_raw;
    }

    // ── Dimensions ───────────────────────────────────────────────────────────

    int         pop_size()   const { return pop_size_;     }
    int         vars_n()     const { return num_vars_;     }
    int         bin_vars_n() const { return num_bin_vars_; }
    int         objs_n()     const { return num_objs_;     }
    int         lims_n()     const { return num_lims_;     }
    std::size_t active_n()   const { return active_.size(); }

    // parents_n() — the number of currently active individuals.
    // A semantic synonym of active_n() for (µ+λ) algorithms (RVEA, θ-DEA):
    // called at the start of step() BEFORE expand(), capturing the parent
    // population size. After RVEA selection active_ may be < pop_size_, so
    // the algorithm must read the up-to-date value, not pop_size().
    int         parents_n()  const { return static_cast<int>(active_.size()); }

    const std::vector<std::pair<std::optional<double>, std::optional<double>>>&
    get_bounds() const { return problem_.bounds; }

    // The problem's is_stochastic() where it has one (problem.hpp), false
    // otherwise: the cores' caches of points evaluated are off by default
    // on a stochastic problem (task 17, supplement 2, §1.3).
    bool stochastic() const {
        if constexpr (has_is_stochastic<Problem<Ind_type>>::value) return problem_.is_stochastic();
        else return false;
    }

    // ── Active population management ─────────────────────────────────────────

    // Appends k new slots to the end of active_.
    // expand(pop_size_) — the old expand() behavior (doubles the population).
    // expand(1)         — for NIMMO (a single scratch slot).
    // If free_ does not have enough slots, the buffer grows automatically.
    //
    // RETURNS: the index of the first added slot. The new slots occupy the
    // range [base, base+k) in active_. Algorithms should use this value
    // instead of manually computing `n+i`, which is wrong when the number
    // of active individuals at step() entry != pop_size() (see mibea/B5).
    // Old code that ignores the return value remains correct.
    //
    // SCRATCH SLOTS AND CONSUMERS. Steady-state cores (adaw, dhea, hlmea and
    // others) keep a persistent scratch slot at active index pop_size() for
    // the duration of the run. A consumer reading the answer set must
    // therefore iterate [0, pop_size()), NOT [0, active_n()) — and must not
    // touch the scratch slot before the first step(), because it is
    // unevaluated and reading its objectives consumes a function evaluation.
    int expand(int k) {
        if (k <= 0) return static_cast<int>(active_.size());
        std::size_t base = active_.size();
        std::size_t need = static_cast<std::size_t>(k);
        if (free_.size() < need) grow_buf(need - free_.size());
        active_.reserve(active_.size() + need);
        for (std::size_t i = 0; i < need; ++i) {
            std::size_t r = free_.back(); free_.pop_back();
            init_slot(r);
            active_.push_back(r);
        }
        return static_cast<int>(base);
    }

    // Keeps the first n active individuals, returns the rest to free_.
    void reduce(int n) {
        if (n < 0 || static_cast<std::size_t>(n) > active_.size())
            throw std::out_of_range("DataVault::reduce out of range");
        for (std::size_t v = static_cast<std::size_t>(n); v < active_.size(); ++v)
            free_.push_back(active_[v]);
        active_.resize(static_cast<std::size_t>(n));
    }

    void swap_active(std::size_t va, std::size_t vb) {
        std::swap(active_.at(va), active_.at(vb));
    }

    // ── Variable setters ─────────────────────────────────────────────────────
    // A value outside the box is clamped here, the last guard. An operator
    // repairs its offspring before writing them (operators/bound_repair.hpp),
    // so a clamp here means one did not: it is booked like any repair, as
    // "vault"/"clip" in the operator log and in the out-of-box counts
    // (oob_share), so that no clamp goes unseen (task 5, 2026-10-01: EMyO/C,
    // HCCA's DE and MaOEA-IAMD's noise used to rely on it without a count).

    void set_variables(std::size_t v, const std::vector<double>& vars) {
        std::size_t r = real_idx(v);
        if (static_cast<int>(vars.size()) != num_vars_)
            throw std::invalid_argument("set_variables: wrong size");
        int clamped = 0;
        for (int j = 0; j < num_vars_; ++j) {
            double val = vars[j];
            const auto& b = problem_.bounds[j];
            if (b.first  && val < *b.first)  { val = *b.first;  ++clamped; }
            if (b.second && val > *b.second) { val = *b.second; ++clamped; }
            buf_[r].variables[j] = val;
        }
        if (clamped > 0) book_clamp(clamped);
        if (snap_) snap_(buf_[r].variables);
        dirty_[r] = 1;
    }

    // Integer and discrete variables (task 7 of 2026-10-10, §10): a function
    // that rounds them in place, applied after the clamp at every write — the
    // Lamarckian repair, the rounded value written into the genome, for every
    // algorithm that writes through set_variables. Unset: every variable real.
    void set_snap(std::function<void(std::vector<double>&)> f) { snap_ = std::move(f); }

    void set_all_variables(std::size_t v,
                           const std::vector<double>& rvars,
                           const std::vector<int>&    bvars)
    {
        std::size_t r = real_idx(v);
        if (static_cast<int>(rvars.size()) != num_vars_)
            throw std::invalid_argument("set_all_variables: wrong real size");
        int clamped = 0;
        for (int j = 0; j < num_vars_; ++j) {
            double val = rvars[j];
            const auto& b = problem_.bounds[j];
            if (b.first  && val < *b.first)  { val = *b.first;  ++clamped; }
            if (b.second && val > *b.second) { val = *b.second; ++clamped; }
            buf_[r].variables[j] = val;
        }
        if (clamped > 0) book_clamp(clamped);
        if (snap_) snap_(buf_[r].variables);
        if (static_cast<int>(bvars.size()) != num_bin_vars_)
            throw std::invalid_argument("set_all_variables: wrong binary size");
        for (int j = 0; j < num_bin_vars_; ++j)
            buf_[r].binary_variables[j] = ops::clamp_gene(j, bvars[j]);
        dirty_[r] = 1;
    }

    // One write's clamps, booked as one offspring repaired (see above).
    static void book_clamp(int n) {
        ops::note_operator("vault", "clip");
        ops::RepairTally tally;
        for (int i = 0; i < n; ++i) tally.out();
    }

    // ── Sync ─────────────────────────────────────────────────────────────────

    // Evaluates all dirty slots of the active population:
    // batch_executor_ (if set) or the built-in Problem.calc_objs.
    void sync() {
        if (batch_executor_) {
            run_batch_executor();
        } else {
            for (std::size_t v = 0; v < active_.size(); ++v) {
                std::size_t r = active_[v];
                if (dirty_[r]) calc_one(r);
            }
        }
        if (post_sync_hook_) post_sync_hook_();
    }

    // ── Getters ──────────────────────────────────────────────────────────────

    // FIX 2026-07-08: DV-2 (WARN).
    // Validation of the component index j (previously [j] into the component
    // vector was unchecked → OOB for j outside [0,num_vars_)). Not a hot path
    // (single-component getters are called during output/saving, not in an
    // algorithm's inner loop), so an explicit throw is performance-safe.
    double get_variable(std::size_t v, int j) const {
        if (j < 0 || j >= num_vars_)
            throw std::out_of_range("DataVault::get_variable: j=" + std::to_string(j) +
                                    " out of [0," + std::to_string(num_vars_) + ")");
        return buf_[real_idx(v)].variables[j];
    }
    int get_bin_variable(std::size_t v, int j) const {
        if (j < 0 || j >= num_bin_vars_)
            throw std::out_of_range("DataVault::get_bin_variable: j=" + std::to_string(j) +
                                    " out of [0," + std::to_string(num_bin_vars_) + ")");
        return buf_[real_idx(v)].binary_variables[j];
    }

    // FIX (2026-06 simplification): cv and limits are now lazily safe — without
    // the EAGER mode, freshness after set_variables is guaranteed by ensure_ready.
    double get_cv(std::size_t v) {
        std::size_t r = real_idx(v); ensure_ready(r); return buf_[r].cv;
    }
    const std::vector<double>& limits_of(std::size_t v) {
        std::size_t r = real_idx(v); ensure_ready(r); return buf_[r].limits;
    }

    const std::vector<double>& objectives_of(std::size_t v) {
        std::size_t r = real_idx(v); ensure_ready(r); return buf_[r].objectives;
    }
    const std::vector<double>& variables_of(std::size_t v) const {
        return buf_[real_idx(v)].variables;
    }
    const std::vector<int>& binary_variables_of(std::size_t v) const {
        return buf_[real_idx(v)].binary_variables;
    }

    void refresh_objectives(std::size_t v) {
        std::size_t r = real_idx(v);
        dirty_[r] = 1;
        if (batch_executor_) run_batch_single(r);
        else                 calc_one(r);
    }

    // Whether active individual v still waits for its evaluation. For an
    // observer that must not trigger one: objectives_of() evaluates a dirty
    // slot on the spot, which would spend an evaluation the algorithm never
    // asked for.
    bool is_dirty(std::size_t v) const { return dirty_[real_idx(v)] != 0; }

    Ind_type&       get_ind(std::size_t v)       { return buf_[real_idx(v)]; }
    const Ind_type& get_ind(std::size_t v) const { return buf_[real_idx(v)]; }

    // ── Points a core keeps (WRITING BACK) ───────────────────────────────────

    // Everything active individual v holds, evaluated first.
    VaultPoint point_of(std::size_t v) {
        std::size_t r = real_idx(v); ensure_ready(r); return read_point(r);
    }
    VaultPoint archive_point_of(std::size_t i) const {
        if (i >= archive_.size()) throw std::out_of_range("DataVault::archive_point_of");
        return read_point(archive_[i]);
    }
    // Writes p into active slot v whole, brought to the vault's present.
    void put_point(std::size_t v, const VaultPoint& p) {
        check_point(p, "put_point");
        write_point(real_idx(v), p);
    }
    // Appends p to the archive, as put_point writes a slot.
    std::size_t archive_push_point(const VaultPoint& p) {
        check_point(p, "archive_push_point");
        std::size_t r = alloc_slot();
        write_point(r, p);
        archive_.push_back(r);
        return archive_.size() - 1;
    }
    // Copies active individual `from` into active slot `to` whole: the
    // "move without re-evaluating" of the MOEA/D family.
    void copy_point(std::size_t from, std::size_t to) { put_point(to, point_of(from)); }
    // A count that moves whenever the vault's present does (the tolerance, the
    // level or the failures' penalty): a core whose copies are many and whose
    // steps are short refreshes them only when it moved.
    unsigned long long present() const { return present_; }
    // Brings a copy the core holds to the vault's present; its cv.
    double refresh_point(VaultPoint& p) const { bring_up(p); return p.cv; }

    // ── Archive ──────────────────────────────────────────────────────────────
    //
    // The archive is a separate pool of slots within the same buf_.
    // Algorithms (MOEA/D EP, SPEA2, HypE) keep auxiliary solution
    // sets here independently of active_.
    //
    // Archive slots are always "clean": archive_push evaluates the source
    // first, archive_push_data takes ready-made objectives.
    //
    // archive_push(v)  — copies active individual v into a new archive slot.
    // archive_pop()    — removes the last archive slot, returning it to free_.
    // archive_clear()  — releases all archive slots.
    // archive_size()   — the number of individuals in the archive.
    // archive_get(i)   — reference to the i-th archived individual (mutable).
    // archive_objectives_of(i) — objectives of the i-th archived individual.
    // archive_variables_of(i)  — variables of the i-th archived individual.
    // archive_cv(i)            — constraint violation of the i-th archived individual.
    //
    // The archive can also be accessed via the raw real index:
    // archive_real_idx(i) — for rare cases of direct access.

    // Appends a copy of active individual v to the end of the archive.
    // Returns the index of the new archive entry.
    std::size_t archive_push(std::size_t v) {
        std::size_t src = real_idx(v);
        // FIX (2026-06 audit): evaluate the individual before copying,
        // so that no unevaluated state ends up in the archive.
        ensure_ready(src);
        std::size_t dst = alloc_slot();
        buf_[dst] = buf_[src];
        dirty_[dst] = 0;
        archive_.push_back(dst);
        return archive_.size() - 1;
    }

    // Adds an individual directly from data (no source in active_).
    // Convenient for initializing the archive from external data.
    std::size_t archive_push_data(const std::vector<double>& vars,
                                  const std::vector<double>& objs,
                                  const std::vector<int>&    bvars = {},
                                  const std::vector<double>& lims  = {})
    {
        // FIX (2026-06 audit): size validation (previously vars[j] was read
        // unchecked — OOB on short input).
        if (static_cast<int>(vars.size()) != num_vars_)
            throw std::invalid_argument(
                "archive_push_data: vars size " + std::to_string(vars.size()) +
                " != " + std::to_string(num_vars_));
        if (static_cast<int>(objs.size()) != num_objs_)
            throw std::invalid_argument(
                "archive_push_data: objs size " + std::to_string(objs.size()) +
                " != " + std::to_string(num_objs_));
        if (num_bin_vars_ > 0 && !bvars.empty() &&
                static_cast<int>(bvars.size()) != num_bin_vars_)
            throw std::invalid_argument("archive_push_data: bvars size mismatch");
        if (!lims.empty() && static_cast<int>(lims.size()) != num_lims_)
            throw std::invalid_argument("archive_push_data: lims size mismatch");
        if (lims.empty() && num_lims_ > 0)
            throw std::invalid_argument(
                "archive_push_data: this problem has " + std::to_string(num_lims_) +
                " constraints and the point carries no values for them (archive_push_point)");

        std::size_t r = alloc_slot();
        // vars + clip
        for (int j = 0; j < num_vars_; ++j) {
            double val = vars[j];
            const auto& b = problem_.bounds[j];
            if (b.first  && val < *b.first)  val = *b.first;
            if (b.second && val > *b.second) val = *b.second;
            buf_[r].variables[j] = val;
        }
        if (num_bin_vars_ > 0) {
            if (bvars.empty()) std::fill(buf_[r].binary_variables.begin(), buf_[r].binary_variables.end(), 0);
            else for (int j = 0; j < num_bin_vars_; ++j) buf_[r].binary_variables[j] = ops::clamp_gene(j, bvars[j]);
        }
        buf_[r].objectives = objs;
        if (num_lims_ > 0) {
            if (lims.empty()) std::fill(buf_[r].limits.begin(), buf_[r].limits.end(), 0.0);
            else buf_[r].limits = lims;
        }
        adopt(buf_[r], !lims.empty());
        settle(r);
        calc_cv_slot(r);
        dirty_[r] = 0;
        archive_.push_back(r);
        return archive_.size() - 1;
    }

    // Removes the last archive element, returning its slot to free_.
    void archive_pop() {
        if (archive_.empty())
            throw std::logic_error("DataVault::archive_pop: archive is empty");
        free_.push_back(archive_.back());
        archive_.pop_back();
    }

    // Removes an archive element by index (swap-with-last).
    void archive_erase(std::size_t i) {
        if (i >= archive_.size())
            throw std::out_of_range("DataVault::archive_erase: i out of range");
        free_.push_back(archive_[i]);
        archive_[i] = archive_.back();
        archive_.pop_back();
    }

    // Releases all archive slots.
    void archive_clear() {
        for (std::size_t r : archive_) free_.push_back(r);
        archive_.clear();
    }

    std::size_t archive_size() const { return archive_.size(); }

    Ind_type&       archive_get(std::size_t i)       {
        if (i >= archive_.size()) throw std::out_of_range("DataVault::archive_get");
        return buf_[archive_[i]];
    }
    const Ind_type& archive_get(std::size_t i) const {
        if (i >= archive_.size()) throw std::out_of_range("DataVault::archive_get");
        return buf_[archive_[i]];
    }

    const std::vector<double>& archive_objectives_of(std::size_t i) const {
        if (i >= archive_.size()) throw std::out_of_range("DataVault::archive_objectives_of");
        return buf_[archive_[i]].objectives;
    }
    const std::vector<double>& archive_variables_of(std::size_t i) const {
        if (i >= archive_.size()) throw std::out_of_range("DataVault::archive_variables_of");
        return buf_[archive_[i]].variables;
    }
    const std::vector<int>& archive_bin_variables_of(std::size_t i) const {
        if (i >= archive_.size()) throw std::out_of_range("DataVault::archive_bin_variables_of");
        return buf_[archive_[i]].binary_variables;
    }
    const std::vector<double>& archive_limits_of(std::size_t i) const {
        if (i >= archive_.size()) throw std::out_of_range("DataVault::archive_limits_of");
        return buf_[archive_[i]].limits;
    }
    double archive_cv(std::size_t i) const {
        if (i >= archive_.size()) throw std::out_of_range("DataVault::archive_cv");
        return buf_[archive_[i]].cv;
    }

    // Raw real index of the i-th archive slot — for rare cases.
    std::size_t archive_real_idx(std::size_t i) const {
        if (i >= archive_.size()) throw std::out_of_range("DataVault::archive_real_idx");
        return archive_[i];
    }

    // ── Seed (resume support) ────────────────────────────────────────────────
    //
    // Writes an individual without calling Problem.calc_objs.
    // dirty_[r] = 0 → ensure_ready will not trigger evaluation.
    // raw: lims are the problem's own values, as an evaluation gives them (a
    // seed population from outside); otherwise the vault's, as limits_of
    // returns them (an algorithm writing back). They differ only in the
    // equalities (EQUALITY CONSTRAINTS).

    void seed_individual(std::size_t v,
                         const std::vector<double>& vars,
                         const std::vector<double>& objs,
                         const std::vector<int>&    bvars = {},
                         const std::vector<double>& lims  = {},
                         bool                       raw   = false)
    {
        std::size_t r = real_idx(v);
        if (static_cast<int>(vars.size()) != num_vars_)
            throw std::invalid_argument(
                "seed_individual: vars size " + std::to_string(vars.size()) +
                " != " + std::to_string(num_vars_));
        if (static_cast<int>(objs.size()) != num_objs_)
            throw std::invalid_argument(
                "seed_individual: objs size " + std::to_string(objs.size()) +
                " != " + std::to_string(num_objs_));
        if (num_bin_vars_ > 0 && !bvars.empty() &&
                static_cast<int>(bvars.size()) != num_bin_vars_)
            throw std::invalid_argument("seed_individual: bvars size mismatch");
        if (!lims.empty() && static_cast<int>(lims.size()) != num_lims_)
            throw std::invalid_argument("seed_individual: lims size mismatch");
        if (lims.empty() && num_lims_ > 0)
            throw std::invalid_argument(
                "seed_individual: this problem has " + std::to_string(num_lims_) +
                " constraints and the point carries no values for them; a core writing "
                "back a point it read keeps it whole (point_of / put_point)");

        for (int j = 0; j < num_vars_; ++j) {
            double val = vars[j];
            const auto& b = problem_.bounds[j];
            if (b.first  && val < *b.first)  val = *b.first;
            if (b.second && val > *b.second) val = *b.second;
            buf_[r].variables[j] = val;
        }
        if (num_bin_vars_ > 0) {
            if (bvars.empty()) std::fill(buf_[r].binary_variables.begin(), buf_[r].binary_variables.end(), 0);
            else for (int j = 0; j < num_bin_vars_; ++j) buf_[r].binary_variables[j] = ops::clamp_gene(j, bvars[j]);
        }
        buf_[r].objectives = objs;
        if (num_lims_ > 0) {
            if (lims.empty()) std::fill(buf_[r].limits.begin(), buf_[r].limits.end(), 0.0);
            else buf_[r].limits = lims;
        }
        if (raw) equalize(buf_[r]);
        else     adopt(buf_[r], !lims.empty());
        settle(r);
        calc_cv_slot(r);
        dirty_[r] = 0;
    }
};

} // namespace mootation

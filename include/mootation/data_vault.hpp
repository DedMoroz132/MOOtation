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
// ============================================================================

#include <algorithm>
#include <cstddef>
#include <deque>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "batch_executor.hpp"
#include "individuals.hpp"
#include "problem.hpp"

namespace mootation {

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
        buf_[r].cv = s;
    }

    void calc_one(std::size_t r) {
        problem_.calc_objs(buf_[r]);
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

    bool has_batch_executor() const { return static_cast<bool>(batch_executor_); }

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

    void set_variables(std::size_t v, const std::vector<double>& vars) {
        std::size_t r = real_idx(v);
        if (static_cast<int>(vars.size()) != num_vars_)
            throw std::invalid_argument("set_variables: wrong size");
        for (int j = 0; j < num_vars_; ++j) {
            double val = vars[j];
            const auto& b = problem_.bounds[j];
            if (b.first  && val < *b.first)  val = *b.first;
            if (b.second && val > *b.second) val = *b.second;
            buf_[r].variables[j] = val;
        }
        dirty_[r] = 1;
    }

    void set_all_variables(std::size_t v,
                           const std::vector<double>& rvars,
                           const std::vector<int>&    bvars)
    {
        std::size_t r = real_idx(v);
        if (static_cast<int>(rvars.size()) != num_vars_)
            throw std::invalid_argument("set_all_variables: wrong real size");
        for (int j = 0; j < num_vars_; ++j) {
            double val = rvars[j];
            const auto& b = problem_.bounds[j];
            if (b.first  && val < *b.first)  val = *b.first;
            if (b.second && val > *b.second) val = *b.second;
            buf_[r].variables[j] = val;
        }
        if (static_cast<int>(bvars.size()) != num_bin_vars_)
            throw std::invalid_argument("set_all_variables: wrong binary size");
        for (int j = 0; j < num_bin_vars_; ++j)
            buf_[r].binary_variables[j] = (bvars[j] != 0) ? 1 : 0;
        dirty_[r] = 1;
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

    Ind_type&       get_ind(std::size_t v)       { return buf_[real_idx(v)]; }
    const Ind_type& get_ind(std::size_t v) const { return buf_[real_idx(v)]; }

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
            else for (int j = 0; j < num_bin_vars_; ++j) buf_[r].binary_variables[j] = (bvars[j] != 0) ? 1 : 0;
        }
        buf_[r].objectives = objs;
        if (num_lims_ > 0) {
            if (lims.empty()) std::fill(buf_[r].limits.begin(), buf_[r].limits.end(), 0.0);
            else buf_[r].limits = lims;
        }
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

    void seed_individual(std::size_t v,
                         const std::vector<double>& vars,
                         const std::vector<double>& objs,
                         const std::vector<int>&    bvars = {},
                         const std::vector<double>& lims  = {})
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

        for (int j = 0; j < num_vars_; ++j) {
            double val = vars[j];
            const auto& b = problem_.bounds[j];
            if (b.first  && val < *b.first)  val = *b.first;
            if (b.second && val > *b.second) val = *b.second;
            buf_[r].variables[j] = val;
        }
        if (num_bin_vars_ > 0) {
            if (bvars.empty()) std::fill(buf_[r].binary_variables.begin(), buf_[r].binary_variables.end(), 0);
            else for (int j = 0; j < num_bin_vars_; ++j) buf_[r].binary_variables[j] = (bvars[j] != 0) ? 1 : 0;
        }
        buf_[r].objectives = objs;
        if (num_lims_ > 0) {
            if (lims.empty()) std::fill(buf_[r].limits.begin(), buf_[r].limits.end(), 0.0);
            else buf_[r].limits = lims;
        }
        calc_cv_slot(r);
        dirty_[r] = 0;
    }
};

} // namespace mootation

#pragma once
// SPDX-License-Identifier: Apache-2.0

#include <utility>

#include "data_vault.hpp"

namespace mootation {

// Tag for deferred setup: Optimizer(vault, defer_setup)
struct DeferSetup {};
inline constexpr DeferSetup defer_setup{};

template <typename Ind_t, typename Alg_t>
class Optimizer {
private:
    Alg_t            algorithm_;
    DataVault<Ind_t> vault_;
    bool             setup_done_ = false;

public:
    // Legacy semantics: setup is called right away in the constructor.
    Optimizer(DataVault<Ind_t> vault) : vault_(std::move(vault)) {
        algorithm_.setup(vault_);
        setup_done_ = true;
    }

    // Deferred initialization: setup is NOT called. The user calls setup()
    // manually after configuring the algorithm (set_seed, set_kappa,
    // batch_executor...).
    Optimizer(DataVault<Ind_t> vault, DeferSetup)
        : vault_(std::move(vault)) {}

    void setup() {
        if (setup_done_) return;
        algorithm_.setup(vault_);
        setup_done_ = true;
    }

    // ============================================================
    //  setup_with_seed — the resume variant of setup.
    //  Plants the population into the DataVault via seed_individual (without
    //  calling Problem.calc_objs), then calls algorithm_.setup_seeded(vault),
    //  which recomputes fitness without touching random generation.
    //
    //  seed_vars[i] / seed_objs[i] — vars and objectives of the i-th
    //  individual. The outer vector length must be == pop_size().
    //  Optional seed_bvars/seed_lims (for problems with bin_vars/limits).
    // ============================================================
    void setup_with_seed(
        const std::vector<std::vector<double>>& seed_vars,
        const std::vector<std::vector<double>>& seed_objs,
        const std::vector<std::vector<int>>&    seed_bvars = {},
        const std::vector<std::vector<double>>& seed_lims  = {})
    {
        if (setup_done_)
            throw std::logic_error(
                "Optimizer::setup_with_seed: setup already done");
        int n = vault_.pop_size();
        if (static_cast<int>(seed_vars.size()) != n)
            throw std::invalid_argument(
                "setup_with_seed: seed_vars.size()=" +
                std::to_string(seed_vars.size()) +
                " != pop_size=" + std::to_string(n));
        if (static_cast<int>(seed_objs.size()) != n)
            throw std::invalid_argument(
                "setup_with_seed: seed_objs.size()=" +
                std::to_string(seed_objs.size()) +
                " != pop_size=" + std::to_string(n));
        if (!seed_bvars.empty() && static_cast<int>(seed_bvars.size()) != n)
            throw std::invalid_argument(
                "setup_with_seed: seed_bvars size mismatch");
        if (!seed_lims.empty() && static_cast<int>(seed_lims.size()) != n)
            throw std::invalid_argument(
                "setup_with_seed: seed_lims size mismatch");

        static const std::vector<int>    EMPTY_BV;
        static const std::vector<double> EMPTY_LM;
        for (int i = 0; i < n; ++i) {
            const std::vector<int>&    bv =
                seed_bvars.empty() ? EMPTY_BV : seed_bvars[i];
            const std::vector<double>& lm =
                seed_lims .empty() ? EMPTY_LM : seed_lims [i];
            vault_.seed_individual(static_cast<std::size_t>(i),
                                   seed_vars[i], seed_objs[i],
                                   bv, lm);
        }
        algorithm_.setup_seeded(vault_);
        setup_done_ = true;
    }

    void optimize(int it) {
        if (!setup_done_) setup();
        for (int i = 0; i < it; ++i) algorithm_.step(vault_);
    }

    // A single step — for cases where the user drives the loop (callback).
    void step() {
        if (!setup_done_) setup();
        algorithm_.step(vault_);
    }

    Alg_t&                  get_algorithm()       { return algorithm_; }
    const Alg_t&            get_algorithm() const { return algorithm_; }
    DataVault<Ind_t>&       get_vault()           { return vault_; }
    const DataVault<Ind_t>& get_vault()     const { return vault_; }
};

} // namespace mootation

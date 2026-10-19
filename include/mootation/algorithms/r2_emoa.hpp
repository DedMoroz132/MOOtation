#pragma once
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../das_dennis.hpp"
#include "sms_emoa.hpp"

namespace mootation {

// ============================================================================
// R2-EMOA — SMS-EMOA's steady state with R2-indicator-based selection
// H. Trautmann, T. Wagner, D. Brockhoff, "R2-EMOA: Focused Multiobjective
// Search Using R2-Indicator-Based Selection", LION 7, LNCS 7997, pp. 70-74,
// 2013, doi:10.1007/978-3-642-44973-4_8 (the authors' version,
// r2-emoa_trautmann2013: Algorithm 1, §3);
// D. Brockhoff, T. Wagner, H. Trautmann, "R2 Indicator-Based Multiobjective
// Search", Evol. Comput. 23(3):369-395, 2015, doi:10.1162/EVCO_a_00135
// (r2+r2-emoa_brockhoff2015: Def. 4-5, §5.1 Algorithm 1, §5.2).
//
// One step is SMS-EMOA's (sms_emoa.hpp): two parents uniformly, SBX and
// polynomial mutation, one offspring, P ∪ {z}, non-dominated sorting; then
// of the worst front R_h the point a* = argmin_a R2(R_h \ {a}; Λ, r*) goes
// (EC 2015 Algorithm 1, lines 6-8), i.e. the one of least contribution
// C_R2(a, R_h) = R2(R_h \ {a}) − R2(R_h) (Def. 5), with
//   R2(A; Λ, r*) = 1/|Λ| Σ_{λ∈Λ} min_{a∈A} max_j λ_j |r*_j − a_j|   (Def. 4).
// The contribution is computed as §5.2 does: for every weight, the best
// point's Tchebycheff value is subtracted from the second best's and added
// to the best point's sum; the least sum goes.
//
// Parameters (EC 2015 §5.2): SBX with crossover probability 0.9 and η_c = 15,
// a variable exchanged with probability 0.5 (sbx_var_prob, the library's
// default), polynomial mutation 1/n and η_m = 20; |Λ| = 500 weights,
// uniform in weight space ("uniweight", Λ_{w,k} of §2) or in angle space
// ("uniangle", Λ_{φ,k}); r* = (0, 0) in the paper's problems.
//
// THE KNOBS:
//   r2_mode    exact (default) | discrete. exact, at two objectives and
//              weights uniform in weight space: the R2 of the continuous
//              uniform distribution of weights, Schäpermeier, Kerschke et al.,
//              "Reinvestigating the R2 Indicator: Achieving Pareto Compliance
//              by Integration", arXiv:2407.01504, 2024 (r2-exact_schapermeier
//              2024, §3.2) — the limit of Λ_{w,k} as k grows (task 9: "where
//              possible, the exact R2"). discrete: Λ as below. Elsewhere
//              (three objectives and more, or angle space) discrete always.
//   r2_weights the number of weights, 500 (§5.2); from three objectives the
//              Das-Dennis lattice (das_dennis.hpp) with the least H that gives
//              at least that many (R2E-2)
//   r2_space   weight | angle: Λ_{w,k} or Λ_{φ,k} (§2; two objectives)
//   r2_offset  r* = z* − r2_offset·(nadir(Q) − z*), z* the ideal point of
//              every evaluation so far (R2E-1); 0
//   normalize  false (default, the paper's raw objectives) | true: the
//              objectives scaled to [0, 1] by ideal(Q) and nadir(Q) first
//
// READINGS (declared):
//   R2E-1. The utopian point. The paper fixes r* per problem ((0, 0); ZDT3
//     (0, −0.7735)); here it is the ideal point of every evaluation so far
//     (each objective's least value, as MOEA/D's z*), less r2_offset of Q's
//     range. Not the ideal point of the population alone (the task's first
//     reading): a utopian point that moves with the set it measures leaves
//     the exact R2 no fixed optimum, and the population drifts along the
//     front instead of spreading over it.
//   R2E-2. More than two objectives: the paper's experiments have two; the
//     Das-Dennis lattice is our choice (task 9).
//   R2E-3. LION 2013's Algorithm 1 takes R2(P \ {x}) over the whole
//     population, the journal's over R_h alone; the journal's is
//     implemented (the corrected later version).
//   R2E-4. A tie of contributions removes the point first in the
//     population, the offspring last (as SMS-3).
//   R2E-5. Everything else — the parents, the variation, constraints (CDP
//     through the non-dominated sorting), the binary genome, survival =
//     stochastic, the inner archive — is SMS-EMOA's (sms_emoa.hpp, SMS-5,
//     -6, -8, -9). SMS-EMOA's hypervolume knobs hv_contrib and hv_algo are
//     not taken (reported ignored): the selection computes no hypervolume.
// ============================================================================

// The exact R2 of a two-objective set for weights uniform on [0, 1]
// (Schäpermeier et al. 2024, §3.2), the utopian point at the origin: the
// points sorted by f1, the dominated ones and copies left out; point n is the
// best Tchebycheff value for the weights w (on f1) between
//   lo_n = y2_n / (y1_{n+1} + y2_n)   (its neighbour of larger f1; 0 for the last)
//   hi_n = y2_{n-1} / (y1_n + y2_{n-1}) (its neighbour of smaller f1; 1 for the first)
// with the switch from (1 − w) y2_n to w y1_n at w_n = y2_n / (y1_n + y2_n):
//   ∫_lo^hi max(w y1, (1 − w) y2) dw
//     = y2 ((1 − lo)² − (1 − w_n)²) / 2 + y1 (hi² − w_n²) / 2.
inline double r2_exact_2d(std::vector<std::pair<double, double>> Y)
{
    std::sort(Y.begin(), Y.end());
    std::vector<std::pair<double, double>> nd;
    for (const auto& y : Y)
        if (nd.empty() || y.second < nd.back().second) nd.push_back(y);
    const std::size_t N = nd.size();
    double r2 = 0.0;
    for (std::size_t n = 0; n < N; ++n) {
        const double y1 = nd[n].first, y2 = nd[n].second;
        if (y1 + y2 <= 0.0) continue;                      // at the utopian point: 0 throughout
        const double wn = y2 / (y1 + y2);
        const double lo = n + 1 < N ? y2 / (nd[n + 1].first + y2) : 0.0;
        const double hi = n > 0 ? nd[n - 1].second / (y1 + nd[n - 1].second) : 1.0;
        r2 += y2 * ((1.0 - lo) * (1.0 - lo) - (1.0 - wn) * (1.0 - wn)) / 2.0
            + y1 * (hi * hi - wn * wn) / 2.0;
    }
    return r2;
}

template <typename Ind_t>
class R2EMOACore : public SMSEMOACore<Ind_t> {
public:
    R2EMOACore()
    {
        this->other_ = true;
        this->set_pc(0.9);                                   // EC 2015 §5.2
    }

    void set_normalize(bool b) { normalize_ = b; }

    bool set_param(const std::string& k, double v)
    {
        if (k == "r2_weights") {
            if (!(v >= 2.0) || v != std::floor(v) || v > 1e7)
                throw std::invalid_argument("r2_weights must be a whole number >= 2");
            n_weights_ = static_cast<int>(v);
            weights_.clear();
            return true;
        }
        if (k == "r2_offset") {
            if (!(v >= 0.0)) throw std::invalid_argument("r2_offset must be >= 0");
            offset_ = v;
            return true;
        }
        return SMSEMOACore<Ind_t>::set_param(k, v);
    }

    bool set_word(const std::string& k, const std::string& w)
    {
        if (k == "r2_mode") {
            if (w != "exact" && w != "discrete")
                throw std::invalid_argument("r2_mode = '" + w + "': exact or discrete");
            exact_ = w == "exact";
            return true;
        }
        if (k == "r2_space") {
            if (w != "weight" && w != "angle")
                throw std::invalid_argument("r2_space = '" + w + "': weight or angle");
            angle_ = w == "angle";
            weights_.clear();
            return true;
        }
        if (k == "hv_contrib" || k == "hv_algo") return false;  // R2 contributions: no hypervolume
        return SMSEMOACore<Ind_t>::set_word(k, w);
    }

    // Λ as the run uses it (tests).
    const std::vector<std::vector<double>>& weights(std::size_t m)
    {
        build_weights(m);
        return weights_;
    }

protected:
    std::vector<double> other_contributions(const std::vector<std::vector<double>>& Fv,
                                            const std::vector<std::vector<double>>& Q) override
    {
        const std::size_t m = Fv.front().size();
        std::vector<double> lo(m, std::numeric_limits<double>::infinity());
        std::vector<double> hi(m, -std::numeric_limits<double>::infinity());
        for (const auto& f : Q)
            for (std::size_t j = 0; j < m; ++j) {
                lo[j] = std::min(lo[j], f[j]);
                hi[j] = std::max(hi[j], f[j]);
            }
        // R2E-1: the ideal point of every evaluation so far (every offspring
        // passes through Q), less r2_offset of Q's range
        if (ideal_.size() != m) ideal_ = lo;
        for (std::size_t j = 0; j < m; ++j) ideal_[j] = std::min(ideal_[j], lo[j]);
        std::vector<double> r(m), scale(m, 1.0);
        for (std::size_t j = 0; j < m; ++j) {
            const double span = hi[j] - std::min(lo[j], ideal_[j]);
            r[j] = ideal_[j] - offset_ * span;
            if (normalize_ && span > 0.0) scale[j] = 1.0 / span;
        }
        // the shifted (and scaled) objectives: y_j = (f_j − r*_j)·scale_j >= 0
        std::vector<std::vector<double>> Y(Fv.size(), std::vector<double>(m));
        for (std::size_t a = 0; a < Fv.size(); ++a)
            for (std::size_t j = 0; j < m; ++j)
                Y[a][j] = std::abs(Fv[a][j] - r[j]) * scale[j];
        if (m == 2 && exact_ && !angle_) return exact_contributions(Y);
        return discrete_contributions(Y);
    }

private:
    int    n_weights_ = 500;           // EC 2015 §5.2
    bool   exact_     = true;          // task 9: the exact R2 where possible
    bool   angle_     = false;         // uniweight
    double offset_    = 0.0;           // R2E-1
    bool   normalize_ = false;         // the paper's raw objectives
    std::vector<std::vector<double>> weights_;
    std::vector<double> ideal_;                              // R2E-1

    void build_weights(std::size_t m)
    {
        if (!weights_.empty() && weights_.front().size() == m) return;
        weights_.clear();
        const int k = n_weights_;
        if (m == 2) {
            const double pi = std::acos(-1.0);
            for (int i = 0; i < k; ++i) {
                if (!angle_) {                                   // Λ_{w,k}
                    const double w = static_cast<double>(i) / (k - 1);
                    weights_.push_back({w, 1.0 - w});
                } else {                                         // Λ_{φ,k}
                    const double phi = (pi / 2.0) * i / (k - 1);
                    if (i == k - 1) { weights_.push_back({1.0, 0.0}); continue; }
                    const double t = std::tan(phi);
                    weights_.push_back({t / (t + 1.0), 1.0 / (t + 1.0)});
                }
            }
            return;
        }
        int H = 1;                                               // R2E-2
        while (das_dennis::n_vectors(static_cast<int>(m), H) < k) ++H;
        weights_ = das_dennis::generate(static_cast<int>(m), H);
    }

    // §5.2: for every weight, (second best − best) to the best point's sum.
    std::vector<double> discrete_contributions(const std::vector<std::vector<double>>& Y)
    {
        build_weights(Y.front().size());
        std::vector<double> c(Y.size(), 0.0);
        for (const auto& w : weights_) {
            double best = std::numeric_limits<double>::infinity(), second = best;
            std::size_t arg = 0;
            for (std::size_t a = 0; a < Y.size(); ++a) {
                double u = 0.0;
                for (std::size_t j = 0; j < w.size(); ++j) u = std::max(u, w[j] * Y[a][j]);
                if (u < best) { second = best; best = u; arg = a; }
                else if (u < second) second = u;
            }
            c[arg] += second - best;
        }
        for (double& x : c) x /= static_cast<double>(weights_.size());
        return c;
    }

    // Def. 5 with the exact R2: R2(R_h \ {a}) − R2(R_h).
    static std::vector<double> exact_contributions(const std::vector<std::vector<double>>& Y)
    {
        std::vector<std::pair<double, double>> P;
        for (const auto& y : Y) P.emplace_back(y[0], y[1]);
        const double all = r2_exact_2d(P);
        std::vector<double> c(Y.size());
        for (std::size_t a = 0; a < Y.size(); ++a) {
            auto without = P;
            without.erase(without.begin() + static_cast<std::ptrdiff_t>(a));
            c[a] = r2_exact_2d(without) - all;
        }
        return c;
    }
};

}   // namespace mootation

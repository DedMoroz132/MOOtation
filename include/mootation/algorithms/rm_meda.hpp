#pragma once
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "../constraint_mode.hpp"
#include "../data_vault.hpp"
#include "../detail/constrained.hpp"
#include "../operators/bound_repair.hpp"
#include "../operators/learned_coords.hpp"

namespace mootation {

// ============================================================================
// RM-MEDA — the regularity model-based multiobjective estimation of
// distribution algorithm
// Q. Zhang, A. Zhou, Y. Jin, "RM-MEDA: A Regularity Model-Based Multiobjective
// Estimation of Distribution Algorithm", IEEE Trans. Evol. Comput. 12(1):41-63,
// 2008, doi:10.1109/TEVC.2007.894202 (§III, §V-B). The erratum (12(3), 2008,
// doi:10.1109/TEVC.2008.923818) replaces Fig. 20 only.
//
// Under the Karush-Kuhn-Tucker conditions the Pareto set of a continuous
// problem with m objectives is a piecewise continuous (m − 1)-D manifold; the
// population is read as points around it, ξ = ζ + ε (Eq. 2). One generation:
//   modeling (Step 2.1-2.2): the (m − 1)-D Local PCA partitions Pop(t) into K
//     clusters S^j; in each, the mean x̄^j and the principal components U_i^j;
//     a_i^j, b_i^j the least and greatest projection (x − x̄^j)ᵀU_i^j (Eqs.
//     3-4), Ψ^j the hyper-rectangle a_i − 0.25(b_i − a_i) <= α_i <= b_i +
//     0.25(b_i − a_i) along U_1..U_{m−1} (the model extended by half);
//     σ_j = (λ_m + ... + λ_n) / (n − m + 1), the mean of the other
//     eigenvalues (Eq. 5);
//   reproduction (Reproduction/S, N times): a cluster τ with probability
//     vol(Ψ^τ) / Σ vol(Ψ^j), a uniform point x' of Ψ^τ, x = x' + ε',
//     ε' ~ N(0, σ_τ I);
//   selection (NDS-selection): Q ∪ Pop(t) sorted into fronts, whole fronts
//     while they fit; then, while more than N, the member of the last front
//     with the smallest crowding distance in what is left of that front goes,
//     the distances recomputed after every removal, a tie drawn at random.
// The (m − 1)-D Local PCA: every L_j an affine (m − 1)-D space through a
// random point of Pop(t); then, until the partition stays, each point to the
// nearest L_j and each L_j the principal subspace of its points.
// The answer is the population (Step 1 returns its non-dominated points).
//
// Parameters: K = n_clusters (5, §V-B); the extension rm_extension (0.25 on
// each side, §III-C); a variable outside the box is put on the bound it
// crossed, bound_repair = clip (RM-8; the text's random reset is
// bound_repair = native or random; resample refused). The paper's population
// is 100 at two objectives and 200 at three (§V-B).
//
// The algorithm leans on a regular Pareto set — an (m − 1)-D manifold in the
// variables; a problem whose set is not one (disconnected pieces, a set of
// full dimension, discrete variables) is outside its assumption.
//
// READINGS (declared):
//   RM-1. The first L_j's directions: an orthonormal basis drawn at random
//     (Gram-Schmidt on standard normal vectors); the paper names only the
//     random point.
//   RM-2. Local PCA stops when the partition repeats, or after 100 rounds.
//     A point equally near two spaces joins the first.
//   RM-3. A cluster with fewer than two points keeps its space from the round
//     before (no covariance to take one from) and has no model: it is never
//     sampled. pop_size must exceed n_clusters, or setup() refuses.
//   RM-4. N(0, σ I) is read as the covariance σ I: ε' has standard deviation
//     √σ_τ in each variable (σ_j of Eq. 5 is a mean of variances).
//   RM-5. vol(Ψ^j) is the product of Ψ^j's side lengths (1 at m = 1). When
//     every model has volume 0 (all its points on one spot), the cluster is
//     drawn uniformly among those with a model.
//   RM-6. The crowding distance is the paper's: the average side length of the
//     cuboid around F(x) in the objectives, not normalized (NSGA-II divides
//     by each objective's range); the extremes of a front count infinite.
//   RM-7. Constraints: Deb's constrained domination in the sorting
//     (constraint_mode not NONE); the paper has none.
//   RM-8. §V-B resets a variable outside the box to a random value inside
//     it, but the paper's own curves (Figs. 3, 4, 11, 12: F1, F2, F5, F6 at
//     10 000 evaluations) are reached with the variable kept at the bound,
//     not with the random reset; and where the Pareto set lies on a bound
//     (ZDT) the random reset keeps the population off it. The default is
//     clip; the text's rule is bound_repair = native (or random).
// Real variables only.
// ============================================================================

template <typename Ind_t>
class RMMEDACore {
public:
    ConstraintMode constraint_mode = ConstraintMode::NONE;

    void set_seed(unsigned s) { rng_.seed(s); }
    void set_n_clusters(int k)
    {
        if (k < 1) throw std::invalid_argument("rm_meda: n_clusters must be >= 1");
        K_ = k;
    }
    void set_bound_repair(ops::BoundRepair r)
    {
        ops::require_repair(r, "rm_meda", false, true);
        repair_ = r == ops::BoundRepair::Native ? ops::BoundRepair::Random : r;
    }

    bool set_param(const std::string& k, double v)
    {
        if (k == "rm_extension") {
            if (!(v >= 0.0) || v > 100.0)
                throw std::invalid_argument("rm_extension must be in [0, 100]");
            ext_ = v;
            return true;
        }
        return false;
    }

    void setup(DataVault<Ind_t>& vault)
    {
        check(vault);
        const auto& b = vault.get_bounds();
        std::uniform_real_distribution<double> U(0.0, 1.0);
        std::vector<double> x(b.size());
        for (int i = 0; i < vault.pop_size(); ++i) {
            for (std::size_t j = 0; j < b.size(); ++j)
                x[j] = *b[j].first + U(rng_) * (*b[j].second - *b[j].first);
            vault.set_variables(static_cast<std::size_t>(i), x);
        }
        vault.sync();
    }

    void setup_seeded(DataVault<Ind_t>& vault) { check(vault); }

    void step(DataVault<Ind_t>& vault)
    {
        const int N = vault.pop_size();
        const int d = std::min(vault.objs_n() - 1, vault.vars_n());       // the manifold's dimension
        std::vector<std::vector<double>> X(static_cast<std::size_t>(N));
        for (int i = 0; i < N; ++i) X[static_cast<std::size_t>(i)] = vault.variables_of(static_cast<std::size_t>(i));
        const auto models = local_pca(X, d);

        // Reproduction/S, N times, in the slots after Pop(t)
        std::vector<double> vol;
        std::vector<std::size_t> with;
        for (std::size_t j = 0; j < models.size(); ++j)
            if (models[j].ok) { with.push_back(j); vol.push_back(models[j].vol); }
        const double total = std::accumulate(vol.begin(), vol.end(), 0.0);
        if (!(total > 0.0)) std::fill(vol.begin(), vol.end(), 1.0);        // RM-5
        std::discrete_distribution<std::size_t> pick(vol.begin(), vol.end());
        std::uniform_real_distribution<double> U(0.0, 1.0);
        std::normal_distribution<double> N01(0.0, 1.0);
        const auto& bounds = vault.get_bounds();
        const std::size_t n = bounds.size();
        const int q0 = vault.expand(N);
        ops::note_operator("rm_meda", ops::bound_repair_name(repair_));
        for (int c = 0; c < N; ++c) {
            const Model& M = models[with[pick(rng_)]];
            std::vector<double> x = M.mean;
            for (int i = 0; i < d; ++i) {
                const double w = M.b[i] - M.a[i];
                const double lo = M.a[i] - ext_ * w, alpha = lo + U(rng_) * (w + 2.0 * ext_ * w);
                for (std::size_t t = 0; t < n; ++t) x[t] += alpha * M.U[i][t];
            }
            const std::vector<double> on_model = x;
            const double sd = std::sqrt(M.sigma);                        // RM-4
            for (std::size_t t = 0; t < n; ++t) x[t] += sd * N01(rng_);
            ops::RepairTally tally;
            for (std::size_t t = 0; t < n; ++t) {
                const double lo = *bounds[t].first, hi = *bounds[t].second;
                if (x[t] < lo || x[t] > hi) {
                    tally.out();
                    x[t] = ops::repair_value(x[t], lo, hi, on_model[t], repair_, rng_);
                }
            }
            vault.set_variables(static_cast<std::size_t>(q0 + c), x);
        }
        vault.sync();

        // NDS-selection of N from Q ∪ Pop(t)
        const auto keep = select(vault, 2 * N, N);
        std::vector<VaultPoint> chosen;
        for (int s : keep) chosen.push_back(vault.point_of(static_cast<std::size_t>(s)));
        for (int i = 0; i < N; ++i) vault.put_point(static_cast<std::size_t>(i), chosen[static_cast<std::size_t>(i)]);
        vault.reduce(N);
    }

private:
    struct Model {
        std::vector<double> mean;
        std::vector<std::vector<double>> U;      // the m − 1 directions, each of n
        std::vector<double> a, b;
        double sigma = 0.0, vol = 0.0;
        bool ok = false;
    };

    int K_ = 5;
    double ext_ = 0.25;
    ops::BoundRepair repair_ = ops::BoundRepair::Clip;                    // RM-8
    std::mt19937 rng_{0};

    void check(DataVault<Ind_t>& vault) const
    {
        if (vault.bin_vars_n() > 0)
            throw std::invalid_argument("rm_meda: continuous variables only");
        for (const auto& b : vault.get_bounds())
            if (!b.first || !b.second)
                throw std::invalid_argument("rm_meda: every variable needs finite bounds");
        if (vault.pop_size() <= K_)
            throw std::invalid_argument("rm_meda: pop_size must exceed n_clusters (" +
                                        std::to_string(K_) + ")");
    }

    // The squared distance from x to the affine space through M.mean along M.U.
    static double dist2(const std::vector<double>& x, const Model& M)
    {
        double all = 0.0;
        for (std::size_t t = 0; t < x.size(); ++t) all += (x[t] - M.mean[t]) * (x[t] - M.mean[t]);
        for (const auto& u : M.U) {
            double p = 0.0;
            for (std::size_t t = 0; t < x.size(); ++t) p += (x[t] - M.mean[t]) * u[t];
            all -= p * p;
        }
        return std::max(all, 0.0);
    }

    // The principal subspace of the points S of X: the mean, the first d
    // components; with eig, all n eigenvalues, largest first.
    static void fit(Model& M, const std::vector<std::vector<double>>& X,
                    const std::vector<std::size_t>& S, int d, std::vector<double>& eig)
    {
        const std::size_t n = X.front().size();
        M.mean.assign(n, 0.0);
        for (std::size_t i : S)
            for (std::size_t t = 0; t < n; ++t) M.mean[t] += X[i][t];
        for (double& v : M.mean) v /= static_cast<double>(S.size());
        eig.clear();
        if (S.size() < 2) return;                                         // RM-3
        std::vector<std::vector<double>> C(n, std::vector<double>(n, 0.0));
        for (std::size_t i : S)
            for (std::size_t r = 0; r < n; ++r) {
                const double dr = X[i][r] - M.mean[r];
                for (std::size_t c = r; c < n; ++c) C[r][c] += dr * (X[i][c] - M.mean[c]);
            }
        for (std::size_t r = 0; r < n; ++r)
            for (std::size_t c = r; c < n; ++c) {
                C[r][c] /= static_cast<double>(S.size() - 1);
                C[c][r] = C[r][c];
            }
        std::vector<std::vector<double>> V;
        ops::coords_detail::jacobi(C, eig, V);
        M.U.assign(static_cast<std::size_t>(d), std::vector<double>(n));
        for (int i = 0; i < d; ++i)
            for (std::size_t t = 0; t < n; ++t) M.U[static_cast<std::size_t>(i)][t] = V[t][static_cast<std::size_t>(i)];
    }

    // The (m − 1)-D Local PCA and the models of Step 2.2.
    std::vector<Model> local_pca(const std::vector<std::vector<double>>& X, int d)
    {
        const std::size_t N = X.size(), n = X.front().size();
        const std::size_t K = static_cast<std::size_t>(K_);
        std::vector<Model> L(K);
        std::uniform_int_distribution<std::size_t> any(0, N - 1);
        std::normal_distribution<double> N01(0.0, 1.0);
        for (auto& M : L) {
            M.mean = X[any(rng_)];
            M.U.assign(static_cast<std::size_t>(d), std::vector<double>(n));
            for (int i = 0; i < d; ++i) {                                 // RM-1
                auto& u = M.U[static_cast<std::size_t>(i)];
                for (double& v : u) v = N01(rng_);
                for (int k = 0; k < i; ++k) {
                    const auto& w = M.U[static_cast<std::size_t>(k)];
                    double p = 0.0;
                    for (std::size_t t = 0; t < n; ++t) p += u[t] * w[t];
                    for (std::size_t t = 0; t < n; ++t) u[t] -= p * w[t];
                }
                double norm = 0.0;
                for (double v : u) norm += v * v;
                norm = std::sqrt(norm);
                for (double& v : u) v /= norm;
            }
        }
        std::vector<std::size_t> part(N, K), next(N);
        std::vector<std::vector<std::size_t>> S(K);
        std::vector<std::vector<double>> eig(K);
        for (int round = 0; round < 100; ++round) {                       // RM-2
            for (std::size_t i = 0; i < N; ++i) {
                std::size_t best = 0;
                double bd = dist2(X[i], L[0]);
                for (std::size_t j = 1; j < K; ++j) {
                    const double dj = dist2(X[i], L[j]);
                    if (dj < bd) { bd = dj; best = j; }
                }
                next[i] = best;
            }
            if (next == part) break;
            part = next;
            for (auto& s : S) s.clear();
            for (std::size_t i = 0; i < N; ++i) S[part[i]].push_back(i);
            for (std::size_t j = 0; j < K; ++j)
                if (!S[j].empty()) {
                    auto U_before = L[j].U;
                    fit(L[j], X, S[j], d, eig[j]);
                    if (S[j].size() < 2) L[j].U = std::move(U_before);    // RM-3
                }
        }
        // Step 2.2: Ψ^j and σ_j of every cluster with two points or more
        for (std::size_t j = 0; j < K; ++j) {
            Model& M = L[j];
            M.ok = S[j].size() >= 2;
            if (!M.ok) continue;
            M.a.assign(static_cast<std::size_t>(d), std::numeric_limits<double>::infinity());
            M.b.assign(static_cast<std::size_t>(d), -std::numeric_limits<double>::infinity());
            for (std::size_t i : S[j])
                for (int k = 0; k < d; ++k) {
                    double p = 0.0;
                    for (std::size_t t = 0; t < n; ++t) p += (X[i][t] - M.mean[t]) * M.U[static_cast<std::size_t>(k)][t];
                    M.a[static_cast<std::size_t>(k)] = std::min(M.a[static_cast<std::size_t>(k)], p);
                    M.b[static_cast<std::size_t>(k)] = std::max(M.b[static_cast<std::size_t>(k)], p);
                }
            double rest = 0.0;
            for (std::size_t t = static_cast<std::size_t>(d); t < eig[j].size(); ++t) rest += std::max(eig[j][t], 0.0);
            M.sigma = n > static_cast<std::size_t>(d) ? rest / static_cast<double>(n - static_cast<std::size_t>(d)) : 0.0;
            M.vol = 1.0;                                                  // RM-5
            for (int k = 0; k < d; ++k)
                M.vol *= (1.0 + 2.0 * ext_) * (M.b[static_cast<std::size_t>(k)] - M.a[static_cast<std::size_t>(k)]);
        }
        return L;
    }

    // NDS-selection: N of the slots 0..total−1.
    std::vector<int> select(DataVault<Ind_t>& vault, int total, int N)
    {
        const bool cons = constraint_mode != ConstraintMode::NONE;
        const std::size_t T = static_cast<std::size_t>(total);
        std::vector<std::vector<double>> F(T);
        std::vector<double> cv(T, 0.0);
        for (std::size_t i = 0; i < T; ++i) {
            F[i] = vault.objectives_of(i);
            if (cons) cv[i] = vault.get_cv(i);
        }
        std::vector<std::vector<std::size_t>> dom(T);
        std::vector<int> count(T, 0);
        for (std::size_t i = 0; i < T; ++i)
            for (std::size_t j = i + 1; j < T; ++j) {
                if (detail::dominates(constraint_mode, F[i], cv[i], F[j], cv[j])) {
                    dom[i].push_back(j); ++count[j];
                } else if (detail::dominates(constraint_mode, F[j], cv[j], F[i], cv[i])) {
                    dom[j].push_back(i); ++count[i];
                }
            }
        std::vector<int> keep;
        std::vector<std::size_t> front;
        for (std::size_t i = 0; i < T; ++i) if (count[i] == 0) front.push_back(i);
        while (static_cast<int>(keep.size()) + static_cast<int>(front.size()) <= N) {
            for (std::size_t i : front) keep.push_back(static_cast<int>(i));
            if (static_cast<int>(keep.size()) == N) return keep;
            std::vector<std::size_t> next;
            for (std::size_t i : front)
                for (std::size_t j : dom[i]) if (--count[j] == 0) next.push_back(j);
            std::stable_sort(next.begin(), next.end());
            front = std::move(next);
        }
        // the front that overflows: one removal at a time
        while (static_cast<int>(keep.size() + front.size()) > N) {
            const auto dc = crowding(F, front);
            const double least = *std::min_element(dc.begin(), dc.end());
            std::vector<std::size_t> ties;
            for (std::size_t t = 0; t < dc.size(); ++t) if (dc[t] == least) ties.push_back(t);
            std::uniform_int_distribution<std::size_t> one(0, ties.size() - 1);
            front.erase(front.begin() + static_cast<std::ptrdiff_t>(ties[one(rng_)]));
        }
        for (std::size_t i : front) keep.push_back(static_cast<int>(i));
        return keep;
    }

    // RM-6: the average side length of the cuboid around each member.
    static std::vector<double> crowding(const std::vector<std::vector<double>>& F,
                                        const std::vector<std::size_t>& front)
    {
        const double inf = std::numeric_limits<double>::infinity();
        const std::size_t m = F.front().size();
        std::vector<double> d(front.size(), 0.0);
        std::vector<std::size_t> o(front.size());
        for (std::size_t k = 0; k < m; ++k) {
            std::iota(o.begin(), o.end(), std::size_t{0});
            std::stable_sort(o.begin(), o.end(), [&](std::size_t a, std::size_t b) {
                return F[front[a]][k] < F[front[b]][k];
            });
            d[o.front()] = d[o.back()] = inf;
            for (std::size_t t = 1; t + 1 < o.size(); ++t)
                d[o[t]] += (F[front[o[t + 1]]][k] - F[front[o[t - 1]]][k]) / static_cast<double>(m);
        }
        return d;
    }
};

}   // namespace mootation

#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// Differential evolution as the crossover of the switchable algorithms
// (task 7 of 2026-10-10, §2.1-2.2): nsga2, nsga2_sdr, ibea_eplus, spea2_sde,
// agemoea and sms_emoa take crossover = "de" or "de_pbest" (real_crossover.hpp).
//
//   de        DE/rand/1/bin: the mutant v = r1 + F·(r2 − r3), crossed
//             binomially with the target r0 — each variable from v with
//             probability CR and variable j_rand from v always (JADE: Zhang &
//             Sanderson, IEEE TEVC 13(5):945-958, 2009, Eqs. 1 and 4; source
//             jade_zhang2009). r0 and r1 are the pair the host's mating
//             selection gave, r2 and r3 two more parents from the same
//             selection, all four different (multi_parent.hpp, MP-6). Two
//             children: (r0, r1, r2, r3) = (p1, p2, d1, d2) and (p2, p1, d2, d1).
//   de_pbest  current-to-pbest/1/bin (JADE, Eqs. 6-7 and Table I): v = x +
//             F·(x_pbest − x) + F·(x_r1 − x̃_r2), x the target (p1 for the first
//             child, p2 for the second), x_pbest a random member of the best
//             ⌈p·N⌉ (de_p, 0.1; JADE's p, "5-20 %"; with SHADE p is drawn
//             per child, below), x_r1 from the host's mating
//             selection and x̃_r2 from the population ∪ the archive. WHAT "BEST"
//             MEANS IS THE HOST'S OWN ORDER, the one its selection uses (nsga2,
//             nsga2_sdr: rank, then crowding distance; ibea_eplus: fitness; spea2_sde:
//             fitness; agemoea: rank, then survival score; sms_emoa: rank, then
//             hypervolume contribution in the last front) — JADE has one scalar f.
//   archive   (de_archive, on by default) the parents the host's selection
//             displaced, at most N, random ones removed beyond (JADE, §IV-A
//             and Table I, line 26): found by their variables, a population
//             member missing from the next population is displaced.
// F = 0.5 and CR = 1 by default (the knobs F and CR): CR = 1 moves the whole
// vector, which keeps the step invariant to a rotation of the variables. A
// variable outside the box is repaired (bound_repair, reflect by default; the
// target's value is the parent for midpoint; resample and native refused).
//
// ADAPTATION (de_adapt = "shade"; "none" by default, F and CR fixed) after
// SHADE (Tanabe & Fukunaga, "Success-history based parameter adaptation for
// differential evolution", CEC 2013, pp. 71-78, doi:10.1109/CEC.2013.6557555;
// source shade_tanabe2013: §V and Algorithm 1):
//   memory   H cells (M_CR, M_F), H = de_memory (5), all 0.5 at the start
//   draw     per child a random cell r; CR ~ N(M_CR[r], 0.1) clipped to [0, 1];
//            F ~ Cauchy(M_F[r], 0.1), drawn again while F ≤ 0, cut to 1 above
//            1 (Eqs. 15-16, out-of-range values as in JADE, §IV-C)
//   p        de_pbest only: per child p = rand[p_min, 0.2], p_min = 2/N (Eq.
//            20, Algorithm 1 line 12), in place of de_p
//   update   after every generation of the host (as many children judged as
//            the population has members) with at least one success:
//            cell k = M_CR ← Σ w·CR (weighted mean), M_F ← Σ w·F² / Σ w·F
//            (weighted Lehmer mean, Eqs. 17-19), k cycling through the cells
//            (lines 27-30); an empty success set leaves the memory alone
// SUCCESS here is multi-objective: the child is in the next population, the
// host's selection kept it (found by its variables) — SHADE's f(u) < f(x) has
// no meaning with several objectives. The weights (de_adapt_weight): "equal"
// (the default) — SHADE weighs by the improvement |f(u) − f(x)|, which does
// not exist here — or "contribution", a surviving child's hypervolume
// contribution to the next population's first front, the objectives
// normalised to that population's range and the reference point 1.1 (exact up
// to three objectives, 10 000 Monte-Carlo samples above), as shares of their
// sum; children outside the first front weigh 0, and when every weight is 0
// they weigh alike.
//
// A HOST'S PART: observe() with the population at the start of every step (the
// verdict on the previous step's children, the archive) and, for de_pbest, a
// function returning a random member of its best ⌈p·N⌉. A steady-state host
// (sms_emoa) observes every step, a generational one every generation.
//
// THE RECORD (de_record(), thread-local like the repair counts): the memory as
// it stands and the CR and F drawn and kept since the binding last read them —
// what operator_stats() reports as de_mu_cr_*, de_mu_f_*, de_cr_mean,
// de_f_mean, de_cr_success_mean and de_f_success_mean.
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../hv_contribution.hpp"
#include "bound_repair.hpp"
#include "sbx.hpp"   // sbx_require_bound
#include "../random.hpp"

namespace mootation::ops {

enum class DEAdapt { None, SHADE };
enum class DEWeight { Equal, Contribution };

inline std::optional<DEAdapt> parse_de_adapt(const std::string& s)
{
    if (s == "none")  return DEAdapt::None;
    if (s == "shade") return DEAdapt::SHADE;
    return std::nullopt;
}

inline std::optional<DEWeight> parse_de_weight(const std::string& s)
{
    if (s == "equal")        return DEWeight::Equal;
    if (s == "contribution") return DEWeight::Contribution;
    return std::nullopt;
}

// The knobs F and CR of the switchable hosts, checked.
inline double checked_de_F(double F)
{
    if (!(F >= 0.0 && F <= 2.0)) throw std::invalid_argument("F must be in [0, 2]");
    return F;
}
inline double checked_de_CR(double CR)
{
    if (!(CR >= 0.0 && CR <= 1.0)) throw std::invalid_argument("CR must be in [0, 1]");
    return CR;
}

// ── the record the binding reads ─────────────────────────────────────────────
struct DERecord {
    bool used = false;
    std::vector<double> mcr, mf;              // the SHADE memory now (empty: none)
    long long drawn = 0;                      // children since the last read ...
    double    cr_sum = 0.0, f_sum = 0.0;      // ... and their CR and F
    long long ok = 0;                         // successes since the last read ...
    double    cr_ok_sum = 0.0, f_ok_sum = 0.0;
    // the same over the whole run
    long long t_drawn = 0, t_ok = 0;
    double    t_cr_sum = 0.0, t_f_sum = 0.0, t_cr_ok_sum = 0.0, t_f_ok_sum = 0.0;
};
inline DERecord& de_record() { thread_local DERecord r; return r; }

// ── one run's state ──────────────────────────────────────────────────────────
struct DESettings {
    double   F = 0.5, CR = 1.0, p = 0.1;
    bool     archive = true;
    DEAdapt  adapt = DEAdapt::None;
    DEWeight weight = DEWeight::Equal;
    int      memory = 5;
};

namespace de_detail {

inline std::string key(const std::vector<double>& x)
{
    return std::string(reinterpret_cast<const char*>(x.data()), x.size() * sizeof(double));
}

// Each child's (CR, F) for one variable vector u, crossed binomially with the
// target x from the mutant v, then repaired.
template <typename RNG>
inline std::vector<double> binomial(const std::vector<double>& x, const std::vector<double>& v,
                                    double CR, const std::vector<std::pair<std::optional<double>,
                                                                           std::optional<double>>>& bounds,
                                    BoundRepair repair, RNG& rng)
{
    const std::size_t n = x.size();
    rnd::uniform_real_distribution<double> uni(0.0, 1.0);
    rnd::uniform_int_distribution<std::size_t> pick(0, n - 1);
    const std::size_t jr = pick(rng);
    std::vector<double> u(x);
    for (std::size_t j = 0; j < n; ++j)
        if (j == jr || uni(rng) < CR) u[j] = v[j];
    RepairTally tally;
    for (std::size_t j = 0; j < n; ++j) {
        const double lo = sbx_require_bound(bounds[j].first,  "lower", static_cast<int>(j));
        const double hi = sbx_require_bound(bounds[j].second, "upper", static_cast<int>(j));
        if (u[j] < lo || u[j] > hi) {
            tally.out();
            u[j] = repair_value(u[j], lo, hi, x[j], repair, rng);
        }
    }
    return u;
}

}   // namespace de_detail

class DEState {
public:
    DESettings set;

    // SHADE's draw (or the fixed F and CR); recorded.
    template <typename RNG>
    std::pair<double, double> draw(RNG& rng)
    {
        auto& rec = de_record();
        rec.used = true;
        double CR = set.CR, F = set.F;
        if (set.adapt == DEAdapt::SHADE) {
            ensure_memory();
            rnd::uniform_int_distribution<int> cell(0, static_cast<int>(mcr_.size()) - 1);
            const int r = cell(rng);
            rnd::normal_distribution<double> N(mcr_[static_cast<std::size_t>(r)], 0.1);
            CR = std::clamp(N(rng), 0.0, 1.0);
            rnd::cauchy_distribution<double> C(mf_[static_cast<std::size_t>(r)], 0.1);
            do { F = C(rng); } while (!(F > 0.0));
            F = std::min(F, 1.0);
        }
        ++rec.drawn;
        rec.cr_sum += CR;
        rec.f_sum  += F;
        ++rec.t_drawn;
        rec.t_cr_sum += CR;
        rec.t_f_sum  += F;
        return {CR, F};
    }

    // A child waiting for the selection's verdict.
    void pending(const std::vector<double>& u, double CR, double F)
    {
        pending_.push_back({u, CR, F});
    }

    // The two children of the last application as the host evaluates them,
    // after its mutation; only_one (0 or 1): the one it keeps.
    void finish(const std::vector<double>& c1, const std::vector<double>& c2, int only_one)
    {
        if (pending_.size() < 2) return;
        const std::size_t a = pending_.size() - 2, b = pending_.size() - 1;
        pending_[a].u = c1;
        pending_[b].u = c2;
        if (only_one == 0) pending_.erase(pending_.begin() + static_cast<std::ptrdiff_t>(b));
        if (only_one == 1) pending_.erase(pending_.begin() + static_cast<std::ptrdiff_t>(a));
    }

    // x̃_r2: a member of P ∪ A, P given by `from_population` (a draw from the
    // population) — the archive's share is |A| / (N + |A|).
    template <typename RNG, typename Draw>
    std::vector<double> population_or_archive(Draw&& from_population, int N, RNG& rng)
    {
        if (!set.archive || archive_.empty()) return from_population();
        rnd::uniform_int_distribution<int> any(0, N + static_cast<int>(archive_.size()) - 1);
        const int i = any(rng);
        if (i < N) return from_population();
        const auto& a = archive_[static_cast<std::size_t>(i - N)];
        return view ? view(a) : a;
    }

    const std::vector<std::vector<double>>& archive() const { return archive_; }

    // Where the variation runs in learned coordinates (variation.hpp): the
    // archive's members, kept in x, handed over in y. Empty otherwise.
    std::function<std::vector<double>(const std::vector<double>&)> view;

    // The population at the start of a step (X its variables, F its
    // objectives): the verdict on the children of the previous step, then the
    // displaced parents into the archive.
    template <typename RNG>
    void observe(const std::vector<std::vector<double>>& X,
                 const std::vector<std::vector<double>>& F, RNG& rng)
    {
        auto& rec = de_record();
        if (!pending_.empty()) {
            std::unordered_map<std::string, int> in_pop, in_par;
            for (const auto& x : X) ++in_pop[de_detail::key(x)];
            for (const auto& x : parents_) ++in_par[de_detail::key(x)];
            std::vector<double> scr, sf;
            std::vector<std::size_t> rows;
            std::unordered_map<std::string, std::size_t> row_of;
            for (std::size_t i = 0; i < X.size(); ++i) row_of.emplace(de_detail::key(X[i]), i);
            for (const auto& c : pending_) {
                const auto k = de_detail::key(c.u);
                auto it = in_pop.find(k);
                // a child equal to a parent that was already there is no success
                const int before = in_par.count(k) ? in_par[k] : 0;
                if (it == in_pop.end() || it->second <= before) continue;
                --it->second;
                scr.push_back(c.CR);
                sf.push_back(c.F);
                rows.push_back(row_of[k]);
                ++rec.ok;
                rec.cr_ok_sum += c.CR;
                rec.f_ok_sum  += c.F;
                ++rec.t_ok;
                rec.t_cr_ok_sum += c.CR;
                rec.t_f_ok_sum  += c.F;
            }
            if (set.adapt == DEAdapt::SHADE) {
                // one update per generation: as many children judged as the
                // population has members (every observe of a generational host,
                // every N-th of a steady-state one)
                std::vector<double> w(scr.size(), 1.0);
                if (set.weight == DEWeight::Contribution && !scr.empty())
                    w = contribution_weights(F, rows, rng);
                acc_cr_.insert(acc_cr_.end(), scr.begin(), scr.end());
                acc_f_.insert(acc_f_.end(), sf.begin(), sf.end());
                acc_w_.insert(acc_w_.end(), w.begin(), w.end());
                judged_ += pending_.size();
                if (judged_ >= X.size()) {
                    update(acc_cr_, acc_f_, acc_w_);
                    acc_cr_.clear(); acc_f_.clear(); acc_w_.clear();
                    judged_ = 0;
                }
            }
            pending_.clear();
        }
        if (set.archive && !parents_.empty()) {
            std::unordered_map<std::string, int> in_pop;
            for (const auto& x : X) ++in_pop[de_detail::key(x)];
            for (const auto& x : parents_) {
                auto it = in_pop.find(de_detail::key(x));
                if (it != in_pop.end() && it->second > 0) { --it->second; continue; }
                archive_.push_back(x);
            }
            const std::size_t N = X.size();
            while (archive_.size() > N) {
                rnd::uniform_int_distribution<std::size_t> any(0, archive_.size() - 1);
                const std::size_t i = any(rng);
                archive_[i] = std::move(archive_.back());
                archive_.pop_back();
            }
        }
        parents_ = X;
        if (set.adapt == DEAdapt::SHADE) {
            ensure_memory();
            rec.mcr = mcr_;
            rec.mf  = mf_;
        }
    }

    // SHADE's update of one cell from the successes and their weights
    // (public for the tests: the paper's means on given sets).
    void update(const std::vector<double>& scr, const std::vector<double>& sf,
                std::vector<double> w)
    {
        if (scr.empty()) return;
        ensure_memory();
        double sw = 0.0;
        for (double v : w) sw += v;
        if (!(sw > 0.0)) { std::fill(w.begin(), w.end(), 1.0); sw = static_cast<double>(w.size()); }
        double cr = 0.0, f2 = 0.0, f1 = 0.0;
        for (std::size_t i = 0; i < scr.size(); ++i) {
            const double wi = w[i] / sw;
            cr += wi * scr[i];
            f2 += wi * sf[i] * sf[i];
            f1 += wi * sf[i];
        }
        mcr_[static_cast<std::size_t>(k_)] = cr;
        if (f1 > 0.0) mf_[static_cast<std::size_t>(k_)] = f2 / f1;
        k_ = (k_ + 1) % static_cast<int>(mcr_.size());
    }

    const std::vector<double>& memory_cr() { ensure_memory(); return mcr_; }
    const std::vector<double>& memory_f()  { ensure_memory(); return mf_; }

private:
    struct Child { std::vector<double> u; double CR, F; };
    std::vector<Child>                 pending_;
    std::vector<std::vector<double>>   parents_, archive_;
    std::vector<double>                mcr_, mf_;
    std::vector<double>                acc_cr_, acc_f_, acc_w_;   // this generation's successes
    std::size_t                        judged_ = 0;
    int                                k_ = 0;

    void ensure_memory()
    {
        const std::size_t H = static_cast<std::size_t>(std::max(1, set.memory));
        if (mcr_.size() != H) { mcr_.assign(H, 0.5); mf_.assign(H, 0.5); k_ = 0; }
    }

    // The surviving children's hypervolume contributions to the population's
    // first front, the objectives normalised to the population's range,
    // reference 1.1 (the header).
    template <typename RNG>
    static std::vector<double> contribution_weights(const std::vector<std::vector<double>>& F,
                                                    const std::vector<std::size_t>& rows, RNG& rng)
    {
        const std::size_t n = F.size(), m = n ? F[0].size() : 0;
        std::vector<double> lo(m, 1e300), hi(m, -1e300);
        for (const auto& f : F)
            for (std::size_t j = 0; j < m; ++j) { lo[j] = std::min(lo[j], f[j]); hi[j] = std::max(hi[j], f[j]); }
        std::vector<std::vector<double>> G(n, std::vector<double>(m));
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < m; ++j)
                G[i][j] = hi[j] > lo[j] ? (F[i][j] - lo[j]) / (hi[j] - lo[j]) : 0.0;
        std::vector<char> first(n, 1);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t q = 0; q < n && first[i]; ++q) {
                if (q == i) continue;
                bool le = true, lt = false;
                for (std::size_t j = 0; j < m && le; ++j) { le = G[q][j] <= G[i][j]; lt = lt || G[q][j] < G[i][j]; }
                if (le && lt) first[i] = 0;
            }
        std::vector<std::vector<double>> P;
        std::vector<std::size_t> at(n, n);
        for (std::size_t i = 0; i < n; ++i) if (first[i]) { at[i] = P.size(); P.push_back(G[i]); }
        const std::vector<double> ref(m, 1.1);
        const auto c = m <= 3 ? hv::contributions_exact(P, ref)
                              : hv::contributions_mc(P, ref, std::max<std::size_t>(1, 10000 / std::max<std::size_t>(1, P.size())), rng);
        std::vector<double> w;
        for (std::size_t r : rows) w.push_back(at[r] < n ? c[at[r]] : 0.0);
        return w;
    }
};

// The two children of one DE application (the header): rand/1 when pbest is
// empty, current-to-pbest/1 otherwise. P holds the pair and two more parents
// for rand/1, the pair and x_r1 for each child of current-to-pbest/1 (the
// caller passes them); draw_archive gives x̃_r2.
template <typename RNG, typename Best, typename DrawR2>
inline void de_children(const std::vector<std::vector<double>>& P, bool pbest, Best&& best,
                        DrawR2&& draw_r2, std::vector<double>& c1, std::vector<double>& c2,
                        const std::vector<std::pair<std::optional<double>,
                                                    std::optional<double>>>& bounds,
                        BoundRepair repair, DEState& st, RNG& rng)
{
    const std::size_t n = P[0].size();
    auto child = [&](const std::vector<double>& x, const std::vector<double>& a,
                     const std::vector<double>& b, const std::vector<double>& c) {
        const auto [CR, F] = st.draw(rng);
        std::vector<double> v(n);
        if (!pbest) {                       // v = a + F (b − c), target x
            for (std::size_t j = 0; j < n; ++j) v[j] = a[j] + F * (b[j] - c[j]);
        } else {                            // v = x + F (pbest − x) + F (r1 − r2)
            const std::vector<double> xp = best();
            const std::vector<double> r2 = draw_r2();
            for (std::size_t j = 0; j < n; ++j)
                v[j] = x[j] + F * (xp[j] - x[j]) + F * (b[j] - r2[j]);
        }
        auto u = de_detail::binomial(x, v, CR, bounds, repair, rng);
        st.pending(u, CR, F);
        return u;
    };
    if (!pbest) {
        c1 = child(P[0], P[1], P[2], P[3]);
        c2 = child(P[1], P[0], P[3], P[2]);
    } else {
        c1 = child(P[0], P[0], P[2], P[0]);
        c2 = child(P[1], P[1], P[3], P[1]);
    }
}

}   // namespace mootation::ops

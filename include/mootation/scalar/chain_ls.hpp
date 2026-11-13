#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// Local searchers for local search chains (task 25, item 24.2.1; АП-2 of the
// task's preparation): Solis-Wets, Subgrouping Solis-Wets (SSW), MTS-LS1 and
// MTS-LS2, each with its state kept outside the call, so that a chain of
// applications continues where the last one stopped — the property MA-LS-
// Chains rests on ("the final configuration ... is used as the initial one
// for the next application"; n applications of I evaluations = one of n I).
//
// Sources:
//   D. Molina, M. Lozano, A.M. Sánchez, F. Herrera, "Memetic algorithms based
//     on local search chains for large scale continuous optimisation
//     problems: MA-SSW-Chains", Soft Comput. 15:2201-2220, 2011 (source
//     memetic-algorithms-local_molina2010; "M11"): Solis-Wets Fig. 2, MTS-LS1
//     Fig. 3, MTS-LS2 Fig. 4, SSW Fig. 5, the states §4.2-4.5.
//   D. Molina, M. Lozano, F. Herrera, "MA-SW-Chains", IEEE CEC 2010 (source
//     ma-sw-chains_molina2010; "M10"): maxSuccesses = 5, maxFailures = 3,
//     rho_0 = 0.2; the state stored with an individual is the success and
//     failure numbers, bias and rho (§III).
//   R. Tanabe, "Benchmarking the Hooke-Jeeves method, MTS-LS1, and BSrr on
//     the large-scale BBOB function set", arXiv 2022 (source large-scale-
//     bbob-hj-mtsls1-bsrr_tanabe2022_arxiv; "T22"): MTS-LS1 as Algorithm 2,
//     sigma_init = 0.4 of the box width "as in" Tseng & Chen 2008.
//   (Tseng & Chen 2008, MTS, is not in the corpus.)
//
// The searchers work in the coordinates they are given; every trial point is
// clipped to [lo, hi] before it is evaluated. With the normalized box [0, 1]^n
// (how the library's single solvers run) the defaults below are fractions of
// the box width. A call spends at most max_evals evaluations of f, one at a
// time, and returns how many; x and fx are the current point and its value,
// in and out (fx must be f(x)); f is only compared by "<" and "==".
//
// SOLIS-WETS (M11 Fig. 2): dif ~ N(0, rho) per coordinate; try x + bias +
// dif; on success bias <- 0.2 bias + 0.4 (dif + bias); else try x - bias -
// dif, on success bias <- bias - 0.4 (dif + bias); else a failure. More than
// max_success successes in a row: rho <- 2 rho; more than max_failed failures
// in a row: rho <- rho / 2 (the counts reset).
// SSW (M11 Fig. 5, §3.4, §4.5): the same, dif drawn only on a subset of
// consecutive coordinates (mod n), 0.2 n of them, at most max_vars = 50; the
// subset is drawn anew every subset_evals evaluations, by default maxEval/10
// of the call; the other coordinates move by bias alone.
// MTS-LS1 (M11 Fig. 3, §3.3; T22 Alg. 2): at the start of a pass, if the last
// pass improved nothing, SR <- SR / 2, and below sr_min SR <- sr_reset; then
// for each coordinate: x_i - SR; better: kept; equal: restored; worse:
// restored and x_i + 0.5 SR tried, kept if better.
// MTS-LS2 (M11 Fig. 4): a pass is n steps; a step draws r_i in {0, 1, 2, 3}
// and D_i in {-1, 1} for every coordinate and moves the coordinates with r_i =
// 0 together, by -SR D_i, then (worse) by +0.5 SR D_i from the restored point.
//
// READINGS:
//   SW-1. "More than": rho doubles after the 6th success in a row and halves
//     after the 4th failure (Fig. 2's "> 5", "> 3"; M10's words
//     maxSuccesses = 5, maxFailures = 3 are the thresholds compared with
//     ">").
//   SW-2. Every evaluation counts: Fig. 2 counts a step as one evaluation
//     though its second trial is another (Fig. 5 counts both).
//   SW-3. rho is the standard deviation of dif (§3.1: "a normal distribution
//     whose standard deviation is given by a parameter rho"; §4.2 calls it
//     "the variance").
//   SW-4. The state is rho, bias and the two counts (M10 §III), and — so that
//     a call may end between the two trials of a step — that step's dif. Fig.
//     2 sets the counts to 0 at every call; M10 keeps them with the
//     individual: kept.
//   SW-5. rho_0 = 0.2 (M10) in the coordinates given, a fifth of the width of
//     the normalized box. M10 took 0.2 on the CEC 2010 boxes of width 10 to
//     200; M11 takes half the distance to the nearest individual, which the
//     strategy that calls the searcher gives as the state's rho.
//   SSW-1. The subset is kept for subset_evals evaluations (§3.4's text);
//     Fig. 5 clears it at every step and fills it only when numEval is a
//     multiple of maxEvalSubSet.
//   SSW-2. A new subset is drawn before a step once subset_evals or more
//     evaluations were spent on the current one (numEval moves by 1 or 2 a
//     step, so Fig. 5's "numEval % maxEvalSubSet = 0" may be stepped over).
//   SSW-3. The subset size is floor(0.2 n), at least 1, at most max_vars;
//     dif is 0 off the subset (§3.4: "there is no difference vector").
//     subset_evals = 0 takes max_evals / 10 of the call (at least 1): a chain
//     is one long run only when subset_evals is given.
//   MTS-1. After a worse first trial x_i' = x_i - SR the coordinate is
//     RESTORED and the second trial is x_i'' = x_i + 0.5 SR (M11 §3.3's text:
//     "the original solution is restored and then the same variable is
//     increased by 0.5 SR"; T22 Alg. 2 line 8). M11's Fig. 3 (checked on the
//     PDF, p. 2207) adds 0.5 SR to the moved coordinate instead: x_i'' =
//     (x_i - SR) + 0.5 SR = x_i - 0.5 SR, a point on the ray just refused;
//     MTS-LS2 (Fig. 4) the same with SR D_i. The text's x_i + 0.5 SR is taken.
//   MTS-2. M11 §4.4's "minimum 0.4, maximum 1e-15" are swapped: SR below
//     sr_min = 1e-15 is reset to sr_reset = 0.4 (Fig. 3; T22: sigma_init =
//     0.4 of the width); sr_init = 0.4 too.
//   MTS-3. "Better", "the same", "worse" compare with the current value; the
//     same value restores the point and moves on (Fig. 3). An MTS-LS2 step
//     whose draw selects no coordinate makes no trial (Fig. 4 would evaluate
//     the unchanged point).
//   MTS-4. A pass's "Improve" is one flag for the whole pass (Fig. 3's
//     Improve[k], k the individual); the state is SR, the flag, the position
//     in the pass, and the selection and signs of an MTS-LS2 step between its
//     two trials.
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <random>
#include <stdexcept>
#include <vector>

#include "../fp_contract.hpp"
#include "../random.hpp"

namespace mootation::chain_ls {

using Objective = std::function<double(const std::vector<double>&)>;

inline void clip(std::vector<double>& y, const std::vector<double>& lo, const std::vector<double>& hi)
{
    for (std::size_t i = 0; i < y.size(); ++i) y[i] = std::min(hi[i], std::max(lo[i], y[i]));
}

inline void check_sizes(const std::vector<double>& x, const std::vector<double>& lo, const std::vector<double>& hi)
{
    if (x.empty() || lo.size() != x.size() || hi.size() != x.size())
        throw std::invalid_argument("chain_ls: x, lo and hi of one nonzero length");
}

// ── Solis-Wets and SSW ──────────────────────────────────────────────────────

struct SWOptions {
    double rho0 = 0.2;            // SW-5
    long long max_success = 5;    // SW-1
    long long max_failed = 3;
    bool subgroups = false;       // SSW
    double subset_share = 0.2;    // SSW-3
    long long max_vars = 50;
    long long subset_evals = 0;   // SSW-3: 0 -> max_evals / 10 of the call
};

struct SWState {
    double rho = -1.0;            // < 0: not started, rho0 is taken
    std::vector<double> bias;
    long long n_success = 0;
    long long n_failed = 0;
    bool pending = false;         // SW-4: the step's second trial is next
    std::vector<double> dif;
    long long sub_start = -1;     // SSW: the subset, -1 none drawn yet
    long long sub_count = 0;
    long long sub_evals = 0;      // evaluations spent on it

    // The state as a vector of doubles (the form a strategy keeps with an
    // individual): rho, n_success, n_failed, pending, sub_start, sub_count,
    // sub_evals, bias (n), dif (n)
    std::vector<double> pack() const
    {
        std::vector<double> v = {rho,
                                 static_cast<double>(n_success),
                                 static_cast<double>(n_failed),
                                 pending ? 1.0 : 0.0,
                                 static_cast<double>(sub_start),
                                 static_cast<double>(sub_count),
                                 static_cast<double>(sub_evals)};
        v.insert(v.end(), bias.begin(), bias.end());
        v.insert(v.end(), dif.begin(), dif.end());
        return v;
    }

    static SWState unpack(const std::vector<double>& v, std::size_t n)
    {
        SWState s;
        if (v.empty()) return s;
        if (v.size() != 7 + 2 * n) throw std::invalid_argument("chain_ls: a Solis-Wets state of another length");
        s.rho = v[0];
        s.n_success = static_cast<long long>(v[1]);
        s.n_failed = static_cast<long long>(v[2]);
        s.pending = v[3] != 0.0;
        s.sub_start = static_cast<long long>(v[4]);
        s.sub_count = static_cast<long long>(v[5]);
        s.sub_evals = static_cast<long long>(v[6]);
        s.bias.assign(v.begin() + 7, v.begin() + 7 + static_cast<std::ptrdiff_t>(n));
        s.dif.assign(v.begin() + 7 + static_cast<std::ptrdiff_t>(n), v.end());
        return s;
    }
};

// One call: at most max_evals evaluations; x, fx and st carry on
inline long long solis_wets(const Objective& f, std::vector<double>& x, double& fx, SWState& st,
                            long long max_evals, const std::vector<double>& lo, const std::vector<double>& hi,
                            std::mt19937& rng, const SWOptions& opt = {})
{
    check_sizes(x, lo, hi);
    const std::size_t n = x.size();
    if (st.rho < 0.0) {
        st.rho = opt.rho0;
        st.bias.assign(n, 0.0);
        st.dif.assign(n, 0.0);
    }
    if (st.bias.size() != n || st.dif.size() != n) throw std::invalid_argument("chain_ls: a state of another length");
    long long period = opt.subset_evals > 0 ? opt.subset_evals : std::max(1LL, max_evals / 10);   // SSW-3

    // SW-1: the step size after a success or a failure
    auto adapt = [&]() {
        if (st.n_success > opt.max_success) {
            st.rho *= 2.0;
            st.n_success = 0;
        } else if (st.n_failed > opt.max_failed) {
            st.rho /= 2.0;
            st.n_failed = 0;
        }
    };

    std::vector<double> y(n);
    long long used = 0;
    while (used < max_evals) {
        if (!st.pending) {
            if (opt.subgroups && (st.sub_start < 0 || st.sub_evals >= period)) {   // SSW-1, SSW-2
                const double share = std::floor(opt.subset_share * static_cast<double>(n));
                st.sub_count = std::min(opt.max_vars, std::max(1LL, static_cast<long long>(share)));
                st.sub_count = std::min(st.sub_count, static_cast<long long>(n));
                st.sub_start = static_cast<long long>(rnd::below(rng, n));
                st.sub_evals = 0;
            }
            std::fill(st.dif.begin(), st.dif.end(), 0.0);
            if (opt.subgroups) {
                for (long long k = 0; k < st.sub_count; ++k) {
                    const std::size_t i = static_cast<std::size_t>((st.sub_start + k) % static_cast<long long>(n));
                    st.dif[i] = st.rho * rnd::normal01(rng);
                }
            } else {
                for (std::size_t i = 0; i < n; ++i) st.dif[i] = st.rho * rnd::normal01(rng);
            }
            for (std::size_t i = 0; i < n; ++i) y[i] = x[i] + st.bias[i] + st.dif[i];
            clip(y, lo, hi);
            const double fy = f(y);
            ++used;
            ++st.sub_evals;
            if (fy < fx) {
                x = y;
                fx = fy;
                for (std::size_t i = 0; i < n; ++i) st.bias[i] = 0.2 * st.bias[i] + 0.4 * (st.dif[i] + st.bias[i]);
                ++st.n_success;
                st.n_failed = 0;
                adapt();
            } else {
                st.pending = true;
            }
            continue;
        }
        for (std::size_t i = 0; i < n; ++i) y[i] = x[i] - st.bias[i] - st.dif[i];
        clip(y, lo, hi);
        const double fy = f(y);
        ++used;
        ++st.sub_evals;
        st.pending = false;
        if (fy < fx) {
            x = y;
            fx = fy;
            for (std::size_t i = 0; i < n; ++i) st.bias[i] = st.bias[i] - 0.4 * (st.dif[i] + st.bias[i]);
            ++st.n_success;
            st.n_failed = 0;
        } else {
            ++st.n_failed;
            st.n_success = 0;
        }
        adapt();
    }
    return used;
}

// ── MTS-LS1 and MTS-LS2 ─────────────────────────────────────────────────────

struct MTSOptions {
    double sr_init = 0.4;         // MTS-2
    double sr_min = 1e-15;
    double sr_reset = 0.4;
};

struct MTSState {
    double sr = -1.0;             // < 0: not started, sr_init is taken
    bool improved = true;         // MTS-4: the last (or this) pass improved
    bool in_pass = false;         // a pass is under way
    long long pos = 0;            // its next coordinate (LS1) or step (LS2)
    bool second = false;          // the second trial of the position is next
    std::vector<double> dir;      // LS2: the step's -1, 0, +1 per coordinate

    // The state as a vector of doubles: sr, improved, in_pass, pos, second,
    // dir (n)
    std::vector<double> pack() const
    {
        std::vector<double> v = {sr, improved ? 1.0 : 0.0, in_pass ? 1.0 : 0.0, static_cast<double>(pos),
                                 second ? 1.0 : 0.0};
        v.insert(v.end(), dir.begin(), dir.end());
        return v;
    }

    static MTSState unpack(const std::vector<double>& v, std::size_t n)
    {
        MTSState s;
        if (v.empty()) return s;
        if (v.size() != 5 + n) throw std::invalid_argument("chain_ls: an MTS state of another length");
        s.sr = v[0];
        s.improved = v[1] != 0.0;
        s.in_pass = v[2] != 0.0;
        s.pos = static_cast<long long>(v[3]);
        s.second = v[4] != 0.0;
        s.dir.assign(v.begin() + 5, v.end());
        return s;
    }
};

namespace detail {

inline void mts_start_pass(MTSState& st, const MTSOptions& opt)
{
    if (!st.improved) {
        st.sr /= 2.0;
        if (st.sr < opt.sr_min) st.sr = opt.sr_reset;
    }
    st.improved = false;
    st.in_pass = true;
    st.pos = 0;
    st.second = false;
}

// One trial along dir (coordinates with dir != 0) by step * SR from x; the
// outcome decides what is next (MTS-1, MTS-3). Returns true when the position
// is done.
inline bool mts_trial(const Objective& f, std::vector<double>& x, double& fx, MTSState& st, double step,
                      const std::vector<double>& dir, const std::vector<double>& lo, const std::vector<double>& hi,
                      std::vector<double>& y)
{
    y = x;
    for (std::size_t i = 0; i < x.size(); ++i)
        if (dir[i] != 0.0) y[i] = x[i] + step * st.sr * dir[i];
    clip(y, lo, hi);
    const double fy = f(y);
    if (fy < fx) {
        x = y;
        fx = fy;
        st.improved = true;
        st.second = false;
        return true;
    }
    if (!st.second && !(fy == fx)) {   // worse (or NaN): the second trial is next
        st.second = true;
        return false;
    }
    st.second = false;                  // the same value, or the second trial failed: restored
    return true;
}

}   // namespace detail

// MTS-LS1: one call, at most max_evals evaluations; passes follow each other
inline long long mts_ls1(const Objective& f, std::vector<double>& x, double& fx, MTSState& st, long long max_evals,
                         const std::vector<double>& lo, const std::vector<double>& hi, const MTSOptions& opt = {})
{
    check_sizes(x, lo, hi);
    const std::size_t n = x.size();
    if (st.sr < 0.0) st.sr = opt.sr_init;
    if (st.dir.size() != n) st.dir.assign(n, 0.0);
    std::vector<double> y(n);
    long long used = 0;
    while (used < max_evals) {
        if (!st.in_pass) detail::mts_start_pass(st, opt);
        std::fill(st.dir.begin(), st.dir.end(), 0.0);
        st.dir[static_cast<std::size_t>(st.pos)] = 1.0;   // the unit vector of the coordinate
        const bool done = detail::mts_trial(f, x, fx, st, st.second ? 0.5 : -1.0, st.dir, lo, hi, y);
        ++used;
        if (done && ++st.pos == static_cast<long long>(n)) st.in_pass = false;
    }
    return used;
}

// MTS-LS2: one call, at most max_evals evaluations; a pass is n steps
inline long long mts_ls2(const Objective& f, std::vector<double>& x, double& fx, MTSState& st, long long max_evals,
                         const std::vector<double>& lo, const std::vector<double>& hi, std::mt19937& rng,
                         const MTSOptions& opt = {})
{
    check_sizes(x, lo, hi);
    const std::size_t n = x.size();
    if (st.sr < 0.0) st.sr = opt.sr_init;
    if (st.dir.size() != n) st.dir.assign(n, 0.0);
    std::vector<double> y(n);
    long long used = 0;
    while (used < max_evals) {
        if (!st.in_pass) detail::mts_start_pass(st, opt);
        if (!st.second) {
            bool any = false;
            for (std::size_t i = 0; i < n; ++i) {
                const bool pick = rnd::below(rng, 4) == 0;            // r_i = 0
                const double sign = rnd::below(rng, 2) == 0 ? -1.0 : 1.0;   // D_i
                st.dir[i] = pick ? sign : 0.0;
                any = any || pick;
            }
            if (!any) {   // MTS-3: no trial; the chance of this is (3/4)^n
                if (++st.pos == static_cast<long long>(n)) st.in_pass = false;
                continue;
            }
        }
        const bool done = detail::mts_trial(f, x, fx, st, st.second ? 0.5 : -1.0, st.dir, lo, hi, y);
        ++used;
        if (done && ++st.pos == static_cast<long long>(n)) st.in_pass = false;
    }
    return used;
}

}   // namespace mootation::chain_ls

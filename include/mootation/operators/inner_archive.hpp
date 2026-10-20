#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// An archive the algorithm sees (task 7 of 2026-10-10, supplement 2, §2): the
// knobs inner_archive, inner_archive_cap and archive_reuse_p of sms_emoa,
// nsga2, nsga3 and ibea_eplus. The run layer's archive (run/archive.py) is the
// answer's and stays apart: the algorithm never sees it.
//
//   inner_archive = "nd"   every non-dominated point the algorithm evaluated,
//                  updated after every evaluation; a point whose objective
//                  vector equals a member's exactly is not added. "off" (the
//                  default): none.
//   inner_archive_cap      0: no limit (a binary problem's front is finite);
//                  N > 0: beyond N members the archive is cut to N by DSS
//                  (dss.hpp, as the answer is) after every addition, not at
//                  the end — Cui et al. (2025, from the abstract): that way is
//                  usually better. Unset: 10 μ for a real-valued genome, none
//                  for a purely binary one. A cut orders all N + 1 members by
//                  dss::order, which gives dss::order_plain's order bit for
//                  bit but asks a k-d tree which points a step can change
//                  instead of computing every pair (dss.hpp): the members and
//                  their order after every addition are those of the plain
//                  cut (tests/test_survival.cpp), only cheaper.
//   archive_reuse_p        each parent, with this probability, is a member of
//                  the archive drawn uniformly, otherwise the host's own
//                  choice; 0 by default. p = 1/2 is Ren, Liang, Li & Qian's
//                  (AAAI 2026, arXiv 2508.16993, from the abstract), whose
//                  archive without parents drawn from it can fail. The
//                  multi-parent crossovers' further parents and the binary
//                  part of the genome go the same way. p > 0 without the
//                  archive is an error.
// With constraints (constraint_mode not NONE) only feasible points enter, as
// in the answer's archive (archive.py); without, every point. A failed
// evaluation (data_vault.hpp, FAILED EVALUATIONS) never enters. Under
// constraint handling a member keeps its point whole and is judged again
// whenever the vault's present moves (data_vault.hpp, WRITING BACK: the
// equalities' tolerance, the epsilon level): one no longer feasible leaves.
//
// Why (from the papers' abstracts): without an archive the stochastic
// population update loses good solutions and needs a large population (Ren,
// Liang, Li & Qian, IJCAI 2025); an archive lets μ be 2-4 and speeds NSGA-II
// and SMS-EMOA up by Θ(n) on OneMinMax and LOTZ (Bian, Ren, Li & Qian, IJCAI
// 2024, doi:10.24963/ijcai.2024/763).
//
// THE RECORD (inner_archive_record(), thread-local like the repair counts):
// the parents drawn and those from the archive, for operator_stats
// (archive_parent_share) and the archive's size.
// ============================================================================

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../data_vault.hpp"
#include "../dss.hpp"
#include "../random.hpp"

namespace mootation::ops {

struct InnerArchiveMember {
    std::vector<double> f, x;
    std::vector<int> b;
    VaultPoint pt{};        // under constraint handling: the point whole, judged again
};

struct InnerArchiveRecord {
    long long drawn = 0, from_archive = 0;         // since the last read
    long long t_drawn = 0, t_from_archive = 0;     // over the run
    long long size = -1;                           // -1: no archive
    // the archive itself while its run lasts (the binding's
    // inner_archive_state(), for cover_gen); null otherwise
    const std::vector<InnerArchiveMember>* live = nullptr;
};
inline InnerArchiveRecord& inner_archive_record() { thread_local InnerArchiveRecord r; return r; }

// {name: value} for operator_stats (totals: over the run), the read counters reset.
inline std::vector<std::pair<std::string, double>> inner_archive_stats(bool totals)
{
    std::vector<std::pair<std::string, double>> out;
    auto& r = inner_archive_record();
    if (r.size < 0) return out;
    out.emplace_back("inner_archive_size", static_cast<double>(r.size));
    const long long n = totals ? r.t_drawn : r.drawn, k = totals ? r.t_from_archive : r.from_archive;
    if (n > 0) out.emplace_back("archive_parent_share", static_cast<double>(k) / static_cast<double>(n));
    if (!totals) r.drawn = r.from_archive = 0;
    return out;
}

struct InnerArchive {
    using Member = InnerArchiveMember;

    bool on = false;
    long long cap = -1;              // -1: 10 μ (real genome) or none (binary)
    double reuse_p = 0.0;

    bool set_param(const std::string& k, double v)
    {
        if (k == "inner_archive_cap") {
            if (!(v >= 0.0) || v != std::floor(v))
                throw std::invalid_argument("inner_archive_cap must be an integer >= 0");
            cap = static_cast<long long>(v);
            return true;
        }
        if (k == "archive_reuse_p") {
            if (!(v >= 0.0 && v <= 1.0))
                throw std::invalid_argument("archive_reuse_p must be in [0, 1]");
            reuse_p = v;
            return true;
        }
        return false;
    }
    bool set_word(const std::string& k, const std::string& w)
    {
        if (k != "inner_archive") return false;
        if (w != "off" && w != "nd") throw std::invalid_argument("inner_archive = '" + w + "': off or nd");
        on = w == "nd";
        return true;
    }

    // At setup: the checks, and the cap for this run (μ, the genome).
    void start(int mu, bool binary_only)
    {
        if (reuse_p > 0.0 && !on)
            throw std::invalid_argument("archive_reuse_p > 0 needs inner_archive = \"nd\"");
        limit_ = cap >= 0 ? cap : binary_only ? 0 : 10LL * mu;
        members_.clear();
        if (on) {
            inner_archive_record().size = 0;
            inner_archive_record().live = &members_;
        }
    }

    // One evaluated point (its objectives, variables, binary variables; under
    // constraint handling the point whole, to judge it again later).
    void add(const std::vector<double>& f, const std::vector<double>& x,
             const std::vector<int>& b = {}, VaultPoint pt = {})
    {
        if (!on) return;
        auto dominates = [](const std::vector<double>& a, const std::vector<double>& c) {
            bool lt = false;
            for (std::size_t j = 0; j < a.size(); ++j) {
                if (a[j] > c[j]) return false;
                lt = lt || a[j] < c[j];
            }
            return lt;
        };
        for (const auto& m : members_)
            if (m.f == f || dominates(m.f, f)) return;
        members_.erase(std::remove_if(members_.begin(), members_.end(),
                                      [&](const Member& m) { return dominates(f, m.f); }),
                       members_.end());
        members_.push_back({f, x, b, std::move(pt)});
        if (limit_ > 0 && static_cast<long long>(members_.size()) > limit_) {
            std::vector<std::vector<double>> F;
            F.reserve(members_.size());
            for (const auto& m : members_) F.push_back(m.f);
            const auto keep = dss::order(F, static_cast<std::size_t>(limit_));
            std::vector<Member> kept;
            kept.reserve(keep.size());
            for (std::size_t i : keep) kept.push_back(std::move(members_[i]));
            members_ = std::move(kept);
        }
        inner_archive_record().size = static_cast<long long>(members_.size());
    }

    // Is this parent the archive's? Draws nothing when archive_reuse_p is 0;
    // counts every parent drawn when the archive is on.
    template <typename RNG>
    bool reuse(RNG& rng)
    {
        if (!on) return false;
        auto& r = inner_archive_record();
        ++r.drawn; ++r.t_drawn;
        if (!(reuse_p > 0.0) || members_.empty()) return false;
        rnd::uniform_real_distribution<double> u(0.0, 1.0);
        if (u(rng) >= reuse_p) return false;
        ++r.from_archive; ++r.t_from_archive;
        return true;
    }

    // A member drawn uniformly.
    template <typename RNG>
    const Member& pick(RNG& rng) const
    {
        rnd::uniform_int_distribution<std::size_t> any(0, members_.size() - 1);
        return members_[any(rng)];
    }

    const std::vector<Member>& members() const { return members_; }

    // Under constraint handling, before a step: when the vault's present moved,
    // every member judged again at it; one no longer feasible leaves.
    template <typename Vault>
    void refresh(const Vault& vault)
    {
        if (!on || vault.present() == present_seen_) return;
        present_seen_ = vault.present();
        members_.erase(std::remove_if(members_.begin(), members_.end(),
                                      [&](Member& m) { return vault.refresh_point(m.pt) > 0.0; }),
                       members_.end());
        inner_archive_record().size = static_cast<long long>(members_.size());
    }

private:
    long long limit_ = 0;
    std::vector<Member> members_;
    unsigned long long present_seen_ = 0;   // the vault's present() the members are judged at
};

}   // namespace mootation::ops

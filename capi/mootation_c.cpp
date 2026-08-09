// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// The C ABI implementation. One translation unit, compiled into a shared
// library; everything else in MOOtation stays header-only.
//
// The whole job here is to keep C++ inside: no exception, no std::string, no
// container ever crosses the boundary. Every entry point is wrapped in the
// same try/catch, which stores the message on the session and returns a
// negative code, so a caller in Python or C# sees an error value and a
// readable string rather than a terminate().
// ============================================================================

#include <cstring>
#include <exception>
#include <memory>
#include <new>
#include <string>
#include <vector>

#include "mootation/capi.h"
#include "mootation/embed.hpp"
#include "mootation/version.hpp"

namespace {

// The last moo_open failure, which has no session to hang off.
thread_local std::string g_open_error;

} // namespace

// Opaque to the caller; a plain struct here.
struct moo_session {
    mootation::Settings              settings;
    std::unique_ptr<mootation::Session> sess;
    mootation::Result                result;
    bool                             have_result = false;
    std::string                      error;
    // moo_ask_count and moo_ask must agree, so the batch is snapshotted once
    // and both read the snapshot. Re-querying between the two calls would let
    // the count change under a caller that allocated for the first answer.
    std::vector<std::vector<double>> batch;
    bool                             batch_valid = false;
};

namespace {

// Every entry point funnels through here.
template <typename Fn>
int guarded(moo_session* s, Fn&& fn) {
    if (!s) {
        g_open_error = "mootation: null session handle";
        return -1;
    }
    try {
        s->error.clear();
        return fn();
    } catch (const std::exception& e) {
        s->error = e.what();
        return -1;
    } catch (...) {
        s->error = "mootation: unknown error";
        return -1;
    }
}

void refresh_batch(moo_session* s) {
    if (s->batch_valid) return;
    s->batch       = s->sess->ask();
    s->batch_valid = true;
}

} // namespace

extern "C" {

int moo_version(void) { return MOOTATION_VERSION; }

const char* moo_version_string(void) {
    // mootation::version() already returns a string literal with static
    // storage duration, so there is nothing to keep alive here.
    return mootation::version();
}

int moo_algorithm_count(void) {
    try {
        return static_cast<int>(mootation::algorithm_names().size());
    } catch (...) {
        return -1;
    }
}

const char* moo_algorithm_name(int i) {
    try {
        // Built once and kept alive for the process: the caller gets a stable
        // pointer it never has to free, which is what an FFI wants.
        static const std::vector<std::string> names = mootation::algorithm_names();
        if (i < 0 || i >= static_cast<int>(names.size())) return nullptr;
        return names[static_cast<std::size_t>(i)].c_str();
    } catch (...) {
        return nullptr;
    }
}

moo_session* moo_open(const char* text) {
    g_open_error.clear();
    if (!text) {
        g_open_error = "mootation: settings text is null";
        return nullptr;
    }
    try {
        auto s = std::make_unique<moo_session>();
        s->settings = mootation::Settings::from_string(text, "<moo_open>");
        s->sess     = std::make_unique<mootation::Session>(s->settings);
        return s.release();
    } catch (const std::exception& e) {
        g_open_error = e.what();
        return nullptr;
    } catch (...) {
        g_open_error = "mootation: unknown error while opening";
        return nullptr;
    }
}

moo_session* moo_open_file(const char* path) {
    g_open_error.clear();
    if (!path) {
        g_open_error = "mootation: path is null";
        return nullptr;
    }
    try {
        mootation::Settings st = mootation::Settings::from_file(path);
        auto s = std::make_unique<moo_session>();
        s->settings = std::move(st);
        s->sess     = std::make_unique<mootation::Session>(s->settings);
        return s.release();
    } catch (const std::exception& e) {
        g_open_error = e.what();
        return nullptr;
    } catch (...) {
        g_open_error = "mootation: unknown error while opening";
        return nullptr;
    }
}

void moo_close(moo_session* s) {
    // ~Session aborts the worker and joins it, so closing mid-run is safe.
    delete s;
}

int moo_n_vars(const moo_session* s) { return s ? s->settings.n_vars() : -1; }
int moo_n_objs(const moo_session* s) { return s ? s->settings.n_objs   : -1; }
int moo_n_cons(const moo_session* s) { return s ? s->settings.n_cons   : -1; }

int moo_generation(const moo_session* s) {
    if (!s || !s->sess) return -1;
    try {
        return const_cast<mootation::Session&>(*s->sess).generation();
    } catch (...) {
        return -1;
    }
}

int moo_ask_count(moo_session* s) {
    return guarded(s, [&] {
        if (s->have_result) return 0;
        refresh_batch(s);
        return static_cast<int>(s->batch.size());
    });
}

int moo_ask(moo_session* s, double* x, int x_len) {
    return guarded(s, [&] {
        if (s->have_result) return 0;
        refresh_batch(s);
        const int n  = static_cast<int>(s->batch.size());
        const int nv = s->settings.n_vars();
        if (n == 0) return 0;
        if (!x) throw std::invalid_argument("moo_ask: x is null");
        if (x_len < n * nv)
            throw std::invalid_argument(
                "moo_ask: x_len = " + std::to_string(x_len) + " but the batch needs " +
                std::to_string(n * nv) + " doubles (" + std::to_string(n) +
                " candidates x " + std::to_string(nv) + " variables)");
        for (int i = 0; i < n; ++i)
            std::memcpy(x + static_cast<std::size_t>(i) * static_cast<std::size_t>(nv),
                        s->batch[static_cast<std::size_t>(i)].data(),
                        static_cast<std::size_t>(nv) * sizeof(double));
        return n;
    });
}

int moo_tell(moo_session* s, const double* f, int f_len,
             const double* g, int g_len) {
    return guarded(s, [&] {
        if (s->have_result)
            throw std::logic_error("moo_tell: the run has already finished");
        refresh_batch(s);
        const int n  = static_cast<int>(s->batch.size());
        const int no = s->settings.n_objs;
        const int nc = s->settings.n_cons;
        if (n == 0)
            throw std::logic_error("moo_tell: there is no batch waiting");
        if (!f) throw std::invalid_argument("moo_tell: f is null");
        if (f_len < n * no)
            throw std::invalid_argument(
                "moo_tell: f_len = " + std::to_string(f_len) + " but the batch needs " +
                std::to_string(n * no) + " doubles (" + std::to_string(n) +
                " candidates x " + std::to_string(no) + " objectives)");
        if (nc > 0) {
            if (!g) throw std::invalid_argument(
                "moo_tell: n_cons = " + std::to_string(nc) + " but g is null");
            if (g_len < n * nc)
                throw std::invalid_argument(
                    "moo_tell: g_len = " + std::to_string(g_len) + " but the batch needs " +
                    std::to_string(n * nc) + " doubles");
        }

        std::vector<std::vector<double>> F(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i)
            F[static_cast<std::size_t>(i)].assign(
                f + static_cast<std::size_t>(i) * static_cast<std::size_t>(no),
                f + static_cast<std::size_t>(i + 1) * static_cast<std::size_t>(no));

        std::vector<std::vector<double>> G;
        if (nc > 0) {
            G.resize(static_cast<std::size_t>(n));
            for (int i = 0; i < n; ++i)
                G[static_cast<std::size_t>(i)].assign(
                    g + static_cast<std::size_t>(i) * static_cast<std::size_t>(nc),
                    g + static_cast<std::size_t>(i + 1) * static_cast<std::size_t>(nc));
        }

        s->batch       = s->sess->tell(F, G);
        s->batch_valid = true;
        if (s->batch.empty()) {
            s->result      = s->sess->result();
            s->have_result = true;
        }
        return static_cast<int>(s->batch.size());
    });
}

int moo_result_count(moo_session* s) {
    return guarded(s, [&] {
        if (!s->have_result) {
            s->result      = s->sess->result();   // throws if a batch is pending
            s->have_result = true;
        }
        return static_cast<int>(s->result.size());
    });
}

int moo_result(moo_session* s, double* x, int x_len,
               double* f, int f_len, double* cv, int cv_len) {
    return guarded(s, [&] {
        if (!s->have_result) {
            s->result      = s->sess->result();
            s->have_result = true;
        }
        const int n  = static_cast<int>(s->result.size());
        const int nv = s->settings.n_vars();
        const int no = s->settings.n_objs;
        if (x && x_len < n * nv)
            throw std::invalid_argument(
                "moo_result: x_len = " + std::to_string(x_len) + " needs " +
                std::to_string(n * nv));
        if (f && f_len < n * no)
            throw std::invalid_argument(
                "moo_result: f_len = " + std::to_string(f_len) + " needs " +
                std::to_string(n * no));
        if (cv && cv_len < n)
            throw std::invalid_argument(
                "moo_result: cv_len = " + std::to_string(cv_len) + " needs " +
                std::to_string(n));
        for (int i = 0; i < n; ++i) {
            const std::size_t ui = static_cast<std::size_t>(i);
            if (x)
                std::memcpy(x + ui * static_cast<std::size_t>(nv),
                            s->result.variables[ui].data(),
                            static_cast<std::size_t>(nv) * sizeof(double));
            if (f)
                std::memcpy(f + ui * static_cast<std::size_t>(no),
                            s->result.objectives[ui].data(),
                            static_cast<std::size_t>(no) * sizeof(double));
            if (cv) cv[ui] = s->result.cv[ui];
        }
        return n;
    });
}

int moo_ignored_count(moo_session* s) {
    return guarded(s, [&] {
        if (!s->have_result) {
            s->result      = s->sess->result();
            s->have_result = true;
        }
        return static_cast<int>(s->result.ignored.size());
    });
}

const char* moo_ignored_name(moo_session* s, int i) {
    if (!s || !s->have_result) return nullptr;
    if (i < 0 || i >= static_cast<int>(s->result.ignored.size())) return nullptr;
    return s->result.ignored[static_cast<std::size_t>(i)].c_str();
}

const char* moo_last_error(const moo_session* s) {
    if (!s) return g_open_error.empty() ? nullptr : g_open_error.c_str();
    return s->error.empty() ? nullptr : s->error.c_str();
}

} // extern "C"

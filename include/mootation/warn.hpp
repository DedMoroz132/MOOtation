#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// A one-line diagnostic channel.
//
// Some algorithms silently substitute a parameter the caller asked for: a
// requested subregion count that is not an attainable Das-Dennis lattice size
// comes back rounded up, a schedule that needs a real generation budget runs
// against a default one. Every such substitution is declared in the file's
// header block, but a header only helps a reader who is already suspicious.
// This makes it audible at runtime.
//
// Deliberately minimal: no severity levels, no formatting, no state beyond a
// single function pointer. It is a diagnostic aid, not a logging framework, and
// a library whose selling point is "nothing beyond the standard library"
// should not grow one.
//
// Default: silent. Nothing is printed unless the caller installs a handler.
//
//   mootation::set_warn_handler([](const std::string& m){
//       std::cerr << "mootation: " << m << '\n';
//   });
//
// Not thread-safe to INSTALL — set the handler once, before the run. Emitting
// is as thread-safe as the handler the caller provides.
// ============================================================================

#include <functional>
#include <string>

namespace mootation {

using WarnHandler = std::function<void(const std::string&)>;

namespace detail {
inline WarnHandler& warn_handler() {
    static WarnHandler h;   // empty = silent
    return h;
}
} // namespace detail

// Install (or clear, by passing {}) the handler. Returns the previous one.
inline WarnHandler set_warn_handler(WarnHandler h) {
    WarnHandler prev = detail::warn_handler();
    detail::warn_handler() = std::move(h);
    return prev;
}

// Emit a diagnostic. No-op when no handler is installed — and the message is
// not built either, when callers pass a lambda (see warn_lazy below).
inline void warn(const std::string& message) {
    const auto& h = detail::warn_handler();
    if (h) h(message);
}

// For messages that cost something to build. The functor is only invoked when
// a handler is installed.
template <typename Fn>
inline void warn_lazy(Fn&& make_message) {
    const auto& h = detail::warn_handler();
    if (h) h(make_message());
}

} // namespace mootation

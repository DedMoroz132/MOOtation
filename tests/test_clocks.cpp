// SPDX-License-Identifier: Apache-2.0
// The clocks of max_seconds (task 31, layer 1; scalar/clocks.hpp CLOCK-3):
// on Windows, <windows.h> first and clocks.hpp after it — its own
// declarations of GetCurrentProcess, GetProcessTimes,
// QueryUnbiasedInterruptTime, GetModuleHandleW and GetProcAddress must be
// windows.h's, or this does not compile (test_stop_rule includes the two the
// other way round) — windows.h's own calls agree with clocks.hpp's, and
// QueryUnbiasedInterruptTimePrecise is found at run time (CLOCK-4); nothing
// but the default libraries is linked. Elsewhere the clocks alone. Each
// clock goes on with work.

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

#include <mootation/scalar/clocks.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>

namespace {

int g_failed = 0;
volatile double g_sink = 0.0;                  // keeps the work from being optimized away

void check(bool ok, const std::string& what)
{
    std::printf("%s  %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++g_failed;
}

namespace sc = mootation::scalar;

// about `ms` milliseconds of work on the steady clock
double work(double ms)
{
    const double t0 = sc::clock_seconds();
    double s = 0.0;
    while ((sc::clock_seconds() - t0) * 1e3 < ms)
        for (int k = 0; k < 200; ++k) s += std::sqrt(static_cast<double>(k) + s * 1e-9);
    return s;
}

}   // namespace

int main()
{
    const double w0 = sc::clock_seconds(), u0 = sc::unbiased_seconds(), c0 = sc::cpu_seconds();
    g_sink = work(60.0);
    const double w = sc::clock_seconds() - w0, u = sc::unbiased_seconds() - u0, c = sc::cpu_seconds() - c0;
    std::printf("      60 ms of work: wall %.4f, unbiased %.4f, cpu %.4f\n", w, u, c);
    check(w >= 0.06, "the wall clock goes on");
    check(u >= 0.05 && u <= w + 0.02, "the unbiased clock goes on, as the wall clock while awake");
    check(c >= 0.03 && c <= w + 0.05, "the processor time goes on with the work (Windows: ticks of ~15.6 ms)");
    check(sc::clock_seconds(sc::Clock::wall) >= w0 && sc::clock_seconds(sc::Clock::cpu) >= c0 &&
              sc::clock_seconds(sc::Clock::unbiased) >= u0,
          "clock_seconds(c) reads each clock");
#if defined(_WIN32)
    // windows.h's own types and calls beside clocks.hpp's (CLOCK-3)
    FILETIME created, ended, kernel, user;
    const BOOL ok = GetProcessTimes(GetCurrentProcess(), &created, &ended, &kernel, &user);
    ULARGE_INTEGER k, us;
    k.LowPart = kernel.dwLowDateTime;
    k.HighPart = kernel.dwHighDateTime;
    us.LowPart = user.dwLowDateTime;
    us.HighPart = user.dwHighDateTime;
    const double theirs = static_cast<double>(k.QuadPart + us.QuadPart) * 1e-7, ours = sc::cpu_seconds();
    check(ok != 0 && ours >= theirs && ours - theirs < 0.05, "GetProcessTimes through windows.h = cpu_seconds()");
    // CLOCK-4: the precise clock found at run time (Windows 10 and later, the CI's Server 2022 among them),
    // and the same clock as windows.h's QueryUnbiasedInterruptTime (kernel32.lib; the timer's tick)
    check(sc::precise_unbiased_clock() != nullptr, "QueryUnbiasedInterruptTimePrecise found in KernelBase.dll");
    ULONGLONG t = 0;
    const BOOL got = QueryUnbiasedInterruptTime(&t);
    const double theirs_u = static_cast<double>(t) * 1e-7, ours_u = sc::unbiased_seconds();
    check(got != 0 && ours_u >= theirs_u - 0.02 && ours_u - theirs_u < 0.05,
          "QueryUnbiasedInterruptTime through windows.h = unbiased_seconds() within a tick");
#endif
    std::printf("%s\n", g_failed ? "FAILED" : "all passed");
    return g_failed ? 1 : 0;
}

#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// THE CLOCKS (task 31, layer 1; item 30.12 of task 30): the clock a run's
// deadline (max_seconds; scalar.hpp TIME-1) is on, the knob time_clock of
// scalar::run (Problem::time_clock):
//   wall      a steady clock (std::chrono::steady_clock), as before: it goes
//             on while the machine sleeps or hibernates;
//   unbiased  the time the machine was awake: Windows'
//             QueryUnbiasedInterruptTimePrecise, Linux's CLOCK_MONOTONIC
//             (it stops in a suspend), macOS's CLOCK_UPTIME_RAW;
//   cpu       the processor time of the whole process, every thread's:
//             Windows' GetProcessTimes (kernel and user), elsewhere
//             CLOCK_PROCESS_CPUTIME_ID.
//
// READINGS:
//   CLOCK-1. Off by default (wall: the steady clock of before, the same
//     call). The deadline is a reading of the run's clock plus max_seconds,
//     and every check reads the same clock (evaluator.hpp past_deadline).
//   CLOCK-2. Off by default (time_clock = cpu only). cpu counts every
//     thread of the process: a strategy on threads > 1 spends it faster
//     than the wall clock, and work of the process outside the run (another
//     job of a campaign in another thread) counts too.
//   CLOCK-3. Off by default (time_clock = unbiased or cpu only). Windows'
//     functions are declared here, not by <windows.h> (its macros min, max,
//     near, far): with windows.h's own types, so that a program including
//     both compiles. QueryUnbiasedInterruptTimePrecise is in Windows 10 and
//     later, in mincore (named for the linker by a #pragma under MSVC; its
//     kernel32.lib does not have it).
// ============================================================================

#include <chrono>
#include <stdexcept>
#include <string>

#if defined(_WIN32)
struct _FILETIME;
extern "C" __declspec(dllimport) void* __stdcall GetCurrentProcess(void);
extern "C" __declspec(dllimport) int __stdcall GetProcessTimes(void* process, _FILETIME* creation, _FILETIME* exit,
                                                                 _FILETIME* kernel, _FILETIME* user);
extern "C" __declspec(dllimport) void __stdcall QueryUnbiasedInterruptTimePrecise(unsigned long long* time);
#if defined(_MSC_VER)
#pragma comment(lib, "mincore")
#endif
#else
#include <time.h>
#endif

namespace mootation::scalar {

enum class Clock { wall, unbiased, cpu };

// time_clock's word: wall, unbiased, cpu
inline Clock clock_of(const std::string& word)
{
    if (word == "wall") return Clock::wall;
    if (word == "unbiased") return Clock::unbiased;
    if (word == "cpu") return Clock::cpu;
    throw std::invalid_argument("time_clock = '" + word + "': wall, unbiased, cpu");
}

inline const char* clock_name(Clock c)
{
    return c == Clock::wall ? "wall" : c == Clock::unbiased ? "unbiased" : "cpu";
}

// a steady clock's seconds (THE TIME)
inline double clock_seconds()
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// the process's processor time, every thread's (CLOCK-2)
inline double cpu_seconds()
{
#if defined(_WIN32)
    unsigned long long created = 0, ended = 0, kernel = 0, user = 0;   // FILETIMEs: 100 ns
    GetProcessTimes(GetCurrentProcess(), reinterpret_cast<_FILETIME*>(&created), reinterpret_cast<_FILETIME*>(&ended),
                    reinterpret_cast<_FILETIME*>(&kernel), reinterpret_cast<_FILETIME*>(&user));
    return static_cast<double>(kernel + user) * 1e-7;
#else
    timespec t{};
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t);
    return static_cast<double>(t.tv_sec) + static_cast<double>(t.tv_nsec) * 1e-9;
#endif
}

// the seconds the machine was awake
inline double unbiased_seconds()
{
#if defined(_WIN32)
    unsigned long long t = 0;                                           // 100 ns
    QueryUnbiasedInterruptTimePrecise(&t);
    return static_cast<double>(t) * 1e-7;
#else
    timespec t{};
#if defined(__APPLE__)
    clock_gettime(CLOCK_UPTIME_RAW, &t);
#else
    clock_gettime(CLOCK_MONOTONIC, &t);
#endif
    return static_cast<double>(t.tv_sec) + static_cast<double>(t.tv_nsec) * 1e-9;
#endif
}

// the seconds of clock c
inline double clock_seconds(Clock c)
{
    if (c == Clock::cpu) return cpu_seconds();
    if (c == Clock::unbiased) return unbiased_seconds();
    return clock_seconds();
}

}   // namespace mootation::scalar

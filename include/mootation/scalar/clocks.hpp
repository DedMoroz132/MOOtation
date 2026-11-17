#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// THE CLOCKS (task 31, layer 1; item 30.12 of task 30): the clock a run's
// deadline (max_seconds; scalar.hpp TIME-1) is on, the knob time_clock of
// scalar::run (Problem::time_clock):
//   wall      a steady clock (std::chrono::steady_clock), as before: it goes
//             on while the machine sleeps or hibernates;
//   unbiased  the time the machine was awake: Windows'
//             QueryUnbiasedInterruptTimePrecise (Windows 10 and later; else
//             QueryUnbiasedInterruptTime, CLOCK-4), Linux's CLOCK_MONOTONIC
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
//     near, far): with windows.h's own types (STRICT's HMODULE, FARPROC), so
//     that a program including both compiles; all of them are kernel32.lib's,
//     the default library, nothing else is linked.
//   CLOCK-4. Off by default (time_clock = unbiased only).
//     QueryUnbiasedInterruptTimePrecise is not in kernel32.lib (an api set's,
//     Windows 10 and later): it is looked up once, at the first reading, in
//     KernelBase.dll (which every process has loaded; kernel32.dll does not
//     export it) by GetProcAddress, a function-local static (thread-safe).
//     Where it is absent (before Windows 10) the clock is
//     QueryUnbiasedInterruptTime, the system timer's tick (15.6 ms by
//     default): a deadline then ends up to a tick late.
// ============================================================================

#include <chrono>
#include <stdexcept>
#include <string>

#if defined(_WIN32)
// windows.h's declarations, its types spelt out (CLOCK-3): HANDLE void*,
// LPFILETIME _FILETIME*, BOOL int, PULONGLONG unsigned long long*, HMODULE
// HINSTANCE__* (STRICT), FARPROC INT_PTR (WINAPI*)(), INT_PTR __int64 on 64 bits
struct _FILETIME;
struct HINSTANCE__;
extern "C" __declspec(dllimport) void* __stdcall GetCurrentProcess(void);
extern "C" __declspec(dllimport) int __stdcall GetProcessTimes(void* process, _FILETIME* creation, _FILETIME* exit,
                                                                 _FILETIME* kernel, _FILETIME* user);
extern "C" __declspec(dllimport) int __stdcall QueryUnbiasedInterruptTime(unsigned long long* time);
extern "C" __declspec(dllimport) HINSTANCE__* __stdcall GetModuleHandleW(const wchar_t* name);
#if defined(_WIN64)
extern "C" __declspec(dllimport) long long(__stdcall* __stdcall GetProcAddress(HINSTANCE__* module, const char* name))();
#else
extern "C" __declspec(dllimport) int(__stdcall* __stdcall GetProcAddress(HINSTANCE__* module, const char* name))();
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

#if defined(_WIN32)
// QueryUnbiasedInterruptTimePrecise, looked up once; nullptr where the system has none (CLOCK-4)
using UnbiasedClock = void(__stdcall*)(unsigned long long*);
inline UnbiasedClock precise_unbiased_clock()
{
    static const UnbiasedClock f = []() -> UnbiasedClock {
        HINSTANCE__* base = GetModuleHandleW(L"kernelbase.dll");
        if (base == nullptr) return nullptr;
        // through void (*)(), the type every function pointer may pass by
        return reinterpret_cast<UnbiasedClock>(
            reinterpret_cast<void (*)()>(GetProcAddress(base, "QueryUnbiasedInterruptTimePrecise")));
    }();
    return f;
}
#endif

// the seconds the machine was awake
inline double unbiased_seconds()
{
#if defined(_WIN32)
    unsigned long long t = 0;                                           // 100 ns
    if (const UnbiasedClock f = precise_unbiased_clock()) f(&t);
    else QueryUnbiasedInterruptTime(&t);                                // CLOCK-4: the timer's tick
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

#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// WORKERS BY FREE MEMORY (task 32, layer 5): workers = auto_mem. A trial run
// of one job measures its peak memory (the process's working set); then,
// before every batch,
//
//     workers = min(threads, floor(mem_share x (free + held) / peak)), at least 1,
//
// free being the operating system's free memory, held what this process holds
// above its level before the trial (AUTO-MEM-3). The campaign engine
// (campaign/engine.hpp, `mootation run --workers auto_mem`) uses it for its
// pool; a pool of one's own — a BatchExecutor (batch_executor.hpp) that hands
// a batch to worker processes — can too: start_trial(), one item, end_trial()
// (or set_peak() with a child process's peak measured by the caller), then
// decide() before every batch.
//
// THE READINGS OF THE SYSTEM
//   free memory   Windows: GlobalMemoryStatusEx's ullAvailPhys; Linux:
//                 /proc/meminfo's MemAvailable; elsewhere sysconf's available
//                 pages where the system has them (_SC_AVPHYS_PAGES), else
//                 none (AUTO-MEM-4).
//   the process   Windows: GetProcessMemoryInfo's WorkingSetSize and
//                 PeakWorkingSetSize; Linux: /proc/self/status's VmRSS and
//                 VmHWM; elsewhere getrusage's ru_maxrss for the peak (bytes
//                 on macOS, kilobytes on the other systems) and no current
//                 level (AUTO-MEM-4).
//
// READINGS:
//   AUTO-MEM-1. Off by default: nothing here runs unless a caller asks
//     (EngineOptions::auto_mem; the engine's `--workers N` as before).
//   AUTO-MEM-2. A job's peak is the process's peak after the trial less its
//     working set before it. The process's peak is at least every level the
//     job reached, so this is at least the job's own rise; it is more where
//     the process had peaked higher before the trial (loading a registry, a
//     cache), which only lowers the count. At least one byte (no division by
//     zero).
//   AUTO-MEM-3. decide() counts what the process holds now above its level
//     before the trial (held: the running jobs' memory, and what the
//     allocator kept of the finished ones) as free: it is this pool's to
//     reuse. Without it a pool of n jobs at their peak would see n x peak
//     less free memory at its next decision and shrink itself. Other
//     programs' memory is not held: what they take lowers the count at the
//     next decision, what they give back raises it.
//   AUTO-MEM-4. Where the free memory or the process's peak cannot be read
//     (a system without either reading), the count is `threads`: the
//     protection is off, as memory_share's is (run/memory_guard.py, MEM-2),
//     and the decision says so (free or peak unknown). Without the current
//     level (the fallback's getrusage) the level before the trial is 0: the
//     peak is the whole process's, the larger reading.
//   AUTO-MEM-5. Small free memory gives 1 worker, never a refusal: a job that
//     alone needs more than mem_share x free still runs, alone.
//   AUTO-MEM-6. Windows' functions are declared here, not by <windows.h>
//     (its macros min, max, near, far): GlobalMemoryStatusEx with windows.h's
//     own type (_MEMORYSTATUSEX), so that a program including both compiles;
//     it is kernel32.lib's, the default library. GetProcessMemoryInfo is
//     psapi's (psapi.lib before Windows 7); its K32GetProcessMemoryInfo is
//     kernel32.dll's since Windows 7 and is looked up there once, by
//     GetProcAddress (as scalar/clocks.hpp's CLOCK-4), a function-local
//     static. Nothing else is linked.
// ============================================================================

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

#if defined(_WIN32)
// windows.h's declarations, its types spelt out (AUTO-MEM-6): BOOL int,
// LPMEMORYSTATUSEX _MEMORYSTATUSEX*, HMODULE HINSTANCE__* (STRICT), FARPROC
// INT_PTR (WINAPI*)(), INT_PTR __int64 on 64 bits; the same as clocks.hpp's
struct _MEMORYSTATUSEX;
struct HINSTANCE__;
extern "C" __declspec(dllimport) int __stdcall GlobalMemoryStatusEx(_MEMORYSTATUSEX* buffer);
extern "C" __declspec(dllimport) void* __stdcall GetCurrentProcess(void);
extern "C" __declspec(dllimport) HINSTANCE__* __stdcall GetModuleHandleW(const wchar_t* name);
#if defined(_WIN64)
extern "C" __declspec(dllimport) long long(__stdcall* __stdcall GetProcAddress(HINSTANCE__* module, const char* name))();
#else
extern "C" __declspec(dllimport) int(__stdcall* __stdcall GetProcAddress(HINSTANCE__* module, const char* name))();
#endif
#else
#include <sys/resource.h>
#include <unistd.h>
#include <fstream>
#endif

namespace mootation::mem {

// ── the readings of the system ──────────────────────────────────────────────

#if defined(_WIN32)
namespace detail {

// MEMORYSTATUSEX's layout: DWORD, DWORD, then seven DWORDLONGs
struct MemoryStatus {
    unsigned long length = sizeof(MemoryStatus), load = 0;
    unsigned long long total_phys = 0, avail_phys = 0, total_page_file = 0, avail_page_file = 0, total_virtual = 0,
                       avail_virtual = 0, avail_extended_virtual = 0;
};

// PROCESS_MEMORY_COUNTERS's layout: DWORD, DWORD, then eight SIZE_Ts
struct ProcessCounters {
    unsigned long cb = sizeof(ProcessCounters), page_faults = 0;
    std::size_t peak_working_set = 0, working_set = 0, quota_peak_paged_pool = 0, quota_paged_pool = 0,
                quota_peak_non_paged_pool = 0, quota_non_paged_pool = 0, page_file = 0, peak_page_file = 0;
};

using ProcessMemoryInfo = int(__stdcall*)(void*, ProcessCounters*, unsigned long);

// K32GetProcessMemoryInfo, looked up once; nullptr where kernel32.dll has none (AUTO-MEM-6)
inline ProcessMemoryInfo process_memory_info()
{
    static const ProcessMemoryInfo f = []() -> ProcessMemoryInfo {
        HINSTANCE__* k32 = GetModuleHandleW(L"kernel32.dll");
        if (k32 == nullptr) return nullptr;
        // through void (*)(), the type every function pointer may pass by
        return reinterpret_cast<ProcessMemoryInfo>(
            reinterpret_cast<void (*)()>(GetProcAddress(k32, "K32GetProcessMemoryInfo")));
    }();
    return f;
}

}   // namespace detail
#else
namespace detail {

// a "Name:   123 kB" line of a /proc file, in bytes; none if absent
inline std::optional<std::uint64_t> proc_kb(const char* file, const std::string& name)
{
    std::ifstream in(file);
    if (!in) return std::nullopt;
    for (std::string line; std::getline(in, line);) {
        if (line.compare(0, name.size(), name) != 0) continue;
        std::uint64_t kb = 0;
        bool digits = false;
        for (std::size_t k = name.size(); k < line.size(); ++k) {
            const char c = line[k];
            if (c >= '0' && c <= '9') {
                kb = kb * 10 + static_cast<std::uint64_t>(c - '0');
                digits = true;
            } else if (digits) {
                break;
            }
        }
        if (!digits) return std::nullopt;
        return kb * 1024;
    }
    return std::nullopt;
}

}   // namespace detail
#endif

// the machine's free physical memory in bytes; none where it cannot be read
inline std::optional<std::uint64_t> free_bytes()
{
#if defined(_WIN32)
    detail::MemoryStatus st;
    if (GlobalMemoryStatusEx(reinterpret_cast<_MEMORYSTATUSEX*>(&st)) == 0) return std::nullopt;
    return st.avail_phys;
#else
#if defined(__linux__)
    if (auto b = detail::proc_kb("/proc/meminfo", "MemAvailable:")) return b;
#endif
#if defined(_SC_AVPHYS_PAGES) && defined(_SC_PAGESIZE)
    const long pages = sysconf(_SC_AVPHYS_PAGES), page = sysconf(_SC_PAGESIZE);
    if (pages > 0 && page > 0) return static_cast<std::uint64_t>(pages) * static_cast<std::uint64_t>(page);
#endif
    return std::nullopt;
#endif
}

// a process's memory: its working set now (none where it cannot be read) and
// its peak, in bytes
struct ProcessMemory {
    std::optional<std::uint64_t> current;
    std::uint64_t peak = 0;
};

// this process's memory; none where it cannot be read
inline std::optional<ProcessMemory> process_memory()
{
#if defined(_WIN32)
    const detail::ProcessMemoryInfo f = detail::process_memory_info();
    if (f == nullptr) return std::nullopt;
    detail::ProcessCounters c;
    if (f(GetCurrentProcess(), &c, c.cb) == 0) return std::nullopt;
    return ProcessMemory{static_cast<std::uint64_t>(c.working_set), static_cast<std::uint64_t>(c.peak_working_set)};
#else
#if defined(__linux__)
    const auto rss = detail::proc_kb("/proc/self/status", "VmRSS:");
    const auto hwm = detail::proc_kb("/proc/self/status", "VmHWM:");
    if (rss && hwm) return ProcessMemory{rss, (*hwm > *rss ? *hwm : *rss)};
#endif
    rusage u{};
    if (getrusage(RUSAGE_SELF, &u) != 0 || u.ru_maxrss <= 0) return std::nullopt;
#if defined(__APPLE__)
    const std::uint64_t peak = static_cast<std::uint64_t>(u.ru_maxrss);             // bytes
#else
    const std::uint64_t peak = static_cast<std::uint64_t>(u.ru_maxrss) * 1024;      // kilobytes
#endif
    return ProcessMemory{std::nullopt, peak};                                       // AUTO-MEM-4
#endif
}

// where the readings come from: the system's by default; a test, or a caller
// that measures otherwise, gives its own
struct Readers {
    std::function<std::optional<std::uint64_t>()> free_memory = free_bytes;
    std::function<std::optional<ProcessMemory>()> process = process_memory;
};

// ── the count ───────────────────────────────────────────────────────────────

// min(threads, floor(share x room / peak)), at least 1 (AUTO-MEM-5)
inline int workers_for(int threads, double share, std::uint64_t room, std::uint64_t peak)
{
    const int cap = (std::max)(1, threads);
    const double n = share * static_cast<double>(room) / static_cast<double>((std::max)(peak, std::uint64_t{1}));
    if (!(n >= 1.0)) return 1;
    if (n >= static_cast<double>(cap)) return cap;
    return static_cast<int>(n);                       // n >= 1: the cast is the floor
}

// one decision: the count and what it was made of
struct Decision {
    int workers = 1;                                  // the count chosen
    int threads = 1;                                  // its cap
    double mem_share = 0.5;
    std::optional<std::uint64_t> peak;                // a job's peak, bytes; none: unknown (AUTO-MEM-4)
    std::optional<std::uint64_t> free_memory;         // the system's free memory, bytes; none: unknown
    std::uint64_t held = 0;                           // this process above its level before the trial (AUTO-MEM-3)
};

class AutoMem {
public:
    AutoMem(int threads, double mem_share, Readers readers = {})
        : threads_(threads), share_(mem_share), readers_(std::move(readers))
    {
        if (threads < 1) throw std::invalid_argument("auto_mem: threads must be >= 1, got " + std::to_string(threads));
        if (!(mem_share > 0.0 && mem_share <= 1.0))
            throw std::invalid_argument("auto_mem: mem_share must be in (0, 1], got " + std::to_string(mem_share));
    }

    // before the trial job: the process's level
    void start_trial()
    {
        const auto m = readers_.process();
        base_ = m && m->current ? *m->current : 0;    // AUTO-MEM-4: no current level, the whole peak
        started_ = true;
    }

    // after it: its peak (AUTO-MEM-2)
    void end_trial()
    {
        const auto m = readers_.process();
        if (!m || !started_) return;                  // unknown: AUTO-MEM-4
        peak_ = m->peak > base_ ? m->peak - base_ : std::uint64_t{1};
    }

    // a peak measured otherwise (a child process's, a known size), bytes
    void set_peak(std::uint64_t bytes) { peak_ = (std::max)(bytes, std::uint64_t{1}); }

    std::optional<std::uint64_t> peak() const { return peak_; }
    int threads() const { return threads_; }
    double mem_share() const { return share_; }

    // the count for the next batch, read now
    Decision decide() const
    {
        Decision d;
        d.threads = threads_;
        d.mem_share = share_;
        d.peak = peak_;
        d.free_memory = readers_.free_memory();
        const auto m = readers_.process();
        if (m && m->current && *m->current > base_) d.held = *m->current - base_;
        if (!d.free_memory || !d.peak) {
            d.workers = threads_;                     // AUTO-MEM-4: the protection is off
            return d;
        }
        d.workers = workers_for(threads_, share_, *d.free_memory + d.held, *d.peak);
        return d;
    }

private:
    int threads_;
    double share_;
    Readers readers_;
    std::uint64_t base_ = 0;
    bool started_ = false;
    std::optional<std::uint64_t> peak_;
};

}   // namespace mootation::mem

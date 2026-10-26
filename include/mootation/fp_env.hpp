#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// The thread's floating-point environment (task 14 of 2026-10-20, §5 item 3):
// denormals flushed to zero (FTZ) or read as zero (DAZ) change results, and a
// library built with -ffast-math may switch them on for the whole process
// when it is loaded. Read, never set. A campaign records it in meta.json
// (fp_env): the extension's _core.fp_env() and the C++ engine read it here.
// ============================================================================

#include <cstdint>
#include <string>

#if defined(_M_X64) || defined(__x86_64__) || defined(_M_IX86) || defined(__i386__)
#include <xmmintrin.h>                    // _mm_getcsr
#elif defined(_M_ARM64)
#include <intrin.h>                       // _ReadStatusReg
#endif

namespace mootation {

struct FpEnv {
    std::string arch = "other";           // "x86", "arm64" or "other"
    bool known = false;                   // ftz and daz read (not on "other")
    bool ftz = false, daz = false;
    std::uint64_t reg = 0;                // MXCSR on x86, FPCR on ARM64
};

inline FpEnv fp_env()
{
    FpEnv e;
#if defined(_M_X64) || defined(__x86_64__) || defined(_M_IX86) || defined(__i386__)
    const unsigned csr = _mm_getcsr();    // MXCSR: FTZ bit 15, DAZ bit 6
    e.arch = "x86";
    e.known = true;
    e.ftz = (csr & 0x8000u) != 0;
    e.daz = (csr & 0x0040u) != 0;
    e.reg = csr;
#elif defined(__aarch64__) && !defined(_MSC_VER)
    std::uint64_t fpcr = 0;               // FPCR: FZ bit 24 flushes both ways
    __asm__ volatile("mrs %0, fpcr" : "=r"(fpcr));
    e.arch = "arm64";
    e.known = true;
    e.ftz = e.daz = (fpcr & (1ull << 24)) != 0;
    e.reg = fpcr;
#elif defined(_M_ARM64)
    const auto fpcr = static_cast<std::uint64_t>(_ReadStatusReg(ARM64_FPCR));
    e.arch = "arm64";
    e.known = true;
    e.ftz = e.daz = (fpcr & (1ull << 24)) != 0;
    e.reg = fpcr;
#endif
    return e;
}

}   // namespace mootation

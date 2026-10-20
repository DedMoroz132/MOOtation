#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// No contraction into fused multiply-adds (task 14 of 2026-10-20, §3). A fused
// a * b + c rounds once where the source rounds twice, and whether a compiler
// fuses depends on the compiler, its dialect and the target: GCC contracts in
// its GNU dialects, Clang within an expression, and AArch64 has the
// instruction in its base set — so the same seed gave other runs on another
// platform. The build says no (CMakeLists.txt: -ffp-contract=off on the
// MOOtation target, /fp:precise with MSVC); these pragmas say it again for a
// translation unit compiled without it, with Clang and MSVC (GCC has no such
// pragma). They hold for the rest of the translation unit, the code that
// includes the library too. Where the library wants a fused multiply-add it
// says so (std::fma; suite/bbob.hpp). -ffast-math, -Ofast and /fp:fast break
// the guarantee whatever these say (and a build with them is warned of).
// MSVC 19.44 contracts only under /fp:contract, which the pragma below
// switches off again; checked on the disassembly of every core built with
// /arch:AVX2 (no vfmadd), and with GCC and Clang for -march=haswell.
// ============================================================================

#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(_MSC_VER)
#pragma fp_contract(off)
#endif

// A build with -ffast-math / -Ofast (GCC, Clang: __FAST_MATH__) or /fp:fast
// (MSVC: _M_FP_FAST) reorders and fuses the arithmetic: it is warned of here
// (task 14, supplement 1, Д4), not refused, as it may be the user's choice.
// 32-bit x86 with the x87 unit's arithmetic (80-bit registers, double
// rounding) gives other bits than SSE2's: only SSE2 arithmetic is supported
// there (task 14, §5 item 4; GCC/Clang: -msse2 -mfpmath=sse; MSVC's default
// /arch:SSE2). 64-bit builds always use SSE2.
#if defined(__i386__) && !defined(__SSE2_MATH__) && !defined(_MSC_VER)
#error "MOOtation on 32-bit x86 needs SSE2 floating point: build with -msse2 -mfpmath=sse"
#endif

#if defined(__FAST_MATH__) && (defined(__GNUC__) || defined(__clang__))
#warning "MOOtation built with -ffast-math or -Ofast: runs are not reproducible across platforms"
#elif defined(_M_FP_FAST)
#pragma message("MOOtation built with /fp:fast: runs are not reproducible across platforms")
#endif

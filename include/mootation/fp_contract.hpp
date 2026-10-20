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
// the guarantee whatever these say.
// ============================================================================

#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(_MSC_VER)
#pragma fp_contract(off)
#endif

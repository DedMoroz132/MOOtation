#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// M_PI is POSIX, not ISO C++. libstdc++ and libc++ define it in <cmath> by
// default, MSVC does not unless _USE_MATH_DEFINES is defined *before* the first
// <cmath> include — an ordering constraint a header-only library cannot impose
// on its users.
//
// Defining the macro here, guarded, sidesteps the ordering problem entirely and
// keeps the "nothing beyond the standard library" promise honest on every
// toolchain. Any header that uses M_PI includes this one.
// ============================================================================

#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

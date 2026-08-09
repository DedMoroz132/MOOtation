#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// MOOtation version.
//
// MOOTATION_VERSION is a single comparable integer, so a dependent project can
// guard on a feature the way it would with any C library:
//
//     #if MOOTATION_VERSION >= MOOTATION_VERSION_NUM(0, 2, 0)
//         // use an API that only exists from 0.2.0 on
//     #endif
//
// Until 1.0.0 the public API may change in a minor release.
// ============================================================================

#define MOOTATION_VERSION_MAJOR 0
#define MOOTATION_VERSION_MINOR 1
#define MOOTATION_VERSION_PATCH 0

#define MOOTATION_VERSION_NUM(major, minor, patch) \
    ((major) *10000 + (minor) *100 + (patch))

#define MOOTATION_VERSION                                                   \
    MOOTATION_VERSION_NUM(MOOTATION_VERSION_MAJOR, MOOTATION_VERSION_MINOR, \
                          MOOTATION_VERSION_PATCH)

#define MOOTATION_STRINGIFY_IMPL(x) #x
#define MOOTATION_STRINGIFY(x)      MOOTATION_STRINGIFY_IMPL(x)

#define MOOTATION_VERSION_STRING                       \
    MOOTATION_STRINGIFY(MOOTATION_VERSION_MAJOR)       \
    "." MOOTATION_STRINGIFY(MOOTATION_VERSION_MINOR)   \
    "." MOOTATION_STRINGIFY(MOOTATION_VERSION_PATCH)

namespace mootation {

// Compile-time version string, e.g. "0.1.0".
inline constexpr const char* version() noexcept
{
    return MOOTATION_VERSION_STRING;
}

}   // namespace mootation

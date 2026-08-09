# SPDX-License-Identifier: Apache-2.0
#
# The C ABI. This is the only compiled artefact in the project: everything else
# is header-only. It exists so that languages other than C++ can drive the
# optimizer through an FFI — ctypes, cffi, P/Invoke, Rust's `extern "C"`,
# Julia's ccall, MATLAB's loadlibrary, and so on.

# The install destinations below need GNUInstallDirs, and this directory is
# added before the top-level install block includes it.
include(GNUInstallDirs)

add_library(mootation_c SHARED mootation_c.cpp)
add_library(MOOtation::c ALIAS mootation_c)

# Same reason as the header-only target: keep the installed name identical to
# the build-tree alias, so MOOtation::c works either way.
set_target_properties(mootation_c PROPERTIES EXPORT_NAME c)

target_link_libraries(mootation_c PRIVATE MOOtation::MOOtation mootation_warnings)
target_include_directories(mootation_c PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

# MOOTATION_C_BUILD flips capi.h from dllimport to dllexport. PRIVATE on
# purpose: a consumer must NOT inherit it, or it would try to export the
# symbols itself instead of importing them.
target_compile_definitions(mootation_c PRIVATE MOOTATION_C_BUILD)

set_target_properties(mootation_c PROPERTIES
    OUTPUT_NAME              mootation
    VERSION                  ${PROJECT_VERSION}
    SOVERSION                ${PROJECT_VERSION_MAJOR}
    # Only the extern "C" entry points are exported; nothing from the C++
    # innards leaks into the shared library's symbol table.
    CXX_VISIBILITY_PRESET    hidden
    VISIBILITY_INLINES_HIDDEN ON
    WINDOWS_EXPORT_ALL_SYMBOLS OFF)

if(MOOTATION_BUILD_TESTS)
    # C is enabled HERE and not in project(), so that configuring MOOtation
    # without the C API needs a C++ compiler and nothing else. The smoke test
    # is the only C source in the repository.
    enable_language(C)

    # A C consumer, compiled as C and not C++, so a C++-ism that crept into
    # capi.h fails here rather than in someone else's project.
    add_executable(capi_smoke smoke.c)
    target_link_libraries(capi_smoke PRIVATE mootation_c)

    # smoke.c evaluates ZDT1, so it calls sqrt(). A C program on glibc must ask
    # for libm explicitly — C++ gets it for free through libstdc++, which is why
    # nothing else here needs it, and why MSVC (math in the CRT) and macOS (math
    # in libSystem) link fine without it. Found rather than hardcoded as "-lm"
    # so the platforms that have no separate libm simply skip it.
    find_library(MOOTATION_LIBM m)
    if(MOOTATION_LIBM)
        target_link_libraries(capi_smoke PRIVATE ${MOOTATION_LIBM})
    endif()
    set_target_properties(capi_smoke PROPERTIES C_STANDARD 99 FOLDER capi)
    add_test(NAME capi.smoke COMMAND capi_smoke)
    set_tests_properties(capi.smoke PROPERTIES TIMEOUT 300)

    # The same run driven from Python through ctypes — no build step, no
    # binding, just the shared library. This is the proof that the ABI is
    # reachable from another language.
    find_package(Python3 COMPONENTS Interpreter)
    if(Python3_Interpreter_FOUND)
        add_test(NAME capi.ctypes
                 COMMAND ${Python3_EXECUTABLE}
                         ${CMAKE_CURRENT_SOURCE_DIR}/ctypes_demo.py
                         $<TARGET_FILE:mootation_c>)
        set_tests_properties(capi.ctypes PROPERTIES TIMEOUT 300)
    endif()
endif()

if(MOOTATION_INSTALL)
    install(TARGETS mootation_c EXPORT MOOtationTargets
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()

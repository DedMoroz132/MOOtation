# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.14)

# CMP0092: do not put /W3 in the default MSVC flags. Without this, adding /W4
# produces "command line warning D9025: overriding '/W3' with '/W4'" on every
# translation unit. Guarded so CMake < 3.15 still configures.
if(POLICY CMP0092)
    cmake_policy(SET CMP0092 NEW)
endif()

# ── Version: single source of truth is include/mootation/version.hpp ─────────
# Parsing it here rather than repeating the number keeps the two from drifting.
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/include/mootation/version.hpp" _mootation_version_hpp)
string(REGEX MATCH "#define MOOTATION_VERSION_MAJOR ([0-9]+)" _m "${_mootation_version_hpp}")
set(_MOOTATION_VER_MAJOR "${CMAKE_MATCH_1}")
string(REGEX MATCH "#define MOOTATION_VERSION_MINOR ([0-9]+)" _m "${_mootation_version_hpp}")
set(_MOOTATION_VER_MINOR "${CMAKE_MATCH_1}")
string(REGEX MATCH "#define MOOTATION_VERSION_PATCH ([0-9]+)" _m "${_mootation_version_hpp}")
set(_MOOTATION_VER_PATCH "${CMAKE_MATCH_1}")
unset(_mootation_version_hpp)
unset(_m)

project(MOOtation
    VERSION   ${_MOOTATION_VER_MAJOR}.${_MOOTATION_VER_MINOR}.${_MOOTATION_VER_PATCH}
    LANGUAGES CXX
    DESCRIPTION "Header-only C++17 library of multi- and many-objective evolutionary algorithms"
    HOMEPAGE_URL "https://github.com/DedMoroz132/MOOtation")

# Is this the top-level project, or are we being consumed via add_subdirectory /
# FetchContent? A consumed project must not build tests, install anything, or
# add options to the parent's cache.
if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    set(MOOTATION_IS_TOP_LEVEL ON)
else()
    set(MOOTATION_IS_TOP_LEVEL OFF)
endif()

option(MOOTATION_BUILD_TESTS    "Build the test suite"           ${MOOTATION_IS_TOP_LEVEL})
option(MOOTATION_BUILD_EXAMPLES "Build the examples"             ${MOOTATION_IS_TOP_LEVEL})
option(MOOTATION_INSTALL        "Generate the install target"    ${MOOTATION_IS_TOP_LEVEL})
option(MOOTATION_SANITIZE       "Build tests with ASan + UBSan"  OFF)
option(MOOTATION_WERROR         "Treat compiler warnings as errors" OFF)
# Off by default on purpose: the library is header-only C++ and must stay
# buildable with nothing but a compiler. Turning this on pulls in pybind11.
option(MOOTATION_BUILD_PYTHON   "Build the pybind11 module"      OFF)
# The C ABI is the bridge for every language that is not C++ (ctypes, cffi,
# P/Invoke, Rust extern "C", Julia ccall, MATLAB loadlibrary). It is the only
# compiled artefact the project produces, so it is opt-in for the same reason
# the Python module is: `cmake --build` with nothing enabled must stay a
# header-only, compile-nothing affair.
option(MOOTATION_BUILD_C_API    "Build the C ABI shared library" OFF)

# ── The library ─────────────────────────────────────────────────────────────
# Header-only, so an INTERFACE target: it carries include paths and the required
# language standard, and compiles nothing of its own.
add_library(mootation INTERFACE)
add_library(MOOtation::MOOtation ALIAS mootation)

# Without EXPORT_NAME the installed package would offer MOOtation::mootation
# (namespace + the raw target name) while the build tree offers the alias
# MOOtation::MOOtation. A project that worked via add_subdirectory or
# FetchContent would then fail on find_package with "target was not found",
# which is the least obvious way for a package to break.
set_target_properties(mootation PROPERTIES EXPORT_NAME MOOtation)

# GNUInstallDirs must be included BEFORE this line, not down in the install
# block. CMAKE_INSTALL_INCLUDEDIR is expanded here and now: undefined, the
# generator expression collapses to $<INSTALL_INTERFACE:>, the exported target
# ships with no INTERFACE_INCLUDE_DIRECTORIES at all, and every consumer of the
# installed package fails with "cannot open include file: mootation/...". The
# build tree keeps working the whole time, which is what makes it easy to miss.
include(GNUInstallDirs)

target_include_directories(mootation INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

# C++17 is the floor. Consumers may compile at 20 or 23; both are tested in CI.
target_compile_features(mootation INTERFACE cxx_std_17)

# mootation/embed.hpp's Session runs the algorithm on a worker thread, so a
# consumer that includes it needs the platform's threading library. Everything
# else in the library is single-threaded and would not need this; carrying it on
# the INTERFACE target is what keeps `#include <mootation/embed.hpp>` from
# failing to link on GCC/libstdc++, where <thread> is not self-sufficient.
find_package(Threads REQUIRED)
target_link_libraries(mootation INTERFACE Threads::Threads)

if(MSVC)
    # /Zc:__cplusplus  — without it MSVC reports __cplusplus as 199711L
    #                    regardless of /std, which breaks feature detection in
    #                    downstream code.
    # /EHsc            — the library throws std::invalid_argument on bad setup;
    #                    MSVC needs standard C++ exception semantics stated.
    target_compile_options(mootation INTERFACE /Zc:__cplusplus /EHsc)
endif()

# ── Warning flags, applied to our own targets only ──────────────────────────
# Deliberately NOT on the INTERFACE target: a consumer chooses their own warning
# level and should never inherit ours.
add_library(mootation_warnings INTERFACE)
if(MSVC)
    target_compile_options(mootation_warnings INTERFACE /W4 /permissive-)
    if(MOOTATION_WERROR)
        target_compile_options(mootation_warnings INTERFACE /WX)
    endif()
else()
    target_compile_options(mootation_warnings INTERFACE -Wall -Wextra -Wpedantic)
    if(MOOTATION_WERROR)
        target_compile_options(mootation_warnings INTERFACE -Werror)
    endif()
endif()

# ── Sanitizers ──────────────────────────────────────────────────────────────
add_library(mootation_sanitize INTERFACE)
if(MOOTATION_SANITIZE)
    if(MSVC)
        target_compile_options(mootation_sanitize INTERFACE /fsanitize=address)
    else()
        target_compile_options(mootation_sanitize INTERFACE
            -fsanitize=address,undefined -fno-omit-frame-pointer -g)
        target_link_options(mootation_sanitize INTERFACE
            -fsanitize=address,undefined)
    endif()
endif()

# ── Subdirectories ──────────────────────────────────────────────────────────
# enable_testing() must come first: add_test() calls made before it are
# silently discarded, which quietly turns the examples into build-only targets.
if(MOOTATION_BUILD_TESTS)
    enable_testing()
endif()

if(MOOTATION_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(MOOTATION_BUILD_TESTS)
    add_subdirectory(tests)
endif()

if(MOOTATION_BUILD_C_API)
    add_subdirectory(capi)
endif()

if(MOOTATION_BUILD_PYTHON)
    # LOCATION, not CMake's default of VERSION. The default picks the HIGHEST
    # Python it can find anywhere on the machine, which is almost never the one
    # you are building for: on a CI runner with a pinned 3.11 on PATH and a
    # 3.12 in /usr, it builds a cp312 extension that the 3.11 running the tests
    # cannot import. An extension module belongs to the interpreter you invoked,
    # so resolve by position on PATH.
    set(Python_FIND_STRATEGY LOCATION)
    # actions/setup-python exports this; CMake reads it as a hint only when it
    # is a CMake variable, so promote it.
    if(NOT DEFINED Python_ROOT_DIR AND DEFINED ENV{Python_ROOT_DIR})
        set(Python_ROOT_DIR "$ENV{Python_ROOT_DIR}")
    endif()
    find_package(Python COMPONENTS Interpreter Development REQUIRED)
    message(STATUS "MOOtation: building the extension for ${Python_EXECUTABLE} (${Python_VERSION})")
    add_subdirectory(python)
endif()

# ── Install / package ───────────────────────────────────────────────────────
if(MOOTATION_INSTALL)
    include(GNUInstallDirs)
    include(CMakePackageConfigHelpers)

    # The exclusions are load-bearing, not tidiness. This rule copies by
    # filesystem glob, so it sees the working tree — including any local-only
    # work that .gitignore keeps out of the git index. Ignoring a file and not
    # installing it are two different guarantees, and only the second one
    # applies here.
    #
    # The patterns are generic on purpose (matching .gitignore), so that a
    # published CMakeLists never names unpublished work.
    #
    # Verified by the install-consume CI job, which fails if any *_local* path
    # reaches the install prefix.
    # *.def is not decoration: algorithms.def is the X-macro list embed.hpp
    # includes three times to build its run-time dispatch. Installing only
    # *.hpp would ship a tree where <mootation/embed.hpp> does not compile.
    install(DIRECTORY include/mootation
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
            FILES_MATCHING
                PATTERN "*_local"     EXCLUDE
                PATTERN "*_local.hpp" EXCLUDE
                PATTERN "*.hpp"
                PATTERN "*.def")

    install(TARGETS mootation EXPORT MOOtationTargets)

    install(EXPORT MOOtationTargets
            FILE        MOOtationTargets.cmake
            NAMESPACE   MOOtation::
            DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MOOtation)

    configure_package_config_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/MOOtationConfig.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/MOOtationConfig.cmake"
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MOOtation)

    # ARCH_INDEPENDENT: a header-only library is usable from any architecture,
    # so the consumer's pointer size must not be compared against ours.
    write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/MOOtationConfigVersion.cmake"
        VERSION       ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion
        ARCH_INDEPENDENT)

    install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/MOOtationConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/MOOtationConfigVersion.cmake"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MOOtation)

    install(FILES LICENSE NOTICE
            DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/licenses/MOOtation)
endif()

if(MOOTATION_IS_TOP_LEVEL)
    message(STATUS "MOOtation ${PROJECT_VERSION} "
                   "(tests=${MOOTATION_BUILD_TESTS} "
                   "examples=${MOOTATION_BUILD_EXAMPLES} "
                   "c_api=${MOOTATION_BUILD_C_API} "
                   "sanitize=${MOOTATION_SANITIZE} "
                   "python=${MOOTATION_BUILD_PYTHON})")
endif()

# SPDX-License-Identifier: Apache-2.0

set(MOOTATION_TESTS
    test_das_dennis
    test_operators
    test_reproducibility
    test_convergence
    test_constraints
    test_warm_start)

foreach(_t IN LISTS MOOTATION_TESTS)
    add_executable(${_t} ${_t}.cpp)
    target_link_libraries(${_t} PRIVATE
        MOOtation::MOOtation
        mootation_warnings
        mootation_sanitize)
    target_include_directories(${_t} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    set_target_properties(${_t} PROPERTIES FOLDER tests)

    if(MSVC)
        # Several of these instantiate all 60 algorithms in one translation
        # unit, which overruns the COFF section limit in a Debug build:
        #   fatal error C1128: number of sections exceeded object file format
        # A Release build folds enough to stay under it, which is why the
        # Windows CI job never hit this and every local Debug build does.
        target_compile_options(${_t} PRIVATE /bigobj)
    endif()

    add_test(NAME ${_t} COMMAND ${_t})
endforeach()

# ── Labels: which tests every push pays for, and which run nightly ──────────
# test_convergence and test_constraints each instantiate all 60 algorithms and
# run them; together they are ~10 minutes of the ~11-minute suite. Paying that
# on every push, across a six-way compiler/standard matrix, means every pull
# request waits on the slowest thing in the repository.
#
# They are labelled `nightly` rather than deleted or shortened, because what
# they check — that all 60 converge, and that every constraint_mode switch
# actually does something — is exactly what must not silently rot. CI runs
# `-LE nightly` on push and the full set on a schedule.
#
#   ctest -LE nightly     everything quick (the default for a push)
#   ctest -L  nightly     only the long ones
#   ctest                 all of it
set_tests_properties(test_convergence test_constraints PROPERTIES LABELS nightly)

# Wall-clock limits, scaled for the build they run in.
#
# A sanitizer build is roughly an order of magnitude slower, and the nightly job
# stacks a Debug build on top of that. One number cannot serve both: the limit
# that is generous for an optimized build expires under ASan+UBSan, and an
# expired limit is reported exactly like a broken test. The nightly suite spent
# ten consecutive nights red for that reason, with nothing actually wrong.
#
# The scale is deliberately blunt. Tuning per test would mean re-tuning every
# time an algorithm gets slower, and the only thing these numbers have to do is
# distinguish "slow" from "hung".
#
# Measured, not guessed: test_convergence under a Debug sanitizer build takes
# 2 h 19 min end to end — against the thirty-minute limit it used to have, which
# is why the nightly job had been red for ten nights. Eight gives four hours,
# comfortably above the measurement rather than a few minutes above it, because
# a runner slower than the machine that measured it must not turn a passing test
# into a red night again. The nightly job runs the two long tests concurrently,
# so even both expiring at once finishes inside the runner's own six-hour
# ceiling and is reported as a timeout rather than as a job GitHub killed.
if(MOOTATION_SANITIZE)
    set(_moo_timeout_scale 8)
else()
    set(_moo_timeout_scale 1)
endif()

function(mootation_test_timeout _test _seconds)
    math(EXPR _scaled "${_seconds} * ${_moo_timeout_scale}")
    set_tests_properties(${_test} PROPERTIES TIMEOUT ${_scaled})
endfunction()

# test_convergence instantiates all 60 algorithms in one translation unit and
# runs each for 200 generations. It is by far the slowest target, in both
# compilation and execution.
mootation_test_timeout(test_convergence 1800)
mootation_test_timeout(test_reproducibility 1800)
# test_constraints instantiates all 60 algorithms twice (mode off / mode on).
mootation_test_timeout(test_constraints 1800)
mootation_test_timeout(test_das_dennis 300)
mootation_test_timeout(test_operators 300)
# Cheap to run, expensive to compile: it includes embed.hpp, which instantiates
# all 60 algorithms for the run-time dispatch. Stays in the push tier because
# the property it guards — that a warm start costs no function evaluations — is
# the whole reason to have one.
mootation_test_timeout(test_warm_start 600)

# A compile-only check that the umbrella header is self-contained: it must pull
# in everything it needs without the user having included anything first.
add_executable(test_umbrella_selfcontained umbrella_tu.cpp)
target_link_libraries(test_umbrella_selfcontained PRIVATE
    MOOtation::MOOtation mootation_warnings)
set_target_properties(test_umbrella_selfcontained PROPERTIES FOLDER tests)
add_test(NAME test_umbrella_selfcontained COMMAND test_umbrella_selfcontained)

# ── The TOML/runner layer (pure Python, no dependencies) ────────────────────
# Registered here rather than under python/ because it needs no pybind11 and no
# build step: it is a plain interpreter run, and skipping it when Python is
# missing is better than making Python a build requirement.
find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND AND NOT Python3_VERSION VERSION_LESS 3.11)
    add_test(NAME python.mootation_run
             COMMAND ${Python3_EXECUTABLE}
                     ${CMAKE_CURRENT_SOURCE_DIR}/../python/test_run_layer.py)
    set_tests_properties(python.mootation_run PROPERTIES TIMEOUT 300)

    # --check must stay green on the shipped example, because that example is
    # the first thing anyone copies.
    add_test(NAME python.mootation_run_check
             COMMAND ${Python3_EXECUTABLE} -m mootation.run --check
                     ${CMAKE_CURRENT_SOURCE_DIR}/../python/examples/demo.toml)
    set_tests_properties(python.mootation_run_check PROPERTIES
        ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/../python"
        TIMEOUT 120)

    # The terminal interface, driven headless through Textual's run_test. Skips
    # itself (exit 0) when Textual is absent, so it never makes a UI library a
    # requirement for building the C++ library.
    add_test(NAME python.mootation_tui
             COMMAND ${Python3_EXECUTABLE}
                     ${CMAKE_CURRENT_SOURCE_DIR}/../python/test_tui.py)
    set_tests_properties(python.mootation_tui PROPERTIES TIMEOUT 300)

    # The examples need the COMPILED extension, so they are registered in
    # python/CMakeLists.txt instead, next to the interpreter it was built for.
    # Registering them here would run them under whatever Python this directory
    # happened to find, which is not necessarily the same one.
endif()

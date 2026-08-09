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

# test_convergence instantiates all 60 algorithms in one translation unit and
# runs each for 200 generations. It is by far the slowest target, in both
# compilation and execution, and needs a longer leash under the sanitizers.
set_tests_properties(test_convergence PROPERTIES TIMEOUT 1800)
set_tests_properties(test_reproducibility PROPERTIES TIMEOUT 1800)
# test_constraints instantiates all 60 algorithms twice (mode off / mode on).
set_tests_properties(test_constraints PROPERTIES TIMEOUT 1800)
set_tests_properties(test_das_dennis PROPERTIES TIMEOUT 300)
set_tests_properties(test_operators  PROPERTIES TIMEOUT 300)
# Cheap to run, expensive to compile: it includes embed.hpp, which instantiates
# all 60 algorithms for the run-time dispatch. Stays in the push tier because
# the property it guards — that a warm start costs no function evaluations — is
# the whole reason to have one.
set_tests_properties(test_warm_start PROPERTIES TIMEOUT 600)

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

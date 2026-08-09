# SPDX-License-Identifier: Apache-2.0
#
# Optional Python bindings. Off by default: the library is header-only C++ and
# must stay buildable with nothing but a compiler.
#
#   cmake -S . -B build -DMOOTATION_BUILD_PYTHON=ON
#   cmake --build build
#
# pybind11 is found either from an installed package or via FetchContent, so a
# clean machine needs no manual setup.

# ONE interpreter for the whole project. Without this, pybind11 falls back to
# the deprecated FindPythonInterp/PythonLibs modules and may resolve a
# DIFFERENT Python than the top-level find_package(Python) did — here it built
# the extension against 3.12 while the tests ran under 3.13, and a .pyd carries
# an ABI tag, so it simply will not import. PYBIND11_FINDPYTHON=ON makes
# pybind11 use the modern FindPython results, which is the same interpreter
# everything else in this project sees.
set(PYBIND11_FINDPYTHON ON)

find_package(pybind11 CONFIG QUIET)

if(NOT pybind11_FOUND)
    include(FetchContent)
    FetchContent_Declare(pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG        v2.13.6)
    FetchContent_MakeAvailable(pybind11)
endif()

pybind11_add_module(mootation_python mootation_module.cpp)

# The extension is mootation._core, not a top-level `mootation`: the package
# of that name is the Python source next door, and it re-exports from here.
# Building it straight into python/mootation/ means the source tree is
# importable as-is after a build, with no copy step and no PYTHONPATH games.
set_target_properties(mootation_python PROPERTIES
    OUTPUT_NAME _core
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/mootation
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/mootation
    FOLDER python)

target_link_libraries(mootation_python PRIVATE MOOtation::MOOtation)

# Building into the source tree above is a developer convenience — it makes the
# checkout importable with no copy step. It does NOT put the extension in a
# wheel: scikit-build-core fills the wheel from CMake's install rules, and
# `wheel.packages` contributes the pure-Python files only. Without this rule
# `pip install .` yields a package whose every compiled entry point raises
# ImportError, which is exactly what CI caught.
if(DEFINED SKBUILD)
    install(TARGETS mootation_python LIBRARY DESTINATION mootation)
endif()

# The dispatch is generated from include/mootation/algorithms.def, so the binding cannot
# drift from the library: adding an algorithm there makes it callable here.
target_include_directories(mootation_python PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include)

# The extension is built against ONE interpreter and is only importable by
# that one: a .pyd carries an ABI tag (cp312) and a different Python will not
# load it. `${Python_EXECUTABLE}` is not reliably set here — pybind11's own
# find writes PYTHON_EXECUTABLE, and the CMake-module name depends on which
# find_package ran first — and an EMPTY interpreter variable does not fail
# loudly: the test command becomes the .py path alone, Windows resolves it
# through the file association, and some other Python runs it. That is exactly
# how this test passed for a while against the wrong interpreter.
#
# So: pin the interpreter explicitly, preferring whatever pybind11 resolved.
if(DEFINED Python_EXECUTABLE)
    set(_MOOTATION_PY "${Python_EXECUTABLE}")
elseif(DEFINED PYTHON_EXECUTABLE)
    set(_MOOTATION_PY "${PYTHON_EXECUTABLE}")
elseif(TARGET Python::Interpreter)
    set(_MOOTATION_PY "$<TARGET_FILE:Python::Interpreter>")
else()
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    set(_MOOTATION_PY "${Python3_EXECUTABLE}")
endif()
message(STATUS "MOOtation: Python module will be tested with ${_MOOTATION_PY}")

if(MOOTATION_BUILD_TESTS)
    add_test(NAME python.smoke
        COMMAND ${CMAKE_COMMAND} -E env
                "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}"
                ${_MOOTATION_PY} ${CMAKE_CURRENT_SOURCE_DIR}/test_binding.py)
    set_tests_properties(python.smoke PROPERTIES TIMEOUT 600)

    # The examples and the run-layer tests import the SAME extension, so they
    # must use the same interpreter. Registered here rather than in tests/ for
    # that reason: tests/ has no idea which Python the module was built for.
    add_test(NAME python.examples_built
        COMMAND ${CMAKE_COMMAND} -E env
                "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}"
                ${_MOOTATION_PY} ${CMAKE_CURRENT_SOURCE_DIR}/test_examples.py)
    set_tests_properties(python.examples_built PROPERTIES
        LABELS nightly
        TIMEOUT 1800)
endif()

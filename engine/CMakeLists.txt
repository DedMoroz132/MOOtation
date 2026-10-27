# SPDX-License-Identifier: Apache-2.0
#
# The campaign engine in C++: the `mootation` executable (engine/mootation.cpp;
# task 17, stage 5). One translation unit that instantiates every core
# (run_core.hpp), as the Python module and embed.hpp do.

add_executable(mootation_engine mootation.cpp)
set_target_properties(mootation_engine PROPERTIES OUTPUT_NAME mootation)
target_link_libraries(mootation_engine PRIVATE MOOtation::MOOtation mootation_warnings)
find_package(Threads REQUIRED)
target_link_libraries(mootation_engine PRIVATE Threads::Threads)

# The source revision it is built from, for --version and meta.json's
# revision: python/git_stamp.cmake, rewritten on every build.
find_package(Git QUIET)
set(_MOOTATION_ENGINE_STAMP_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
add_custom_target(mootation_engine_git_stamp
    COMMAND ${CMAKE_COMMAND}
            -DSRC=${PROJECT_SOURCE_DIR}
            -DOUT=${_MOOTATION_ENGINE_STAMP_DIR}/mootation_git_stamp.h
            -DGIT=${GIT_EXECUTABLE}
            -P ${PROJECT_SOURCE_DIR}/python/git_stamp.cmake
    BYPRODUCTS ${_MOOTATION_ENGINE_STAMP_DIR}/mootation_git_stamp.h
    COMMENT "MOOtation: stamping the engine's source revision"
    VERBATIM)
add_dependencies(mootation_engine mootation_engine_git_stamp)
target_include_directories(mootation_engine PRIVATE ${_MOOTATION_ENGINE_STAMP_DIR})

# Built with the Python module, a copy goes beside it, python/mootation/bin/,
# where `python -m mootation.run.campaign` finds it (--engine auto): the
# jobs it can run go to it, the rest to the Python engine.
if(MOOTATION_BUILD_PYTHON)
    add_custom_command(TARGET mootation_engine POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${PROJECT_SOURCE_DIR}/python/mootation/bin
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:mootation_engine>
                ${PROJECT_SOURCE_DIR}/python/mootation/bin/$<TARGET_FILE_NAME:mootation_engine>
        VERBATIM)
endif()

if(MOOTATION_BUILD_TESTS)
    # the executable answers; the engine itself is tests/test_engine.cpp's
    add_test(NAME engine.version COMMAND mootation_engine --version)
    add_test(NAME engine.usage COMMAND mootation_engine --help)
endif()

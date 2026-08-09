# SPDX-License-Identifier: Apache-2.0
#
# Every example is built AND run in CI. An example that compiles but crashes is
# worse than no example, because it is the first code a new user copies.

set(MOOTATION_EXAMPLES
    nsga2_zdt1
    ibea_zdt1
    ibea_mixed
    custom_problem
    batch_demo
    benchmark_problem
    embed_asktell)

foreach(_ex IN LISTS MOOTATION_EXAMPLES)
    add_executable(${_ex} ${_ex}.cpp)
    target_link_libraries(${_ex} PRIVATE MOOtation::MOOtation mootation_warnings)
    set_target_properties(${_ex} PROPERTIES FOLDER examples)

    if(MOOTATION_BUILD_TESTS)
        add_test(NAME example.${_ex} COMMAND ${_ex})
        # The examples write their populations into the working directory.
        set_tests_properties(example.${_ex} PROPERTIES
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            TIMEOUT 300)
    endif()
endforeach()

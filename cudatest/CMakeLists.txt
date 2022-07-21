################################################################################
# Tests
################################################################################
set(test_sources
	main.cpp
	cudatest.cu
)
add_executable(cuda_test ${test_sources})

################################################################################
# Required Libraries
################################################################################
include(catch2)
target_link_libraries(cuda_test PUBLIC Catch2::Catch2)

target_link_libraries(cuda_test PUBLIC polysolve::polysolve)

include(polysolve_warnings)
target_link_libraries(cuda_test PRIVATE polysolve::warnings)
################################################################################
# Register tests
################################################################################
foreach(source IN ITEMS ${test_sources})
    source_group("tests" FILES "${source}")
endforeach()

# Register tests
set(PARSE_CATCH_TESTS_ADD_TO_CONFIGURE_DEPENDS ON)
catch_discover_tests(cuda_test)

################################################################################
# Data
################################################################################
set(DATA_DIR "${CMAKE_SOURCE_DIR}/tests/data")
target_compile_definitions(cuda_test PUBLIC -DPOLYSOLVE_DATA_DIR=\"${DATA_DIR}\")
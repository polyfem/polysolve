################################################################################
# Tests
################################################################################

set(test_sources
	main.cpp
	test_solver.cpp
)
add_executable(unit_tests ${test_sources})
################################################################################
# Required Libraries
################################################################################

include(catch2)
target_link_libraries(unit_tests PUBLIC Catch2::Catch2)

target_link_libraries(unit_tests PUBLIC polysolve::polysolve)

include(polysolve_warnings)
target_link_libraries(unit_tests PRIVATE polysolve::warnings)

if(POLYSOLVE_WITH_SANITIZERS)
	add_sanitizers(unit_tests)
endif()

################################################################################
# Register tests
################################################################################

foreach(source IN ITEMS ${test_sources})
    source_group("tests" FILES "${source}")
endforeach()

# Register tests
set(PARSE_CATCH_TESTS_ADD_TO_CONFIGURE_DEPENDS ON)
catch_discover_tests(unit_tests)

################################################################################
# Data
################################################################################

set(DATA_DIR "${CMAKE_SOURCE_DIR}/tests/data")
include(ExternalProject)
include(FetchContent)
FetchContent_Declare(
    data
    SOURCE_DIR ${DATA_DIR}
    GIT_REPOSITORY https://github.com/polyfem/polyfem-data
    GIT_TAG 9c1bdd5bd02215e80bc1668547e5dbeb5484a527
)
FetchContent_GetProperties(data)
if(NOT data_POPULATED)
	FetchContent_Populate(data)
endif()

target_compile_definitions(unit_tests PUBLIC -DPOLYSOLVE_DATA_DIR=\"${DATA_DIR}\")

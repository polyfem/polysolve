cmake_minimum_required(VERSION 3.1)
################################################################################

set(test_sources
	main.cpp
	test_solver.cpp
)


add_executable(unit_tests ${test_sources})

foreach(source IN ITEMS ${test_sources})
    source_group("tests" FILES "${source}")
endforeach()

target_link_libraries(unit_tests PUBLIC polysolve catch warnings::all)

set(DATA_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_definitions(unit_tests PUBLIC -DPOLYSOLVE_DATA_DIR=\"${DATA_DIR}\")


if(POLYSOLVE_WITH_SANITIZERS)
	add_sanitizers(unit_tests)
endif()

# Register tests
set(PARSE_CATCH_TESTS_ADD_TO_CONFIGURE_DEPENDS ON)
include(Catch)
catch_discover_tests(unit_tests)

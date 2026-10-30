################################################################################
# Tests
################################################################################

set(test_sources
	main.cpp
	test_linear_solver.cpp
	test_nonlinear_solver.cpp
	test_json.cpp
)
add_executable(unit_tests ${test_sources})

# Standalone CLI driver: load a Matrix Market matrix/rhs, run any solver from
# a JSON config, and report timing/residual -- for reproducing/benchmarking a
# solve outside of the full polyfem integration.
# Uses POSIX-only APIs (unistd.h, sys/resource.h), so it can't build on Windows.
if(NOT WIN32)
	add_executable(linear_solve linear_solve.cpp)
endif()

################################################################################
# Required Libraries
################################################################################

include(catch2)
target_link_libraries(unit_tests PUBLIC Catch2::Catch2)

target_link_libraries(unit_tests PUBLIC polysolve::polysolve)

include(argparse)
if(TARGET linear_solve)
	target_link_libraries(linear_solve PUBLIC polysolve::polysolve)
	if(TARGET argparse::argparse)
		target_link_libraries(linear_solve PRIVATE argparse::argparse)
	endif()
	if(DEFINED argparse_SOURCE_DIR)
		target_include_directories(linear_solve SYSTEM PRIVATE "${argparse_SOURCE_DIR}/include")
	endif()
endif()

include(polysolve_warnings)
target_link_libraries(unit_tests PRIVATE polysolve::warnings)
if(TARGET linear_solve)
	target_link_libraries(linear_solve PRIVATE polysolve::warnings)
endif()

if(POLYSOLVE_WITH_SANITIZERS)
	add_sanitizers(unit_tests)
	if(TARGET linear_solve)
		add_sanitizers(linear_solve)
	endif()
endif()

if(POLYSOLVE_WITH_AMGCL)
    target_compile_definitions(unit_tests PRIVATE -DPOLYSOLVE_WITH_AMGCL)
endif()

include(polyfem-data)
target_link_libraries(unit_tests PRIVATE polyfem::data)

################################################################################
# Register tests
################################################################################

foreach(source IN ITEMS ${test_sources})
    source_group("tests" FILES "${source}")
endforeach()
source_group("tests" FILES "linear_solve.cpp")

# Register tests
set(PARSE_CATCH_TESTS_ADD_TO_CONFIGURE_DEPENDS ON)
catch_discover_tests(unit_tests)

# Standalone SPMD driver. Every rank enters the
# solve, which is how the hybrid solver is meant to be driven -- the Catch2
# binary drives it from rank 0 only, so it cannot time the multi-rank path.
add_executable(bench_spmd bench_spmd.cpp)
target_link_libraries(bench_spmd PUBLIC polysolve::polysolve)

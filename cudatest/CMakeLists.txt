################################################################################
# Tests
################################################################################
add_executable(cudatest main.cpp cudatest.cu )
target_link_libraries(cudatest PUBLIC polysolve::polysolve)

include(catch2)
target_link_libraries(cudatest PUBLIC Catch2::Catch2)
set(PARSE_CATCH_TESTS_ADD_TO_CONFIGURE_DEPENDS ON)
catch_discover_tests(cudatest)
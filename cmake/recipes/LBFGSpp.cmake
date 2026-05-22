# LBFGSpp (https://github.com/yixuan/LBFGSpp)
# License: MIT

if(TARGET LBFGSpp::LBFGSpp)
    return()
endif()

include(polysolve_optional_dependency)

set(POLYSOLVE_LBFGSPP_CHECK_SOURCE [[
#include <LBFGSpp/BFGSMat.h>
int main()
{
    LBFGSpp::BFGSMat<double> bfgs;
    bfgs.reset(2, 1);
    return 0;
}
]])

polysolve_find_system_dependency(LBFGSPP_SYSTEM_FOUND
    NAME LBFGSpp
    PACKAGE lbfgspp
    TARGET LBFGSpp::LBFGSpp
    CONFIG
    SOURCE_VAR POLYSOLVE_LBFGSPP_CHECK_SOURCE
)
if(LBFGSPP_SYSTEM_FOUND)
    return()
endif()

polysolve_find_system_dependency(LBFGSPP_LOWERCASE_SYSTEM_FOUND
    NAME LBFGSpp
    PACKAGE lbfgspp
    TARGET lbfgspp
    CONFIG
    SOURCE_VAR POLYSOLVE_LBFGSPP_CHECK_SOURCE
)
if(LBFGSPP_LOWERCASE_SYSTEM_FOUND)
    add_library(LBFGSpp::LBFGSpp ALIAS lbfgspp)
    return()
endif()

polysolve_should_fetch_dependency(LBFGSPP_SHOULD_FETCH LBFGSpp)
if(NOT LBFGSPP_SHOULD_FETCH)
    message(FATAL_ERROR "LBFGSpp is required to build PolySolve.")
endif()

message(STATUS "Third-party: creating target 'LBFGSpp::LBFGSpp'")

include(CPM)
CPMAddPackage(
    NAME lbfgspp
    GITHUB_REPOSITORY yixuan/LBFGSpp
    GIT_TAG v0.2.0
    DOWNLOAD_ONLY TRUE
)

add_library(LBFGSpp INTERFACE)
target_include_directories(LBFGSpp INTERFACE "${lbfgspp_SOURCE_DIR}/include")
add_library(LBFGSpp::LBFGSpp ALIAS LBFGSpp)

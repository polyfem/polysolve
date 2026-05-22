# SPQR solver

if(TARGET SuiteSparse::SPQR)
    return()
endif()

include(polysolve_optional_dependency)

set(POLYSOLVE_SPQR_CHECK_SOURCE [[
#include <SuiteSparseQR.hpp>
int main()
{
    cholmod_common common;
    cholmod_l_start(&common);
    SuiteSparseQR_factorization<double> *qr = nullptr;
    SuiteSparseQR_free(&qr, &common);
    cholmod_l_finish(&common);
    return 0;
}
]])

message(STATUS "Third-party: creating targets 'SuiteSparse::SPQR'")

# We do not have a build recipe for this, so find it as a system installed library.
polysolve_find_system_dependency(SPQR_SUITESPARSE_SYSTEM_FOUND
    NAME SPQR
    PACKAGE SuiteSparse
    TARGET SuiteSparse::SPQR
    CONFIG
    SOURCE_VAR POLYSOLVE_SPQR_CHECK_SOURCE
)
if(SPQR_SUITESPARSE_SYSTEM_FOUND)
    return()
endif()

polysolve_find_system_dependency(SPQR_SYSTEM_FOUND
    NAME SPQR
    PACKAGE SPQR
    TARGET SPQR::SPQR
    SOURCE_VAR POLYSOLVE_SPQR_CHECK_SOURCE
)

if(SPQR_SYSTEM_FOUND AND NOT TARGET SuiteSparse::SPQR)
    add_library(SuiteSparse::SPQR INTERFACE IMPORTED GLOBAL)
    target_link_libraries(SuiteSparse::SPQR INTERFACE SPQR::SPQR)
endif()

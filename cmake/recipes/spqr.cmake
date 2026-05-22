# SPQR solver

if(TARGET SuiteSparse::SPQR)
    return()
endif()

message(STATUS "Third-party: creating targets 'SuiteSparse::SPQR'")

# We do not have a build recipe for this, so find it as a system installed library.
find_package(SPQR QUIET)

if(TARGET SPQR::SPQR AND NOT TARGET SuiteSparse::SPQR)
    add_library(SuiteSparse::SPQR INTERFACE IMPORTED GLOBAL)
    target_link_libraries(SuiteSparse::SPQR INTERFACE SPQR::SPQR)
endif()

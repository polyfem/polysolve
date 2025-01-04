# SPQR solver

if(TARGET SparseSuite::SPQR)
    return()
endif()

message(STATUS "Third-party: creating targets 'SuiteSparse::SPQR'")

# We do not have a build recipe for this, so find it as a system installed library.
find_package(SPQR)


# UMFPACK solver

if(TARGET UMFPACK::UMFPACK)
    return()
endif()

message(STATUS "Third-party: creating targets 'UMFPACK::UMFPACK'")

# We do not have a build recipe for this, so find it as a system installed library.
find_package(UMFPACK)

if(UMFPACK_FOUND)
    add_library(UMFPACK_UMFPACK INTERFACE)
    add_library(UMFPACK::UMFPACK ALIAS UMFPACK_UMFPACK)
    target_include_directories(UMFPACK_UMFPACK INTERFACE ${UMFPACK_INCLUDES})
    target_link_libraries(UMFPACK_UMFPACK INTERFACE ${UMFPACK_LIBRARIES})
endif()

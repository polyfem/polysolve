# Pardiso solver

if(TARGET Pardiso::Pardiso)
    return()
endif()

message(STATUS "Third-party: creating targets 'Pardiso::Pardiso'")

# We do not have a build recipe for this, so find it as a system installed library.
find_package(Pardiso)

if(PARDISO_FOUND)

    # find_package(OpenMP)
    # if( OpenMP_CXX_FOUND )
    #   target_link_libraries(polysolve PUBLIC ${OpenMP_CXX_LIBRARIES})
    #   # target_compile_definitions(polysolve PUBLIC ${OpenMP_CXX_FLAGS})
    # else()
    #   message(FATAL_ERROR "unable to find omp")
    # endif()

    add_library(Pardiso_Pardiso INTERFACE)
    add_library(Pardiso::Pardiso ALIAS Pardiso_Pardiso)

    target_link_libraries(Pardiso_Pardiso INTERFACE ${PARDISO_LIBRARIES})

    find_package(LAPACK)
    if(LAPACK_FOUND)
        target_link_libraries(Pardiso_Pardiso INTERFACE ${LAPACK_LIBRARIES})
    else()
        message(FATAL_ERROR "unable to find lapack")
    endif()
endif()

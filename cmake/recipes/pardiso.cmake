# Pardiso solver

include(polysolve_optional_dependency)

if(TARGET Pardiso::Pardiso)
    return()
endif()
if(TARGET Pardiso_Pardiso)
    return()
endif()

message(STATUS "Third-party: creating targets 'Pardiso::Pardiso'")

# We do not have a build recipe for this, so find it as a system installed library.
find_package(Pardiso QUIET)

if(Pardiso_FOUND OR PARDISO_FOUND)
    find_package(LAPACK QUIET)
    if(NOT LAPACK_FOUND)
        message(WARNING "Pardiso found but LAPACK was not found.")
        return()
    endif()

    # find_package(OpenMP)
    # if( OpenMP_CXX_FOUND )
    #   target_link_libraries(polysolve PUBLIC ${OpenMP_CXX_LIBRARIES})
    #   # target_compile_definitions(polysolve PUBLIC ${OpenMP_CXX_FLAGS})
    # else()
    #   message(FATAL_ERROR "unable to find omp")
    # endif()

    add_library(Pardiso_Pardiso INTERFACE IMPORTED GLOBAL)
    target_link_libraries(Pardiso_Pardiso INTERFACE ${PARDISO_LIBRARIES} ${LAPACK_LIBRARIES})

    set(POLYSOLVE_PARDISO_CHECK_SOURCE [[
extern "C" void pardisoinit(void *, int *, int *, int *, double *, int *);
int main()
{
    auto *symbol = &pardisoinit;
    return symbol == nullptr ? 1 : 0;
}
]])
    polysolve_check_linkable_target(POLYSOLVE_PARDISO_LINKABLE
        NAME Pardiso
        TARGET Pardiso_Pardiso
        SOURCE_VAR POLYSOLVE_PARDISO_CHECK_SOURCE
    )

    if(NOT POLYSOLVE_PARDISO_LINKABLE)
        polysolve_note_disabled_dependency("Pardiso" "${POLYSOLVE_PARDISO_LINKABLE_REASON}")
        return()
    endif()

    add_library(Pardiso::Pardiso ALIAS Pardiso_Pardiso)
endif()

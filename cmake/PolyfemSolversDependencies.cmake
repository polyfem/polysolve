# Prepare dependencies
#
# For each third-party library, if the appropriate target doesn't exist yet,
# download it via external project, and add_subdirectory to build it alongside
# this project.

### Configuration
set(POLYSOLVE_ROOT     "${CMAKE_CURRENT_LIST_DIR}/..")
set(POLYSOLVE_EXTERNAL ${THIRD_PARTY_DIR})

# Download and update 3rdparty libraries
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR})
list(REMOVE_DUPLICATES CMAKE_MODULE_PATH)
include(PolyfemSolversDownloadExternal)

################################################################################
# Required libraries
################################################################################

# Sanitizers
if(POLYSOLVE_WITH_SANITIZERS)
    polyfem_solvers_download_sanitizers()
	find_package(Sanitizers)

    add_sanitizers(polysolve)
endif()



################################################################################
# Required libraries
################################################################################

# Eigen
# Uncomment to use the system's version of Eigen (e.g. to use Eigen 3.3)
if(NOT TARGET Eigen3::Eigen)
    polyfem_solvers_download_eigen()
    add_library(eigen INTERFACE)
    target_include_directories(eigen SYSTEM INTERFACE
        $<BUILD_INTERFACE:${POLYSOLVE_EXTERNAL}/eigen>
        $<INSTALL_INTERFACE:include>
    )
    set_property(TARGET eigen PROPERTY EXPORT_NAME Eigen3::Eigen)
    add_library(Eigen3::Eigen ALIAS eigen)
endif()
target_link_libraries(polysolve PUBLIC Eigen3::Eigen)



# Catch2
if(${POLYSOLVE_TOPLEVEL_PROJECT})
    polyfem_solvers_download_catch2()
    add_library(catch INTERFACE)
    target_include_directories(catch SYSTEM INTERFACE ${THIRD_PARTY_DIR}/Catch2/single_include/catch2)
endif()

# Hypre
if(POLYSOLVE_WITH_HYPRE)
    polyfem_solvers_download_hypre()
    include(hypre)
    target_link_libraries(polysolve PUBLIC HYPRE)
    target_compile_definitions(polysolve PUBLIC -DPOLYSOLVE_WITH_HYPRE)
endif()

# Json
polyfem_solvers_download_json()
add_library(json INTERFACE)
target_include_directories(json SYSTEM INTERFACE ${THIRD_PARTY_DIR}/json/include)
target_link_libraries(polysolve PUBLIC json)


################################################################################
# Optional libraries
################################################################################

# Cholmod solver
if(POLYSOLVE_WITH_CHOLMOD)
    find_package(Cholmod)
    if(CHOLMOD_FOUND)
        target_include_directories(polysolve PUBLIC ${CHOLMOD_INCLUDES})
        target_link_libraries(polysolve PUBLIC ${CHOLMOD_LIBRARIES})
        target_compile_definitions(polysolve PUBLIC -DPOLYSOLVE_WITH_CHOLMOD)
    endif()
endif()

# MKL library
if(POLYSOLVE_WITH_MKL)
    if ("$ENV{MKLROOT}" STREQUAL "")
        message(WARNING "MKLROOT is not set. Please source the Intel MKL mklvars.sh file.")
    endif()

    # user defined options for MKL
    option(MKL_USE_parallel "Use MKL parallel" ON)
    option(MKL_USE_sdl "Single Dynamic Library or static/dynamic" OFF)
    set(MKL_USE_interface "lp64" CACHE STRING "for Intel(R)64 compatible arch: ilp64/lp64 or for ia32 arch: cdecl/stdcall")


    SET(INTEL_COMPILER_DIR $ENV{MKLROOT}/..)

    find_package(MKL)

    if(MKL_FOUND)
        target_include_directories(polysolve PUBLIC ${MKL_INCLUDE_DIR})
        target_link_libraries(polysolve PUBLIC ${MKL_LIBRARIES})
        target_compile_definitions(polysolve PUBLIC -DPOLYSOLVE_WITH_MKL)
        target_compile_definitions(polysolve PUBLIC -DEIGEN_USE_MKL_ALL)
        target_compile_definitions(polysolve PUBLIC -DPOLYSOLVE_WITH_PARDISO)
    else()
        MESSAGE(WARNING "unable to find MKL")
    endif()
endif()

# Pardiso solver
if(POLYSOLVE_WITH_PARDISO AND NOT (POLYSOLVE_WITH_MKL AND MKL_FOUND))
    find_package(Pardiso)
    if(PARDISO_FOUND)
        find_package(LAPACK)
        if(LAPACK_FOUND)
            target_link_libraries(polysolve PUBLIC ${LAPACK_LIBRARIES})
        else()
            message(FATAL_ERROR "unable to find lapack")
        endif()

        # find_package(OpenMP)
        # if( OpenMP_CXX_FOUND )
        #   target_link_libraries(polysolve PUBLIC ${OpenMP_CXX_LIBRARIES})
        #   # target_compile_definitions(polysolve PUBLIC ${OpenMP_CXX_FLAGS})
        # else()
        #   message(FATAL_ERROR "unable to find omp")
        # endif()

        target_link_libraries(polysolve PUBLIC ${PARDISO_LIBRARIES})
        target_compile_definitions(polysolve PUBLIC -DPOLYSOLVE_WITH_PARDISO)
    else()
        message(WARNING "Pardiso not found, solver will not be available.")
    endif()
endif()

# UmfPack solver
if(POLYSOLVE_WITH_UMFPACK)
    find_package(Umfpack)
    if(UMFPACK_FOUND)
        target_include_directories(polysolve PUBLIC ${UMFPACK_INCLUDES})
        target_link_libraries(polysolve PUBLIC ${UMFPACK_LIBRARIES})
        target_compile_definitions(polysolve PUBLIC -DPOLYSOLVE_WITH_UMFPACK)
    endif()
endif()

# SuperLU solver
if(POLYSOLVE_WITH_SUPERLU AND NOT (POLYSOLVE_WITH_MKL AND MKL_FOUND))
    find_package(SuperLU)
    if(SUPERLU_FOUND)
        target_include_directories(polysolve PUBLIC ${SUPERLU_INCLUDES})
        target_link_libraries(polysolve PUBLIC ${SUPERLU_LIBRARIES})
        target_compile_definitions(polysolve PUBLIC ${SUPERLU_DEFINES})
        target_compile_definitions(polysolve PUBLIC -DPOLYSOLVE_WITH_SUPERLU)
    endif()
endif()


# amgcl solver
if(POLYSOLVE_WITH_AMGCL)
    find_package(Boost COMPONENTS
    program_options
    serialization
    unit_test_framework
    )
    if(Boost_FOUND)
        polyfem_solvers_download_amgcl()
        add_subdirectory(${THIRD_PARTY_DIR}/amgcl amgcl)
        target_link_libraries(polysolve PUBLIC amgcl::amgcl)
        target_compile_definitions(polysolve PUBLIC -DPOLYSOLVE_WITH_AMGCL)
    else()
        MESSAGE(WARNING "Boost not found, AMGCL requires boost thus it has beend disabled")
    endif()
endif()


# Spectra
if(POLYSOLVE_WITH_SPECTRA)
    polyfem_solvers_download_spectra()
    add_library(spectra INTERFACE)
    target_include_directories(spectra SYSTEM INTERFACE ${THIRD_PARTY_DIR}/spectra/include)
    target_link_libraries(polysolve PUBLIC spectra)
    target_compile_definitions(polysolve PUBLIC -DPOLYSOLVE_WITH_SPECTRA)
endif()
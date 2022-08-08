# Detects whether this is a top-level project
get_directory_property(HAS_PARENT PARENT_DIRECTORY)
if(HAS_PARENT)
    set(POLYSOLVE_TOPLEVEL_PROJECT OFF)
else()
    set(POLYSOLVE_TOPLEVEL_PROJECT ON)
endif()

# Check required CMake version
set(REQUIRED_CMAKE_VERSION "3.14.0")
if(POLYSOLVE_TOPLEVEL_PROJECT)
    cmake_minimum_required(VERSION ${REQUIRED_CMAKE_VERSION})
else()
    # Don't use cmake_minimum_required here to avoid implicitly overriding parent policies
    if(${CMAKE_VERSION} VERSION_LESS ${REQUIRED_CMAKE_VERSION})
        message(FATAL_ERROR "CMake required version to build PolySolve is ${REQUIRED_CMAKE_VERSION}")
    endif()
endif()

# Include user-provided default options if available. We do that before the main
# `project()` so that we can define the C/C++ compilers from the option file.
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/PolySolveOptions.cmake)
    message(STATUS "Using local options file: ${CMAKE_CURRENT_SOURCE_DIR}/PolySolveOptions.cmake")
    include(${CMAKE_CURRENT_SOURCE_DIR}/PolySolveOptions.cmake)
endif()

################################################################################

project(PolySolve
        DESCRIPTION "Easy-to-use wrapper for linear solver"
        LANGUAGES CXX)

# Polysolve options
option(POLYSOLVE_WITH_SANITIZERS    "Enable sanitizers in compilation targets"   OFF)
# Polysolve options for enabling/disabling optional libraries
option(POLYSOLVE_WITH_CHOLMOD       "Enable Cholmod library"                      ON)
option(POLYSOLVE_WITH_UMFPACK       "Enable UmfPack library"                      ON)
option(POLYSOLVE_WITH_SUPERLU       "Enable SuperLU library"                      ON)
option(POLYSOLVE_WITH_MKL           "Enable MKL library"                          ON)

option(POLYSOLVE_WITH_PARDISO       "Enable Pardiso library"                     OFF)
option(POLYSOLVE_WITH_HYPRE         "Enable hypre"                                ON)
option(POLYSOLVE_WITH_AMGCL         "Use AMGCL"                                   ON)
option(POLYSOLVE_WITH_SPECTRA       "Enable computing spectrum"                   ON)
# Sanitizer options
option(POLYSOLVE_SANITIZE_ADDRESS   "Sanitize Address"                           OFF)
option(POLYSOLVE_SANITIZE_MEMORY    "Sanitize Memory"                            OFF)
option(POLYSOLVE_SANITIZE_THREAD    "Sanitize Thread"                            OFF)
option(POLYSOLVE_SANITIZE_UNDEFINED "Sanitize Undefined"                         OFF)
# Misc.
option(POLYSOLVE_LARGE_INDEX        "Build for large indices"                    OFF)
option(POLYSOLVE_WITH_TESTS         "Build unit-tests" ${POLYSOLVE_TOPLEVEL_PROJECT})

include(CMakeDependentOption)
cmake_dependent_option(SUITE_SPARSE_WITH_MKL "Build SuiteSparse using MKL" ON "POLYSOLVE_WITH_MKL" OFF)
cmake_dependent_option(EIGEN_WITH_MKL "Use Eigen with MKL" ON "POLYSOLVE_WITH_MKL" OFF)

# Set default minimum C++ standard
if(POLYSOLVE_TOPLEVEL_PROJECT)
    set(CMAKE_CXX_STANDARD 14)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)
endif()

if (MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /bigobj")
endif()

### Configuration
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/polysolve/")
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/recipes/")
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/find/")

# Color output
include(polysolve_use_colors)

# IPC Toolkit utils
include(polysolve_prepend_current_path)
include(polysolve_set_source_group)

# Sort projects inside the solution
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Generate position independent code by default
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

################################################################################
# PolySolve Library
################################################################################

# Add an empty library and fill in the list of sources in `src/CMakeLists.txt`.
add_library(polysolve)
add_library(polysolve::polysolve ALIAS polysolve)

add_subdirectory(src)

# Public include directory for Polysolve
target_include_directories(polysolve PUBLIC ${PROJECT_SOURCE_DIR}/src)

################################################################################
# Definitions
################################################################################

if(POLYSOLVE_LARGE_INDEX)
    target_compile_definitions(polysolve PUBLIC -DPOLYSOLVE_LARGE_INDEX)
endif()

# No limit yay
target_compile_definitions(polysolve PUBLIC -DEIGEN_STACK_ALLOCATION_LIMIT=0)

# 8MB
# target_compile_definitions(polysolve PUBLIC -DEIGEN_STACK_ALLOCATION_LIMIT=8388608)

################################################################################
# Dependencies
################################################################################

# Extra warnings
include(polysolve_warnings)
target_link_libraries(polysolve PRIVATE polysolve::warnings)

# Sanitizers
if(POLYSOLVE_WITH_SANITIZERS)
    include(sanitizers)
    add_sanitizers(polysolve)
endif()

include(eigen)
target_link_libraries(polysolve PUBLIC Eigen3::Eigen)

# Hypre (GNU Lesser General Public License)
if(POLYSOLVE_WITH_HYPRE)
    include(hypre)
    target_link_libraries(polysolve PUBLIC HYPRE::HYPRE)
    target_compile_definitions(polysolve PUBLIC -DPOLYSOLVE_WITH_HYPRE)
    if(HYPRE_WITH_MPI)
        target_compile_definitions(polysolve PUBLIC -DHYPRE_WITH_MPI)
    endif()
endif()

# Json (MIT)
include(json)
target_link_libraries(polysolve PUBLIC nlohmann_json::nlohmann_json)

# CHOLMOD solver
if(POLYSOLVE_WITH_CHOLMOD)
    include(suitesparse)
    target_link_libraries(polysolve PUBLIC SuiteSparse::CHOLMOD)
    target_compile_definitions(polysolve PUBLIC -DPOLYSOLVE_WITH_CHOLMOD)
endif()

# MKL library
if(POLYSOLVE_WITH_MKL)
    # include(eigen) will add mkl for us
    target_compile_definitions(polysolve PUBLIC -DPOLYSOLVE_WITH_MKL)
endif()

# Pardiso solver
if(POLYSOLVE_WITH_PARDISO)
    include(pardiso)
    if(TARGET Pardiso::Pardiso)
        target_link_libraries(polysolve PUBLIC Pardiso::Pardiso)
        target_compile_definitions(polysolve PUBLIC -DPOLYSOLVE_WITH_PARDISO)
    else()
        message(WARNING "Pardiso not found, solver will not be available.")
    endif()
endif()

# UmfPack solver
if(POLYSOLVE_WITH_UMFPACK)
    include(umfpack)
    if(TARGET UMFPACK::UMFPACK)
        target_link_libraries(polysolve PUBLIC UMFPACK::UMFPACK)
        target_compile_definitions(polysolve PUBLIC -DPOLYSOLVE_WITH_UMFPACK)
    else()
        message(WARNING "UMFPACK not found, solver will not be available.")
    endif()
endif()

# SuperLU solver
if(POLYSOLVE_WITH_SUPERLU)
    include(superlu)
    if(TARGET SuperLU::SuperLU)
        target_link_libraries(polysolve PUBLIC SuperLU::SuperLU)
        target_compile_definitions(polysolve PUBLIC -DPOLYSOLVE_WITH_SUPERLU)
    else()
        message(WARNING "SuperLU not found, solver will not be available.")
    endif()
endif()


# AMGCL solver
if(POLYSOLVE_WITH_AMGCL)
    include(amgcl)
    target_link_libraries(polysolve PUBLIC amgcl::amgcl)
    target_compile_definitions(polysolve PUBLIC -DPOLYSOLVE_WITH_AMGCL)
endif()

# Spectra (MPL 2.0)
if(POLYSOLVE_WITH_SPECTRA)
    include(spectra)
    target_link_libraries(polysolve PUBLIC spectra::spectra)
    target_compile_definitions(polysolve PUBLIC -DPOLYSOLVE_WITH_SPECTRA)
endif()

################################################################################
# Compiler options
################################################################################

# Use C++14
target_compile_features(polysolve PUBLIC cxx_std_14)

################################################################################
# Tests
################################################################################

# Compile extras only if this is a top-level project
if(POLYSOLVE_WITH_TESTS)
    # Unit tests
    include(CTest)
    enable_testing()

    # Include Catch2 and provide function `catch_discover_tests` to register tests.
    include(catch2)
    FetchContent_GetProperties(catch2)
    include("${catch2_SOURCE_DIR}/contrib/Catch.cmake")

    add_subdirectory(tests)
endif()

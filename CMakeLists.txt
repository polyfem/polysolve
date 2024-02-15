# Detects whether this is a top-level project
get_directory_property(HAS_PARENT PARENT_DIRECTORY)
if(HAS_PARENT)
    set(POLYSOLVE_TOPLEVEL_PROJECT OFF)
else()
    set(POLYSOLVE_TOPLEVEL_PROJECT ON)
endif()

# Check required CMake version
set(REQUIRED_CMAKE_VERSION "3.18.0")
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

# Enable ccache if available
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    option(POLYSOLVE_WITH_CCACHE "Enable ccache when building Polysolve" ${POLYSOLVE_TOPLEVEL_PROJECT})
else()
    option(POLYSOLVE_WITH_CCACHE "Enable ccache when building Polysolve" OFF)
endif()
if(POLYSOLVE_WITH_CCACHE AND CCACHE_PROGRAM)
    set(ccacheEnv
        CCACHE_BASEDIR=${CMAKE_BINARY_DIR}
        CCACHE_SLOPPINESS=clang_index_store,include_file_ctime,include_file_mtime,locale,pch_defines,time_macros
    )
    foreach(lang IN ITEMS C CXX)
        set(CMAKE_${lang}_COMPILER_LAUNCHER
            ${CMAKE_COMMAND} -E env ${ccacheEnv} ${CCACHE_PROGRAM}
        )
    endforeach()
endif()

################################################################################
# CMake Policies
################################################################################

cmake_policy(SET CMP0054 NEW) # Only interpret if() arguments as variables or keywords when unquoted.
cmake_policy(SET CMP0076 NEW) # target_sources() command converts relative paths to absolute.
set(CMAKE_POLICY_DEFAULT_CMP0091 NEW) # MSVC runtime library flags are selected by an abstraction.
set(CMAKE_POLICY_DEFAULT_CMP0135 NEW) # Set the timestamps of all extracted contents to the time of the extraction.

################################################################################

project(PolySolve
        DESCRIPTION "Easy-to-use wrapper for linear solver"
        LANGUAGES CXX)

if(CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "arm64" AND APPLE)
    set(POLYSOLVE_NOT_ON_APPLE_SILICON OFF)
    set(POLYSOLVE_ON_APPLE_SILICON ON)
else()
    set(POLYSOLVE_NOT_ON_APPLE_SILICON ON)
    set(POLYSOLVE_ON_APPLE_SILICON OFF)
endif()

# Polysolve options
option(POLYSOLVE_WITH_SANITIZERS    "Enable sanitizers in compilation targets"          OFF)

# Polysolve options for enabling/disabling optional libraries
option(POLYSOLVE_WITH_ACCELERATE    "Enable Apple Accelerate" ${POLYSOLVE_ON_APPLE_SILICON})
option(POLYSOLVE_WITH_CHOLMOD       "Enable Cholmod library"                             ON)
option(POLYSOLVE_WITH_UMFPACK       "Enable UmfPack library"                             ON)
option(POLYSOLVE_WITH_SUPERLU       "Enable SuperLU library"                             ON)
option(POLYSOLVE_WITH_MKL           "Enable MKL library"  ${POLYSOLVE_NOT_ON_APPLE_SILICON})
option(POLYSOLVE_WITH_CUSOLVER      "Enable cuSOLVER library"                           OFF)
option(POLYSOLVE_WITH_PARDISO       "Enable Pardiso library"                            OFF)
option(POLYSOLVE_WITH_HYPRE         "Enable hypre"                                       ON)
option(POLYSOLVE_WITH_AMGCL         "Use AMGCL"                                          ON)
option(POLYSOLVE_WITH_SPECTRA       "Enable Spectra library"                             ON)

# Sanitizer options
option(POLYSOLVE_SANITIZE_ADDRESS   "Sanitize Address"                                  OFF)
option(POLYSOLVE_SANITIZE_MEMORY    "Sanitize Memory"                                   OFF)
option(POLYSOLVE_SANITIZE_THREAD    "Sanitize Thread"                                   OFF)
option(POLYSOLVE_SANITIZE_UNDEFINED "Sanitize Undefined"                                OFF)

# Misc.
option(POLYSOLVE_LARGE_INDEX        "Build for large indices"                           OFF)
option(POLYSOLVE_WITH_TESTS         "Build unit-tests"        ${POLYSOLVE_TOPLEVEL_PROJECT})

include(CMakeDependentOption)
cmake_dependent_option(EIGEN_WITH_MKL "Use Eigen with MKL" ON "POLYSOLVE_WITH_MKL" OFF)

option(POLYSOLVE_CODE_COVERAGE "Enable coverage reporting" OFF)

add_library(polysolve_coverage_config INTERFACE)
if(POLYSOLVE_CODE_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  # Add required flags (GCC & LLVM/Clang)
  target_compile_options(polysolve_coverage_config INTERFACE
    -g         # generate debug info
    --coverage # sets all required flags
  )
  target_link_options(polysolve_coverage_config INTERFACE --coverage)
endif()

# Set default minimum C++ standard
if(POLYSOLVE_TOPLEVEL_PROJECT)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)
endif()

if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /bigobj")
endif()

### Configuration
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/polysolve/")
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/recipes/")
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/find/")

# General CMake utils
include(polysolve_cpm_cache)
include(polysolve_use_colors)

# Sort projects inside the solution
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Generate position independent code by default
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Since MKL comes precompiled with /MD on Windows, we need to use the same MSVC runtime library flag
# globally for the whole project (otherwise in Debug we will have a mismatch between /MD and /MDd).
if(POLYSOLVE_WITH_MKL)
    # Set MSVC runtime library globally for all targets
    message(STATUS "Forcing /MD globally due MKL being enabled")
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL" CACHE STRING "Select the MSVC runtime library")
endif()

################################################################################
# PolySolve Library
################################################################################

# Add an empty library and fill in the list of sources in `src/CMakeLists.txt`.
add_library(polysolve_linear)
add_library(polysolve::linear ALIAS polysolve_linear)
add_subdirectory(src/polysolve)
add_subdirectory(src/polysolve/linear)

add_library(polysolve)
add_library(polysolve::polysolve ALIAS polysolve)
add_subdirectory(src/polysolve/nonlinear)

target_link_libraries(polysolve_linear PUBLIC polysolve_coverage_config)
target_link_libraries(polysolve PUBLIC polysolve_coverage_config)

target_compile_features(polysolve_linear PUBLIC cxx_std_17)
target_compile_features(polysolve PUBLIC cxx_std_17)

# Public include directory for Polysolve
target_include_directories(polysolve_linear PUBLIC ${PROJECT_SOURCE_DIR}/src)
target_include_directories(polysolve PUBLIC ${PROJECT_SOURCE_DIR}/src)

################################################################################
# Definitions
################################################################################

if(POLYSOLVE_LARGE_INDEX)
    target_compile_definitions(polysolve_linear PUBLIC POLYSOLVE_LARGE_INDEX)
endif()

target_compile_definitions(polysolve_linear PRIVATE POLYSOLVE_LINEAR_SPEC="${PROJECT_SOURCE_DIR}/linear-solver-spec.json")
target_compile_definitions(polysolve PRIVATE POLYSOLVE_NON_LINEAR_SPEC="${PROJECT_SOURCE_DIR}/nonlinear-solver-spec.json")
target_compile_definitions(polysolve_linear PUBLIC POLYSOLVE_JSON_SPEC_DIR="${PROJECT_SOURCE_DIR}")


################################################################################
# Dependencies
################################################################################

# ------
# Linear
# ------

# Accelerate solver (Include before Eigen)
if(POLYSOLVE_WITH_ACCELERATE)
    include(CPM)
    CPMAddPackage(
        NAME eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 969c31eefcdfaab11da763bea3f7502086673ab0
        DOWNLOAD_ONLY ON
    )
    set(BLA_VENDOR Apple)
    find_package(BLAS REQUIRED)
    find_package(LAPACK REQUIRED)
    target_link_libraries(polysolve_linear PRIVATE BLAS::BLAS LAPACK::LAPACK)
    target_compile_definitions(polysolve_linear PUBLIC POLYSOLVE_WITH_ACCELERATE)
endif()

include(eigen)
target_link_libraries(polysolve_linear PUBLIC Eigen3::Eigen)

# spdlog
include(spdlog)
target_link_libraries(polysolve_linear PUBLIC spdlog::spdlog)

# JSON (MIT)
include(json)
target_link_libraries(polysolve_linear PUBLIC nlohmann_json::nlohmann_json)

# JSON Specification Engine library
include(jse)
target_link_libraries(polysolve_linear PUBLIC jse::jse)

# Hypre (GNU Lesser General Public License)
if(POLYSOLVE_WITH_HYPRE)
    include(hypre)
    target_link_libraries(polysolve_linear PUBLIC HYPRE::HYPRE)
    target_compile_definitions(polysolve_linear PUBLIC POLYSOLVE_WITH_HYPRE)
    if(HYPRE_WITH_MPI)
        target_compile_definitions(polysolve_linear PUBLIC HYPRE_WITH_MPI)
    endif()
endif()

# CHOLMOD solver
if(POLYSOLVE_WITH_CHOLMOD)
    include(suitesparse)
    target_link_libraries(polysolve_linear PRIVATE SuiteSparse::CHOLMOD)
    target_compile_definitions(polysolve_linear PUBLIC POLYSOLVE_WITH_CHOLMOD)
endif()

# MKL library
if(POLYSOLVE_WITH_MKL)
    include(mkl)
    target_link_libraries(polysolve_linear PRIVATE mkl::mkl)
    target_compile_definitions(polysolve_linear PUBLIC POLYSOLVE_WITH_MKL)
endif()

# Pardiso solver
if(POLYSOLVE_WITH_PARDISO)
    include(pardiso)
    if(TARGET Pardiso::Pardiso)
        target_link_libraries(polysolve_linear PRIVATE Pardiso::Pardiso)
        target_compile_definitions(polysolve_linear PUBLIC POLYSOLVE_WITH_PARDISO)
    else()
        message(WARNING "Pardiso not found, solver will not be available.")
    endif()
endif()

# UmfPack solver
if(POLYSOLVE_WITH_UMFPACK)
    include(suitesparse)
    target_link_libraries(polysolve_linear PRIVATE SuiteSparse::UMFPACK)
    target_compile_definitions(polysolve_linear PUBLIC POLYSOLVE_WITH_UMFPACK)
endif()

# SuperLU solver
if(POLYSOLVE_WITH_SUPERLU)
    include(superlu)
    if(TARGET SuperLU::SuperLU)
        target_link_libraries(polysolve_linear PRIVATE SuperLU::SuperLU)
        target_compile_definitions(polysolve_linear PUBLIC POLYSOLVE_WITH_SUPERLU)
    else()
        message(WARNING "SuperLU not found, solver will not be available.")
    endif()
endif()

# AMGCL solver
if(POLYSOLVE_WITH_AMGCL)
    include(amgcl)
    target_link_libraries(polysolve_linear PUBLIC amgcl::amgcl)
    target_compile_definitions(polysolve_linear PUBLIC POLYSOLVE_WITH_AMGCL)
endif()

# Spectra (MPL 2.0)
if(POLYSOLVE_WITH_SPECTRA)
    include(spectra)
    target_link_libraries(polysolve_linear PUBLIC Spectra::Spectra)
    target_compile_definitions(polysolve_linear PUBLIC POLYSOLVE_WITH_SPECTRA)
endif()

# cuSolver solvers
if(POLYSOLVE_WITH_CUSOLVER)
    include(cusolverdn)
    if(TARGET CUDA::cusolver)
        target_link_libraries(polysolve_linear PUBLIC CUDA::cusolver)
        target_compile_definitions(polysolve_linear PUBLIC POLYSOLVE_WITH_CUSOLVER)
    else()
        message(WARNING "cuSOLVER not found, solver will not be available.")
    endif()
endif()

# Sanitizers
if(POLYSOLVE_WITH_SANITIZERS)
    include(sanitizers)
    add_sanitizers(polysolve_linear)
endif()

# Extra warnings (include last for highest priority)
include(polysolve_warnings)
target_link_libraries(polysolve_linear PRIVATE polysolve::warnings)

# ---------
# Nonlinear
# ---------

# polysolve::linear
target_link_libraries(polysolve PUBLIC polysolve::linear)

# LBFGSpp
include(LBFGSpp)
target_link_libraries(polysolve PRIVATE LBFGSpp::LBFGSpp)

# finite-diff (include this after eigen)
include(finite-diff)
target_link_libraries(polysolve PRIVATE finitediff::finitediff)

# Sanitizers
if(POLYSOLVE_WITH_SANITIZERS)
    include(sanitizers)
    add_sanitizers(polysolve)
endif()

# Extra warnings (include last for highest priority)
include(polysolve_warnings)
target_link_libraries(polysolve PRIVATE polysolve::warnings)


################################################################################
# Compiler options
################################################################################


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

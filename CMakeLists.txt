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

# ###############################################################################
# CMake Policies
# ###############################################################################
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
    cmake_policy(SET CMP0135 NEW) # https://cmake.org/cmake/help/latest/policy/CMP0135.html
endif()

# ###############################################################################
project(PolySolve
    DESCRIPTION "Easy-to-use wrapper for linear solver"
    LANGUAGES CXX)

# Polysolve options
option(POLYSOLVE_WITH_SANITIZERS "Enable sanitizers in compilation targets" OFF)

# Polysolve options for enabling/disabling optional libraries
option(POLYSOLVE_WITH_CHOLMOD "Enable Cholmod library" ON)
option(POLYSOLVE_WITH_UMFPACK "Enable UmfPack library" ON)
option(POLYSOLVE_WITH_SUPERLU "Enable SuperLU library" ON)
option(POLYSOLVE_WITH_MKL "Enable MKL library" ON)

option(POLYSOLVE_WITH_CUSOLVER "Enable cuSOLVER library" ON)

option(POLYSOLVE_WITH_CUDA "Enable CUDA" ON)
option(POLYSOLVE_WITH_PETSC "Enable PETSC" ON)
option(POLYSOLVE_WITH_PARDISO "Enable Pardiso library" ON)
option(POLYSOLVE_WITH_HYPRE "Enable hypre" OFF)
option(HYPRE_WITH_MPI "Enable hypre MPI" OFF)
option(POLYSOLVE_WITH_AMGCL "Use AMGCL" ON)
option(POLYSOLVE_WITH_SPECTRA "Enable computing spectrum" ON)

# Sanitizer options
option(POLYSOLVE_SANITIZE_ADDRESS "Sanitize Address" OFF)
option(POLYSOLVE_SANITIZE_MEMORY "Sanitize Memory" OFF)
option(POLYSOLVE_SANITIZE_THREAD "Sanitize Thread" OFF)
option(POLYSOLVE_SANITIZE_UNDEFINED "Sanitize Undefined" OFF)

# Misc.
option(POLYSOLVE_LARGE_INDEX "Build for large indices" OFF)
option(POLYSOLVE_WITH_TESTS "Build unit-tests" ${POLYSOLVE_TOPLEVEL_PROJECT})

include(CMakeDependentOption)
cmake_dependent_option(SUITE_SPARSE_WITH_MKL "Build SuiteSparse using MKL" ON "POLYSOLVE_WITH_MKL" OFF)
cmake_dependent_option(EIGEN_WITH_MKL "Use Eigen with MKL" ON "POLYSOLVE_WITH_MKL" OFF)

# Set default minimum C++ standard
if(POLYSOLVE_TOPLEVEL_PROJECT)
    set(CMAKE_CXX_STANDARD 14)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)
endif()

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /bigobj")
endif()

# ## Configuration
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/polysolve/")
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/recipes/")
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/find/")

# Color output
include(polysolve_use_colors)

# IPC Toolkit utils
include(polysolve_prepend_current_path)
include(polysolve_set_source_group)
include(polysolve_target_link_system_libraries)

# Sort projects inside the solution
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Generate position independent code by default
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ###############################################################################
# PolySolve Library
# ###############################################################################

# Add an empty library and fill in the list of sources in `src/CMakeLists.txt`.
add_library(polysolve)
add_library(polysolve::polysolve ALIAS polysolve)

add_subdirectory(src)

# Public include directory for Polysolve
target_include_directories(polysolve PUBLIC ${PROJECT_SOURCE_DIR}/src)

# ###############################################################################
# Definitions
# ###############################################################################
if(POLYSOLVE_LARGE_INDEX)
    target_compile_definitions(polysolve PUBLIC -DPOLYSOLVE_LARGE_INDEX)
endif()

# No limit yay
target_compile_definitions(polysolve PUBLIC -DEIGEN_STACK_ALLOCATION_LIMIT=0)

# 8MB
# target_compile_definitions(polysolve PUBLIC -DEIGEN_STACK_ALLOCATION_LIMIT=8388608)

# ###############################################################################
# Dependencies
# ###############################################################################

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

# CUDA
if(POLYSOLVE_WITH_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    set(CMAKE_CUDA_ARCHITECTURES "61")
    find_package(CUDAToolkit)

    if(CUDAToolkit_FOUND)
        message(STATUS "Found CUDAToolkit")
        set(CMAKE_CUDA_COMPILER ${CUDAToolkit_NVCC_EXECUTABLE})
        enable_language(CUDA)
        target_compile_definitions(polysolve PUBLIC -DPOLYSOLVE_WITH_CUDA)
    else()
        message(FATAL_ERROR "No CUDA support found!")
    endif()

    # We need to explicitly state that we need all CUDA files in the particle
    # library to be built with -dc as the member functions could be called by
    # other libraries and executables.
    set_target_properties(polysolve PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

    # Nvidia RTX8000 -> compute_75
    # Nvidia V100 -> compute_70
    # Nvidia 1080/1080Ti -> compute_61
    # Nvidia 3080Ti -> compute_86
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 60 70 75 86)
    endif()

    set_target_properties(polysolve PROPERTIES CUDA_ARCHITECTURES "60;70;75;86")

    if(APPLE)
        # We need to add the path to the driver (libcuda.dylib) as an rpath,
        # so that the static cuda runtime can find it at runtime.
        set_property(TARGET polysolve
            PROPERTY
            BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
    endif()

    polysolve_target_link_system_libraries(polysolve PRIVATE CUDA::cudart)
    polysolve_target_link_system_libraries(polysolve PRIVATE CUDA::cusparse)
endif()

# Hypre (GNU Lesser General Public License)
if(POLYSOLVE_WITH_HYPRE)
    include(hypre)
    target_link_libraries(polysolve PUBLIC HYPRE::HYPRE)
    target_compile_definitions(polysolve PUBLIC -DPOLYSOLVE_WITH_HYPRE)

    if(HYPRE_WITH_MPI)
        target_compile_definitions(polysolve PUBLIC HYPRE_WITH_MPI)
        message(STATUS "HYPRE WITH MPI.")
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

# ###############################################################################

# cuSolver solvers
if(POLYSOLVE_WITH_CUSOLVER)
    include(cusolverdn)

    if(TARGET CUDA::cusolver)
        target_link_libraries(polysolve PUBLIC CUDA::cusolver)
        target_compile_definitions(polysolve PUBLIC -DPOLYSOLVE_WITH_CUSOLVER)
        message(STATUS "cuSOLVER found!")
    else()
        message(WARNING "cuSOLVER not found, solver will not be available.")
    endif()
endif()

# ###############################################################################

# PETSC solvers
if(POLYSOLVE_WITH_PETSC)
    # include(PETSC PETSc)
    find_package(MPI)

    # TODO: CHECK IF PETSC IS AVAILABLE
    # if(TARGET PETSC::PETSc)
    target_link_libraries(polysolve PUBLIC -lpetsc MPI::MPI_CXX)
    target_compile_definitions(polysolve PUBLIC -DPOLYSOLVE_WITH_PETSC)
    message(STATUS "PETSc found!")

    # else()
    # message(WARNING "PETSc not found, solver will not be available.")
    # endif()
endif()

# ###############################################################################

# Compiler options
# ###############################################################################

# Use C++14
target_compile_features(polysolve PUBLIC cxx_std_14)

# ###############################################################################
# Tests
# ###############################################################################

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

    # Cuda test
    if(POLYSOLVE_WITH_CUDA)
        add_subdirectory(cudatest)
    endif()
endif()

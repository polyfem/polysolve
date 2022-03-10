#
# Copyright 2020 Adobe. All rights reserved.
# This file is licensed to you under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR REPRESENTATIONS
# OF ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.
#
#---------------------------------------------------------------------------------------------------
# WARNING: By default, the GPL module Cholmod/Supernodal is enabled. This leads to a 2x speedup
# compared to simplicial mode. This is optional and can be disabled by setting WITH_GPL to OFF.
#---------------------------------------------------------------------------------------------------

if(TARGET SuiteSparse::SuiteSparse)
    return()
endif()

message(STATUS "Third-party: creating targets 'SuiteSparse::SuiteSparse'")

include(FetchContent)
FetchContent_Declare(
    suitesparse
    GIT_REPOSITORY https://github.com/sergiud/SuiteSparse.git
    GIT_TAG 5.10.1-cmake.1
)

FetchContent_GetProperties(suitesparse)
if(NOT suitesparse_POPULATED)
    FetchContent_Populate(suitesparse)
endif()

include(CMakeDependentOption)
option(BUILD_CXSPARSE "Build CXSparse" OFF)
option(WITH_FORTRAN "Enables Fortran support" OFF)
option(WITH_DEMOS "Build demos" OFF)
option(WITH_PRINT "Print diagnostic messages" ON)
option(WITH_TBB "Enables Intel Threading Building Blocks support" OFF)
option(WITH_LGPL "Enable GNU LGPL modules" ON)
option(WITH_CUDA "Enable CUDA support" OFF)
option(WITH_OPENMP "Enable OpenMP support" OFF)
cmake_dependent_option(WITH_GPL "Enable GNU GPL modules" ON "WITH_LGPL" OFF)
cmake_dependent_option(WITH_CAMD "Enable interfaces to CAMD, CCOLAMD, CSYMAMD in Partition module" OFF "WITH_LGPL" OFF)
cmake_dependent_option(WITH_PARTITION "Enable the Partition module" OFF "WITH_LGPL AND METIS_FOUND" OFF)
cmake_dependent_option(WITH_CHOLESKY "Enable the Cholesky module" ON "WITH_LGPL" OFF)
cmake_dependent_option(WITH_CHECK "Enable the Check module" OFF "WITH_LGPL" OFF)
cmake_dependent_option(WITH_MODIFY "Enable the Modify module" OFF "WITH_GPL" OFF)
cmake_dependent_option(WITH_MATRIXOPS "Enable the MatrixOps module" OFF "WITH_GPL" OFF)
cmake_dependent_option(WITH_SUPERNODAL "Enable the Supernodal module" ON "WITH_GPL" OFF)

option(SUITE_SPARSE_WITH_MKL "Build SuiteSparse using MKL" ON)

if(SUITE_SPARSE_WITH_MKL)
    include(mkl)
    if(NOT TARGET blas)
        add_library(blas INTERFACE IMPORTED GLOBAL)
    endif()
    if(NOT TARGET lapack)
        add_library(lapack INTERFACE IMPORTED GLOBAL)
    endif()

    function(suitesparse_import_target)
        macro(ignore_package NAME)
            file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/${NAME}/${NAME}Config.cmake "")
            set(${NAME}_DIR ${CMAKE_CURRENT_BINARY_DIR}/${NAME} CACHE PATH "" FORCE)
        endmacro()

        # Prefer Config mode before Module mode
        set(CMAKE_FIND_PACKAGE_PREFER_CONFIG TRUE)

        ignore_package(BLAS)
        ignore_package(CBLAS)
        ignore_package(LAPACK)

        add_subdirectory(${suitesparse_SOURCE_DIR} ${suitesparse_BINARY_DIR} EXCLUDE_FROM_ALL)
    endfunction()

    suitesparse_import_target()

    target_link_libraries(cholmod PUBLIC mkl::mkl)
else()
    function(suitesparse_import_target)
        macro(unignore_package NAME)
            file(REMOVE_RECURSE ${CMAKE_CURRENT_BINARY_DIR}/${NAME})
            if(DEFINED ${NAME}_DIR)
                unset(${NAME}_DIR CACHE)
            endif()
        endmacro()

        # Prefer Config mode before Module mode
        set(CMAKE_FIND_PACKAGE_PREFER_CONFIG FALSE)

        unignore_package(BLAS)
        unignore_package(CBLAS)
        unignore_package(LAPACK)

        add_subdirectory(${suitesparse_SOURCE_DIR} ${suitesparse_BINARY_DIR} EXCLUDE_FROM_ALL)
    endfunction()

    suitesparse_import_target()
endif()

macro(suitesparse_add_library LIBRARY_NAME)
    add_library(SuiteSparse_${LIBRARY_NAME} INTERFACE)
    add_library(SuiteSparse::${LIBRARY_NAME} ALIAS SuiteSparse_${LIBRARY_NAME})
    foreach(name IN ITEMS ${LIBRARY_NAME})
        if(NOT TARGET ${name})
            message(FATAL_ERROR "${name} is not a valid CMake target. Please check your config!")
        endif()
        target_link_libraries(SuiteSparse_${LIBRARY_NAME} INTERFACE ${name})
    endforeach()
endmacro()

suitesparse_add_library(cholmod)
# suitesparse_add_library(umfpack)

add_library(SuiteSparse_SuiteSparse INTERFACE)
add_library(SuiteSparse::SuiteSparse ALIAS SuiteSparse_SuiteSparse)
target_link_libraries(SuiteSparse_SuiteSparse INTERFACE SuiteSparse::cholmod)
# target_link_libraries(SuiteSparse_SuiteSparse INTERFACE SuiteSparse::umfpack)
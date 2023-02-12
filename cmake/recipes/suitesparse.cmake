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
    URL https://github.com/sergiud/SuiteSparse/archive/refs/tags/5.12.0-cmake.3.zip
    URL_HASH MD5=73a6fbfc949d43f37c2c8749662fb35e
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
set (WITH_LICENSE "GPL" CACHE STRING "Software license the binary distribution should adhere")
set_property (CACHE WITH_LICENSE PROPERTY STRINGS "Minimal;GPL;LGPL")

option(SUITE_SPARSE_WITH_MKL "Build SuiteSparse using MKL" ON)

message(STATUS "SuiteSparse with MKL is ${SUITE_SPARSE_WITH_MKL}")

if(SUITE_SPARSE_WITH_MKL)
    include(mkl)
    if(NOT TARGET blas)
        add_library(blas INTERFACE IMPORTED GLOBAL)
        add_library(BLAS::BLAS ALIAS blas) # This alias is used by SuiteSparse
    endif()
    if(NOT TARGET lapack)
        add_library(lapack INTERFACE IMPORTED GLOBAL)
        add_library(LAPACK::LAPACK ALIAS lapack) # This alias is used by SuiteSparse
    endif()

    # This is copied from SuiteSparse/CMakeLists.txt. Needs to use blas and
    # lapack instead of BLAS::BLAS and LAPACK::LAPACK
    include(CMakePushCheckState)
    include (CheckFunctionExists)
    cmake_push_check_state (RESET)
    set(CMAKE_REQUIRED_LIBRARIES blas lapack)
    check_function_exists (sgemm HAVE_BLAS_NO_UNDERSCORE)
    check_function_exists (sgemm_ HAVE_BLAS_UNDERSCORE)
    cmake_pop_check_state ()

    function(suitesparse_import_target)
        macro(ignore_package NAME)
            file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/${NAME}/${NAME}Config.cmake "")
            set(${NAME}_DIR ${CMAKE_CURRENT_BINARY_DIR}/${NAME} CACHE PATH "" FORCE)
        endmacro()

        # Prefer Config mode before Module mode
        set(CMAKE_FIND_PACKAGE_PREFER_CONFIG TRUE)

        # Copy over the TBB version file so SuiteSparse wont complain
        include(onetbb)
        FetchContent_GetProperties(tbb)
        ignore_package(TBB)
        file(COPY "${tbb_BINARY_DIR}/TBBConfigVersion.cmake" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/TBB")
        set (CMAKE_DISABLE_FIND_PACKAGE_TBB ON) # Disable find TBB completly (only SPQR uses it in SuiteSparse)

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

add_library(SuiteSparse_SuiteSparse INTERFACE)
add_library(SuiteSparse::SuiteSparse ALIAS SuiteSparse_SuiteSparse)
target_link_libraries(SuiteSparse_SuiteSparse INTERFACE SuiteSparse::CHOLMOD)
# target_link_libraries(SuiteSparse_SuiteSparse INTERFACE SuiteSparse::umfpack)

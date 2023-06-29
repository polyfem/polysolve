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

#---------------------------------------------------------------------------------------------------
# WARNING: By default, the GPL module Cholmod/Supernodal is enabled. This leads to a 2x speedup
# compared to simplicial mode. This is optional and can be disabled by setting WITH_LICENSE to LGPL.
#---------------------------------------------------------------------------------------------------

if(TARGET SuiteSparse::CHOLMOD)
    return()
endif()

message(STATUS "Third-party: creating targets 'SuiteSparse::CHOLMOD'")

include(CMakeDependentOption)
option(BUILD_CXSPARSE "Build CXSparse" OFF)
option(WITH_DEMOS "Build demos" OFF)
option(WITH_METIS "Enables METIS support" OFF)
option(WITH_OPENMP "Enable OpenMP support" OFF)
option(WITH_PRINT "Print diagnostic messages" ON)
option(WITH_TBB "Enables Intel Threading Building Blocks support" OFF)
set(WITH_LICENSE "GPL" CACHE STRING "Software license the binary distribution should adhere")
option(WITH_CUDA "Enable CUDA support" OFF)
option(WITH_FORTRAN "Enables Fortran support" OFF)
cmake_dependent_option (WITH_LGPL "Enable GNU LGPL modules" ON "WITH_LICENSE MATCHES GPL" OFF)
cmake_dependent_option (WITH_GPL "Enable GNU GPL modules" ON "WITH_LICENSE STREQUAL GPL" OFF)
cmake_dependent_option (WITH_CHOLMOD "Enable CHOLMOD" ON "WITH_LGPL" OFF)
cmake_dependent_option (WITH_CHOLESKY "Enable the Cholesky module" ON "WITH_CHOLMOD" OFF)
cmake_dependent_option (WITH_CAMD "Enable interfaces to CAMD, CCOLAMD, CSYMAMD in Partition module" ON "WITH_LGPL" OFF)
cmake_dependent_option (WITH_CHECK "Enable the Check module" ON "WITH_LGPL" OFF)
cmake_dependent_option (WITH_MATRIXOPS "Enable the MatrixOps module" ON "WITH_GPL AND WITH_CHOLMOD" OFF)
cmake_dependent_option (WITH_MODIFY "Enable the Modify module" ON "WITH_GPL AND WITH_CHOLMOD" OFF)
cmake_dependent_option (WITH_SUPERNODAL "Enable the Supernodal module" ON
  "WITH_CHOLESKY AND CMAKE_CXX_COMPILER AND WITH_GPL" OFF)

include(blas)
include(lapack)

if(NOT TARGET BLAS::BLAS OR NOT TARGET LAPACK::LAPACK)
    message(FATAL_ERROR "BLAS/LAPACK configuration error")
endif()

function(suitesparse_import_target)
    macro(push_variable var value)
        if(DEFINED CACHE{${var}})
            set(SUITESPARSE_OLD_${var}_VALUE "${${var}}")
            set(SUITESPARSE_OLD_${var}_TYPE CACHE_TYPE)
        elseif(DEFINED ${var})
            set(SUITESPARSE_OLD_${var}_VALUE "${${var}}")
            set(SUITESPARSE_OLD_${var}_TYPE NORMAL_TYPE)
        else()
            set(SUITESPARSE_OLD_${var}_TYPE NONE_TYPE)
        endif()
        set(${var} "${value}" CACHE PATH "" FORCE)
    endmacro()

    macro(pop_variable var)
        if(SUITESPARSE_OLD_${var}_TYPE STREQUAL CACHE_TYPE)
            set(${var} "${SUITESPARSE_OLD_${var}_VALUE}" CACHE PATH "" FORCE)
        elseif(SUITESPARSE_OLD_${var}_TYPE STREQUAL NORMAL_TYPE)
            unset(${var} CACHE)
            set(${var} "${SUITESPARSE_OLD_${var}_VALUE}")
        elseif(SUITESPARSE_OLD_${var}_TYPE STREQUAL NONE_TYPE)
            unset(${var} CACHE)
        else()
            message(FATAL_ERROR "Trying to pop a variable that has not been pushed: ${var}")
        endif()
    endmacro()

    macro(ignore_package NAME)
        file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/${NAME}/${NAME}Config.cmake "")
        set(${NAME}_DIR ${CMAKE_CURRENT_BINARY_DIR}/${NAME} CACHE PATH "" FORCE)
    endmacro()

    # Prefer Config mode before Module mode
    set(CMAKE_FIND_PACKAGE_PREFER_CONFIG TRUE)

    ignore_package(BLAS)
    ignore_package(CBLAS)
    ignore_package(LAPACK)
    # ignore_package(TBB)
    # ignore_package(Metis)

    # Bypass SuiteSpars's check_function_exists() calls, as they do not work with non-imported targets
    set(HAVE_BLAS_NO_UNDERSCORE ON)
    set(HAVE_BLAS_UNDERSCORE OFF)

    # SuiteSparse is composed of multiple libraries. If LGPL is enabled, we need to build shared
    # libraries to comply with the license. If GPL is enabled, it is the user's responsibility to
    # comply with the license by release the parts of their code that depends on SuiteSparse under GPL.
    if(WITH_LICENSE STREQUAL LGPL)
        push_variable(BUILD_SHARED_LIBS ON)
    endif()
    include(CPM)
    CPMAddPackage(
        NAME suitesparse
        GITHUB_REPOSITORY sergiud/SuiteSparse
        GIT_TAG 0ba07264225518a487a0a9a8e675f6e36c96a68a
        EXCLUDE_FROM_ALL ON
    )
    if(WITH_LICENSE STREQUAL LGPL)
        pop_variable(BUILD_SHARED_LIBS)
    endif()
endfunction()

suitesparse_import_target()

foreach(name IN ITEMS cholmod SuiteSparse::CHOLMOD)
    if(NOT TARGET ${name})
        message(FATAL_ERROR "SuiteSparse error: missing '${name}' target.")
    endif()
endforeach()

# Ensure that cholmod is linked as a shared library due to LGPL licensing
if(WITH_LICENSE STREQUAL LGPL)
    get_target_property(target_type cholmod TYPE)
    if(NOT ${target_type} STREQUAL "SHARED_LIBRARY")
        message(FATAL_ERROR "SuiteSparse error: 'cholmod' should be SHARED_LIBRARY, but is ${target_type}.")
    endif()
endif()

# Set folders for MSVC
foreach(name IN ITEMS
        amd
        amd_int
        amd_long
        camd
        camd_int
        camd_long
        ccolamd
        ccolamd_int
        ccolamd_long
        cholmod
        cholmod_int
        cholmod_long
        colamd
        colamd_int
        colamd_long
        suitesparsebase
        suitesparseconfig
    )
    if(TARGET ${name})
        set_target_properties(${name} PROPERTIES FOLDER third_party/suitesparse)
    endif()
endforeach()

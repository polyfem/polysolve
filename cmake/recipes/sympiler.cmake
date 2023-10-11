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
if(TARGET sympiler::sym_sparse_blas)
    return()
endif()

message(STATUS "Third-party (external): creating target 'sympiler::sym_sparse_blas'")

if("${CMAKE_SYSTEM_PROCESSOR}" MATCHES "arm64" OR "${CMAKE_OSX_ARCHITECTURES}" MATCHES "arm64")
    # apple M1
    set(SYMPILER_BLAS_BACKEND "Apple" CACHE STRING "BLAS implementation for SYMPILER to use")
else()
    # windows, linux, apple intel
    set(SYMPILER_BLAS_BACKEND "MKL" CACHE STRING "BLAS implementation for SYMPILER to use")
endif()

if(SYMPILER_BLAS_BACKEND STREQUAL "MKL")
    message("INCLUDING MKL")
    include(mkl)
    if(NOT TARGET MKL::MKL)
        add_library(MKL::MKL ALIAS mkl::mkl)
    endif()
endif()

include(CPM)
CPMAddPackage(
    NAME sympiler
    GITHUB_REPOSITORY ryansynk/sympiler
    GIT_TAG 2fe1ef1d72b551acedd848d64b51fa5e90251804
)

add_library(sympiler::sym_sparse_blas ALIAS sym_sparse_blas)

include(GNUInstallDirs)
target_include_directories(sym_sparse_blas INTERFACE
    "$<BUILD_INTERFACE:${sympiler_SOURCE_DIR}>/include"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)

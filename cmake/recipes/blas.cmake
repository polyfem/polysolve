#
# Copyright 2022 Adobe. All rights reserved.
# This file is licensed to you under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR REPRESENTATIONS
# OF ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.
#
if(TARGET BLAS::BLAS)
    return()
endif()

message(STATUS "Third-party: creating target 'BLAS::BLAS'")

if("${CMAKE_SYSTEM_PROCESSOR}" MATCHES "arm64" OR "${CMAKE_OSX_ARCHITECTURES}" MATCHES "arm64")
    # Use Accelerate on macOS M1
    set(BLA_VENDOR Apple)
    find_package(BLAS QUIET)
elseif(POLYSOLVE_WITH_MKL AND TARGET mkl::mkl)
    # Use MKL if enabled
    include(mkl)
    if(TARGET mkl::mkl)
        add_library(BLAS::BLAS INTERFACE IMPORTED GLOBAL)
        target_link_libraries(BLAS::BLAS INTERFACE mkl::mkl)
    endif()
else()
    # otherwise find system version
    find_package(BLAS QUIET)
endif()

if(NOT TARGET BLAS::BLAS)
    message(WARNING "BLAS not found.")
endif()

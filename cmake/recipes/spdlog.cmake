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

# spdlog (https://github.com/gabime/spdlog)
# License: MIT

if(TARGET spdlog::spdlog)
    return()
endif()

include(polysolve_optional_dependency)

set(POLYSOLVE_SPDLOG_CHECK_SOURCE [[
#include <spdlog/spdlog.h>
int main()
{
    spdlog::info("polysolve dependency check");
    return 0;
}
]])

polysolve_find_system_dependency(SPDLOG_SYSTEM_FOUND
    NAME spdlog
    PACKAGE spdlog
    TARGET spdlog::spdlog
    CONFIG
    SOURCE_VAR POLYSOLVE_SPDLOG_CHECK_SOURCE
)
if(SPDLOG_SYSTEM_FOUND)
    return()
endif()

polysolve_should_fetch_dependency(SPDLOG_SHOULD_FETCH spdlog)
if(NOT SPDLOG_SHOULD_FETCH)
    message(FATAL_ERROR "spdlog is required to build PolySolve.")
endif()

message(STATUS "Third-party: creating target 'spdlog::spdlog'")

option(SPDLOG_INSTALL "Generate the install target" ON)
set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME "spdlog")

include(CPM)
CPMAddPackage("gh:gabime/spdlog@1.9.2")

set_target_properties(spdlog PROPERTIES POSITION_INDEPENDENT_CODE ON)

set_target_properties(spdlog PROPERTIES FOLDER external)

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang" OR
   "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    target_compile_options(spdlog PRIVATE
        "-Wno-sign-conversion"
    )
endif()

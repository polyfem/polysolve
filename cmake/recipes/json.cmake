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
if(TARGET nlohmann_json::nlohmann_json)
    return()
endif()

include(polysolve_optional_dependency)

set(POLYSOLVE_NLOHMANN_JSON_CHECK_SOURCE [[
#include <nlohmann/json.hpp>
int main()
{
    nlohmann::json data = {{"ok", true}};
    return data["ok"].get<bool>() ? 0 : 1;
}
]])

polysolve_find_system_dependency(NLOHMANN_JSON_SYSTEM_FOUND
    NAME nlohmann_json
    PACKAGE nlohmann_json
    TARGET nlohmann_json::nlohmann_json
    CONFIG
    SOURCE_VAR POLYSOLVE_NLOHMANN_JSON_CHECK_SOURCE
)
if(NLOHMANN_JSON_SYSTEM_FOUND)
    return()
endif()

polysolve_should_fetch_dependency(NLOHMANN_JSON_SHOULD_FETCH nlohmann_json)
if(NOT NLOHMANN_JSON_SHOULD_FETCH)
    message(FATAL_ERROR "nlohmann_json is required to build PolySolve.")
endif()

message(STATUS "Third-party: creating target 'nlohmann_json::nlohmann_json'")

# nlohmann_json is a big repo for a single header, so we just download the release archive
set(NLOHMANNJSON_VERSION "v3.11.2")

include(CPM)
CPMAddPackage(
    NAME nlohmann_json
    URL "https://github.com/nlohmann/json/releases/download/${NLOHMANNJSON_VERSION}/include.zip"
    URL_HASH SHA256=e5c7a9f49a16814be27e4ed0ee900ecd0092bfb7dbfca65b5a421b774dccaaed
)

add_library(nlohmann_json INTERFACE)
add_library(nlohmann_json::nlohmann_json ALIAS nlohmann_json)

include(GNUInstallDirs)
target_include_directories(nlohmann_json INTERFACE
    "$<BUILD_INTERFACE:${nlohmann_json_SOURCE_DIR}>/include"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)

# Install rules
set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME nlohmann_json)
install(DIRECTORY ${nlohmann_json_SOURCE_DIR}/include/nlohmann DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(TARGETS nlohmann_json EXPORT NlohmannJson_Targets)
install(EXPORT NlohmannJson_Targets DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/nlohmann_json NAMESPACE nlohmann_json::)
export(EXPORT NlohmannJson_Targets FILE "${CMAKE_CURRENT_BINARY_DIR}/NlohmannJsonTargets.cmake")

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

if(TARGET amgcl::amgcl)
    return()
endif()

include(polysolve_optional_dependency)

set(POLYSOLVE_AMGCL_CHECK_SOURCE [[
#include <amgcl/backend/builtin.hpp>
#include <amgcl/make_solver.hpp>
#include <amgcl/preconditioner/runtime.hpp>
#include <amgcl/solver/runtime.hpp>
int main()
{
    using Backend = amgcl::backend::builtin<double>;
    using Solver = amgcl::make_solver<amgcl::runtime::preconditioner<Backend>, amgcl::runtime::solver::wrapper<Backend>>;
    return sizeof(Solver) > 0 ? 0 : 1;
}
]])

polysolve_find_system_dependency(AMGCL_SYSTEM_FOUND
    NAME AMGCL
    PACKAGE amgcl
    TARGET amgcl::amgcl
    CONFIG
    SOURCE_VAR POLYSOLVE_AMGCL_CHECK_SOURCE
)
if(AMGCL_SYSTEM_FOUND)
    return()
endif()

polysolve_should_fetch_dependency(AMGCL_SHOULD_FETCH AMGCL)
if(NOT AMGCL_SHOULD_FETCH)
    polysolve_note_disabled_dependency("AMGCL" "AMGCL was not found and fetching is disabled.")
    return()
endif()

message(STATUS "Third-party: creating target 'amgcl::amgcl'")

function(amgcl_import_target)
    macro(ignore_package NAME VERSION_NUM)
        include(CMakePackageConfigHelpers)
        file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/${NAME}/${NAME}Config.cmake "")
        write_basic_package_version_file(
            ${CMAKE_CURRENT_BINARY_DIR}/${NAME}/${NAME}ConfigVersion.cmake
            VERSION ${VERSION_NUM}
            COMPATIBILITY AnyNewerVersion
        )
        set(${NAME}_DIR ${CMAKE_CURRENT_BINARY_DIR}/${NAME} CACHE PATH "")
        set(${NAME}_ROOT ${CMAKE_CURRENT_BINARY_DIR}/${NAME} CACHE PATH "")
    endmacro()

    include(boost)

    ignore_package(Boost 1.71.0)
    set(Boost_ROOT "")
    set(Boost_INCLUDE_DIRS "")
    set(Boost_LIBRARIES "")

    # Prefer Config mode before Module mode to prevent lib from loading its own FindXXX.cmake
    set(CMAKE_FIND_PACKAGE_PREFER_CONFIG TRUE)

    # Ready to include third-party lib
    include(CPM)
    CPMAddPackage(
        NAME amgcl
        GITHUB_REPOSITORY ddemidov/amgcl
        GIT_TAG 1.4.3
    )

    target_link_libraries(amgcl INTERFACE Boost::boost)
endfunction()

amgcl_import_target()

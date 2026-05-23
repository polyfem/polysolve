include_guard(GLOBAL)

option(POLYSOLVE_FIND_SYSTEM_DEPENDENCIES "Look for package-manager provided dependencies before fetching vendored copies" ON)
option(POLYSOLVE_FETCH_MISSING_DEPENDENCIES "Fetch vendored dependency copies when package-manager provided dependencies are unavailable" ON)

function(polysolve_note_disabled_dependency name reason)
    message(WARNING "${name} requested but unavailable; disabling ${name}. ${reason}")
endfunction()

function(polysolve_find_system_dependency out_var)
    set(options CONFIG MODULE)
    set(one_value_args NAME PACKAGE VERSION TARGET SOURCE SOURCE_VAR)
    set(multi_value_args COMPONENTS COMPILE_DEFINITIONS LINK_LIBRARIES)
    cmake_parse_arguments(POLYSOLVE_FIND
        "${options}"
        "${one_value_args}"
        "${multi_value_args}"
        ${ARGN}
    )

    if(NOT POLYSOLVE_FIND_NAME)
        set(POLYSOLVE_FIND_NAME "${POLYSOLVE_FIND_PACKAGE}")
    endif()

    if(NOT POLYSOLVE_FIND_PACKAGE)
        message(FATAL_ERROR "polysolve_find_system_dependency requires PACKAGE")
    endif()

    if(NOT POLYSOLVE_FIND_TARGET)
        message(FATAL_ERROR "polysolve_find_system_dependency requires TARGET")
    endif()

    set(${out_var} OFF PARENT_SCOPE)
    set(${out_var}_REASON "" PARENT_SCOPE)

    if(NOT POLYSOLVE_FIND_SYSTEM_DEPENDENCIES)
        return()
    endif()

    set(find_mode)
    if(POLYSOLVE_FIND_CONFIG)
        set(find_mode CONFIG)
    elseif(POLYSOLVE_FIND_MODULE)
        set(find_mode MODULE)
    endif()

    if(TARGET ${POLYSOLVE_FIND_TARGET})
        if(POLYSOLVE_FIND_SOURCE OR POLYSOLVE_FIND_SOURCE_VAR)
            if(POLYSOLVE_FIND_SOURCE_VAR)
                polysolve_check_linkable_target(POLYSOLVE_FIND_LINKABLE
                    NAME ${POLYSOLVE_FIND_NAME}
                    TARGET ${POLYSOLVE_FIND_TARGET}
                    SOURCE_VAR ${POLYSOLVE_FIND_SOURCE_VAR}
                    COMPILE_DEFINITIONS ${POLYSOLVE_FIND_COMPILE_DEFINITIONS}
                    LINK_LIBRARIES ${POLYSOLVE_FIND_LINK_LIBRARIES}
                )
            else()
                polysolve_check_linkable_target(POLYSOLVE_FIND_LINKABLE
                    NAME ${POLYSOLVE_FIND_NAME}
                    TARGET ${POLYSOLVE_FIND_TARGET}
                    SOURCE "${POLYSOLVE_FIND_SOURCE}"
                    COMPILE_DEFINITIONS ${POLYSOLVE_FIND_COMPILE_DEFINITIONS}
                    LINK_LIBRARIES ${POLYSOLVE_FIND_LINK_LIBRARIES}
                )
            endif()

            if(NOT POLYSOLVE_FIND_LINKABLE)
                set(${out_var}_REASON "${POLYSOLVE_FIND_LINKABLE_REASON}" PARENT_SCOPE)
                return()
            endif()
        endif()

        message(STATUS "Third-party: using target '${POLYSOLVE_FIND_TARGET}' for ${POLYSOLVE_FIND_NAME}")
        set(${out_var} ON PARENT_SCOPE)
        return()
    endif()

    set(find_args ${POLYSOLVE_FIND_PACKAGE})
    if(POLYSOLVE_FIND_VERSION)
        list(APPEND find_args ${POLYSOLVE_FIND_VERSION})
    endif()
    if(find_mode)
        list(APPEND find_args ${find_mode})
    endif()
    list(APPEND find_args QUIET)
    if(POLYSOLVE_FIND_COMPONENTS)
        list(APPEND find_args COMPONENTS ${POLYSOLVE_FIND_COMPONENTS})
    endif()

    if(POLYSOLVE_FIND_SOURCE OR POLYSOLVE_FIND_SOURCE_VAR)
        if(POLYSOLVE_FIND_SOURCE_VAR)
            set(check_source_content "${${POLYSOLVE_FIND_SOURCE_VAR}}")
        else()
            set(check_source_content "${POLYSOLVE_FIND_SOURCE}")
        endif()

        string(REGEX REPLACE "[^A-Za-z0-9_]" "_" check_id
            "${POLYSOLVE_FIND_NAME}_${POLYSOLVE_FIND_PACKAGE}_${POLYSOLVE_FIND_TARGET}")
        set(check_root "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/polysolve_system_dependency_checks")
        set(check_source_dir "${check_root}/${check_id}-src")
        set(check_binary_dir "${check_root}/${check_id}-build")
        set(check_log "${check_root}/${check_id}.log")
        set(check_source "${check_source_dir}/main.cpp")
        set(check_cmake_file "${check_source_dir}/CMakeLists.txt")

        file(MAKE_DIRECTORY "${check_source_dir}")
        file(WRITE "${check_source}" "${check_source_content}")

        set(check_find_package_args ${POLYSOLVE_FIND_PACKAGE})
        if(POLYSOLVE_FIND_VERSION)
            list(APPEND check_find_package_args ${POLYSOLVE_FIND_VERSION})
        endif()
        if(find_mode)
            list(APPEND check_find_package_args ${find_mode})
        endif()
        list(APPEND check_find_package_args QUIET)
        if(POLYSOLVE_FIND_COMPONENTS)
            list(APPEND check_find_package_args COMPONENTS ${POLYSOLVE_FIND_COMPONENTS})
        endif()

        string(REPLACE ";" " " check_find_package_args_line "${check_find_package_args}")
        string(REPLACE ";" " " check_compile_definitions_line "${POLYSOLVE_FIND_COMPILE_DEFINITIONS}")
        string(REPLACE ";" " " check_link_libraries_line "${POLYSOLVE_FIND_LINK_LIBRARIES}")

        set(check_find_package_hints)
        foreach(package_hint_suffix IN ITEMS DIR ROOT)
            set(package_hint_var "${POLYSOLVE_FIND_PACKAGE}_${package_hint_suffix}")
            if(DEFINED ${package_hint_var})
                string(APPEND check_find_package_hints
                    "set(${package_hint_var} [==[${${package_hint_var}}]==])\n")
            endif()
        endforeach()

        set(check_disable_find_package_var "CMAKE_DISABLE_FIND_PACKAGE_${POLYSOLVE_FIND_PACKAGE}")
        if(DEFINED ${check_disable_find_package_var})
            string(APPEND check_find_package_hints
                "set(${check_disable_find_package_var} [==[${${check_disable_find_package_var}}]==])\n")
        endif()

        file(WRITE "${check_cmake_file}" "
cmake_minimum_required(VERSION 3.18)
project(polysolve_system_dependency_check LANGUAGES CXX)

set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS TRUE CACHE INTERNAL \"\")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ${CMAKE_POSITION_INDEPENDENT_CODE})
set(CMAKE_MODULE_PATH [==[${CMAKE_MODULE_PATH}]==])
set(CMAKE_PREFIX_PATH [==[${CMAKE_PREFIX_PATH}]==])
set(CMAKE_FIND_PACKAGE_PREFER_CONFIG [==[${CMAKE_FIND_PACKAGE_PREFER_CONFIG}]==])
${check_find_package_hints}

find_package(${check_find_package_args_line})
if(TARGET ${POLYSOLVE_FIND_TARGET})
    add_executable(polysolve_system_dependency_check main.cpp)
    target_compile_definitions(polysolve_system_dependency_check PRIVATE ${check_compile_definitions_line})
    target_link_libraries(polysolve_system_dependency_check PRIVATE ${POLYSOLVE_FIND_TARGET} ${check_link_libraries_line})
else()
    file(WRITE \"\${CMAKE_CURRENT_BINARY_DIR}/missing_target.cpp\" \"#error Target '${POLYSOLVE_FIND_TARGET}' was not created.\\n\")
    add_executable(polysolve_system_dependency_check \"\${CMAKE_CURRENT_BINARY_DIR}/missing_target.cpp\")
endif()
")

        set(check_cmake_flags
            "-DCMAKE_POSITION_INDEPENDENT_CODE=${CMAKE_POSITION_INDEPENDENT_CODE}"
        )
        if(DEFINED CMAKE_OSX_ARCHITECTURES AND NOT CMAKE_OSX_ARCHITECTURES STREQUAL "")
            string(REPLACE ";" "\\;" check_osx_architectures "${CMAKE_OSX_ARCHITECTURES}")
            list(APPEND check_cmake_flags "-DCMAKE_OSX_ARCHITECTURES=${check_osx_architectures}")
        endif()

        if(DEFINED CMAKE_OSX_DEPLOYMENT_TARGET AND NOT CMAKE_OSX_DEPLOYMENT_TARGET STREQUAL "")
            list(APPEND check_cmake_flags "-DCMAKE_OSX_DEPLOYMENT_TARGET=${CMAKE_OSX_DEPLOYMENT_TARGET}")
        endif()

        if(DEFINED CMAKE_OSX_SYSROOT AND NOT CMAKE_OSX_SYSROOT STREQUAL "")
            list(APPEND check_cmake_flags "-DCMAKE_OSX_SYSROOT=${CMAKE_OSX_SYSROOT}")
        endif()

        set(check_result_var "POLYSOLVE_${check_id}_SYSTEM_LINKABLE")
        unset(${check_result_var} CACHE)
        try_compile(${check_result_var}
            "${check_binary_dir}"
            "${check_source_dir}"
            polysolve_system_dependency_check
            polysolve_system_dependency_check
            CMAKE_FLAGS ${check_cmake_flags}
            OUTPUT_VARIABLE check_output
        )

        if(NOT ${check_result_var})
            file(WRITE "${check_log}" "${check_output}")
            set(${out_var}_REASON "A system package compile/link check failed; see ${check_log}." PARENT_SCOPE)
            return()
        endif()
    endif()

    find_package(${find_args})

    if(TARGET ${POLYSOLVE_FIND_TARGET})
        message(STATUS "Third-party: using system target '${POLYSOLVE_FIND_TARGET}' for ${POLYSOLVE_FIND_NAME}")
        set(${out_var} ON PARENT_SCOPE)
    else()
        set(${out_var}_REASON "Target '${POLYSOLVE_FIND_TARGET}' was not created." PARENT_SCOPE)
    endif()
endfunction()

function(polysolve_should_fetch_dependency out_var name)
    if(POLYSOLVE_FETCH_MISSING_DEPENDENCIES)
        set(${out_var} ON PARENT_SCOPE)
    else()
        message(WARNING "${name} was not found and POLYSOLVE_FETCH_MISSING_DEPENDENCIES is OFF.")
        set(${out_var} OFF PARENT_SCOPE)
    endif()
endfunction()

function(polysolve_check_linkable_target out_var)
    set(options)
    set(one_value_args NAME TARGET SOURCE SOURCE_VAR)
    set(multi_value_args COMPILE_DEFINITIONS LINK_LIBRARIES)
    cmake_parse_arguments(POLYSOLVE_CHECK
        "${options}"
        "${one_value_args}"
        "${multi_value_args}"
        ${ARGN}
    )

    if(NOT POLYSOLVE_CHECK_NAME)
        message(FATAL_ERROR "polysolve_check_linkable_target requires NAME")
    endif()

    if(NOT POLYSOLVE_CHECK_TARGET)
        message(FATAL_ERROR "polysolve_check_linkable_target requires TARGET")
    endif()

    if(NOT POLYSOLVE_CHECK_SOURCE AND NOT POLYSOLVE_CHECK_SOURCE_VAR)
        message(FATAL_ERROR "polysolve_check_linkable_target requires SOURCE or SOURCE_VAR")
    endif()

    if(POLYSOLVE_CHECK_SOURCE_VAR)
        set(check_source_content "${${POLYSOLVE_CHECK_SOURCE_VAR}}")
    else()
        set(check_source_content "${POLYSOLVE_CHECK_SOURCE}")
    endif()

    if(NOT TARGET ${POLYSOLVE_CHECK_TARGET})
        set(${out_var} OFF PARENT_SCOPE)
        set(${out_var}_REASON "Target '${POLYSOLVE_CHECK_TARGET}' was not created." PARENT_SCOPE)
        return()
    endif()

    foreach(link_dependency IN LISTS POLYSOLVE_CHECK_LINK_LIBRARIES)
        if(link_dependency MATCHES "::" AND NOT TARGET ${link_dependency})
            set(${out_var} OFF PARENT_SCOPE)
            set(${out_var}_REASON "Target '${link_dependency}' was not created." PARENT_SCOPE)
            return()
        endif()
    endforeach()

    string(REGEX REPLACE "[^A-Za-z0-9_]" "_" check_id
        "${POLYSOLVE_CHECK_NAME}_${POLYSOLVE_CHECK_TARGET}")
    set(check_root "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/polysolve_dependency_checks")
    set(check_source "${check_root}/${check_id}.cpp")
    set(check_binary_dir "${check_root}/${check_id}")
    set(check_log "${check_root}/${check_id}.log")

    file(MAKE_DIRECTORY "${check_root}")
    file(WRITE "${check_source}" "${check_source_content}")

    set(check_cmake_flags
        "-DCMAKE_CXX_STANDARD=17"
        "-DCMAKE_CXX_STANDARD_REQUIRED=ON"
        "-DCMAKE_CXX_EXTENSIONS=OFF"
        "-DCMAKE_POSITION_INDEPENDENT_CODE=${CMAKE_POSITION_INDEPENDENT_CODE}"
    )

    if(DEFINED CMAKE_OSX_ARCHITECTURES AND NOT CMAKE_OSX_ARCHITECTURES STREQUAL "")
        string(REPLACE ";" "\\;" check_osx_architectures "${CMAKE_OSX_ARCHITECTURES}")
        list(APPEND check_cmake_flags "-DCMAKE_OSX_ARCHITECTURES=${check_osx_architectures}")
    endif()

    if(DEFINED CMAKE_OSX_DEPLOYMENT_TARGET AND NOT CMAKE_OSX_DEPLOYMENT_TARGET STREQUAL "")
        list(APPEND check_cmake_flags "-DCMAKE_OSX_DEPLOYMENT_TARGET=${CMAKE_OSX_DEPLOYMENT_TARGET}")
    endif()

    if(DEFINED CMAKE_OSX_SYSROOT AND NOT CMAKE_OSX_SYSROOT STREQUAL "")
        list(APPEND check_cmake_flags "-DCMAKE_OSX_SYSROOT=${CMAKE_OSX_SYSROOT}")
    endif()

    set(check_result_var "POLYSOLVE_${check_id}_LINKABLE")
    unset(${check_result_var} CACHE)
    try_compile(${check_result_var}
        "${check_binary_dir}"
        "${check_source}"
        CMAKE_FLAGS ${check_cmake_flags}
        COMPILE_DEFINITIONS ${POLYSOLVE_CHECK_COMPILE_DEFINITIONS}
        LINK_LIBRARIES ${POLYSOLVE_CHECK_TARGET} ${POLYSOLVE_CHECK_LINK_LIBRARIES}
        OUTPUT_VARIABLE check_output
    )

    if(${check_result_var})
        set(${out_var} ON PARENT_SCOPE)
        set(${out_var}_REASON "" PARENT_SCOPE)
    else()
        file(WRITE "${check_log}" "${check_output}")
        set(${out_var} OFF PARENT_SCOPE)
        set(${out_var}_REASON "A compile/link check failed; see ${check_log}." PARENT_SCOPE)
    endif()
endfunction()

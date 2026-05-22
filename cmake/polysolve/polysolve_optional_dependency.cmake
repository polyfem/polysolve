include_guard(GLOBAL)

function(polysolve_note_disabled_dependency name reason)
    message(WARNING "${name} requested but unavailable; disabling ${name}. ${reason}")
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

# Try to define mkl::mkl from an activated Intel oneAPI MKL installation;
# this is indicated by the presence of MKL environment variables.
# This is intentionally optional: if no MKL root is configured, it leaves
# mkl::mkl undefined so the caller can fall back to the old CPM download route.

if(TARGET mkl::mkl)
    return()
endif()

set(MKL_ROOT "" CACHE PATH "Path to an existing Intel oneAPI MKL installation")
if(MKL_ROOT)
    set(_mkl_root "${MKL_ROOT}")
elseif(DEFINED ENV{MKL_ROOT})
    set(_mkl_root "$ENV{MKL_ROOT}")
elseif(DEFINED ENV{MKLROOT})
    set(_mkl_root "$ENV{MKLROOT}")
else()
    return()
endif()

get_filename_component(_mkl_root "${_mkl_root}" ABSOLUTE)

function(_system_mkl_assert_is_found var)
    if(NOT ${var})
        message(FATAL_ERROR "Could not find ${var} in MKL root: ${_mkl_root}")
    endif()
endfunction()

message(STATUS "Using system MKL from: ${_mkl_root}")

find_path(MKL_INCLUDE_DIR
    NAMES mkl.h
    HINTS "${_mkl_root}/include"
    NO_DEFAULT_PATH
)
_system_mkl_assert_is_found(MKL_INCLUDE_DIR)
message(STATUS "MKL include dir: ${MKL_INCLUDE_DIR}")

set(_mkl_lib_hints
    "${_mkl_root}/lib/intel64"
    "${_mkl_root}/lib"
)

if(MKL_LINKING STREQUAL static)
    set(_mkl_type STATIC)
else()
    set(_mkl_type SHARED)
endif()

add_library(mkl::mkl INTERFACE IMPORTED GLOBAL)
target_include_directories(mkl::mkl INTERFACE ${MKL_INCLUDE_DIR})

function(_system_mkl_add_imported_library name)
    string(TOUPPER mkl_${name}_library _libvar)
    set(_old_find_library_suffixes ${CMAKE_FIND_LIBRARY_SUFFIXES})
    if(MKL_LINKING STREQUAL static)
        set(CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_STATIC_LIBRARY_SUFFIX})
    elseif(WIN32)
        set(CMAKE_FIND_LIBRARY_SUFFIXES ".lib")
    else()
        set(CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_SHARED_LIBRARY_SUFFIX})
    endif()
    find_library(${_libvar}
        NAMES mkl_${name}
        HINTS ${_mkl_lib_hints}
        NO_DEFAULT_PATH
    )
    set(CMAKE_FIND_LIBRARY_SUFFIXES ${_old_find_library_suffixes})
    _system_mkl_assert_is_found(${_libvar})
    message(STATUS "Creating target mkl::${name} for lib file: ${${_libvar}}")

    add_library(mkl::${name} ${_mkl_type} IMPORTED GLOBAL)
    set_target_properties(mkl::${name} PROPERTIES
        IMPORTED_LOCATION "${${_libvar}}"
        IMPORTED_LINK_INTERFACE_LANGUAGES CXX
    )

    target_link_libraries(mkl::mkl INTERFACE mkl::${name})
endfunction()

function(_system_mkl_set_static_dependencies)
    if(NOT MKL_LINKING STREQUAL static)
        return()
    endif()

    set(_shifted ${ARGN})
    list(POP_FRONT _shifted _first_item)
    list(APPEND _shifted ${_first_item})
    foreach(_a _b IN ZIP_LISTS ARGN _shifted)
        set_target_properties(mkl::${_a} PROPERTIES INTERFACE_LINK_LIBRARIES mkl::${_b})
    endforeach()

    set_property(TARGET mkl::${_first_item} PROPERTY IMPORTED_LINK_INTERFACE_MULTIPLICITY 4)
endfunction()

_system_mkl_add_imported_library(core)
_system_mkl_add_imported_library(intel_${MKL_INTERFACE})

if(MKL_THREADING STREQUAL sequential)
    _system_mkl_add_imported_library(sequential)
    _system_mkl_set_static_dependencies(core intel_${MKL_INTERFACE} sequential)
else()
    _system_mkl_add_imported_library(tbb_thread)
    _system_mkl_set_static_dependencies(core intel_${MKL_INTERFACE} tbb_thread)
    include(onetbb)
    target_link_libraries(mkl::tbb_thread INTERFACE TBB::tbb)
endif()

if(MKL_INTERFACE STREQUAL "ilp64")
    target_compile_definitions(mkl::mkl INTERFACE MKL_ILP64)
endif()

if(NOT MSVC)
    find_package(Threads REQUIRED)
    find_library(LIBM_LIBRARY m)
    _system_mkl_assert_is_found(LIBM_LIBRARY)
    target_link_libraries(mkl::mkl INTERFACE Threads::Threads ${LIBM_LIBRARY} ${CMAKE_DL_LIBS})
endif()

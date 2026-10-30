# METIS (https://github.com/KarypisLab/METIS), with GKlib (https://github.com/KarypisLab/GKlib)
# License: Apache 2.0

if(TARGET METIS::metis)
    return()
endif()

message(STATUS "Third-party: creating target 'METIS::metis'")

# Neither project can be add_subdirectory()'d: METIS's CMake expects GKlib to
# be installed under CMAKE_INSTALL_PREFIX already, and a metis.h that its
# `make config` step fills in. So both are fetched as plain sources and
# compiled here into a single static library.
include(CPM)
CPMAddPackage(
    NAME gklib
    GITHUB_REPOSITORY KarypisLab/GKlib
    # Pinned: GKlib's master has since moved its sources into src/ and include/.
    GIT_TAG 8bd6bad750b2b0d90800c632cf18e8ee93ad72d7
    DOWNLOAD_ONLY YES
)
CPMAddPackage(
    NAME metis
    GITHUB_REPOSITORY KarypisLab/METIS
    GIT_TAG v5.2.1
    DOWNLOAD_ONLY YES
)

enable_language(C)

# metis.h ships with its index and real widths commented out, for `make config`
# to fill in. Fill in 32 bits, matching the int indices of the Eigen matrices
# every caller here passes in. file(CONFIGURE) rewrites the header only when
# its content changes, so a reconfigure does not trigger a rebuild.
file(READ "${metis_SOURCE_DIR}/include/metis.h" METIS_HEADER)
string(REPLACE "//#define IDXTYPEWIDTH 32" "#define IDXTYPEWIDTH 32" METIS_HEADER "${METIS_HEADER}")
string(REPLACE "//#define REALTYPEWIDTH 32" "#define REALTYPEWIDTH 32" METIS_HEADER "${METIS_HEADER}")
file(CONFIGURE OUTPUT "${metis_BINARY_DIR}/include/metis.h" CONTENT "${METIS_HEADER}" @ONLY)

file(GLOB GKLIB_SOURCES "${gklib_SOURCE_DIR}/*.c")
file(GLOB METIS_SOURCES "${metis_SOURCE_DIR}/libmetis/*.c")

add_library(METIS_metis STATIC ${GKLIB_SOURCES} ${METIS_SOURCES})
add_library(METIS::metis ALIAS METIS_metis)
set_target_properties(METIS_metis PROPERTIES FOLDER third_party/metis)

target_include_directories(METIS_metis
    PUBLIC
        "${metis_BINARY_DIR}/include"
    PRIVATE
        "${gklib_SOURCE_DIR}"
        "${metis_SOURCE_DIR}/libmetis"
)

# The definitions GKlib's own build passes (see its GKlibSystem.cmake), minus
# -march=native and -Werror.
if(MSVC)
    target_include_directories(METIS_metis PRIVATE "${gklib_SOURCE_DIR}/win32")
    target_compile_definitions(METIS_metis PRIVATE WIN32 MSC _CRT_SECURE_NO_DEPRECATE USE_GKREGEX "__thread=__declspec(thread)")
elseif(MINGW)
    target_compile_definitions(METIS_metis PRIVATE USE_GKREGEX)
else()
    target_compile_definitions(METIS_metis PRIVATE LINUX _FILE_OFFSET_BITS=64)
endif()
target_compile_definitions(METIS_metis PRIVATE NDEBUG NDEBUG2)

if(UNIX)
    target_link_libraries(METIS_metis PRIVATE m)
endif()

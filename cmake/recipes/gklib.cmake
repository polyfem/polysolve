if(TARGET GKlib::GKlib)
    return()
endif()

message(STATUS "Third-party: creating target 'GKlib::GKlib'")

# Note: the version here is chosen to match `wildmeshing-toolkit`.
include(FetchContent)
FetchContent_Declare(
    gklib
    GIT_REPOSITORY https://github.com/KarypisLab/GKlib.git
    GIT_TAG        a7f8172703cf6e999dd0710eb279bba513da4fec
)

FetchContent_GetProperties(gklib)
if(NOT gklib_POPULATED)
    FetchContent_Populate(gklib)
endif()

file(GLOB INC_FILES "${gklib_SOURCE_DIR}/*.h")
file(GLOB SRC_FILES "${gklib_SOURCE_DIR}/*.c")
if(NOT MSVC)
    list(REMOVE_ITEM SRC_FILES "${gklib_SOURCE_DIR}/gkregex.c")
endif()

add_library(GKlib STATIC ${INC_FILES} ${SRC_FILES})
add_library(GKlib::GKlib ALIAS GKlib)

if(MSVC)
    target_compile_definitions(GKlib PUBLIC USE_GKREGEX)
    target_compile_definitions(GKlib PUBLIC "__thread=__declspec(thread)")
endif()

file(MAKE_DIRECTORY "${gklib_BINARY_DIR}/include/gklib")
configure_file("${gklib_SOURCE_DIR}/gk_ms_stdint.h" "${gklib_BINARY_DIR}/include/gklib/ms_stdint.h" COPYONLY)

target_include_directories(GKlib SYSTEM PUBLIC
    "$<BUILD_INTERFACE:${gklib_SOURCE_DIR}>"
    "$<BUILD_INTERFACE:${gklib_BINARY_DIR}/include/gklib>"
)

set_target_properties(GKlib PROPERTIES FOLDER third_party)

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang" OR
   "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    target_compile_options(GKlib PRIVATE
        "-Wno-unused-variable"
        "-Wno-sometimes-uninitialized"
        "-Wno-absolute-value"
        "-Wno-shadow"
    )
elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
    target_compile_options(GKlib PRIVATE "-w")
elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL "MSVC")
    target_compile_options(GKlib PRIVATE "/w")
endif()

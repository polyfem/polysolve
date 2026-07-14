if(TARGET metis::metis)
    return()
endif()

message(STATUS "Third-party: creating target 'metis::metis'")

# Note: the version/compilation settings defined in here are chosen to match
# those in `wildmeshing-toolkit`.
include(FetchContent)
FetchContent_Declare(
    metis
    GIT_REPOSITORY https://github.com/KarypisLab/METIS.git
    GIT_TAG        94c03a6e2d1860128c2d0675cbbb86ad4f261256
)

FetchContent_GetProperties(metis)
if(NOT metis_POPULATED)
    FetchContent_Populate(metis)
endif()

file(GLOB INC_FILES "${metis_SOURCE_DIR}/libmetis/*.h")
file(GLOB SRC_FILES "${metis_SOURCE_DIR}/libmetis/*.c")

add_library(metis STATIC ${INC_FILES} ${SRC_FILES})
add_library(metis::metis ALIAS metis)

include(gklib)
target_link_libraries(metis PRIVATE GKlib::GKlib)

target_include_directories(metis PRIVATE "${metis_SOURCE_DIR}/libmetis")
target_include_directories(metis SYSTEM PUBLIC "$<BUILD_INTERFACE:${metis_SOURCE_DIR}/include>")

target_compile_definitions(metis PUBLIC -DIDXTYPEWIDTH=32)
target_compile_definitions(metis PUBLIC -DREALTYPEWIDTH=32)

set(POLYSOLVE_METIS_SOURCE_DIR "${metis_SOURCE_DIR}" CACHE INTERNAL "")
set_target_properties(metis PROPERTIES FOLDER third_party)

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang" OR
   "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    target_compile_options(metis PRIVATE
        "-Wno-unused-variable"
        "-Wno-sometimes-uninitialized"
        "-Wno-absolute-value"
        "-Wno-shadow"
    )
elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
    target_compile_options(metis PRIVATE "-w")
elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL "MSVC")
    target_compile_options(metis PRIVATE "/w")
endif()

# Sanitizers MIT (optional)

message(STATUS "Third-party: creating 'Sanitizers'")

include(FetchContent)
FetchContent_Declare(
    sanitizers-cmake
    GIT_REPOSITORY https://github.com/arsenm/sanitizers-cmake.git
    GIT_TAG 6947cff3a9c9305eb9c16135dd81da3feb4bf87f
    GIT_SHALLOW FALSE
)
FetchContent_GetProperties(sanitizers-cmake)
if(NOT sanitizers-cmake_POPULATED)
    FetchContent_Populate(sanitizers-cmake)
endif()

list(APPEND CMAKE_MODULE_PATH ${sanitizers-cmake_SOURCE_DIR}/cmake)

find_package(Sanitizers)

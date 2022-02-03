# Spectra MPL 2.0 (optional)

if(TARGET spectra::spectra)
    return()
endif()

message(STATUS "Third-party: creating target 'spectra::spectra'")

include(FetchContent)
FetchContent_Declare(
    spectra
    GIT_REPOSITORY https://github.com/yixuan/spectra.git
    GIT_TAG v0.6.2
    GIT_SHALLOW TRUE
)
FetchContent_GetProperties(spectra)
if(NOT spectra_POPULATED)
    FetchContent_Populate(spectra)
endif()

add_library(spectra INTERFACE)
add_library(spectra::spectra ALIAS spectra)

target_include_directories(spectra SYSTEM INTERFACE ${spectra_SOURCE_DIR}/include)

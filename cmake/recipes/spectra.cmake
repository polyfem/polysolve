# Spectra MPL 2.0 (optional)

if(TARGET Spectra::Spectra)
    return()
endif()

message(STATUS "Third-party: creating target 'Spectra::Spectra'")

include(CPM)
CPMAddPackage(
    NAME spectra
    GITHUB_REPOSITORY yixuan/spectra
    GIT_TAG v0.6.2
    DOWNLOAD_ONLY ON
)

add_library(Spectra INTERFACE)
add_library(Spectra::Spectra ALIAS Spectra)

target_include_directories(Spectra SYSTEM INTERFACE ${spectra_SOURCE_DIR}/include)

include(eigen)
target_link_libraries(Spectra INTERFACE Eigen3::Eigen)

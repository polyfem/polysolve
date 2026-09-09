# cuDSS solver

if(TARGET cudss)
    return()
endif()

message(STATUS "Third-party: creating target 'cudss'")

set(CUDSS_URL
    "https://developer.download.nvidia.com/compute/cudss/redist/libcudss/linux-x86_64/libcudss-linux-x86_64-0.7.1.4_cuda13-archive.tar.xz"
    CACHE STRING "cuDSS download URL")
set(CUDSS_URL_SHA256
    "84b34ebe7fad40ec10f2aab2957a63b6070bd8ce16e3ada3e6bcac7317256347"
    CACHE STRING "cuDSS download URL SHA256 checksum")

include(CPM)
CPMAddPackage(
    NAME cudss
    URL ${CUDSS_URL}
    URL_HASH SHA256=${CUDSS_URL_SHA256}
    DOWNLOAD_ONLY ON
)

find_package(cudss CONFIG REQUIRED
    PATHS "${cudss_SOURCE_DIR}/lib/cmake/cudss"
    NO_DEFAULT_PATH
)

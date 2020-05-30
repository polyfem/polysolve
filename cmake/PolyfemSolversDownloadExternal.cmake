################################################################################
include(DownloadProject)

# With CMake 3.8 and above, we can hide warnings about git being in a
# detached head by passing an extra GIT_CONFIG option
if(NOT (${CMAKE_VERSION} VERSION_LESS "3.8.0"))
    set(POLYFEM_SOLVERS_EXTRA_OPTIONS "GIT_CONFIG advice.detachedHead=false")
else()
    set(POLYFEM_SOLVERS_EXTRA_OPTIONS "")
endif()

# Shortcut function
function(polyfem_solvers_download_project name)
    download_project(
        PROJ         ${name}
        SOURCE_DIR   ${POLYFEM_SOLVERS_EXTERNAL}/${name}
        DOWNLOAD_DIR ${POLYFEM_SOLVERS_EXTERNAL}/.cache/${name}
        QUIET
        ${POLYFEM_SOLVERS_EXTRA_OPTIONS}
        ${ARGN}
    )
endfunction()

################################################################################

## Catch2 BSL 1.0 optional
function(polyfem_solvers_download_catch2)
    polyfem_solvers_download_project(Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG        v2.12.2
    )
endfunction()


## spectra MPL 2.0 optional
function(polyfem_solvers_download_spectra)
    polyfem_solvers_download_project(spectra
        GIT_REPOSITORY https://github.com/yixuan/spectra.git
        GIT_TAG        v0.6.2
    )
endfunction()


## hypre GNU Lesser General Public License
function(polyfem_solvers_download_hypre)
    polyfem_solvers_download_project(hypre
        GIT_REPOSITORY https://github.com/LLNL/hypre.git
        GIT_TAG        v2.15.1
    )

    file(REMOVE ${POLYFEM_SOLVERS_EXTERNAL}/hypre/src/utilities/version)
endfunction()


## Sanitizers MIT optional
function(polyfem_solvers_download_sanitizers)
    polyfem_solvers_download_project(sanitizers-cmake
        GIT_REPOSITORY https://github.com/arsenm/sanitizers-cmake.git
        GIT_TAG        6947cff3a9c9305eb9c16135dd81da3feb4bf87f
    )
endfunction()


## amgcl mit
function(polyfem_solvers_download_amgcl)
    polyfem_solvers_download_project(amgcl
        GIT_REPOSITORY https://github.com/ddemidov/amgcl.git
        GIT_TAG        a2fab1037946de87e448e5fc7539277cd6fb9ec3
    )
endfunction()

## Json MIT
function(polyfem_solvers_download_json)
    polyfem_solvers_download_project(json
        GIT_REPOSITORY https://github.com/jdumas/json
        GIT_TAG        0901d33bf6e7dfe6f70fd9d142c8f5c6695c6c5b
    )
endfunction()
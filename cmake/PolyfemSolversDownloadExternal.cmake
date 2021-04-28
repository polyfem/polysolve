################################################################################
include(DownloadProject)

# With CMake 3.8 and above, we can hide warnings about git being in a
# detached head by passing an extra GIT_CONFIG option
if(NOT (${CMAKE_VERSION} VERSION_LESS "3.8.0"))
    set(POLYSOLVE_EXTRA_OPTIONS "GIT_CONFIG advice.detachedHead=false")
else()
    set(POLYSOLVE_EXTRA_OPTIONS "")
endif()

# Shortcut function
function(polyfem_solvers_download_project name)
    download_project(
        PROJ         ${name}
        SOURCE_DIR   ${POLYSOLVE_EXTERNAL}/${name}
        DOWNLOAD_DIR ${POLYSOLVE_EXTERNAL}/.cache/${name}
        QUIET
        ${POLYSOLVE_EXTRA_OPTIONS}
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

function(polyfem_solvers_download_eigen)
	polyfem_solvers_download_project(eigen
		GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
		GIT_TAG        3.3.7
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
        GIT_REPOSITORY https://github.com/hypre-space/hypre.git
        GIT_TAG        v2.15.1
    )

    file(REMOVE ${POLYSOLVE_EXTERNAL}/hypre/src/utilities/version)
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
        GIT_TAG        b1c1ec55da829ebcbf9d854b641606ff415ee6bb
    )
endfunction()

## Json MIT
function(polyfem_solvers_download_json)
    polyfem_solvers_download_project(json
        URL     https://github.com/nlohmann/json/releases/download/v3.9.1/include.zip
        URL_MD5 d2f66c608af689e21d69a33c220e974e
    )
endfunction()


## data
function(polyfem_solvers_download_polyfem_data)
    polyfem_solvers_download_project(data
        GIT_REPOSITORY https://github.com/polyfem/polyfem-data
        GIT_TAG        1e9a93d5c7ab5f6f386edd6b7ff1a78871553af6
    )
endfunction()

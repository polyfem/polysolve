if(TARGET polyfem::data)
    return()
endif()

include(ExternalProject)
include(FetchContent)

set(POLYFEM_DATA_ROOT "${PROJECT_SOURCE_DIR}/data/" CACHE PATH "Where should we download test data?")

ExternalProject_Add(
    polyfem_data_download
    PREFIX "${FETCHCONTENT_BASE_DIR}/polyfem-test-data"
    SOURCE_DIR ${POLYFEM_DATA_ROOT}

    GIT_REPOSITORY https://github.com/polyfem/polyfem-data
    GIT_TAG 8a2eff19a33ccfd395c9294342e2c5c2babf2626

    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    LOG_DOWNLOAD ON
)

# Create a dummy target for convenience
add_library(polyfem_data INTERFACE)
add_library(polyfem::data ALIAS polyfem_data)

add_dependencies(polyfem_data polyfem_data_download)

target_compile_definitions(polyfem_data INTERFACE POLYFEM_DATA_DIR=\"${POLYFEM_DATA_ROOT}\")

# metis (https://github.com/KarypisLab/METIS)
# License: Apache 2

if(TARGET metis::metis)
    return()
endif()

message(STATUS "Third-party: creating target 'metis::metis'")

include(CPM)
# This fork provides proper cmake setup for METIS 5.2.1.
# The upstream CMakeLists.txt is not usable.
CPMAddPackage(
    NAME METIS
    GITHUB_REPOSITORY scivision/METIS
    GIT_TAG 777472ae3cd15a8e6d1e5b7d6c347d21947e3ab2
)

# MeshFEMSparse

if(TARGET MeshFEMSparse)
    return()
endif()

message(STATUS "Third-party: creating target 'MeshFEMSparse'")

set(MESHFEMSPARSE_GIT_REPOSITORY "https://github.com/MeshFEM/MeshFEMSparse.git" CACHE STRING "MeshFEMSparse git repository")
set(MESHFEMSPARSE_GIT_TAG "main" CACHE STRING "MeshFEMSparse git revision")

include(CPM)
CPMAddPackage(
    NAME MeshFEMSparse
    GIT_REPOSITORY ${MESHFEMSPARSE_GIT_REPOSITORY}
    GIT_TAG ${MESHFEMSPARSE_GIT_TAG}
)

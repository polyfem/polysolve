# HYPRE GNU Lesser General Public License

if(TARGET HYPRE::HYPRE)
    return()
endif()

message(STATUS "Third-party: creating target 'HYPRE::HYPRE'")

set(HYPRE_WITH_MPI    OFF CACHE INTERNAL "" FORCE)
set(HYPRE_PRINT_ERRORS  ON CACHE INTERNAL "" FORCE)
set(HYPRE_BIGINT        ON CACHE INTERNAL "" FORCE)
set(HYPRE_USING_FEI    OFF CACHE INTERNAL "" FORCE)
set(HYPRE_USING_OPENMP OFF CACHE INTERNAL "" FORCE)
set(HYPRE_SHARED       OFF CACHE INTERNAL "" FORCE)

include(CPM)
CPMAddPackage(
    NAME hypre
    GITHUB_REPOSITORY hypre-space/hypre
    GIT_TAG v2.28.0
    SOURCE_SUBDIR src
)
file(REMOVE "${hypre_SOURCE_DIR}/src/utilities/version")

# HYPRE GNU Lesser General Public License

if(TARGET HYPRE::HYPRE)
    return()
endif()

message(STATUS "Third-party: creating target 'HYPRE::HYPRE'")

set(HYPRE_SEQUENTIAL    ON CACHE INTERNAL "" FORCE)
set(HYPRE_PRINT_ERRORS  ON CACHE INTERNAL "" FORCE)
set(HYPRE_BIGINT        ON CACHE INTERNAL "" FORCE)
set(HYPRE_USING_FEI    OFF CACHE INTERNAL "" FORCE)
set(HYPRE_USING_OPENMP OFF CACHE INTERNAL "" FORCE)
set(HYPRE_SHARED       OFF CACHE INTERNAL "" FORCE)


include(FetchContent)
FetchContent_Declare(
    hypre
    GIT_REPOSITORY https://github.com/hypre-space/hypre.git
    GIT_TAG v2.25.0
    GIT_SHALLOW TRUE
)

FetchContent_MakeAvailable(hypre)

add_subdirectory("${hypre_SOURCE_DIR}/src" ${hypre_BINARY_DIR})
file(REMOVE "${hypre_SOURCE_DIR}/src/utilities/version")
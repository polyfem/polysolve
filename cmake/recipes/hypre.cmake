# HYPRE GNU Lesser General Public License

if(TARGET HYPRE::HYPRE)
    return()
endif()

include(polysolve_optional_dependency)

set(POLYSOLVE_HYPRE_CHECK_SOURCE [[
#include <HYPRE.h>
#include <HYPRE_parcsr_ls.h>
int main()
{
    HYPRE_Solver solver = nullptr;
    HYPRE_BoomerAMGCreate(&solver);
    HYPRE_BoomerAMGDestroy(solver);
    return 0;
}
]])

polysolve_find_system_dependency(HYPRE_SYSTEM_FOUND
    NAME HYPRE
    PACKAGE HYPRE
    TARGET HYPRE::HYPRE
    CONFIG
    SOURCE_VAR POLYSOLVE_HYPRE_CHECK_SOURCE
)
if(HYPRE_SYSTEM_FOUND)
    return()
endif()

polysolve_should_fetch_dependency(HYPRE_SHOULD_FETCH HYPRE)
if(NOT HYPRE_SHOULD_FETCH)
    polysolve_note_disabled_dependency("HYPRE" "HYPRE was not found and fetching is disabled.")
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

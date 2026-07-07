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

set(HYPRE_ENABLE_MPI           OFF CACHE INTERNAL "" FORCE)
set(HYPRE_ENABLE_PRINT_ERRORS  ON  CACHE INTERNAL "" FORCE)
set(HYPRE_ENABLE_BIGINT        OFF CACHE INTERNAL "" FORCE)
set(HYPRE_ENABLE_MIXEDINT      OFF CACHE BOOL     "" FORCE)
set(HYPRE_ENABLE_FEI           OFF CACHE INTERNAL "" FORCE)
set(HYPRE_ENABLE_OPENMP        OFF CACHE INTERNAL "" FORCE)
set(HYPRE_ENABLE_UMPIRE        OFF CACHE INTERNAL "" FORCE)

if (POLYSOLVE_WITH_CUDA)
    set(HYPRE_USING_GPU            ON  CACHE INTERNAL "" FORCE)
    set(HYPRE_ENABLE_CUDA          ON  CACHE INTERNAL "" FORCE)
else()
    set(HYPRE_USING_GPU            OFF CACHE INTERNAL "" FORCE)
    set(HYPRE_ENABLE_CUDA          OFF CACHE INTERNAL "" FORCE)
endif()

# HYPRE unconditionally defines an "uninstall" target, which conflicts with other buggy libraries
# as modern cmake requires unique target name. This is a hacky workaround until upstream is fixed.
macro(add_custom_target _target_name)
  if("${_target_name}" STREQUAL "uninstall" AND TARGET uninstall)
    # skip: HYPRE's uninstall target conflicts with an existing one
  else()
    _add_custom_target(${_target_name} ${ARGN})
  endif()
endmacro()

include(CPM)
CPMAddPackage(
    NAME hypre
    GITHUB_REPOSITORY hypre-space/hypre
    GIT_TAG 7e247a231ebdeb44b06c7c9d3b5bee3bac21123f
    SOURCE_SUBDIR src
)

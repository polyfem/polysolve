# HYPRE GNU Lesser General Public License

if(TARGET HYPRE::HYPRE)
    return()
endif()

message(STATUS "Third-party: creating target 'HYPRE::HYPRE'")

set(HYPRE_ENABLE_MPI           OFF CACHE INTERNAL "" FORCE)
set(HYPRE_ENABLE_PRINT_ERRORS  ON  CACHE INTERNAL "" FORCE)
set(HYPRE_ENABLE_BIGINT        ON  CACHE INTERNAL "" FORCE)
set(HYPRE_ENABLE_FEI           OFF CACHE INTERNAL "" FORCE)
set(HYPRE_ENABLE_OPENMP        OFF CACHE INTERNAL "" FORCE)
set(HYPRE_SHARED               OFF CACHE INTERNAL "" FORCE)

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
    GIT_TAG v3.1.0
    SOURCE_SUBDIR src
)
file(REMOVE "${hypre_SOURCE_DIR}/src/utilities/version")

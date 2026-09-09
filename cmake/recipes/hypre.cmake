# HYPRE GNU Lesser General Public License

if(TARGET HYPRE::HYPRE)
    return()
endif()

message(STATUS "Third-party: creating target 'HYPRE::HYPRE'")

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

# hypre is built as an ordinary MPI build. What is different is the MPI: with
# POLYSOLVE_WITH_MPI on, the nanompi recipe has already put a FindMPI shim on the
# module path, so hypre's own find_package(MPI REQUIRED) resolves to nano-mpi --
# whose ranks are threads of this process. hypre itself needs no special mode.
if (POLYSOLVE_WITH_MPI)
    include(nanompi)
    set(HYPRE_ENABLE_MPI ON  CACHE INTERNAL "" FORCE)
else()
    set(HYPRE_ENABLE_MPI OFF CACHE INTERNAL "" FORCE)
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
    GITHUB_REPOSITORY danielepanozzo/hypre
    GIT_TAG thread-mpi-backend
    SOURCE_SUBDIR src
)

# hypre links MPI::MPI_C, which here carries only the header path -- see the
# FindMPI shim for why it cannot carry the library. Supply the library here.
if (POLYSOLVE_WITH_MPI)
    target_link_libraries(HYPRE PUBLIC nanompi::nanompi)
endif()

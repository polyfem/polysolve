# HYPRE GNU Lesser General Public License

if(TARGET HYPRE::HYPRE)
    return()
endif()

message(STATUS "Third-party: creating target 'HYPRE::HYPRE'")

set(HYPRE_ENABLE_MPI           OFF CACHE INTERNAL "" FORCE)
set(HYPRE_ENABLE_PRINT_ERRORS  ON  CACHE INTERNAL "" FORCE)
# TODO: Enable GPU accelerated HYPRE conditionally.
set(HYPRE_USING_GPU            OFF CACHE INTERNAL "" FORCE)
set(HYPRE_ENABLE_CUDA          OFF  CACHE INTERNAL "" FORCE)
set(HYPRE_ENABLE_BIGINT        OFF CACHE INTERNAL "" FORCE)
set(HYPRE_ENABLE_MIXEDINT      OFF CACHE BOOL     "" FORCE)
set(HYPRE_ENABLE_FEI           OFF CACHE INTERNAL "" FORCE)
set(HYPRE_ENABLE_OPENMP        OFF CACHE INTERNAL "" FORCE)
set(HYPRE_ENABLE_UMPIRE        OFF CACHE INTERNAL "" FORCE)

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

# HYPRE v3.1.0 relies on transitive Thrust includes that were removed in CCCL 3.2+.
# Patch HYPRE source.

function(polysolve_prepend_hypre_include file_path include_line)
    if(NOT EXISTS "${file_path}")
        message(FATAL_ERROR "Expected HYPRE source file does not exist: ${file_path}")
    endif()

    file(READ "${file_path}" file_contents)
    string(FIND "${file_contents}" "${include_line}" include_pos)
    if(include_pos EQUAL -1)
        file(WRITE "${file_path}" "${include_line}\n${file_contents}")
    endif()
endfunction()

polysolve_prepend_hypre_include(
    "${hypre_SOURCE_DIR}/src/utilities/device_utils.c"
    "#include <thrust/pair.h>")
polysolve_prepend_hypre_include(
    "${hypre_SOURCE_DIR}/src/seq_mv/csr_matop_device.c"
    "#include <thrust/pair.h>")
polysolve_prepend_hypre_include(
    "${hypre_SOURCE_DIR}/src/IJ_mv/IJMatrix_parcsr_device.c"
    "#include <thrust/iterator/reverse_iterator.h>")
polysolve_prepend_hypre_include(
    "${hypre_SOURCE_DIR}/src/IJ_mv/IJVector_parcsr_device.c"
    "#include <thrust/iterator/reverse_iterator.h>")

file(REMOVE "${hypre_SOURCE_DIR}/src/utilities/version")

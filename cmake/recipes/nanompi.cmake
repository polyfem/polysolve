# nano-mpi -- MPI ranks as threads of this process (https://github.com/danielepanozzo/nano-mpi)
#
# The hybrid solver wants hypre's MPI algorithms, which are the fast ones, but
# polysolve is a library: it cannot ask the application that links it to relaunch
# itself under mpirun. nano-mpi resolves that by making each rank a thread, so
# an ordinary single-threaded caller gets the domain-decomposed solve anyway.

# From here on, find_package(MPI) inside this build resolves to nano-mpi rather
# than to a system Open MPI. hypre calls it, and would otherwise link an MPI
# whose ranks are processes and then need a launcher we cannot provide.
# include() does not open a scope, so this reaches the subdirectories added
# after it -- which is exactly the set that must see it. Done before the
# already-built guard, so a second include() still gets the module path.
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/nanompi-as-mpi/")
list(REMOVE_DUPLICATES CMAKE_MODULE_PATH)

# Promote to CACHE so the shim is visible in every directory scope, not just
# ones that inherit this list(APPEND) through add_subdirectory(). A dependency
# added from a scope that never saw it (e.g. AMGCL's own optional
# find_package(MPI) call) would otherwise fall through to CMake's real
# FindMPI, which unconditionally overwrites MPI::MPI_C's
# INTERFACE_INCLUDE_DIRECTORIES on every invocation (see _MPI_create_imported_target
# in CMake's own FindMPI.cmake) -- silently re-pointing everyone's
# #include <mpi.h>, hypre included, at a real system MPI nano-mpi cannot
# actually run against.
set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" CACHE STRING "" FORCE)

if(TARGET nanompi::nanompi)
    return()
endif()

message(STATUS "Third-party: creating target 'nanompi::nanompi'")

set(NANOMPI_BUILD_TESTS    OFF CACHE INTERNAL "" FORCE)
set(NANOMPI_BUILD_WRAPPERS OFF CACHE INTERNAL "" FORCE)
# hypre exports its own targets and links this one, so it has to be in an
# export set of its own -- and an installed polysolve linking a static hypre
# needs libnanompi at link time regardless.
set(NANOMPI_INSTALL        ON  CACHE INTERNAL "" FORCE)

include(CPM)
CPMAddPackage(
    NAME nanompi
    GITHUB_REPOSITORY danielepanozzo/nano-mpi
    GIT_TAG v0.1.1
)

# The header path as a plain directory, for the FindMPI shim. It cannot get it
# from the target: an imported MPI::MPI_C must not name a project target in its
# link interface, because try_compile() -- which hypre uses to probe for
# MPI_Comm_f2c -- exports that target into a scratch project where nanompi does
# not exist. So MPI::MPI_C carries includes only, and whoever links MPI also
# links nanompi::nanompi explicitly.
set(NANOMPI_INCLUDE_DIR "${nanompi_SOURCE_DIR}/include"
    CACHE PATH "nano-mpi headers" FORCE)

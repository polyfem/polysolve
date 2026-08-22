#
# Inside this build, "MPI" is nano-mpi.
#
# nano-mpi implements MPI with every rank a thread of this process, so there is
# nothing to search for: it is an ordinary CMake target, built from source
# alongside everything else. This module exists so that dependencies which call
# find_package(MPI) -- hypre does, in HYPRE_CMakeUtilities.cmake -- resolve to
# that target instead of to a system Open MPI, which would drag in a launcher
# polysolve has no way to invoke.
#
# It is only on CMAKE_MODULE_PATH when the nanompi recipe has been included;
# without that, CMake's own FindMPI is found as usual.
#

if(NOT TARGET nanompi::nanompi)
    message(FATAL_ERROR
        "This FindMPI shim is on the module path but nano-mpi is not in the "
        "build. Include the nanompi recipe before anything calls "
        "find_package(MPI).")
endif()

if("Fortran" IN_LIST MPI_FIND_COMPONENTS)
    message(FATAL_ERROR
        "MPI Fortran bindings were requested, and nano-mpi has none. It cannot: "
        "Fortran SAVE and COMMON storage is per-process by language rule, and "
        "nano-mpi's ranks are threads of one process.")
endif()

# MPI::MPI_C carries the include path and NOTHING ELSE, deliberately. hypre
# puts it in CMAKE_REQUIRED_LIBRARIES and runs check_c_source_compiles, and
# try_compile() exports the target into a scratch project -- where a link
# interface naming nanompi::nanompi is a hard error, because that target does
# not exist there. Whoever links MPI links nanompi::nanompi explicitly instead;
# the hypre recipe and polysolve's own CMakeLists both do.
#
# The probe hypre runs is for MPI_Comm_f2c, which nano-mpi does not declare
# (it has no Fortran bindings, and cannot), so it correctly comes out false.
foreach(_lang IN ITEMS C CXX)
    if(NOT TARGET MPI::MPI_${_lang})
        # GLOBAL, so the subdirectories CPM adds later can use it
        add_library(MPI::MPI_${_lang} INTERFACE IMPORTED GLOBAL)
        set_target_properties(MPI::MPI_${_lang} PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${NANOMPI_INCLUDE_DIR}")
    endif()

    set(MPI_${_lang}_FOUND               TRUE)
    set(MPI_${_lang}_COMPILER            "${CMAKE_${_lang}_COMPILER}")
    set(MPI_${_lang}_LIBRARIES           "")
    set(MPI_${_lang}_INCLUDE_DIRS        "${NANOMPI_INCLUDE_DIR}")
    set(MPI_${_lang}_INCLUDE_DIR         "${NANOMPI_INCLUDE_DIR}")
    set(MPI_${_lang}_INCLUDE_PATH        "${NANOMPI_INCLUDE_DIR}")
    set(MPI_${_lang}_COMPILE_OPTIONS     "")
    set(MPI_${_lang}_COMPILE_DEFINITIONS "")
    set(MPI_${_lang}_LINK_FLAGS          "")
    set(MPI_${_lang}_VERSION             "3.1")
endforeach()

set(MPI_FOUND   TRUE)
set(MPI_VERSION "3.1")

# There is no launcher. nanompiexec exists for scripts that expect one, but it
# only translates -n into an environment variable and execs the program.
set(MPIEXEC_EXECUTABLE   "" CACHE FILEPATH "nano-mpi has no launcher: ranks are threads")
set(MPIEXEC_NUMPROC_FLAG "-n")
set(MPIEXEC_MAX_NUMPROCS 1)

message(STATUS "MPI: using nano-mpi (ranks are threads of this process)")

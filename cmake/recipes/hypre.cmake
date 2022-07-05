# HYPRE GNU Lesser General Public License

if(TARGET HYPRE::HYPRE)
    return()
endif()

message(STATUS "Third-party: creating target 'HYPRE::HYPRE'")

include(FetchContent)
FetchContent_Declare(
    hypre
    GIT_REPOSITORY https://github.com/hypre-space/hypre.git
    GIT_TAG v2.15.1
    GIT_SHALLOW TRUE
)

FetchContent_GetProperties(hypre)
if(NOT hypre_POPULATED)
    FetchContent_Populate(hypre)
    file(REMOVE ${hypre_SOURCE_DIR}/src/utilities/version)
endif()

################################################################################

set(HYPRE_SEQUENTIAL    ON CACHE INTERNAL "" FORCE)
set(HYPRE_PRINT_ERRORS  ON CACHE INTERNAL "" FORCE)
set(HYPRE_BIGINT        ON CACHE INTERNAL "" FORCE)
set(HYPRE_USING_FEI    OFF CACHE INTERNAL "" FORCE)
set(HYPRE_USING_OPENMP ON CACHE INTERNAL "" FORCE)
set(HYPRE_SHARED       OFF CACHE INTERNAL "" FORCE)
# set(HYPRE_LONG_DOUBLE ON)

set(HYPRE_BUILD_TYPE "${CMAKE_BUILD_TYPE}" CACHE INTERNAL "" FORCE)

add_subdirectory(${hypre_SOURCE_DIR}/src ${hypre_BINARY_DIR})
add_library(HYPRE::HYPRE ALIAS HYPRE)

set_property(TARGET HYPRE PROPERTY FOLDER "dependencies")

target_include_directories(HYPRE PUBLIC ${hypre_BINARY_DIR})
target_include_directories(HYPRE PUBLIC ${hypre_SOURCE_DIR}/src)
target_include_directories(HYPRE PUBLIC ${hypre_SOURCE_DIR}/src/blas)
target_include_directories(HYPRE PUBLIC ${hypre_SOURCE_DIR}/src/lapack)
target_include_directories(HYPRE PUBLIC ${hypre_SOURCE_DIR}/src/utilities)
target_include_directories(HYPRE PUBLIC ${hypre_SOURCE_DIR}/src/multivector)
target_include_directories(HYPRE PUBLIC ${hypre_SOURCE_DIR}/src/krylov)
target_include_directories(HYPRE PUBLIC ${hypre_SOURCE_DIR}/src/seq_mv)
target_include_directories(HYPRE PUBLIC ${hypre_SOURCE_DIR}/src/parcsr_mv)
target_include_directories(HYPRE PUBLIC ${hypre_SOURCE_DIR}/src/parcsr_block_mv)
target_include_directories(HYPRE PUBLIC ${hypre_SOURCE_DIR}/src/distributed_matrix)
target_include_directories(HYPRE PUBLIC ${hypre_SOURCE_DIR}/src/IJ_mv)
target_include_directories(HYPRE PUBLIC ${hypre_SOURCE_DIR}/src/matrix_matrix)
target_include_directories(HYPRE PUBLIC ${hypre_SOURCE_DIR}/src/distributed_ls)
target_include_directories(HYPRE PUBLIC ${hypre_SOURCE_DIR}/src/distributed_ls/Euclid)
target_include_directories(HYPRE PUBLIC ${hypre_SOURCE_DIR}/src/distributed_ls/ParaSails)
target_include_directories(HYPRE PUBLIC ${hypre_SOURCE_DIR}/src/parcsr_ls)
target_include_directories(HYPRE PUBLIC ${hypre_SOURCE_DIR}/src/struct_mv)
target_include_directories(HYPRE PUBLIC ${hypre_SOURCE_DIR}/src/struct_ls)
target_include_directories(HYPRE PUBLIC ${hypre_SOURCE_DIR}/src/sstruct_mv)
target_include_directories(HYPRE PUBLIC ${hypre_SOURCE_DIR}/src/sstruct_ls)

if(HYPRE_USING_OPENMP)
	find_package(OpenMP QUIET REQUIRED)
	target_link_libraries(HYPRE PUBLIC OpenMP::OpenMP_CXX)
endif()

if(NOT HYPRE_SEQUENTIAL)
	find_package(MPI)
	if(MPI_CXX_FOUND)
		target_link_libraries(HYPRE PUBLIC MPI::MPI_CXX)
	endif()
endif()

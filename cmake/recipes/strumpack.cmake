# STRUMPACK (https://github.com/pghysels/STRUMPACK)
# License: BSD 3-Clause (LBNL)
#
# Sparse multifrontal LU, with optional low-rank (BLR, HSS) compression of its
# frontal matrices. CPUHybridSolver can use it to factorize its subdomains
# (the experimental use_strumpack option).

if(TARGET STRUMPACK::strumpack)
    return()
endif()

message(STATUS "Third-party: creating target 'STRUMPACK::strumpack'")

# STRUMPACK's project() declares Fortran (for BLAS name mangling and its
# Fortran interface), so it cannot even configure without a Fortran compiler.
include(CheckLanguage)
check_language(Fortran)
if(NOT CMAKE_Fortran_COMPILER)
    message(FATAL_ERROR
        "STRUMPACK needs a Fortran compiler, and none was found. Install one "
        "(e.g. gfortran) or turn POLYSOLVE_WITH_STRUMPACK off.")
endif()

include(metis)
include(blas)
include(lapack)

include(CPM)
CPMAddPackage(
    NAME strumpack
    GITHUB_REPOSITORY pghysels/STRUMPACK
    GIT_TAG v8.0.0
    DOWNLOAD_ONLY YES
)

# Unlike the other recipes, STRUMPACK is not add_subdirectory()'d: its CMake
# only works as the top-level project. It finds its own modules and test
# sources through CMAKE_SOURCE_DIR, which would point at polysolve instead, and
# unconditionally adds its ~200 tests to CTest, which would land in polysolve's.
# So it is configured and built in a CMake run of its own, and only the
# resulting static library is imported below.
set(STRUMPACK_INSTALL_DIR "${FETCHCONTENT_BASE_DIR}/strumpack-install")
set(STRUMPACK_LIBRARY "${STRUMPACK_INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}strumpack${CMAKE_STATIC_LIBRARY_SUFFIX}")

# Follow polysolve's build type; with none set, STRUMPACK would be built
# unoptimized.
if(CMAKE_BUILD_TYPE)
    set(STRUMPACK_BUILD_TYPE ${CMAKE_BUILD_TYPE})
else()
    set(STRUMPACK_BUILD_TYPE Release)
endif()

get_target_property(STRUMPACK_METIS_INCLUDE_DIR METIS::metis INTERFACE_INCLUDE_DIRECTORIES)

set(STRUMPACK_CMAKE_ARGS
    -DCMAKE_BUILD_TYPE=${STRUMPACK_BUILD_TYPE}
    -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
    -DCMAKE_INSTALL_LIBDIR=lib
    -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
    -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
    -DCMAKE_Fortran_COMPILER=${CMAKE_Fortran_COMPILER}
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON
    -DBUILD_SHARED_LIBS=OFF

    # Each subdomain is factorized by a single nanompi rank -- a thread of
    # this process -- so STRUMPACK runs sequentially inside it, with neither
    # MPI nor OpenMP of its own.
    -DSTRUMPACK_USE_MPI=OFF
    -DSTRUMPACK_USE_OPENMP=OFF
    -DSTRUMPACK_USE_CUDA=OFF
    -DSTRUMPACK_USE_HIP=OFF
    -DSTRUMPACK_USE_SYCL=OFF

    # No optional third-party libraries: BLR and HSS compression are built in.
    -DTPL_ENABLE_SLATE=OFF
    -DTPL_ENABLE_PARMETIS=OFF
    -DTPL_ENABLE_SCOTCH=OFF
    -DTPL_ENABLE_PTSCOTCH=OFF
    -DTPL_ENABLE_BPACK=OFF
    -DTPL_ENABLE_ZFP=OFF
    -DTPL_ENABLE_SZ3=OFF
    -DTPL_ENABLE_MAGMA=OFF
    -DTPL_ENABLE_KBLAS=OFF
    -DTPL_ENABLE_COMBBLAS=OFF
    -DTPL_ENABLE_PAPI=OFF
    -DTPL_ENABLE_MATLAB=OFF

    # METIS, the one required dependency, is the one built by the metis
    # recipe. Setting METIS_INCLUDE_DIR outright keeps STRUMPACK's FindMETIS
    # from settling on a system metis.h instead. The library itself is only
    # needed once polysolve links, so it does not have to be built yet.
    -DMETIS_INCLUDE_DIR=${STRUMPACK_METIS_INCLUDE_DIR}
    -DTPL_METIS_LIBRARIES=$<TARGET_FILE:METIS::metis>

    # BLAS and LAPACK are left for STRUMPACK's configure to find: it uses
    # them only to check that a test program links. What STRUMPACK actually
    # calls is whatever polysolve links it with, below.
)

# Build the library only: STRUMPACK's default target also builds its test
# programs. With a Makefile generator, $(MAKE) lets the sub-build share the
# outer make's job slots instead of running serially.
if(CMAKE_GENERATOR MATCHES "Makefiles")
    set(STRUMPACK_BUILD_COMMAND $(MAKE) strumpack)
else()
    set(STRUMPACK_BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --target strumpack --config ${STRUMPACK_BUILD_TYPE})
endif()

include(ExternalProject)
ExternalProject_Add(strumpack_build
    SOURCE_DIR "${strumpack_SOURCE_DIR}"
    BINARY_DIR "${strumpack_BINARY_DIR}"
    INSTALL_DIR "${STRUMPACK_INSTALL_DIR}"
    CMAKE_ARGS ${STRUMPACK_CMAKE_ARGS}
    BUILD_COMMAND ${STRUMPACK_BUILD_COMMAND}
    # `cmake --install`, unlike the install target, does not first build
    # everything else.
    INSTALL_COMMAND ${CMAKE_COMMAND} --install <BINARY_DIR> --config ${STRUMPACK_BUILD_TYPE}
    BUILD_BYPRODUCTS "${STRUMPACK_LIBRARY}"
)

# An imported target's include directory has to exist at generate time, well
# before the ExternalProject has installed anything into it.
file(MAKE_DIRECTORY "${STRUMPACK_INSTALL_DIR}/include")

add_library(STRUMPACK::strumpack STATIC IMPORTED GLOBAL)
set_target_properties(STRUMPACK::strumpack PROPERTIES
    IMPORTED_LOCATION "${STRUMPACK_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${STRUMPACK_INSTALL_DIR}/include"
)
# Link polysolve's BLAS and LAPACK (MKL, Accelerate, ...), not whichever ones
# STRUMPACK's configure happened to find.
target_link_libraries(STRUMPACK::strumpack INTERFACE METIS::metis LAPACK::LAPACK BLAS::BLAS)
add_dependencies(STRUMPACK::strumpack strumpack_build)

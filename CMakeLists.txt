################################################################################
cmake_minimum_required(VERSION 3.11)
project(PolyFEMSolvers)
################################################################################

# cmake_policy(SET CMP0063 NEW)
# set(CMAKE_CXX_VISIBILITY_PRESET hidden)

# Detects whether this is a top-level project
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(POLYFEM_SOLVERS_TOPLEVEL_PROJECT ON)
else()
    set(POLYFEM_SOLVERS_TOPLEVEL_PROJECT OFF)
endif()


if(INPUT_THIRD_PARTY_DIR)
    set(THIRD_PARTY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/${INPUT_THIRD_PARTY_DIR}/)
else()
    set(THIRD_PARTY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/)
endif()
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

if(${POLYFEM_SOLVERS_TOPLEVEL_PROJECT})
    list(APPEND CMAKE_MODULE_PATH ${THIRD_PARTY_DIR}/Catch2/contrib)
endif()

# Color output
include(UseColors)

# Prepend function
include(PrependCurrentPath)

# Polyfem utils
include(PolyfemSolversUtils)

# Extra warnings
include(Warnings)

# Use C++11/14
include(CXXFeatures)

# Sort projects inside the solution
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Generate position independent code by default
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

################################################################################



################################################################################

# Polyfem options
option(POLYFEM_SOLVERS_WITH_SANITIZERS "Enable sanitizers in compilation targets"   OFF)


# Polyfem options for enabling/disabling optional libraries
option(POLYFEM_SOLVERS_WITH_CHOLMOD          "Enable Cholmod library"            ON)
option(POLYFEM_SOLVERS_WITH_UMFPACK          "Enable UmfPack library"            ON)
option(POLYFEM_SOLVERS_WITH_SUPERLU          "Enable SuperLU library"            ON)
option(POLYFEM_SOLVERS_WITH_MKL              "Enable MKL library"                ON)
option(POLYFEM_SOLVERS_WITH_PARDISO          "Enable Pardiso library"            OFF)
option(POLYFEM_SOLVERS_WITH_HYPRE            "Enable hypre"                      ON)
option(POLYFEM_SOLVERS_WITH_AMGCL            "Use AMGCL"                         ON)
option(POLYFEM_SOLVERS_WITH_SPECTRA          "Enable computing spectrum"         ON)



# Sanitizer options
option(POLYFEM_SOLVERS_SANITIZE_ADDRESS   "Sanitize Address"       OFF)
option(POLYFEM_SOLVERS_SANITIZE_MEMORY    "Sanitize Memory"        OFF)
option(POLYFEM_SOLVERS_SANITIZE_THREAD    "Sanitize Thread"        OFF)
option(POLYFEM_SOLVERS_SANITIZE_UNDEFINED "Sanitize Undefined"     OFF)

option(POLYFEM_SOLVERS_LARGE_INDEX    "Build for large indices"       OFF)

################################################################################



# Somehow Pardiso 6 conflicts with libmetis that these guys depends on... Grr
if(POLYFEM_SOLVERS_WITH_PARDISO)
    set(POLYFEM_SOLVERS_WITH_CHOLMOD      OFF CACHE BOOL "" FORCE)
    set(POLYFEM_SOLVERS_WITH_UMFPACK      OFF CACHE BOOL "" FORCE)
    set(POLYFEM_SOLVERS_WITH_SUPERLU      OFF CACHE BOOL "" FORCE)
endif()

################################################################################
# Dependencies
################################################################################

# Sanitizers
if(POLYFEM_SOLVERS_WITH_SANITIZERS)
    list(APPEND CMAKE_MODULE_PATH ${THIRD_PARTY_DIR}/sanitizers-cmake/cmake)
endif()


################################################################################
# Polyfem-solver library
################################################################################

add_library(polyfem-solvers)

if(${POLYFEM_SOLVERS_LARGE_INDEX})
    target_compile_definitions(polyfem-solvers PUBLIC -DPOLYFEM_SOLVERS_LARGE_INDEX)
endif()

# Public include directory for polyfem
target_include_directories(polyfem-solvers PUBLIC ${PROJECT_BINARY_DIR}/include)

# Extra warnings
target_link_libraries(polyfem-solvers PRIVATE warnings::all)

# Use C++14
target_compile_features(polyfem-solvers PUBLIC ${CXX14_FEATURES})

# No limit yay
target_compile_definitions(polyfem-solvers PUBLIC -DEIGEN_STACK_ALLOCATION_LIMIT=0)

# 8MB
# target_compile_definitions(polyfem-solvers PUBLIC -DEIGEN_STACK_ALLOCATION_LIMIT=8388608)


# Setup dependencies
include(PolyfemSolversDependencies)

# Other sources
add_subdirectory(src)

# Compile extras only if this is a top-level project
if(POLYFEM_SOLVERS_TOPLEVEL_PROJECT)
    # Unit tests
    include(CTest)
    enable_testing()
    add_subdirectory(tests)
endif()

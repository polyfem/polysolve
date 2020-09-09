################################################################################
cmake_minimum_required(VERSION 3.2)
project(PolySolve)
################################################################################

# cmake_policy(SET CMP0063 NEW)
# set(CMAKE_CXX_VISIBILITY_PRESET hidden)

if (MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /bigobj")
endif()

# Detects whether this is a top-level project
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(POLYSOLVE_TOPLEVEL_PROJECT ON)
else()
    set(POLYSOLVE_TOPLEVEL_PROJECT OFF)
endif()


if(INPUT_THIRD_PARTY_DIR)
    set(THIRD_PARTY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/${INPUT_THIRD_PARTY_DIR}/)
else()
    set(THIRD_PARTY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/)
endif()
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

if(${POLYSOLVE_TOPLEVEL_PROJECT})
    list(APPEND CMAKE_MODULE_PATH ${THIRD_PARTY_DIR}/Catch2/contrib)
endif()

# Color output
include(UseColors)

# Prepend function
include(PrependCurrentPath)

# Polysolve utils
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

# Polysolve options
option(POLYSOLVE_WITH_SANITIZERS "Enable sanitizers in compilation targets"   OFF)


# Polysolve options for enabling/disabling optional libraries
option(POLYSOLVE_WITH_CHOLMOD          "Enable Cholmod library"            ON)
option(POLYSOLVE_WITH_UMFPACK          "Enable UmfPack library"            ON)
option(POLYSOLVE_WITH_SUPERLU          "Enable SuperLU library"            ON)
option(POLYSOLVE_WITH_MKL              "Enable MKL library"                ON)
option(POLYSOLVE_WITH_PARDISO          "Enable Pardiso library"            OFF)
option(POLYSOLVE_WITH_HYPRE            "Enable hypre"                      ON)
option(POLYSOLVE_WITH_AMGCL            "Use AMGCL"                         ON)
option(POLYSOLVE_WITH_SPECTRA          "Enable computing spectrum"         ON)
option(POLYSOLVE_WITH_MATIO            "Use matio for matrix export"       ON)



# Sanitizer options
option(POLYSOLVE_SANITIZE_ADDRESS   "Sanitize Address"       OFF)
option(POLYSOLVE_SANITIZE_MEMORY    "Sanitize Memory"        OFF)
option(POLYSOLVE_SANITIZE_THREAD    "Sanitize Thread"        OFF)
option(POLYSOLVE_SANITIZE_UNDEFINED "Sanitize Undefined"     OFF)

option(POLYSOLVE_LARGE_INDEX    "Build for large indices"       OFF)

################################################################################



# Somehow Pardiso 6 conflicts with libmetis that these guys depends on... Grr
if(POLYSOLVE_WITH_PARDISO)
    set(POLYSOLVE_WITH_CHOLMOD      OFF CACHE BOOL "" FORCE)
    set(POLYSOLVE_WITH_UMFPACK      OFF CACHE BOOL "" FORCE)
    set(POLYSOLVE_WITH_SUPERLU      OFF CACHE BOOL "" FORCE)
endif()

################################################################################
# Dependencies
################################################################################

# Sanitizers
if(POLYSOLVE_WITH_SANITIZERS)
    list(APPEND CMAKE_MODULE_PATH ${THIRD_PARTY_DIR}/sanitizers-cmake/cmake)
endif()


################################################################################
# polysolve library
################################################################################

add_library(polysolve src/FEMSolver.cpp)

if(${POLYSOLVE_LARGE_INDEX})
    target_compile_definitions(polysolve PUBLIC -DPOLYSOLVE_LARGE_INDEX)
endif()

if(POLYSOLVE_WITH_MATIO)
  find_package(MATIO)
  target_link_libraries(polysolve PUBLIC matio)
  target_compile_definitions(polysolve PUBLIC -DPOLYSOLVE_WITH_MATIO)
endif()

# Public include directory for Polysolve
target_include_directories(polysolve PUBLIC ${PROJECT_BINARY_DIR}/include)

# Extra warnings
target_link_libraries(polysolve PRIVATE warnings::all)

# Use C++14
target_compile_features(polysolve PUBLIC ${CXX14_FEATURES})

# No limit yay
target_compile_definitions(polysolve PUBLIC -DEIGEN_STACK_ALLOCATION_LIMIT=0)

# 8MB
# target_compile_definitions(polysolve PUBLIC -DEIGEN_STACK_ALLOCATION_LIMIT=8388608)


# Setup dependencies
include(PolyfemSolversDependencies)

# Other sources
add_subdirectory(src)

# Compile extras only if this is a top-level project
if(POLYSOLVE_TOPLEVEL_PROJECT)
    # Unit tests
    include(CTest)
    enable_testing()
    add_subdirectory(tests)
endif()

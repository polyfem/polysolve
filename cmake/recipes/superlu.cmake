# SuperLU solver

if(TARGET SuperLU::SuperLU)
    return()
endif()

message(STATUS "Third-party: creating targets 'SuperLU::SuperLU'")

# We do not have a build recipe for this, so find it as a system installed library.
find_package(SUPERLU)

if(SUPERLU_FOUND)
    add_library(SuperLU_SuperLU INTERFACE)
    add_library(SuperLU::SuperLU ALIAS SuperLU_SuperLU)
    target_include_directories(SuperLU_SuperLU INTERFACE ${SUPERLU_INCLUDES})
    target_link_libraries(SuperLU_SuperLU INTERFACE ${SUPERLU_LIBRARIES})
    target_compile_definitions(SuperLU_SuperLU INTERFACE ${SUPERLU_DEFINES})
endif()

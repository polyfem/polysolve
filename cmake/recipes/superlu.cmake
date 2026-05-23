# SuperLU solver

include(polysolve_optional_dependency)

if(TARGET SuperLU::SuperLU)
    return()
endif()
if(TARGET SuperLU_SuperLU)
    return()
endif()

message(STATUS "Third-party: creating targets 'SuperLU::SuperLU'")

# We do not have a build recipe for this, so find it as a system installed library.
find_package(SUPERLU QUIET)

if(SUPERLU_FOUND)
    add_library(SuperLU_SuperLU INTERFACE IMPORTED GLOBAL)
    target_include_directories(SuperLU_SuperLU INTERFACE ${SUPERLU_INCLUDES})
    target_link_libraries(SuperLU_SuperLU INTERFACE ${SUPERLU_LIBRARIES})
    target_compile_definitions(SuperLU_SuperLU INTERFACE ${SUPERLU_DEFINES})

    set(POLYSOLVE_SUPERLU_CHECK_SOURCE [[
#include <slu_ddefs.h>
int main()
{
    SuperLUStat_t stat;
    StatInit(&stat);
    StatFree(&stat);
    return 0;
}
]])
    polysolve_check_linkable_target(POLYSOLVE_SUPERLU_LINKABLE
        NAME SuperLU
        TARGET SuperLU_SuperLU
        SOURCE_VAR POLYSOLVE_SUPERLU_CHECK_SOURCE
    )

    if(NOT POLYSOLVE_SUPERLU_LINKABLE)
        polysolve_note_disabled_dependency("SuperLU" "${POLYSOLVE_SUPERLU_LINKABLE_REASON}")
        return()
    endif()

    add_library(SuperLU::SuperLU ALIAS SuperLU_SuperLU)
endif()

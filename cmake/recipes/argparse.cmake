if(TARGET argparse::argparse)
    return()
endif()

message(STATUS "Third-party: creating target 'argparse::argparse'")

include(CPM)
CPMAddPackage(
    NAME argparse
    GITHUB_REPOSITORY p-ranav/argparse
    GIT_TAG v3.2
)

if(TARGET argparse)
    set_target_properties(argparse PROPERTIES SYSTEM ON)
endif()

# json spec engine (https://github.com/geometryprocessing/json-spec-engine)
# License: MIT

if(TARGET jse::jse)
    return()
endif()

message(STATUS "Third-party: creating target 'jse::jse'")

include(CPM)
CPMAddPackage("gh:geometryprocessing/json-spec-engine#11d028ebf54c3665e1a7c25d8ac622a8cb851223")

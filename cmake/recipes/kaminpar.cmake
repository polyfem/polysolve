# KaMinPar (https://github.com/KaHIP/KaMinPar)
# License: MIT

if(TARGET KaMinPar::KaMinPar)
    return()
endif()

message(STATUS "Third-party: creating target 'KaMinPar::KaMinPar'")

include(onetbb)

include(CPM)
CPMAddPackage(
    NAME KaMinPar
    VERSION 3.7.3
    GITHUB_REPOSITORY KaHIP/KaMinPar
    GIT_TAG v3.7.3
    OPTIONS
        "KAMINPAR_BUILD_APPS OFF"
        "KAMINPAR_BUILD_IO OFF"
        "KAMINPAR_BUILD_WITH_SPARSEHASH OFF"
        "KAMINPAR_BUILD_WITH_DEBUG_SYMBOLS OFF"
        "KAMINPAR_64BIT_WEIGHTS ON"
        "KAMINPAR_ENABLE_TIMERS OFF"
        "INSTALL_KAMINPAR OFF"
        # Below hack force kaminpar to use our vendored tbb
        "KAMINPAR_DOWNLOAD_TBB ON"
        "TBB_FOUND TRUE"
)

# Apple Accelerate solver support

include(polysolve_optional_dependency)

if(TARGET polysolve::accelerate)
    return()
endif()

if(NOT APPLE)
    polysolve_note_disabled_dependency("Accelerate" "Accelerate is only supported on macOS.")
    return()
endif()

message(STATUS "Third-party: creating target 'polysolve::accelerate'")

include(CPM)
CPMAddPackage(
    NAME eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 969c31eefcdfaab11da763bea3f7502086673ab0
    DOWNLOAD_ONLY ON
)

set(BLA_VENDOR Apple)
find_package(BLAS QUIET)
find_package(LAPACK QUIET)

set(POLYSOLVE_ACCELERATE_CHECK_SOURCE [[
#include <Accelerate/Accelerate.h>
int main()
{
    double x[1] = {1.0};
    return cblas_ddot(1, x, 1, x, 1) == 1.0 ? 0 : 1;
}
]])
polysolve_check_linkable_target(POLYSOLVE_ACCELERATE_LINKABLE
    NAME Accelerate
    TARGET BLAS::BLAS
    LINK_LIBRARIES LAPACK::LAPACK
    SOURCE_VAR POLYSOLVE_ACCELERATE_CHECK_SOURCE
)

if(NOT POLYSOLVE_ACCELERATE_LINKABLE)
    polysolve_note_disabled_dependency("Accelerate" "${POLYSOLVE_ACCELERATE_LINKABLE_REASON}")
    return()
endif()

add_library(polysolve_accelerate INTERFACE)
add_library(polysolve::accelerate ALIAS polysolve_accelerate)
target_link_libraries(polysolve_accelerate INTERFACE BLAS::BLAS LAPACK::LAPACK)

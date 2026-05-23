# Spectra MPL 2.0 (optional)

if(TARGET Spectra::Spectra)
    return()
endif()

include(polysolve_optional_dependency)

set(POLYSOLVE_SPECTRA_CHECK_SOURCE [[
#include <Eigen/Sparse>
#include <MatOp/SparseSymMatProd.h>
#include <SymEigsSolver.h>
int main()
{
    Eigen::SparseMatrix<double> matrix(2, 2);
    matrix.setIdentity();
    Spectra::SparseSymMatProd<double> op(matrix);
    return op.rows() == 2 ? 0 : 1;
}
]])

polysolve_find_system_dependency(SPECTRA_SYSTEM_FOUND
    NAME Spectra
    PACKAGE Spectra
    TARGET Spectra::Spectra
    CONFIG
    SOURCE_VAR POLYSOLVE_SPECTRA_CHECK_SOURCE
)
if(SPECTRA_SYSTEM_FOUND)
    return()
endif()

polysolve_should_fetch_dependency(SPECTRA_SHOULD_FETCH Spectra)
if(NOT SPECTRA_SHOULD_FETCH)
    polysolve_note_disabled_dependency("Spectra" "Spectra was not found and fetching is disabled.")
    return()
endif()

message(STATUS "Third-party: creating target 'Spectra::Spectra'")

include(CPM)
CPMAddPackage(
    NAME spectra
    GITHUB_REPOSITORY yixuan/spectra
    GIT_TAG v0.6.2
    DOWNLOAD_ONLY ON
)

add_library(Spectra INTERFACE)
add_library(Spectra::Spectra ALIAS Spectra)

target_include_directories(Spectra SYSTEM INTERFACE ${spectra_SOURCE_DIR}/include)

include(eigen)
target_link_libraries(Spectra INTERFACE Eigen3::Eigen)

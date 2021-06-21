set(SOURCES
    FEMSolver.cpp
    FEMSolver.hpp
    LbfgsSolver.hpp
    LinearSolver.cpp
    LinearSolver.hpp
    LinearSolverEigen.h
    LinearSolverEigen.hpp
    LinearSolverHypre.cpp
    LinearSolverHypre.hpp
    SaddlePointSolver.cpp
    SaddlePointSolver.hpp
)

if(POLYSOLVE_HAS_AMGCL)
    set(SOURCES
    ${SOURCES}
    LinearSolverAMGCL.cpp
    LinearSolverAMGCL.hpp
    )
endif()


if(POLYSOLVE_HAS_PARDISO)
    set(SOURCES
    ${SOURCES}
    LinearSolverPardiso.cpp
    LinearSolverPardiso.hpp
    )
endif()

prepend_current_path(SOURCES)
polyfem_solvers_copy_headers(${SOURCES})
polyfem_solvers_set_source_group(${SOURCES})
target_sources(polysolve PRIVATE ${SOURCES})

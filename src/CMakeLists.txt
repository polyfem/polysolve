set(SOURCES
    FEMSolver.cpp
    FEMSolver.hpp
    LbfgsSolver.hpp
    LinearSolver.cpp
    LinearSolver.hpp
    LinearSolverAMGCL.cpp
    LinearSolverAMGCL.hpp
    LinearSolverEigen.h
    LinearSolverEigen.hpp
    LinearSolverHypre.cpp
    LinearSolverHypre.hpp
    LinearSolverPardiso.cpp
)

prepend_current_path(SOURCES)
polyfem_solvers_copy_headers(${SOURCES})
polyfem_solvers_set_source_group(${SOURCES})
target_sources(polysolve PRIVATE ${SOURCES})

set(SOURCES
    FEMSolver.cpp
    polysolve/FEMSolver.hpp
    polysolve/LbfgsSolver.hpp
    LinearSolver.cpp
    polysolve/LinearSolver.hpp
    LinearSolverAMGCL.cpp
    polysolve/LinearSolverAMGCL.hpp
    polysolve/LinearSolverEigen.hpp
    polysolve/LinearSolverEigen.tpp
    LinearSolverHypre.cpp
    polysolve/LinearSolverHypre.hpp
    LinearSolverPardiso.cpp
    polysolve/LinearSolverPardiso.hpp
    SaddlePointSolver.cpp
    polysolve/SaddlePointSolver.hpp
)

polysolve_prepend_current_path(SOURCES)
polysolve_set_source_group(${SOURCES})
target_sources(polysolve PRIVATE ${SOURCES})

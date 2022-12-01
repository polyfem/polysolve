set(SOURCES
    polysolve/FEMSolver.cpp
    polysolve/FEMSolver.hpp
    polysolve/LbfgsSolver.hpp
    polysolve/LinearSolver.cpp
    polysolve/LinearSolver.hpp
    polysolve/LinearSolverAMGCL.cpp
    polysolve/LinearSolverAMGCL.hpp
    polysolve/LinearSolverCuSolverDN.cu
    polysolve/LinearSolverCuSolverDN.cuh
    polysolve/LinearSolverEigen.hpp
    polysolve/LinearSolverEigen.tpp
    polysolve/LinearSolverHypre.cpp
    polysolve/LinearSolverHypre.hpp
    polysolve/LinearSolverPardiso.cpp
    polysolve/LinearSolverPardiso.hpp
    polysolve/SaddlePointSolver.cpp
    polysolve/SaddlePointSolver.hpp
)

polysolve_prepend_current_path(SOURCES)
polysolve_set_source_group(${SOURCES})
target_sources(polysolve PRIVATE ${SOURCES})

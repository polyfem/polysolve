set(SOURCES
    FEMSolver.cpp
    FEMSolver.hpp
    LbfgsSolver.hpp
    LinearSolver.cpp
    LinearSolver.hpp
    LinearSolverAMGCL.cpp
    LinearSolverAMGCL.hpp
    LinearSolverEigen.hpp
    LinearSolverEigen.tpp
    LinearSolverHypre.cpp
    LinearSolverHypre.hpp
    LinearSolverPardiso.cpp
    LinearSolverPardiso.hpp
    SaddlePointSolver.cpp
    SaddlePointSolver.hpp
    CHOLMODSolver.cpp
    CHOLMODSolver.hpp
)

polysolve_prepend_current_path(SOURCES)
polysolve_copy_headers(${SOURCES})
polysolve_set_source_group(${SOURCES})
target_sources(polysolve PRIVATE ${SOURCES})

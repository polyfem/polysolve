set(SOURCES
    FEMSolver.cpp
    FEMSolver.hpp
    LbfgsSolver.hpp
    LinearSolver.cpp
    LinearSolver.hpp
    LinearSolverAMGCL.cpp
    LinearSolverAMGCL.hpp
    LinearSolverCuSolverDN.cu
    LinearSolverCuSolverDN.cuh
    LinearSolverEigen.hpp
    LinearSolverEigen.tpp
    LinearSolverHypre.cpp
    LinearSolverHypre.hpp
    LinearSolverPardiso.cpp
    LinearSolverPardiso.hpp
    LinearSolverSympiler.cpp
    LinearSolverSympiler.hpp
    SaddlePointSolver.cpp
    SaddlePointSolver.hpp
)

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" PREFIX "Source Files" FILES ${SOURCES})
target_sources(polysolve PRIVATE ${SOURCES})

set(SOURCES
    FEMSolver.cpp
    FEMSolver.hpp
    Solver.cpp
    Solver.hpp
    AMGCL.cpp
    AMGCL.hpp
    CuSolverDN.cu
    CuSolverDN.cuh
    EigenSolver.hpp
    EigenSolver.tpp
    HypreSolver.cpp
    HypreSolver.hpp
    Pardiso.cpp
    Pardiso.hpp
    SaddlePointSolver.cpp
    SaddlePointSolver.hpp
)

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" PREFIX "Source Files" FILES ${SOURCES})
target_sources(polysolve PRIVATE ${SOURCES})

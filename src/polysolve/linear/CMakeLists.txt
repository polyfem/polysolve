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

if(POLYSOLVE_WITH_CUDA)
    list(APPEND SOURCES
        PCGSolver.cu
        PCGSolver.hpp
        mas_utils/BSRAdjacency.cpp
        mas_utils/BSRAdjacency.hpp
        mas_utils/BSRMatrix.cpp
        mas_utils/BSRMatrix.hpp
        mas_utils/CuSparseWrapper.hpp
        mas_utils/CudaUtils.cuh
        mas_utils/InnerProduct.cu
        mas_utils/InnerProduct.hpp
        mas_utils/MASPreconditioner.cu
        mas_utils/MASPreconditioner.hpp
        mas_utils/MetisPartition.cpp
        mas_utils/MetisPartition.hpp
    )

endif()

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" PREFIX "Source Files" FILES ${SOURCES})
target_sources(polysolve_linear PRIVATE ${SOURCES})

#pragma once

#include "Solver.hpp" 
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <cudss.h>
#include <cuda_runtime.h>
#include <vector>

namespace polysolve::linear {
    class cuDSSSolver : public Solver {
    public:
        cuDSSSolver();
        ~cuDSSSolver();

        void analyze_pattern(const StiffnessMatrix &A, const int precond_num) override;
        void factorize(const StiffnessMatrix &A) override;
        void solve(const Ref<const VectorXd> b, Ref<VectorXd> x) override;
        
        std::string name() const override { return "cuDSS"; }

    private:
        void free_device_memory();

        double pattern_analysis_time, factorization_time, solve_time;
        double reordering_time, symbolic_time;

        cudssHandle_t cudss_handle = nullptr;
        cudssConfig_t config = nullptr;
        cudssData_t solverData = nullptr;
        
        cudssMatrix_t MatrixA = nullptr;
        cudssMatrix_t MatrixX = nullptr;
        cudssMatrix_t MatrixB = nullptr;

        int m_nrows = 0;
        int m_ncols = 0;
        int m_nnz = 0;

        int* d_csrRowOffsets = nullptr;
        int* d_csrColIndices = nullptr;
        double* d_csrValues = nullptr;
        double* d_x = nullptr;
        double* d_b = nullptr;
    };
}
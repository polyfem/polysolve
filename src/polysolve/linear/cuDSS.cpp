#include "cuDSS.hpp"

#include <chrono>
#include <iostream>
#include <stdexcept>

#include <spdlog/spdlog.h>

#ifndef CHECK_CUDA
#define CHECK_CUDA(func)                                                       \
    {                                                                          \
        cudaError_t status = (func);                                           \
        if (status != cudaSuccess) {                                           \
            std::cerr << "CUDA Error: " << cudaGetErrorString(status)          \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;   \
            throw std::runtime_error("CUDA Error");                            \
        }                                                                      \
    }
#endif

#ifndef CHECK_CUDSS
#define CHECK_CUDSS(func)                                                      \
    {                                                                          \
        cudssStatus_t status = (func);                                         \
        if (status != CUDSS_STATUS_SUCCESS) {                                  \
            std::cerr << "cuDSS Error Code: " << status                        \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;   \
            throw std::runtime_error("cuDSS Error");                           \
        }                                                                      \
    }
#endif

namespace polysolve::linear {

    namespace
    {
        using clock = std::chrono::steady_clock;

        double elapsed_seconds(const std::chrono::time_point<clock> &begin)
        {
            return std::chrono::duration<double>(clock::now() - begin).count();
        }
    }

    cuDSSSolver::cuDSSSolver() {
        CHECK_CUDSS(cudssCreate(&cudss_handle));
    }

    cuDSSSolver::~cuDSSSolver() {
        free_device_memory();
        if (cudss_handle) {
            CHECK_CUDSS(cudssDestroy(cudss_handle));
            cudss_handle = nullptr;
        }
    }

    void cuDSSSolver::free_device_memory() {
        if (MatrixA) { CHECK_CUDSS(cudssMatrixDestroy(MatrixA)); MatrixA = nullptr; }
        if (MatrixX) { CHECK_CUDSS(cudssMatrixDestroy(MatrixX)); MatrixX = nullptr; }
        if (MatrixB) { CHECK_CUDSS(cudssMatrixDestroy(MatrixB)); MatrixB = nullptr; }
        if (solverData)   { CHECK_CUDSS(cudssDataDestroy(cudss_handle, solverData)); solverData = nullptr; }
        if (config)       { CHECK_CUDSS(cudssConfigDestroy(config)); config = nullptr; }

        if (d_csrRowOffsets) { CHECK_CUDA(cudaFree(d_csrRowOffsets)); d_csrRowOffsets = nullptr; }
        if (d_csrColIndices) { CHECK_CUDA(cudaFree(d_csrColIndices)); d_csrColIndices = nullptr; }
        if (d_csrValues)     { CHECK_CUDA(cudaFree(d_csrValues)); d_csrValues = nullptr; }
        if (d_x)             { CHECK_CUDA(cudaFree(d_x)); d_x = nullptr; }
        if (d_b)             { CHECK_CUDA(cudaFree(d_b)); d_b = nullptr; }
    }

    void cuDSSSolver::analyze_pattern(const StiffnessMatrix &A, const int precond_num) {
        free_device_memory();

        {
            auto phase_begin = clock::now();
            CHECK_CUDSS(cudssConfigCreate(&config));
            CHECK_CUDSS(cudssDataCreate(cudss_handle, &solverData));
            SPDLOG_TRACE("[cuDSS] [create_solver_objects] [{:.6f}]", elapsed_seconds(phase_begin));
        }

        {
            auto phase_begin = clock::now();
            m_nrows = A.rows();
            m_ncols = A.cols();
            m_nnz   = A.nonZeros();

            CHECK_CUDA(cudaMalloc(&d_csrRowOffsets, (m_nrows + 1) * sizeof(int)));
            CHECK_CUDA(cudaMalloc(&d_csrColIndices, m_nnz * sizeof(int)));
            CHECK_CUDA(cudaMalloc(&d_csrValues, m_nnz * sizeof(double)));

            CHECK_CUDA(cudaMemcpy(d_csrRowOffsets, A.outerIndexPtr(), (m_nrows + 1) * sizeof(int), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(d_csrColIndices, A.innerIndexPtr(), m_nnz * sizeof(int), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(d_csrValues, A.valuePtr(), m_nnz * sizeof(double), cudaMemcpyHostToDevice));
            
            CHECK_CUDSS(cudssMatrixCreateCsr(
                &MatrixA, m_nrows, m_ncols, m_nnz, 
                d_csrRowOffsets, nullptr, d_csrColIndices, d_csrValues, 
                CUDA_R_32I, CUDA_R_64F, CUDSS_MTYPE_SYMMETRIC, 
                CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO
            ));

            CHECK_CUDA(cudaDeviceSynchronize());
            SPDLOG_TRACE("[cuDSS] [copy_sparse_matrix] [{:.6f}]", elapsed_seconds(phase_begin));
        }

        {
            auto phase_begin = clock::now();
            
            CHECK_CUDSS(cudssExecute(cudss_handle, CUDSS_PHASE_REORDERING, config, solverData, 
                                MatrixA, nullptr, nullptr));
            CHECK_CUDSS(cudssExecute(cudss_handle, CUDSS_PHASE_SYMBOLIC_FACTORIZATION, config, solverData, 
                                MatrixA, nullptr, nullptr));
            CHECK_CUDA(cudaDeviceSynchronize());

            SPDLOG_TRACE("[cuDSS] [pattern_analysis] [{:.6f}]", elapsed_seconds(phase_begin));
        }
    }

    void cuDSSSolver::factorize(const StiffnessMatrix &A) {
        auto phase_begin = clock::now();
        CHECK_CUDSS(cudssExecute(cudss_handle, CUDSS_PHASE_FACTORIZATION, config, solverData, 
                                MatrixA, nullptr, nullptr));
        CHECK_CUDA(cudaDeviceSynchronize());
        SPDLOG_TRACE("[cuDSS] [numerical_factorization] [{:.6f}]", elapsed_seconds(phase_begin));
    }

    void cuDSSSolver::solve(const Ref<const VectorXd> b, Ref<VectorXd> x) {
        {
            auto phase_begin = clock::now();
            if (d_x == nullptr || d_b == nullptr) {
                CHECK_CUDA(cudaMalloc(&d_x, m_nrows * sizeof(double)));
                CHECK_CUDA(cudaMalloc(&d_b, m_nrows * sizeof(double)));

                CHECK_CUDSS(cudssMatrixCreateDn(
                    &MatrixX, m_nrows, 1, m_nrows, 
                    d_x, CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR
                ));

                CHECK_CUDSS(cudssMatrixCreateDn(
                    &MatrixB, m_nrows, 1, m_nrows, 
                    d_b, CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR
                ));
            }

            CHECK_CUDA(cudaMemcpy(d_b, b.data(), m_nrows * sizeof(double), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(d_x, x.data(), m_nrows * sizeof(double), cudaMemcpyHostToDevice));

            CHECK_CUDA(cudaDeviceSynchronize());
            SPDLOG_TRACE("[cuDSS] [copy_vectors] [{:.6f}]", elapsed_seconds(phase_begin));
        }

        {
            auto phase_begin = clock::now();
            CHECK_CUDSS(cudssExecute(cudss_handle, CUDSS_PHASE_SOLVE, config, solverData, 
                                    MatrixA, MatrixX, MatrixB));
            CHECK_CUDA(cudaDeviceSynchronize());
            SPDLOG_TRACE("[cuDSS] [solve] [{:.6f}]", elapsed_seconds(phase_begin));
        }

        CHECK_CUDA(cudaMemcpy(x.data(), d_x, m_nrows * sizeof(double), cudaMemcpyDeviceToHost));
    }
}
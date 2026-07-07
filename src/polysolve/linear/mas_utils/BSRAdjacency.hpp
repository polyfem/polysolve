#pragma once

#include <polysolve/linear/mas_utils/CudaUtils.cuh>
#include <polysolve/Types.hpp>

#include <cstdint>
#include <vector>

namespace polysolve::linear::mas
{
    /// Host CSR adjacency for graph partitioning, built using CUDA.
    /// Self-excluding, with symmetrized + quantized positive integer weights.
    class BSRAdjacency
    {
    public:
        BSRAdjacency() = default;
        BSRAdjacency(const StiffnessMatrix &A, int block_dim, CudaRuntime rt);

        std::vector<int> row_ptr;
        std::vector<int> cols;
        std::vector<int64_t> weights;
    };

} // namespace polysolve::linear::mas

#pragma once

#include <polysolve/Types.hpp>

#include <cstdint>
#include <vector>

namespace polysolve::linear::mas
{
    /// Host-only CSR adjacency for graph partitioning.
    /// Self-excluding, with symmetrized + quantized positive integer weights.
    class BSRAdjacency
    {
    public:
        BSRAdjacency() = default;
        BSRAdjacency(const StiffnessMatrix &A, int block_dim);

        std::vector<int> row_ptr;
        std::vector<int> cols;
        std::vector<int64_t> weights;
    };

} // namespace polysolve::linear::mas

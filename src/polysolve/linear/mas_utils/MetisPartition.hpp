#pragma once

#include <cuda/std/span>
#include <vector>

#include <polysolve/linear/mas_utils/CudaUtils.cuh>

namespace polysolve::linear::mas
{
    /// @brief K-way graph partition.
    /// @param row_ptr[in] CSR graph topology.
    /// @param cols[in] CSR graph topology.
    /// @param weights[in] CSR graph edge weights. Empty span implies unweighted graph.
    /// @param max_part_size[in] Maximum partition size.
    /// @param part_num[out] Total partition num.
    /// @param part_id[out] Partition id of each graph node.
    void metis_partition(ctd::span<const int> row_ptr,
                         ctd::span<const int> cols,
                         ctd::span<const int> weights,
                         int max_part_size,
                         int &part_num,
                         std::vector<int> &part_id);
} // namespace polysolve::linear::mas

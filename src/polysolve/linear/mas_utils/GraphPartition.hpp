#pragma once

#include <cuda/std/span>
#include <cstdint>
#include <vector>

#include <polysolve/linear/mas_utils/CudaUtils.cuh>

namespace polysolve::linear::mas
{
    /// @brief K-way graph partition.
    /// @param row_ptr[in] CSR graph topology. Will be modified!
    /// @param cols[in] CSR graph topology. Will be modified!
    /// @param weights[in] CSR graph edge weights. Empty span implies unweighted graph. Will be modified!
    /// @param max_part_size[in] Maximum partition size.
    /// @param part_num[out] Total partition num.
    /// @param part_id[out] Partition id of each graph node.
    /// @param num_threads[in] Threads for KaMinPar; <= 0 uses all hardware threads.
    void graph_partition(ctd::span<int> row_ptr,
                         ctd::span<int> cols,
                         ctd::span<int64_t> weights,
                         int max_part_size,
                         int &part_num,
                         std::vector<int> &part_id,
                         int num_threads = 0);
} // namespace polysolve::linear::mas

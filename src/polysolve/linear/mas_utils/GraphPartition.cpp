#include <polysolve/linear/mas_utils/GraphPartition.hpp>

#include <polysolve/linear/mas_utils/CudaUtils.cuh>

#include <cassert>
#include <cuda/std/span>
#include <cstdint>
#include <thread>
#include <vector>
#include <iostream>

#include <ckaminpar.h>

namespace polysolve::linear::mas
{
    void graph_partition(ctd::span<int> row_ptr,
                         ctd::span<int> cols,
                         ctd::span<int64_t> weights,
                         int max_part_size,
                         int &part_num,
                         std::vector<int> &part_id)
    {
        int node_num = row_ptr.size() - 1;
        assert(node_num >= 0);
        assert(max_part_size > 0);
        assert(weights.empty() || weights.size() == cols.size());

        part_id.resize(node_num, 0);
        if (node_num <= max_part_size)
        {
            part_num = 1;
            return;
        }

        static_assert(sizeof(int) == sizeof(kaminpar_node_id_t), "int and KaMinPar node IDs differ in size");
        static_assert(sizeof(int) == sizeof(kaminpar_edge_id_t), "int and KaMinPar edge IDs differ in size");
        static_assert(sizeof(int64_t) == sizeof(kaminpar_edge_weight_t),
                      "int64 and KaMinPar edge weights differ in size (enable KAMINPAR_64BIT_WEIGHTS?)");

        auto *xadj = reinterpret_cast<kaminpar_edge_id_t *>(row_ptr.data());
        auto *adjncy = reinterpret_cast<kaminpar_node_id_t *>(cols.data());
        auto *adjwgt = weights.empty()
                           ? nullptr
                           : reinterpret_cast<kaminpar_edge_weight_t *>(weights.data());

        kaminpar_context_t *ctx = kaminpar_create_context_by_preset_name("default");
        assert(ctx);

        int thread_num = (std::thread::hardware_concurrency() != 0)
                             ? std::thread::hardware_concurrency()
                             : 1;
        kaminpar_t *partitioner = kaminpar_create(thread_num, ctx);
        assert(partitioner);

        kaminpar_set_output_level(partitioner, KAMINPAR_OUTPUT_LEVEL_QUIET);
        kaminpar_borrow_and_mutate_graph(
            partitioner,
            node_num,
            xadj,
            adjncy,
            nullptr,
            adjwgt);

        // Impl Eq.7 of arXiv:2411.06224

        // Forcing tight K for k-way algorithm leads to low quality parition.
        // It seems like max_part_size - 2 is a good balance.
        part_num = div_round_up(node_num, max_part_size - 2);
        assert(part_num >= 1);
        std::vector<kaminpar_block_weight_t> max_block_weights(
            part_num,
            static_cast<kaminpar_block_weight_t>(max_part_size));
        std::vector<kaminpar_block_id_t> partition(node_num);

        // K way partition with absolution max block weight.
        // Since our node weight is uniform, this is equivalent to limiting parition size.
        kaminpar_compute_partition_with_max_block_weights(
            partitioner,
            part_num,
            max_block_weights.data(),
            partition.data());

        kaminpar_context_free(ctx);
        kaminpar_free(partitioner);

        for (int i = 0; i < node_num; ++i)
        {
            part_id[i] = static_cast<int>(partition[i]);
        }
    }

} // namespace polysolve::linear::mas

#include <polysolve/linear/mas_utils/GraphPartition.hpp>

#include <polysolve/linear/mas_utils/CudaUtils.cuh>

#include <algorithm>
#include <cassert>
#include <cuda/std/span>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include <ckaminpar.h>

namespace polysolve::linear::mas
{

    // Debug code to ensure input adjacency is valid.
#ifndef NDEBUG
    namespace
    {
        [[noreturn]] void throw_invalid_kaminpar_graph(const std::string &reason)
        {
            throw std::runtime_error("[MAS] Invalid graph for KaMinPar: " + reason);
        }

        void guard_graph_for_kaminpar(ctd::span<int> row_ptr,
                                      ctd::span<int> cols,
                                      ctd::span<int64_t> weights)
        {
            if (row_ptr.empty())
            {
                throw_invalid_kaminpar_graph("row_ptr is empty");
            }
            if (row_ptr.front() != 0)
            {
                std::ostringstream oss;
                oss << "row_ptr[0] must be 0, got " << row_ptr.front();
                throw_invalid_kaminpar_graph(oss.str());
            }

            const int node_num = static_cast<int>(row_ptr.size()) - 1;
            const int edge_num = row_ptr.back();
            if (edge_num < 0)
            {
                std::ostringstream oss;
                oss << "row_ptr.back() is negative: " << edge_num;
                throw_invalid_kaminpar_graph(oss.str());
            }
            if (edge_num != static_cast<int>(cols.size()))
            {
                std::ostringstream oss;
                oss << "row_ptr.back() (" << edge_num << ") != cols.size() (" << cols.size() << ")";
                throw_invalid_kaminpar_graph(oss.str());
            }
            if (!weights.empty() && weights.size() != cols.size())
            {
                std::ostringstream oss;
                oss << "weights.size() (" << weights.size() << ") != cols.size() (" << cols.size() << ")";
                throw_invalid_kaminpar_graph(oss.str());
            }

            const bool weighted = !weights.empty();
            std::vector<std::pair<int, int64_t>> entries;
            int max_degree = 0;
            for (int u = 0; u < node_num; ++u)
            {
                const int begin = row_ptr[u];
                const int end = row_ptr[u + 1];
                if (begin > end)
                {
                    std::ostringstream oss;
                    oss << "row_ptr is not monotone at row " << u << ": " << begin << " > " << end;
                    throw_invalid_kaminpar_graph(oss.str());
                }
                max_degree = std::max(max_degree, end - begin);
            }
            entries.reserve(max_degree);

            // Sort each row in-place so the subsequent reverse-edge check is deterministic.
            for (int u = 0; u < node_num; ++u)
            {
                const int begin = row_ptr[u];
                const int end = row_ptr[u + 1];
                entries.clear();
                for (int e = begin; e < end; ++e)
                {
                    const int v = cols[e];
                    if (v < 0 || v >= node_num)
                    {
                        std::ostringstream oss;
                        oss << "neighbor out of range at row " << u << ", edge " << e << ": " << v
                            << " not in [0, " << node_num << ")";
                        throw_invalid_kaminpar_graph(oss.str());
                    }
                    if (u == v)
                    {
                        std::ostringstream oss;
                        oss << "self-loop at row " << u << ", edge " << e;
                        throw_invalid_kaminpar_graph(oss.str());
                    }
                    if (weighted && weights[e] <= 0)
                    {
                        std::ostringstream oss;
                        oss << "non-positive edge weight at row " << u << ", edge " << e << " (" << u
                            << " -> " << v << "): " << weights[e];
                        throw_invalid_kaminpar_graph(oss.str());
                    }
                    entries.emplace_back(v, weighted ? weights[e] : int64_t{1});
                }

                std::sort(
                    entries.begin(),
                    entries.end(),
                    [](const auto &lhs, const auto &rhs) {
                        if (lhs.first != rhs.first)
                            return lhs.first < rhs.first;
                        return lhs.second < rhs.second;
                    });

                for (int i = 1; i < static_cast<int>(entries.size()); ++i)
                {
                    if (entries[i - 1].first == entries[i].first)
                    {
                        std::ostringstream oss;
                        oss << "duplicate neighbor " << entries[i].first << " in row " << u;
                        throw_invalid_kaminpar_graph(oss.str());
                    }
                }

                for (int i = 0; i < static_cast<int>(entries.size()); ++i)
                {
                    cols[begin + i] = entries[i].first;
                    if (weighted)
                    {
                        weights[begin + i] = entries[i].second;
                    }
                }
            }

            for (int u = 0; u < node_num; ++u)
            {
                for (int e = row_ptr[u]; e < row_ptr[u + 1]; ++e)
                {
                    const int v = cols[e];
                    const auto *reverse_begin = cols.data() + row_ptr[v];
                    const auto *reverse_end = cols.data() + row_ptr[v + 1];
                    const auto *reverse_it = std::lower_bound(reverse_begin, reverse_end, u);
                    if (reverse_it == reverse_end || *reverse_it != u)
                    {
                        std::ostringstream oss;
                        oss << "missing reverse edge for " << u << " -> " << v;
                        throw_invalid_kaminpar_graph(oss.str());
                    }

                    if (weighted)
                    {
                        const int reverse_edge = static_cast<int>(reverse_it - cols.data());
                        if (weights[e] != weights[reverse_edge])
                        {
                            std::ostringstream oss;
                            oss << "weight mismatch for edge pair " << u << " <-> " << v
                                << ": " << weights[e] << " vs " << weights[reverse_edge];
                            throw_invalid_kaminpar_graph(oss.str());
                        }
                    }
                }
            }
        }
    } // namespace
#endif

    void graph_partition(ctd::span<int> row_ptr,
                         ctd::span<int> cols,
                         ctd::span<int64_t> weights,
                         int max_part_size,
                         int &part_num,
                         std::vector<int> &part_id)
    {
#ifndef NDEBUG
        guard_graph_for_kaminpar(row_ptr, cols, weights);
#endif
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

        static_assert(std::is_same_v<kaminpar_node_id_t, std::uint32_t>,
                      "KaMinPar node IDs are expected to be uint32_t");
        static_assert(std::is_same_v<kaminpar_edge_id_t, std::uint32_t>,
                      "KaMinPar edge IDs are expected to be uint32_t");
        static_assert(std::is_same_v<kaminpar_block_id_t, std::uint32_t>,
                      "KaMinPar block IDs are expected to be uint32_t");
        static_assert(std::is_same_v<kaminpar_edge_weight_t, std::int64_t>,
                      "KaMinPar edge weights are expected to be int64_t (enable KAMINPAR_64BIT_WEIGHTS)");
        static_assert(std::is_same_v<kaminpar_block_weight_t, std::int64_t>,
                      "KaMinPar block weights are expected to be int64_t (enable KAMINPAR_64BIT_WEIGHTS)");

        std::vector<kaminpar_edge_id_t> xadj(row_ptr.size());
        for (size_t i = 0; i < row_ptr.size(); ++i)
        {
            if (row_ptr[i] < 0)
            {
                std::ostringstream oss;
                oss << "[MAS] Invalid graph for KaMinPar: row_ptr[" << i << "] is negative: " << row_ptr[i];
                throw std::runtime_error(oss.str());
            }
            xadj[i] = static_cast<kaminpar_edge_id_t>(row_ptr[i]);
        }

        std::vector<kaminpar_node_id_t> adjncy(cols.size());
        for (size_t i = 0; i < cols.size(); ++i)
        {
            if (cols[i] < 0)
            {
                std::ostringstream oss;
                oss << "[MAS] Invalid graph for KaMinPar: cols[" << i << "] is negative: " << cols[i];
                throw std::runtime_error(oss.str());
            }
            adjncy[i] = static_cast<kaminpar_node_id_t>(cols[i]);
        }

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
            static_cast<kaminpar_node_id_t>(node_num),
            xadj.data(),
            adjncy.data(),
            nullptr,
            adjwgt);

        // Impl Eq.7 of arXiv:2411.06224

        // Forcing tight K for k-way algorithm leads to low quality parition.
        // It seems like max_part_size - 2 is a good balance.
        part_num = div_round_up(node_num, max_part_size - 2);
        assert(part_num >= 1);
        const auto k = static_cast<kaminpar_block_id_t>(part_num);
        std::vector<kaminpar_block_weight_t> max_block_weights(
            k,
            static_cast<kaminpar_block_weight_t>(max_part_size));
        std::vector<kaminpar_block_id_t> partition(node_num);

        // K way partition with absolution max block weight.
        // Since our node weight is uniform, this is equivalent to limiting parition size.
        kaminpar_compute_partition_with_max_block_weights(
            partitioner,
            k,
            max_block_weights.data(),
            partition.data());

        kaminpar_context_free(ctx);
        kaminpar_free(partitioner);

        for (int i = 0; i < node_num; ++i)
        {
            if (partition[i] >= k)
            {
                std::ostringstream oss;
                oss << "[MAS] Invalid partition id from KaMinPar at node " << i
                    << ": " << partition[i] << " not in [0, " << part_num << ")";
                throw std::runtime_error(oss.str());
            }
            part_id[i] = static_cast<int>(partition[i]);
        }
    }

} // namespace polysolve::linear::mas

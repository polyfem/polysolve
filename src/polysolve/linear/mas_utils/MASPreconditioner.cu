#include <polysolve/linear/mas_utils/MASPreconditioner.hpp>

// #ifndef SPDLOG_ACTIVE_LEVEL
// #define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE
// #endif

#include <cub/cub.cuh>
#include <cuda/algorithm>
#include <cuda/std/array>
#include <cuda/std/bit>
#include <cuda/std/utility>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <vector>
#include <cassert>

namespace polysolve::linear::mas
{
    namespace
    {
        using clock = std::chrono::steady_clock;

        double elapsed_seconds(const std::chrono::time_point<clock> &begin)
        {
            return std::chrono::duration<double>(clock::now() - begin).count();
        }

        // This is not tunable.
        constexpr int BANK_SIZE = 32;

        struct HostPaddedTopology
        {
            std::vector<int> real_to_padded;
            std::vector<int> padded_to_real;
            std::vector<int> row_ptr;
            std::vector<int> cols;
            int node_num = 0;
            int padded_node_num = 0;
        };

        /// @brief Convenience wrapper for CoarseMatrices.
        struct CoarseMatricesRef
        {
            int mat_dim;
            int mat_storage_size;
            ctd::array<ctd::span<double>, MAS_LEVEL_COUNT> matrix_per_level;

            CoarseMatricesRef(CoarseMatrices &mats)
            {
                mat_dim = mats.mat_dim;
                mat_storage_size = mats.mat_storage_size;

                int scalar_offset = 0;
                for (int i = 0; i < MAS_LEVEL_COUNT; ++i)
                {
                    int level_size = mats.matrix_counts[i] * mats.mat_storage_size;
                    matrix_per_level[i] =
                        ctd::span<double>(mats.data->data() + scalar_offset, level_size);
                    scalar_offset += level_size;
                }
            }
        };

        HostPaddedTopology build_padded_topology(TopologyView topo, ctd::span<const int> part_offsets)
        {
            int node_num = topo.row_ptr.size() - 1;
            int part_num = part_offsets.size() - 1;
            int padded_node_num = part_num * BANK_SIZE;

            HostPaddedTopology out;
            out.real_to_padded.resize(node_num, -1);
            out.padded_to_real.resize(padded_node_num, -1);
            out.row_ptr.resize(padded_node_num + 1, 0);
            out.node_num = node_num;
            out.padded_node_num = padded_node_num;

            // Fill element num for padded space row.
            for (int part = 0; part < part_num; ++part)
            {
                int part_begin = part_offsets[part];
                int part_end = part_offsets[part + 1];
                int part_size = part_end - part_begin;
                int padded_begin = part * BANK_SIZE;
                assert(part_size <= BANK_SIZE);

                for (int local_id = 0; local_id < part_size; ++local_id)
                {
                    int real_id = part_begin + local_id;
                    int padded_id = padded_begin + local_id;
                    out.real_to_padded[real_id] = padded_id;
                    out.padded_to_real[padded_id] = real_id;
                    out.row_ptr[padded_id + 1] = topo.row_ptr[real_id + 1] - topo.row_ptr[real_id];
                }
            }

            // Prefix sum to complete padded CSR row ptr.
            for (int i = 0; i < out.padded_node_num; ++i)
            {
                out.row_ptr[i + 1] += out.row_ptr[i];
            }

            // Fill padded space cols.
            out.cols.resize(out.row_ptr.back());
            for (int real_id = 0; real_id < node_num; ++real_id)
            {
                int padded_id = out.real_to_padded[real_id];
                int dst = out.row_ptr[padded_id];
                for (int n = topo.row_ptr[real_id]; n < topo.row_ptr[real_id + 1]; ++n)
                {
                    out.cols[dst] = out.real_to_padded[topo.cols[n]];
                    ++dst;
                }
            }

            return out;
        }

        /// @brief Get coarse space CCO id.
        /// @param map Coarse space map.
        /// @param vid Vertex id.
        /// @param level Coarse space level. Start at lv 0.
        /// @return CCO id.
        __both__ int get_coarse_space_id(ctd::span<const int> map, int vid, int level)
        {
            return map[vid * MAS_MAX_COARSE_LEVEL + level];
        }

        /// @brief Build local CCO mapping from input space -> coarse space lv 0 and collapse topology.
        /// @param row_ptr CSR graph topology (does not include self).
        /// @param cols CSR graph topology (does not include self).
        /// @param padded_to_real Padded id -> real id. -1 denotes padding.
        /// @param cco_num_per_bank CCO number per bank.
        /// @param local_cco_ids Bank local CCO id at coarse space lv 0.
        __global__ void build_local_cco_lv0(ctd::span<const int> row_ptr,
                                            ctd::span<int> cols,
                                            ctd::span<const int> padded_to_real,
                                            ctd::span<int> cco_num_per_bank,
                                            ctd::span<int> local_cco_ids)
        {
            // Bank local neighbor masks.
            __shared__ uint32_t neighbors[128];

            int btid = threadIdx.x;                   // block local thread id
            int tid = blockDim.x * blockIdx.x + btid; // global thread id
            int wid = threadIdx.x / BANK_SIZE;        // block local warp id
            int lid = threadIdx.x % BANK_SIZE;        // lane id
            int bid = tid / BANK_SIZE;                // bank id

            if (lid == 0 && bid < cco_num_per_bank.size())
            {
                cco_num_per_bank[bid] = 0;
            }

            int node_num = row_ptr.size() - 1;
            if (tid >= node_num)
            {
                return;
            }
            // Skip virtual padding.
            if (padded_to_real[tid] == -1)
            {
                local_cco_ids[tid] = -1;
                return;
            }

            // Build neighbor mask (including self) and collapse topology.
            int row_begin = row_ptr[tid];
            int row_end = row_ptr[tid + 1];
            uint32_t neighbor = (1u << lid);
            int out_of_bank_neighbor_count = 0;
            for (int n = row_begin; n < row_end; ++n)
            {
                int neighbor_vid = cols[n];
                int neighbor_bid = neighbor_vid / BANK_SIZE;
                if (bid == neighbor_bid)
                {
                    neighbor |= (1u << (neighbor_vid % BANK_SIZE));
                }
                else
                {
                    // Compact topology to remove in-bank neighbors so we can skip
                    // additional filtering when building future coarse spaces.
                    cols[row_begin + out_of_bank_neighbor_count] = neighbor_vid;
                    ++out_of_bank_neighbor_count;
                }
            }
            if (row_begin + out_of_bank_neighbor_count < row_end)
            {
                // -1 denotes neighbor list ending.
                cols[row_begin + out_of_bank_neighbor_count] = -1;
            }
            neighbors[btid] = neighbor;
            __syncwarp();

            // Build connectivity mask (including self).
            uint32_t connection = neighbor;
            uint32_t visited = (1u << lid);
            uint32_t to_visit = connection ^ visited;
            while (to_visit)
            {
                int visiting = ctd::countr_zero(to_visit);
                connection |= neighbors[visiting + wid * BANK_SIZE];
                visited |= (1u << visiting);
                to_visit = connection ^ visited;
            }

            // Find bank (warp) local connected component (CCO).
            uint32_t cco_lead_lid = ctd::countr_zero(connection);
            bool is_lead = (cco_lead_lid == lid);
            uint32_t lead_lanes = __ballot_sync(0xFFFFFFFFu, is_lead);
            uint32_t before_cco_lead_mask = static_cast<uint32_t>((1ull << cco_lead_lid) - 1ull);

            // Write cco info back.
            local_cco_ids[tid] = ctd::popcount(lead_lanes & before_cco_lead_mask);
            if (lid == 0)
            {
                cco_num_per_bank[bid] = ctd::popcount(lead_lanes);
            }
        }

        /// @brief Build bank neighbor mask and collapse topology.
        /// @param level Target coarse space level.
        /// @param coarse_space_map Coarse space map.
        /// @param row_ptr CSR graph topology (does not include self).
        /// @param cols CSR graph topology (does not include self).
        /// @param padded_to_real Padded id -> real id. -1 denotes padding.
        /// @param neighbors Bank local neighbor mask.
        __global__ void build_neighbor_masks_lvx(int level,
                                                 ctd::span<const int> coarse_space_map,
                                                 ctd::span<const int> row_ptr,
                                                 ctd::span<int> cols,
                                                 ctd::span<const int> padded_to_real,
                                                 ctd::span<uint32_t> neighbors)
        {
            int tid = blockDim.x * blockIdx.x + threadIdx.x; // global thread id
            int node_num = row_ptr.size() - 1;
            if (tid >= node_num)
            {
                return;
            }
            // Skip virtual padding.
            if (padded_to_real[tid] == -1)
            {
                return;
            }

            // Build neighbor mask (including self) and collapse topology.
            int row_begin = row_ptr[tid];
            int row_end = row_ptr[tid + 1];
            int real_id = padded_to_real[tid];
            int cco_id = get_coarse_space_id(coarse_space_map, real_id, level - 1);
            uint32_t neighbor = (1u << (cco_id % BANK_SIZE));
            int out_of_bank_neighbor_count = 0;
            for (int n = row_begin; n < row_end; ++n)
            {
                int neighbor_vid = cols[n];
                if (neighbor_vid == -1)
                {
                    break;
                }

                int neighbor_real_id = padded_to_real[neighbor_vid];
                int neighbor_cco_id =
                    get_coarse_space_id(coarse_space_map, neighbor_real_id, level - 1);
                if (cco_id / BANK_SIZE == neighbor_cco_id / BANK_SIZE)
                {
                    neighbor |= (1u << (neighbor_cco_id % BANK_SIZE));
                }
                else
                {
                    cols[row_begin + out_of_bank_neighbor_count] = neighbor_vid;
                    ++out_of_bank_neighbor_count;
                }
            }
            if (row_begin + out_of_bank_neighbor_count < row_end)
            {
                cols[row_begin + out_of_bank_neighbor_count] = -1;
            }

            atomicOr(neighbors.data() + cco_id, neighbor);
        }

        /// @brief Build local CCO mapping from coarse space level x-1 -> coarse space level x.
        /// @param cco_num Total CCO number at coarse space lv x-1.
        /// @param cco_num_per_bank Independent CCO number per bank at coarse space lv x.
        /// @param local_cco_ids Bank local CCO id at coarse space lv x.
        __global__ void build_local_cco_lvx(int cco_num,
                                            ctd::span<const uint32_t> neighbors,
                                            ctd::span<int> cco_num_per_bank,
                                            ctd::span<int> local_cco_ids)
        {
            int tid = blockDim.x * blockIdx.x + threadIdx.x; // global thread id
            int bid = tid / BANK_SIZE;                       // bank id
            int lid = tid % BANK_SIZE;                       // lane id
            if (tid >= cco_num)
            {
                return;
            }

            // Build connectivity mask (including self).
            uint32_t connection = neighbors[tid];
            uint32_t visited = (1u << lid);
            uint32_t to_visit = connection ^ visited;
            while (to_visit)
            {
                int visiting = ctd::countr_zero(to_visit);
                connection |= neighbors[visiting + bid * BANK_SIZE];
                visited |= (1u << visiting);
                to_visit = connection ^ visited;
            }

            // Find bank (warp) local connected component (CCO).
            uint32_t cco_lead_lid = ctd::countr_zero(connection);
            bool is_lead = (cco_lead_lid == lid);
            uint32_t lead_lanes = __ballot_sync(0xFFFFFFFFu, is_lead);
            uint32_t before_cco_lead_mask = static_cast<uint32_t>((1ull << cco_lead_lid) - 1ull);

            // Write cco info back.
            local_cco_ids[tid] = ctd::popcount(lead_lanes & before_cco_lead_mask);
            if (lid == 0)
            {
                cco_num_per_bank[bid] = ctd::popcount(lead_lanes);
            }
        }

        /// @brief Compute global CCO id and write it to coarse space map.
        ///
        /// Compute global CCO id offset from prefix sum, then write result to coarse_space_map.
        __global__ void aggregate_coarse_space_map_lv0(ctd::span<const int> local_cco_ids,
                                                       ctd::span<const int> cco_num_per_bank,
                                                       ctd::span<const int> cco_num_per_bank_summed,
                                                       ctd::span<const int> real_to_padded,
                                                       ctd::span<int> coarse_space_map)
        {
            int real_id = blockDim.x * blockIdx.x + threadIdx.x;
            if (real_id >= real_to_padded.size())
            {
                return;
            }

            int padded_id = real_to_padded[real_id];
            int bid = padded_id / BANK_SIZE;
            int cco_id =
                local_cco_ids[padded_id] + cco_num_per_bank_summed[bid] - cco_num_per_bank[bid];
            coarse_space_map[real_id * MAS_MAX_COARSE_LEVEL] = cco_id;
        }

        /// @brief Compute global CCO id and write it to coarse space map.
        ///
        /// Compute global CCO id offset from prefix sum, then write result to coarse_space_map.
        __global__ void aggregate_coarse_space_map_lvx(int level,
                                                       ctd::span<const int> local_cco_ids,
                                                       ctd::span<const int> cco_num_per_bank,
                                                       ctd::span<const int> cco_num_per_bank_summed,
                                                       ctd::span<int> coarse_space_map)
        {
            int real_id = blockDim.x * blockIdx.x + threadIdx.x;
            int node_num = coarse_space_map.size() / MAS_MAX_COARSE_LEVEL;
            if (real_id >= node_num)
            {
                return;
            }

            int prev_cco_id = get_coarse_space_id(coarse_space_map, real_id, level - 1);
            int bid = prev_cco_id / BANK_SIZE;
            int cco_id =
                local_cco_ids[prev_cco_id] + cco_num_per_bank_summed[bid] - cco_num_per_bank[bid];
            coarse_space_map[real_id * MAS_MAX_COARSE_LEVEL + level] = cco_id;
        }

        /// @brief Build coarse space map.
        ///
        /// Coarse space map real node id layout:
        /// | node0_lv0 node0_lv1 ... node0_lv_max | node1_lv0 ... node1_lv_max | ... |
        ///
        /// @param row_ptr CSR graph topology (does not include self).
        /// @param cols CSR graph topology (does not include self).
        /// @param real_to_padded Real id -> padded id.
        /// @param padded_to_real Padded id -> real id. -1 denotes virtual padding.
        /// @rt Cuda runtime config.
        /// @warning Will modify cols as side effect!
        CoarseSpace build_coarse_space(ctd::span<const int> row_ptr,
                                       ctd::span<int> cols,
                                       ctd::span<const int> real_to_padded,
                                       ctd::span<const int> padded_to_real,
                                       CudaRuntime rt)
        {
            ctd::array<int, MAS_MAX_COARSE_LEVEL> cco_nums{};

            int padded_node_num = row_ptr.size() - 1;
            int real_node_num = real_to_padded.size();
            int max_bank_per_level = div_round_up(padded_node_num, BANK_SIZE);
            // Bank local CCO id.
            auto local_cco_ids =
                cu::make_buffer<int>(rt.stream, rt.mr, padded_node_num, cu::no_init);
            auto cco_num_per_bank =
                cu::make_buffer<int>(rt.stream, rt.mr, max_bank_per_level, cu::no_init);
            // Inclusive prefix sum of cco_num_per_bank.
            auto cco_num_per_bank_summed =
                cu::make_buffer<int>(rt.stream, rt.mr, max_bank_per_level, cu::no_init);
            auto coarse_space_map =
                cu::make_buffer<int>(rt.stream, rt.mr, real_node_num * MAS_MAX_COARSE_LEVEL, cu::no_init);

            // Build coarse map level 0.
            int grid_num = div_round_up(padded_node_num, 128);
            build_local_cco_lv0<<<grid_num, 128, 0, rt.stream.get()>>>(
                row_ptr, cols, padded_to_real, cco_num_per_bank, local_cco_ids);

            // Prefix sum to compute global CCO id offset.
            size_t cub_tmp_size = 0;
            cub::DeviceScan::InclusiveSum(nullptr,
                                          cub_tmp_size,
                                          cco_num_per_bank.data(),
                                          cco_num_per_bank_summed.data(),
                                          max_bank_per_level,
                                          rt.stream.get());
            auto cub_tmp =
                cu::make_buffer<char>(rt.stream, rt.mr, cub_tmp_size, cu::no_init);
            cub::DeviceScan::InclusiveSum(cub_tmp.data(),
                                          cub_tmp_size,
                                          cco_num_per_bank.data(),
                                          cco_num_per_bank_summed.data(),
                                          max_bank_per_level,
                                          rt.stream.get());

            aggregate_coarse_space_map_lv0<<<div_round_up(real_node_num, 128), 128, 0, rt.stream.get()>>>(
                local_cco_ids, cco_num_per_bank, cco_num_per_bank_summed, real_to_padded, coarse_space_map);

            int cco_num = device2host(cco_num_per_bank_summed.data() + max_bank_per_level - 1, rt);
            cco_nums[0] = cco_num;
            int level_num = 1;

            // Build coarse map level 1 ... MAS_MAX_COARSE_LEVEL-1 recursively.
            // We do:
            //  1. Build bank local neighbbor bit mask.
            //  2. Build bank local connectivity mask and find local CCO partition.
            //  3. Compute global CCO id and write to coarse space map.
            auto neighbors = cu::make_buffer<uint32_t>(rt.stream, rt.mr, padded_node_num, cu::no_init);
            for (int lv = 1; lv < MAS_MAX_COARSE_LEVEL; ++lv)
            {
                cudaMemsetAsync(neighbors.data(), 0, cco_num * sizeof(uint32_t), rt.stream.get());

                grid_num = div_round_up(padded_node_num, 128);
                build_neighbor_masks_lvx<<<grid_num, 128, 0, rt.stream.get()>>>(
                    lv,
                    coarse_space_map,
                    row_ptr,
                    cols,
                    padded_to_real,
                    ctd::span<uint32_t>(neighbors.data(), cco_num));

                int bank_num = div_round_up(cco_num, BANK_SIZE);
                grid_num = div_round_up(cco_num, 128);
                build_local_cco_lvx<<<grid_num, 128, 0, rt.stream.get()>>>(
                    cco_num,
                    ctd::span<const uint32_t>(neighbors.data(), cco_num),
                    ctd::span<int>(cco_num_per_bank.data(), bank_num),
                    ctd::span<int>(local_cco_ids.data(), cco_num));

                cub::DeviceScan::InclusiveSum(cub_tmp.data(),
                                              cub_tmp_size,
                                              cco_num_per_bank.data(),
                                              cco_num_per_bank_summed.data(),
                                              bank_num,
                                              rt.stream.get());
                aggregate_coarse_space_map_lvx<<<div_round_up(real_node_num, 128), 128, 0, rt.stream.get()>>>(
                    lv,
                    ctd::span<const int>(local_cco_ids.data(), cco_num),
                    ctd::span<const int>(cco_num_per_bank.data(), bank_num),
                    ctd::span<const int>(cco_num_per_bank_summed.data(), bank_num),
                    coarse_space_map);

                // No more CCO to merge.
                int next_cco_num = device2host(cco_num_per_bank_summed.data() + bank_num - 1, rt);
                if (next_cco_num == cco_num)
                {
                    break;
                }

                cco_num = next_cco_num;
                cco_nums[lv] = cco_num;
                level_num = lv + 1;
            }

            CoarseSpace cs;
            cs.cco_nums = cco_nums;
            cs.level_num = level_num;
            cs.map = std::move(coarse_space_map);
            return cs;
        }

        /// @brief Compute offset of row major upper triangular dim x dim matrix.
        __both__ int index_upper_mat(int dim, int i, int j)
        {
            if (i > j)
            {
                ctd::swap(i, j);
            }
            return i * dim - (i * (i + 1) / 2) + j;
        }

        /// @brief Build coarse space matrices by accumulating padded space hessian.
        __global__ void fill_coarse_matrices(BSRView mat_in,
                                             CoarseMatricesRef mat_out,
                                             ctd::span<const int> coarse_space_map,
                                             ctd::span<const int> real_to_padded,
                                             int coarse_level_num)
        {
            int tid = blockDim.x * blockIdx.x + threadIdx.x; // global thread id
            if (tid >= mat_in.dim)
            {
                return;
            }

            int padded_i = real_to_padded[tid];
            int block_size = mat_in.block_dim * mat_in.block_dim;
            for (int nz = mat_in.rows[tid]; nz < mat_in.rows[tid + 1]; ++nz)
            {
                int j = mat_in.cols[nz];
                // Assume input mat is symmetric (Which is required by CG anyway). Skip lower.
                if (tid > j)
                {
                    continue;
                }

                int padded_j = real_to_padded[j];
                int block_offset = nz * block_size;
                // Padded space matrices.
                if (padded_i / BANK_SIZE == padded_j / BANK_SIZE)
                {
                    int mat_id = padded_i / BANK_SIZE;
                    int scalar_i_root = (padded_i % BANK_SIZE) * mat_in.block_dim;
                    int scalar_j_root = (padded_j % BANK_SIZE) * mat_in.block_dim;
                    double *mat = mat_out.matrix_per_level[0].data() + mat_id * mat_out.mat_storage_size;
                    for (int bi = 0; bi < mat_in.block_dim; ++bi)
                    {
                        for (int bj = 0; bj < mat_in.block_dim; ++bj)
                        {
                            int row = scalar_i_root + bi;
                            int col = scalar_j_root + bj;
                            // Write upper part of coarse matrix only.
                            if (row > col)
                            {
                                continue;
                            }
                            // For diagonal block (tid == j), write upper part only.
                            if (tid == j && bi > bj)
                            {
                                continue;
                            }

                            double val = mat_in.vals[block_offset + bi * mat_in.block_dim + bj];
                            atomicAdd(mat + index_upper_mat(mat_out.mat_dim, row, col), val);
                        }
                    }
                }
                // Coarse space matrices.
                for (int lv = 0; lv < coarse_level_num; ++lv)
                {
                    int cco_i = coarse_space_map[tid * MAS_MAX_COARSE_LEVEL + lv];
                    int cco_j = coarse_space_map[j * MAS_MAX_COARSE_LEVEL + lv];
                    if (cco_i / BANK_SIZE != cco_j / BANK_SIZE)
                    {
                        continue;
                    }

                    int mat_id = cco_i / BANK_SIZE;
                    int scalar_i_root = (cco_i % BANK_SIZE) * mat_in.block_dim;
                    int scalar_j_root = (cco_j % BANK_SIZE) * mat_in.block_dim;
                    double *mat = mat_out.matrix_per_level[lv + 1].data() + mat_id * mat_out.mat_storage_size;
                    for (int bi = 0; bi < mat_in.block_dim; ++bi)
                    {
                        for (int bj = 0; bj < mat_in.block_dim; ++bj)
                        {
                            int row = scalar_i_root + bi;
                            int col = scalar_j_root + bj;
                            // Write upper part of coarse matrix only.
                            if (row > col)
                            {
                                continue;
                            }
                            // For diagonal block (tid == j), write upper part only.
                            if (tid == j && bi > bj)
                            {
                                continue;
                            }

                            double val = mat_in.vals[block_offset + bi * mat_in.block_dim + bj];
                            // When two different fine nodes collapse to the same coarse node,
                            // the coarse diagonal block must receive both A_ij and A_ji = A_ij^T.
                            if (tid != j && cco_i == cco_j)
                            {
                                val += mat_in.vals[block_offset + bj * mat_in.block_dim + bi];
                            }
                            atomicAdd(mat + index_upper_mat(mat_out.mat_dim, row, col), val);
                        }
                    }
                }
            }
        }

        __global__ void gather_multi_level_r(
            ctd::span<const double> r,
            ctd::span<double> multi_level_r,
            ctd::span<const int> real_to_padded,
            ctd::span<const int> coarse_space_map,
            ctd::array<int, MAS_LEVEL_COUNT> level_offsets,
            int block_dim,
            int coarse_level_num)
        {
            int real_id = blockDim.x * blockIdx.x + threadIdx.x;
            if (real_id >= real_to_padded.size())
            {
                return;
            }

            int padded_id = real_to_padded[real_id];
            int src_root = real_id * block_dim;
            int fine_root = padded_id * block_dim;
            // Gather padded space.
            for (int i = 0; i < block_dim; ++i)
            {
                multi_level_r[fine_root + i] = r[src_root + i];
            }
            // Gather coarse space.
            for (int lv = 0; lv < coarse_level_num; ++lv)
            {
                int cco_id = coarse_space_map[real_id * MAS_MAX_COARSE_LEVEL + lv];
                int dst_root = (level_offsets[lv + 1] + cco_id) * block_dim;
                for (int i = 0; i < block_dim; ++i)
                {
                    atomicAdd(multi_level_r.data() + dst_root + i, r[src_root + i]);
                }
            }
        }

        /// @brief Matrix vector multiplication. A is packed upper row major matrix.
        ///
        /// This is the main bottleneck of MAS preconditioner. But since we are vram bandwidth
        /// bound, there's nothing I could do. The compute / memory ratio is just too low.
        template <int N, int BLOCK>
        __global__ void symv_upper_packed(const double *A_upper,
                                          const double *x,
                                          double *y)
        {
            int mat_id = blockIdx.x;
            int row = threadIdx.x;
            constexpr int L = N * (N + 1) / 2;
            const double *Amat = A_upper + mat_id * L;

            // All threads in block load A and x into shared mem.

            __shared__ double sx[N];
            __shared__ double sA[L];

            for (int i = threadIdx.x; i < N; i += BLOCK)
            {
                sx[i] = x[mat_id * N + i];
            }
            for (int k = threadIdx.x; k < L; k += BLOCK)
            {
                sA[k] = Amat[k];
            }

            __syncthreads();

            // Each thread do one row * x.

            if (row >= N)
            {
                return;
            }

            double sum = 0.0;
            for (int col = 0; col < N; ++col)
            {
                sum += sA[index_upper_mat(N, row, col)] * sx[col];
            }

            y[mat_id * N + row] = sum;
        }

        void apply_inverse(const CoarseMatrices &mats,
                           ctd::span<const double> x,
                           ctd::span<double> y,
                           int block_dim,
                           CudaRuntime rt)
        {
            int mat_num = mats.total_matrix_num;
            if (mat_num == 0)
            {
                return;
            }

            if (block_dim == 1)
            {
                symv_upper_packed<32, 32><<<mat_num, 32, 0, rt.stream.get()>>>(
                    mats.data->data(),
                    x.data(),
                    y.data());
                return;
            }
            if (block_dim == 2)
            {
                symv_upper_packed<64, 64><<<mat_num, 64, 0, rt.stream.get()>>>(
                    mats.data->data(),
                    x.data(),
                    y.data());
                return;
            }
            if (block_dim == 3)
            {
                symv_upper_packed<96, 96><<<mat_num, 96, 0, rt.stream.get()>>>(
                    mats.data->data(),
                    x.data(),
                    y.data());
                return;
            }

            throw std::runtime_error("[CudaPCG] MAS only supports block size 1, 2, or 3.");
        }

        __global__ void gather_multi_level_z(
            ctd::span<const double> multi_level_z,
            ctd::span<double> z,
            ctd::span<const int> real_to_padded,
            ctd::span<const int> coarse_space_map,
            ctd::array<int, MAS_LEVEL_COUNT> level_offsets,
            int block_dim,
            int coarse_level_num)
        {
            int real_id = blockDim.x * blockIdx.x + threadIdx.x;
            if (real_id >= real_to_padded.size())
            {
                return;
            }

            // Padded space.
            int padded_id = real_to_padded[real_id];
            int dst_root = real_id * block_dim;
            for (int i = 0; i < block_dim; ++i)
            {
                z[dst_root + i] = multi_level_z[padded_id * block_dim + i];
            }
            // Coarse space.
            for (int lv = 0; lv < coarse_level_num; ++lv)
            {
                int cco_id = coarse_space_map[real_id * MAS_MAX_COARSE_LEVEL + lv];
                int src_root = (level_offsets[lv + 1] + cco_id) * block_dim;
                for (int i = 0; i < block_dim; ++i)
                {
                    z[dst_root + i] += multi_level_z[src_root + i];
                }
            }
        }

        __global__ void pad_zero_diagonal(double *mats, int mat_num, int mat_dim, int mat_storage_size)
        {
            int tid = blockDim.x * blockIdx.x + threadIdx.x;
            int total_diag = mat_num * mat_dim;
            if (tid >= total_diag)
            {
                return;
            }

            int mat_id = tid / mat_dim;
            int row = tid % mat_dim;
            double *mat = mats + mat_id * mat_storage_size;
            double *diag = mat + index_upper_mat(mat_dim, row, row);
            if (*diag == 0.0)
            {
                *diag = 1.0;
            }
        }

        /// @brief Invert a packed symmetric matrix in-place using symmetric Gauss-Jordan sweeps.
        template <int N>
        __global__ void batched_invert_upper(double *d_matrices,
                                             bool *success)
        {
            int mat_idx = blockIdx.x;
            constexpr int STORAGE = N * (N + 1) / 2;
            double *d_A = d_matrices + mat_idx * STORAGE;

            __shared__ double s_A[STORAGE];
            __shared__ double s_col[N];
            __shared__ double s_pivot;
            int tx = threadIdx.x;

            for (int i = tx; i < STORAGE; i += N)
            {
                s_A[i] = d_A[i];
            }
            __syncthreads();

            for (int pivot = 0; pivot < N; ++pivot)
            {
                if (tx == 0)
                {
                    s_pivot = s_A[index_upper_mat(N, pivot, pivot)];
                    if (!ctd::isfinite(s_pivot) || s_pivot == 0.0)
                    {
                        *success = false;
                    }
                }
                __syncthreads();

                if (tx < N)
                {
                    if (tx == pivot)
                    {
                        s_col[tx] = 0.0;
                    }
                    else
                    {
                        int row = ctd::min(tx, pivot);
                        int col = ctd::max(tx, pivot);
                        s_col[tx] = s_A[index_upper_mat(N, row, col)];
                    }
                }
                __syncthreads();

                if (tx < N && tx != pivot)
                {
                    double a_ik = s_col[tx];
                    for (int col = tx; col < N; ++col)
                    {
                        if (col == pivot)
                        {
                            continue;
                        }

                        double updated =
                            s_A[index_upper_mat(N, tx, col)] - a_ik * s_col[col] / s_pivot;
                        if (!ctd::isfinite(updated))
                        {
                            *success = false;
                        }
                        s_A[index_upper_mat(N, tx, col)] = updated;
                    }
                }
                __syncthreads();

                if (tx == pivot)
                {
                    double updated = -1.0 / s_pivot;
                    if (!ctd::isfinite(updated))
                    {
                        *success = false;
                    }
                    else
                    {
                        s_A[index_upper_mat(N, pivot, pivot)] = updated;
                    }
                }
                else if (tx < N)
                {
                    double updated = s_col[tx] / s_pivot;
                    if (!ctd::isfinite(updated))
                    {
                        *success = false;
                    }
                    else
                    {
                        int row = ctd::min(tx, pivot);
                        int col = ctd::max(tx, pivot);
                        s_A[index_upper_mat(N, row, col)] = updated;
                    }
                }
                __syncthreads();
            }

            for (int i = tx; i < STORAGE; i += N)
            {
                double output = -s_A[i];
                if (!ctd::isfinite(output))
                {
                    *success = false;
                }
                d_A[i] = output;
            }
        }

        void invert_packed_matrices(double *mats, int mat_num, int block_dim, CudaRuntime rt)
        {
            if (mat_num == 0)
            {
                return;
            }

            auto success = cu::make_buffer<bool>(rt.stream, rt.mr, 1, true);

            if (block_dim == 1)
            {
                batched_invert_upper<32><<<mat_num, 32, 0, rt.stream.get()>>>(
                    mats, success.data());
            }
            else if (block_dim == 2)
            {
                batched_invert_upper<64><<<mat_num, 64, 0, rt.stream.get()>>>(
                    mats, success.data());
            }
            else if (block_dim == 3)
            {
                batched_invert_upper<96><<<mat_num, 96, 0, rt.stream.get()>>>(
                    mats, success.data());
            }
            else
            {
                assert(false);
            }

            bool host_success = device2host(success.data(), rt);
            if (!host_success)
            {
                throw std::runtime_error("[CudaPCG] MAS packed inverse failed.");
            }
        }

        /// @brief Build symmetric upper triangular coarse space matrices.
        CoarseMatrices build_sym_coarse_matrices(const CoarseSpace &cs,
                                                 BSRView mat,
                                                 ctd::span<const int> real_to_padded,
                                                 int padded_node_num,
                                                 CudaRuntime rt)
        {
            CoarseMatrices out;
            out.mat_dim = BANK_SIZE * mat.block_dim;
            out.mat_storage_size = out.mat_dim * (out.mat_dim + 1) / 2;
            out.total_matrix_num = 0;
            out.matrix_offsets[0] = 0;
            out.matrix_counts[0] = padded_node_num / BANK_SIZE;
            out.total_matrix_num += out.matrix_counts[0];
            for (int i = 0; i < cs.level_num; ++i)
            {
                out.matrix_offsets[i + 1] = out.total_matrix_num;
                out.matrix_counts[i + 1] = div_round_up(cs.cco_nums[i], BANK_SIZE);
                out.total_matrix_num += out.matrix_counts[i + 1];
            }

            out.data = cu::make_buffer<double>(
                rt.stream,
                rt.mr,
                out.total_matrix_num * out.mat_storage_size,
                0.0);
            CoarseMatricesRef view{out};

            // Gather.
            int grid_num = div_round_up(mat.dim, 128);
            fill_coarse_matrices<<<grid_num, 128, 0, rt.stream.get()>>>(
                mat, view, *cs.map, real_to_padded, cs.level_num);

            // Pad diagonal.
            int total_diag = out.total_matrix_num * out.mat_dim;
            pad_zero_diagonal<<<div_round_up(total_diag, 128), 128, 0, rt.stream.get()>>>(
                out.data->data(), out.total_matrix_num, out.mat_dim, out.mat_storage_size);

            // Compute inverse.
            invert_packed_matrices(out.data->data(), out.total_matrix_num, mat.block_dim, rt);
            return out;
        }
    } // namespace

    void MASPreconditioner::factorize(const BSRMatrix &A,
                                      ctd::span<const int> part_offsets,
                                      CudaRuntime rt)
    {
        assert(part_offsets.size() >= 2);

        BSRView view = A.view();
        assert(view.block_dim >= 1 && view.block_dim <= 3);

        initialized_ = false;
        block_dim_ = view.block_dim;
        vector_size_ = view.dim * view.block_dim;

        auto total_begin = clock::now();
        auto phase_begin = clock::now();

        // Build host padded topology.
        HostPaddedTopology topo = build_padded_topology(A.host_topology_view(), part_offsets);
        SPDLOG_TRACE("CUDA_PCG MAS: build_padded_topology {:.6f}s", elapsed_seconds(phase_begin));

        // Transfer host padded topology to device.
        phase_begin = clock::now();
        padded_topology_.node_num = topo.node_num;
        padded_topology_.padded_node_num = topo.padded_node_num;
        padded_topology_.real_to_padded =
            cu::make_buffer<int>(rt.stream, rt.mr, topo.real_to_padded.size(), cu::no_init);
        padded_topology_.padded_to_real =
            cu::make_buffer<int>(rt.stream, rt.mr, topo.padded_to_real.size(), cu::no_init);
        padded_topology_.rows =
            cu::make_buffer<int>(rt.stream, rt.mr, topo.row_ptr.size(), cu::no_init);
        padded_topology_.cols =
            cu::make_buffer<int>(rt.stream, rt.mr, topo.cols.size(), cu::no_init);
        cu::copy_bytes(rt.stream, topo.real_to_padded, *padded_topology_.real_to_padded);
        cu::copy_bytes(rt.stream, topo.padded_to_real, *padded_topology_.padded_to_real);
        cu::copy_bytes(rt.stream, topo.row_ptr, *padded_topology_.rows);
        cu::copy_bytes(rt.stream, topo.cols, *padded_topology_.cols);
        rt.stream.sync();
        SPDLOG_TRACE("CUDA_PCG MAS: copy_padded_topology {:.6f}s", elapsed_seconds(phase_begin));

        // Build coarse space hierarchy.
        phase_begin = clock::now();
        coarse_space_ = build_coarse_space(
            *(padded_topology_.rows),
            *(padded_topology_.cols),
            *(padded_topology_.real_to_padded),
            *(padded_topology_.padded_to_real),
            rt);
        rt.stream.sync();
        SPDLOG_TRACE("CUDA_PCG MAS: build_coarse_space {:.6f}s", elapsed_seconds(phase_begin));

        // Build coarse space matrices by 1. gather coarse space hessian from fine nodes 2. invert
        phase_begin = clock::now();
        coarse_matrices_ = build_sym_coarse_matrices(
            coarse_space_,
            view,
            *(padded_topology_.real_to_padded),
            padded_topology_.padded_node_num,
            rt);
        rt.stream.sync();
        SPDLOG_TRACE("CUDA_PCG MAS: build_coarse_matrices {:.6f}s", elapsed_seconds(phase_begin));

        // Allocate memory for coarse space residual (r) and preconditioned residual (z).
        phase_begin = clock::now();
        coarse_vectors_.level_offsets[0] = 0;
        coarse_vectors_.level_sizes[0] = padded_topology_.padded_node_num;
        coarse_vectors_.total_level_nodes = padded_topology_.padded_node_num;
        for (int i = 0; i < coarse_space_.level_num; ++i)
        {
            coarse_vectors_.level_offsets[i + 1] = coarse_vectors_.total_level_nodes;
            coarse_vectors_.level_sizes[i + 1] = coarse_matrices_.matrix_counts[i + 1] * BANK_SIZE;
            coarse_vectors_.total_level_nodes += coarse_vectors_.level_sizes[i + 1];
        }
        int total_level_scalars = coarse_vectors_.total_level_nodes * block_dim_;
        coarse_vectors_.multi_level_r =
            cu::make_buffer<double>(rt.stream, rt.mr, total_level_scalars, 0.0);
        coarse_vectors_.multi_level_z =
            cu::make_buffer<double>(rt.stream, rt.mr, total_level_scalars, 0.0);
        rt.stream.sync();
        SPDLOG_TRACE("CUDA_PCG MAS: allocate_coarse_vectors {:.6f}s", elapsed_seconds(phase_begin));
        SPDLOG_TRACE("CUDA_PCG MAS: factorize_total {:.6f}s", elapsed_seconds(total_begin));

        initialized_ = true;
    }

    void MASPreconditioner::apply(
        ctd::span<const double> r,
        ctd::span<double> z,
        CudaRuntime rt)
    {
        if (!initialized_)
        {
            throw std::runtime_error("[CudaPCG] MASPreconditioner is not initialized.");
        }
        if (r.size() != z.size() || r.size() != vector_size_)
        {
            throw std::runtime_error("[CudaPCG] Invalid vector size for MAS preconditioner.");
        }

        cu::fill_bytes(rt.stream, *(coarse_vectors_.multi_level_r), 0);
        cu::fill_bytes(rt.stream, *(coarse_vectors_.multi_level_z), 0);

        int grid_num = div_round_up(padded_topology_.node_num, 128);
        gather_multi_level_r<<<grid_num, 128, 0, rt.stream.get()>>>(
            r,
            *(coarse_vectors_.multi_level_r),
            *(padded_topology_.real_to_padded),
            *(coarse_space_.map),
            coarse_vectors_.level_offsets,
            block_dim_,
            coarse_space_.level_num);

        apply_inverse(
            coarse_matrices_,
            *(coarse_vectors_.multi_level_r),
            *(coarse_vectors_.multi_level_z),
            block_dim_,
            rt);

        gather_multi_level_z<<<grid_num, 128, 0, rt.stream.get()>>>(
            *(coarse_vectors_.multi_level_z),
            z,
            *(padded_topology_.real_to_padded),
            *(coarse_space_.map),
            coarse_vectors_.level_offsets,
            block_dim_,
            coarse_space_.level_num);
    }

} // namespace polysolve::linear::mas

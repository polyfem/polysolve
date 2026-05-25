#include <polysolve/linear/mas_utils/BSRAdjacency.hpp>

#include <cub/cub.cuh>

#include <cuda/algorithm>
#include <cuda/buffer>
#include <cuda/std/functional>
#include <cuda/std/span>

#include <Eigen/SparseCore>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace polysolve::linear::mas
{
    namespace
    {
        // Key bit layout:
        // | block row idx (32) | block col idx (32) |

        __both__ uint64_t pack_key(int row, int col)
        {
            return (static_cast<uint64_t>(static_cast<uint32_t>(row)) << 32) | static_cast<uint32_t>(col);
        }

        __device__ int key_to_row(uint64_t key)
        {
            return static_cast<int>(static_cast<uint32_t>(key >> 32));
        }

        __device__ int key_to_col(uint64_t key)
        {
            return static_cast<int>(static_cast<uint32_t>(key));
        }

        __global__ void build_col_of_nnz(int col_num,
                                         ctd::span<const int> col_ptr,
                                         ctd::span<int> col_of_nnz)
        {
            int col = blockDim.x * blockIdx.x + threadIdx.x;
            if (col >= col_num)
            {
                return;
            }

            int begin = col_ptr[col];
            int end = col_ptr[col + 1];
            for (int k = begin; k < end; ++k)
            {
                col_of_nnz[k] = col;
            }
        }

        // Maps each scalar nnz to canonical undirected block pair (min, max);
        // computes |val| for aggregation into L1 edge weights;
        // flags off-diagonal entries (diagonal blocks are not edges).
        __global__ void build_keys_abs_flags(int nnz,
                                             int block_dim,
                                             ctd::span<const int> col_of_nnz,
                                             ctd::span<const int> row_idx,
                                             ctd::span<const double> vals,
                                             ctd::span<uint64_t> keys,
                                             ctd::span<double> abs_val,
                                             ctd::span<int> keep_flags)
        {
            int tid = blockDim.x * blockIdx.x + threadIdx.x;
            if (tid >= nnz)
            {
                return;
            }

            int block_row = row_idx[tid] / block_dim;
            int block_col = col_of_nnz[tid] / block_dim;
            int u = block_row < block_col ? block_row : block_col;
            int v = block_row < block_col ? block_col : block_row;
            keys[tid] = pack_key(u, v);

            double val = vals[tid];
            abs_val[tid] = fabs(val);
            keep_flags[tid] = (block_row != block_col) ? 1 : 0;
        }

        // Counts how many undirected edges touch each vertex (for CSR row_ptr construction).
        // Each edge (u, v) increments both hist[u] and hist[v].
        __global__ void count_symmetric_rows(ctd::span<const uint64_t> edge_keys,
                                             ctd::span<int> hist)
        {
            int eid = blockDim.x * blockIdx.x + threadIdx.x;
            if (eid >= edge_keys.size())
            {
                return;
            }

            uint64_t key = edge_keys[eid];
            int u = key_to_row(key);
            int v = key_to_col(key);
            atomicAdd(&hist[u], 1);
            atomicAdd(&hist[v], 1);
        }

        // Writes two entries per undirected edge — (u,v) and (v,u) — into symmetric CSR structure.
        __global__ void scatter_symmetric_edges(ctd::span<const uint64_t> edge_keys,
                                                ctd::span<const int64_t> edge_weights,
                                                ctd::span<int> write_ptr,
                                                ctd::span<int> cols,
                                                ctd::span<int64_t> weights)
        {
            int eid = blockDim.x * blockIdx.x + threadIdx.x;
            if (eid >= edge_keys.size())
            {
                return;
            }

            uint64_t key = edge_keys[eid];
            int u = key_to_row(key);
            int v = key_to_col(key);
            int64_t weight = edge_weights[eid];

            int uv_pos = atomicAdd(&write_ptr[u], 1);
            cols[uv_pos] = v;
            weights[uv_pos] = weight;

            int vu_pos = atomicAdd(&write_ptr[v], 1);
            cols[vu_pos] = u;
            weights[vu_pos] = weight;
        }

        // Quantizes continuous weights to integer levels [1, MAX_QUANT] for MAS coarsening.
        __global__ void quantize_weights(ctd::span<const double> weights_in,
                                         double min_weight,
                                         double max_weight,
                                         ctd::span<int64_t> weights_out)
        {
            int tid = blockDim.x * blockIdx.x + threadIdx.x;
            if (tid >= weights_in.size())
            {
                return;
            }

            double range = max_weight - min_weight;
            if (range <= 0.0)
            {
                weights_out[tid] = 1;
                return;
            }

            constexpr int MAX_QUANT = 1000000;
            double scaled = (weights_in[tid] - min_weight) / range * MAX_QUANT;
            int quant = static_cast<int>(scaled);
            if (quant < 1)
            {
                quant = 1;
            }
            else if (quant > MAX_QUANT)
            {
                quant = MAX_QUANT;
            }
            weights_out[tid] = static_cast<int64_t>(quant);
        }

        int build_adjacency_from_csc(const StiffnessMatrix &A_csc,
                                     int block_dim,
                                     std::vector<int> &row_ptr,
                                     std::vector<int> &cols,
                                     std::vector<int64_t> &weights,
                                     CudaRuntime rt)
        {
            const int rows_num = A_csc.rows();
            const int cols_num = A_csc.cols();
            const int nnz = A_csc.nonZeros();
            const int dim_blocks = div_round_up(rows_num, block_dim);

            // Upload input CSC to device.
            Buf<int> d_col_ptr = safe_alloc<int>(cols_num + 1, rt, "MAS topology_adj input_csc");
            Buf<int> d_row_idx = safe_alloc<int>(nnz, rt, "MAS topology_adj input_csc");
            Buf<double> d_vals = safe_alloc<double>(nnz, rt, "MAS topology_adj input_csc");
            cudaMemcpyAsync(
                d_col_ptr->data(),
                A_csc.outerIndexPtr(),
                static_cast<size_t>(cols_num + 1) * sizeof(int),
                cudaMemcpyHostToDevice,
                rt.stream.get());
            cudaMemcpyAsync(
                d_row_idx->data(),
                A_csc.innerIndexPtr(),
                static_cast<size_t>(nnz) * sizeof(int),
                cudaMemcpyHostToDevice,
                rt.stream.get());
            cudaMemcpyAsync(
                d_vals->data(),
                A_csc.valuePtr(),
                static_cast<size_t>(nnz) * sizeof(double),
                cudaMemcpyHostToDevice,
                rt.stream.get());

            // ---------------------------------------------------------------------------
            // Map each nnz index to input col.
            // ---------------------------------------------------------------------------

            Buf<int> col_of_nnz = safe_alloc<int>(nnz, rt, "MAS topology_adj col_of_nnz");
            build_col_of_nnz<<<div_round_up(cols_num, 128), 128, 0, rt.stream.get()>>>(
                cols_num, *d_col_ptr, *col_of_nnz);
            d_col_ptr->destroy();

            // ---------------------------------------------------------------------------
            // Build canonical undirected block-pair keys (min, max) and |val| per scalar nnz.
            // ---------------------------------------------------------------------------

            Buf<uint64_t> all_keys = safe_alloc<uint64_t>(nnz, rt, "MAS topology_adj staging");
            Buf<double> all_abs = safe_alloc<double>(nnz, rt, "MAS topology_adj staging");
            Buf<int> keep_flags = safe_alloc<int>(nnz, rt, "MAS topology_adj staging");
            build_keys_abs_flags<<<div_round_up(nnz, 128), 128, 0, rt.stream.get()>>>(
                nnz,
                block_dim,
                *col_of_nnz,
                *d_row_idx,
                *d_vals,
                *all_keys,
                *all_abs,
                *keep_flags);
            col_of_nnz->destroy();
            d_row_idx->destroy();
            d_vals->destroy();

            Buf<char> cub_tmp;
            auto make_cub_tmp = [&cub_tmp, rt](size_t required_size) {
                if (!cub_tmp || cub_tmp->size() < required_size)
                {
                    cub_tmp = safe_alloc<char>(required_size, rt, "MAS topology_adj cub_tmp");
                }
                return cub_tmp->data();
            };

            // ---------------------------------------------------------------------------
            // Filter off-diagonal entries only (diagonal blocks are not edges).
            // ---------------------------------------------------------------------------

            Buf<uint64_t> offdiag_keys = safe_alloc<uint64_t>(nnz, rt, "MAS topology_adj staging");
            Buf<double> offdiag_abs = safe_alloc<double>(nnz, rt, "MAS topology_adj staging");
            Buf<int> num_selected = safe_alloc<int>(1, rt, "MAS topology_adj staging");

            size_t select_tmp_size = 0;
            cub::DeviceSelect::Flagged(nullptr,
                                       select_tmp_size,
                                       all_keys->data(),
                                       keep_flags->data(),
                                       offdiag_keys->data(),
                                       num_selected->data(),
                                       nnz,
                                       rt.stream.get());
            cub::DeviceSelect::Flagged(make_cub_tmp(select_tmp_size),
                                       select_tmp_size,
                                       all_keys->data(),
                                       keep_flags->data(),
                                       offdiag_keys->data(),
                                       num_selected->data(),
                                       nnz,
                                       rt.stream.get());
            int offdiag_nnz = device2host(num_selected->data(), rt);
            if (offdiag_nnz == 0)
            {
                row_ptr.assign(dim_blocks + 1, 0);
                cols.clear();
                weights.clear();
                return 0;
            }

            cub::DeviceSelect::Flagged(make_cub_tmp(select_tmp_size),
                                       select_tmp_size,
                                       all_abs->data(),
                                       keep_flags->data(),
                                       offdiag_abs->data(),
                                       num_selected->data(),
                                       nnz,
                                       rt.stream.get());
            all_keys->destroy();
            all_abs->destroy();
            keep_flags->destroy();
            num_selected->destroy();

            // ---------------------------------------------------------------------------
            // Sort off-diagonal entries by canonical key.
            // ---------------------------------------------------------------------------

            Buf<uint64_t> offdiag_keys_alt =
                safe_alloc<uint64_t>(offdiag_nnz, rt, "MAS topology_adj staging");
            Buf<double> offdiag_abs_alt =
                safe_alloc<double>(offdiag_nnz, rt, "MAS topology_adj staging");
            cub::DoubleBuffer<uint64_t> d_keys(offdiag_keys->data(), offdiag_keys_alt->data());
            cub::DoubleBuffer<double> d_abs(offdiag_abs->data(), offdiag_abs_alt->data());

            size_t sort_tmp_size = 0;
            cub::DeviceRadixSort::SortPairs(nullptr,
                                            sort_tmp_size,
                                            d_keys,
                                            d_abs,
                                            offdiag_nnz,
                                            0,
                                            64,
                                            rt.stream.get());
            cub::DeviceRadixSort::SortPairs(make_cub_tmp(sort_tmp_size),
                                            sort_tmp_size,
                                            d_keys,
                                            d_abs,
                                            offdiag_nnz,
                                            0,
                                            64,
                                            rt.stream.get());

            Buf<uint64_t> &sorted_key_buf =
                (d_keys.Current() == offdiag_keys->data()) ? offdiag_keys : offdiag_keys_alt;
            Buf<double> &sorted_abs_buf =
                (d_abs.Current() == offdiag_abs->data()) ? offdiag_abs : offdiag_abs_alt;
            Buf<uint64_t> &stale_key_buf =
                (d_keys.Current() == offdiag_keys->data()) ? offdiag_keys_alt : offdiag_keys;
            Buf<double> &stale_abs_buf =
                (d_abs.Current() == offdiag_abs->data()) ? offdiag_abs_alt : offdiag_abs;
            stale_key_buf->destroy();
            stale_abs_buf->destroy();

            // ---------------------------------------------------------------------------
            // Reduce-by-key: sum |val| per undirected block pair to get L1 edge weight.
            // ---------------------------------------------------------------------------

            Buf<uint64_t> unique_keys = safe_alloc<uint64_t>(offdiag_nnz, rt, "MAS topology_adj reduce");
            Buf<double> edge_weights = safe_alloc<double>(offdiag_nnz, rt, "MAS topology_adj reduce");
            Buf<int> edge_count_buf = safe_alloc<int>(1, rt, "MAS topology_adj reduce");

            size_t reduce_tmp_size = 0;
            cub::DeviceReduce::ReduceByKey(nullptr,
                                           reduce_tmp_size,
                                           d_keys.Current(),
                                           unique_keys->data(),
                                           d_abs.Current(),
                                           edge_weights->data(),
                                           edge_count_buf->data(),
                                           ctd::plus<>(),
                                           offdiag_nnz,
                                           rt.stream.get());
            cub::DeviceReduce::ReduceByKey(make_cub_tmp(reduce_tmp_size),
                                           reduce_tmp_size,
                                           d_keys.Current(),
                                           unique_keys->data(),
                                           d_abs.Current(),
                                           edge_weights->data(),
                                           edge_count_buf->data(),
                                           ctd::plus<>(),
                                           offdiag_nnz,
                                           rt.stream.get());
            int edge_count = device2host(edge_count_buf->data(), rt);
            sorted_key_buf->destroy();
            sorted_abs_buf->destroy();
            edge_count_buf->destroy();

            // ---------------------------------------------------------------------------
            // Histogram + ExclusiveSum to get CSR row_ptr.
            // ---------------------------------------------------------------------------

            const int sym_edge_count = 2 * edge_count;
            Buf<int> hist = safe_alloc<int>(dim_blocks + 1, 0, rt, "MAS topology_adj histogram");
            count_symmetric_rows<<<div_round_up(edge_count, 128), 128, 0, rt.stream.get()>>>(
                ctd::span<const uint64_t>(unique_keys->data(), edge_count),
                *hist);
            Buf<int> d_row_ptr = safe_alloc<int>(dim_blocks + 1, rt, "MAS topology_adj histogram");
            size_t scan_tmp_size = 0;
            cub::DeviceScan::ExclusiveSum(nullptr,
                                          scan_tmp_size,
                                          hist->data(),
                                          d_row_ptr->data(),
                                          dim_blocks + 1,
                                          rt.stream.get());
            cub::DeviceScan::ExclusiveSum(make_cub_tmp(scan_tmp_size),
                                          scan_tmp_size,
                                          hist->data(),
                                          d_row_ptr->data(),
                                          dim_blocks + 1,
                                          rt.stream.get());
            hist->destroy();

            // ---------------------------------------------------------------------------
            // Quantize weights for MAS coarsening.
            // ---------------------------------------------------------------------------

            Buf<double> min_weight_buf = safe_alloc<double>(1, rt, "MAS topology_adj quantize");
            Buf<double> max_weight_buf = safe_alloc<double>(1, rt, "MAS topology_adj quantize");

            size_t min_tmp_size = 0;
            cub::DeviceReduce::Min(nullptr,
                                   min_tmp_size,
                                   edge_weights->data(),
                                   min_weight_buf->data(),
                                   edge_count,
                                   rt.stream.get());
            cub::DeviceReduce::Min(make_cub_tmp(min_tmp_size),
                                   min_tmp_size,
                                   edge_weights->data(),
                                   min_weight_buf->data(),
                                   edge_count,
                                   rt.stream.get());

            size_t max_tmp_size = 0;
            cub::DeviceReduce::Max(nullptr,
                                   max_tmp_size,
                                   edge_weights->data(),
                                   max_weight_buf->data(),
                                   edge_count,
                                   rt.stream.get());
            cub::DeviceReduce::Max(make_cub_tmp(max_tmp_size),
                                   max_tmp_size,
                                   edge_weights->data(),
                                   max_weight_buf->data(),
                                   edge_count,
                                   rt.stream.get());

            double min_weight = device2host(min_weight_buf->data(), rt);
            double max_weight = device2host(max_weight_buf->data(), rt);

            Buf<int64_t> quantized_weights =
                safe_alloc<int64_t>(edge_count, rt, "MAS topology_adj quantize");
            quantize_weights<<<div_round_up(edge_count, 128), 128, 0, rt.stream.get()>>>(
                ctd::span<const double>(edge_weights->data(), edge_count),
                min_weight,
                max_weight,
                *quantized_weights);
            edge_weights->destroy();

            // ---------------------------------------------------------------------------
            // Scatter symmetric CSR: write (u,v) and (v,u) per undirected edge.
            // ---------------------------------------------------------------------------

            Buf<int> write_ptr = safe_alloc<int>(dim_blocks, rt, "MAS topology_adj outputs");
            cudaMemcpyAsync(
                write_ptr->data(),
                d_row_ptr->data(),
                dim_blocks * sizeof(int),
                cudaMemcpyDeviceToDevice,
                rt.stream.get());

            Buf<int> d_cols = safe_alloc<int>(sym_edge_count, rt, "MAS topology_adj outputs");
            Buf<int64_t> d_weights = safe_alloc<int64_t>(sym_edge_count, rt, "MAS topology_adj outputs");
            scatter_symmetric_edges<<<div_round_up(edge_count, 128), 128, 0, rt.stream.get()>>>(
                ctd::span<const uint64_t>(unique_keys->data(), edge_count),
                *quantized_weights,
                *write_ptr,
                *d_cols,
                *d_weights);

            row_ptr.resize(dim_blocks + 1);
            cu::copy_bytes(rt.stream, *d_row_ptr, row_ptr);
            cols.resize(sym_edge_count);
            cu::copy_bytes(rt.stream, *d_cols, cols);
            weights.resize(sym_edge_count);
            cu::copy_bytes(rt.stream, *d_weights, weights);
            rt.stream.sync();

            return sym_edge_count;
        }
    } // namespace

    BSRAdjacency::BSRAdjacency(const StiffnessMatrix &A, int block_dim, CudaRuntime rt)
    {
        static_assert(std::is_same_v<StiffnessMatrix::StorageIndex, int>, "MAS only support int32 index type.");
        if (A.cols() != A.rows() || A.cols() == 0 || A.rows() == 0 || A.nonZeros() == 0)
        {
            throw std::runtime_error("[MAS] Factorization failed due to invalid A");
        }
        if (A.cols() > std::numeric_limits<int>::max() || A.rows() > std::numeric_limits<int>::max())
        {
            throw std::runtime_error("[MAS] A is too large. Row/Col number exceeding int32 max.");
        }
        if (A.nonZeros() > std::numeric_limits<int>::max())
        {
            throw std::runtime_error("[MAS] A is too large. Non-zero number exceeding int32 max.");
        }
        if (block_dim < 1 || block_dim > 3)
        {
            throw std::runtime_error("[MAS] MAS only supports block size 1, 2, or 3.");
        }

        const StiffnessMatrix *A_csc = &A;
        StiffnessMatrix compressed_A;
        if (!A.isCompressed())
        {
            compressed_A = A;
            compressed_A.makeCompressed();
            A_csc = &compressed_A;
        }

        build_adjacency_from_csc(*A_csc, block_dim, row_ptr, cols, weights, rt);
    }

} // namespace polysolve::linear::mas

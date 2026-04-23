#include <polysolve/linear/mas_utils/BSRMatrix.hpp>

#include <cub/cub.cuh>

#include <cuda/algorithm>
#include <cuda/buffer>
#include <cuda/std/span>

#include <Eigen/SparseCore>

#include <cassert>
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

        __device__ uint64_t pack_key(int block_row, int block_col)
        {
            return (static_cast<uint64_t>(static_cast<uint32_t>(block_row)) << 32) | static_cast<uint32_t>(block_col);
        }

        __device__ int key_to_row(uint64_t key)
        {
            return static_cast<int>(static_cast<uint32_t>(key >> 32));
        }

        __device__ int key_to_col(uint64_t key)
        {
            return static_cast<int>(static_cast<uint32_t>(key));
        }

        // Payload bit layout:
        // | is padding (1) | block local offset (31) | nnz index (32) |

        /// @brief Pack payload for radix sort.
        /// @param is_padding If true, this is the diagonal padding of trailing block.
        /// @param nnz_index Index of input CSC vals array.
        /// @param local_offset Block local offset. Assumes Row major block.
        __device__ uint64_t pack_payload(bool is_padding, int nnz_index, int local_offset)
        {
            uint64_t payload = 0;
            payload |= static_cast<uint32_t>(nnz_index);
            payload |= (static_cast<uint64_t>(static_cast<uint32_t>(local_offset)) << 32);
            if (is_padding)
            {
                payload |= (uint64_t{1} << 63);
            }
            return payload;
        }

        __device__ bool payload_to_is_padding(uint64_t payload)
        {
            return (payload >> 63) != 0;
        }

        __device__ int payload_to_local_offset(uint64_t payload)
        {
            return static_cast<int>(static_cast<uint32_t>(payload >> 32) & 0x7FFFFFFF);
        }

        __device__ int payload_to_nnz_index(uint64_t payload)
        {
            return static_cast<int>(static_cast<uint32_t>(payload));
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

        __global__ void build_keys_payloads(int nnz,
                                            int padded,
                                            int block_dim,
                                            int dim_blocks,
                                            const int *perm, // nullable
                                            ctd::span<const int> col_of_nnz,
                                            ctd::span<const int> row_idx,
                                            ctd::span<uint64_t> keys,
                                            ctd::span<uint64_t> payloads)
        {
            int tid = blockDim.x * blockIdx.x + threadIdx.x;
            int nnz_total = nnz + padded;
            if (tid >= nnz_total)
            {
                return;
            }

            // Real node from input A. compute permuted block ij then pack.
            if (tid < nnz)
            {
                int old_row = row_idx[tid];
                int old_col = col_of_nnz[tid];

                int br_old = old_row / block_dim;
                int bc_old = old_col / block_dim;
                int br = (perm == nullptr) ? br_old : perm[br_old];
                int bc = (perm == nullptr) ? bc_old : perm[bc_old];

                int offset = (old_row - br_old * block_dim) * block_dim + (old_col - bc_old * block_dim);
                keys[tid] = pack_key(br, bc);
                payloads[tid] = pack_payload(false, tid, offset);
            }
            // Padding for trailing block. Compute permuted block ij for tail block then pack.
            else
            {
                int t = tid - nnz;
                int tail_old = dim_blocks - 1;
                int tail_new = (perm == nullptr) ? tail_old : perm[tail_old];
                int local = (block_dim - padded) + t;
                int offset = local * block_dim + local;

                keys[tid] = pack_key(tail_new, tail_new);
                payloads[tid] = pack_payload(true, 0, offset);
            }
        }

        __global__ void extract_block_rows(ctd::span<const uint64_t> unique_keys,
                                           ctd::span<int> block_rows)
        {
            int tid = blockDim.x * blockIdx.x + threadIdx.x;
            if (tid >= unique_keys.size())
            {
                return;
            }

            block_rows[tid] = key_to_row(unique_keys[tid]);
        }

        __global__ void fill_bsr_cols_vals(int block_dim,
                                           ctd::span<const uint64_t> unique_keys,
                                           ctd::span<const int> payload_offsets,
                                           ctd::span<const uint64_t> payloads,
                                           ctd::span<const double> csc_vals,
                                           ctd::span<int> bsr_cols,
                                           ctd::span<double> bsr_vals)
        {
            int bid = blockDim.x * blockIdx.x + threadIdx.x;
            if (bid >= unique_keys.size())
            {
                return;
            }

            bsr_cols[bid] = key_to_col(unique_keys[bid]);

            int begin = payload_offsets[bid];
            int end = payload_offsets[bid + 1];
            int block_size = block_dim * block_dim;
            int base = bid * block_size;
            for (int i = begin; i < end; ++i)
            {
                uint64_t payload = payloads[i];
                int offset = payload_to_local_offset(payload);
                // We pad diagonal entry to 1.0 for trailing blocks.
                double val = payload_to_is_padding(payload) ? 1.0 : csc_vals[payload_to_nnz_index(payload)];
                bsr_vals[base + offset] = val;
            }
        }

        int build_device_bsr_from_csc(ctd::span<const int> d_col_ptr,
                                      ctd::span<const int> d_row_idx,
                                      ctd::span<const double> d_vals,
                                      int rows_num,
                                      int cols_num,
                                      int block_dim,
                                      const int *d_perm,
                                      Buf<int> &out_rows,
                                      Buf<int> &out_cols,
                                      Buf<double> &out_vals,
                                      CudaRuntime rt)
        {
            int nnz = d_row_idx.size();
            int dim_blocks = div_round_up(rows_num, block_dim);
            int padded = block_dim * dim_blocks - rows_num;
            int nnz_total = nnz + padded;

            // ---------------------------------------------------------------------------
            // Map each nnz index to input col.
            // ---------------------------------------------------------------------------

            auto col_of_nnz = cu::make_buffer<int>(rt.stream, rt.mr, nnz, cu::no_init);
            build_col_of_nnz<<<div_round_up(cols_num, 128), 128, 0, rt.stream.get()>>>(
                cols_num, d_col_ptr, col_of_nnz);

            // ---------------------------------------------------------------------------
            // Convert each non-zero to [key, payload] pair.
            // ---------------------------------------------------------------------------

            auto keys_in = cu::make_buffer<uint64_t>(rt.stream, rt.mr, nnz_total, cu::no_init);
            auto payloads_in = cu::make_buffer<uint64_t>(rt.stream, rt.mr, nnz_total, cu::no_init);
            build_keys_payloads<<<div_round_up(nnz_total, 128), 128, 0, rt.stream.get()>>>(
                nnz,
                padded,
                block_dim,
                dim_blocks,
                d_perm,
                col_of_nnz,
                d_row_idx,
                keys_in,
                payloads_in);

            // ---------------------------------------------------------------------------
            // Radix sort by key. Result should be in row major order.
            // ---------------------------------------------------------------------------

            auto keys_alt = cu::make_buffer<uint64_t>(rt.stream, rt.mr, nnz_total, cu::no_init);
            auto payloads_alt = cu::make_buffer<uint64_t>(rt.stream, rt.mr, nnz_total, cu::no_init);
            cub::DoubleBuffer<uint64_t> d_keys(keys_in.data(), keys_alt.data());
            cub::DoubleBuffer<uint64_t> d_payloads(payloads_in.data(), payloads_alt.data());
            Buf<char> cub_tmp;
            auto make_cub_tmp = [&cub_tmp, rt](size_t required_size) {
                if (!cub_tmp || cub_tmp->size() < required_size)
                {
                    cub_tmp = cu::make_buffer<char>(rt.stream, rt.mr, required_size, cu::no_init);
                }
                return cub_tmp->data();
            };

            size_t sort_tmp_size = 0;
            cub::DeviceRadixSort::SortPairs(nullptr,
                                            sort_tmp_size,
                                            d_keys,
                                            d_payloads,
                                            nnz_total,
                                            0,
                                            64,
                                            rt.stream.get());
            cub::DeviceRadixSort::SortPairs(make_cub_tmp(sort_tmp_size),
                                            sort_tmp_size,
                                            d_keys,
                                            d_payloads,
                                            nnz_total,
                                            0,
                                            64,
                                            rt.stream.get());

            // ---------------------------------------------------------------------------
            // Run-length encode.
            // This step computes non-zero block num + payload count for each block.
            // ---------------------------------------------------------------------------

            auto unique_keys = cu::make_buffer<uint64_t>(rt.stream, rt.mr, nnz_total, cu::no_init);
            auto counts = cu::make_buffer<int>(rt.stream, rt.mr, nnz_total, cu::no_init);
            auto num_runs = cu::make_buffer<int>(rt.stream, rt.mr, 1, cu::no_init);

            size_t rle_tmp_size = 0;
            cub::DeviceRunLengthEncode::Encode(nullptr,
                                               rle_tmp_size,
                                               d_keys.Current(),
                                               unique_keys.data(),
                                               counts.data(),
                                               num_runs.data(),
                                               nnz_total,
                                               rt.stream.get());
            cub::DeviceRunLengthEncode::Encode(make_cub_tmp(rle_tmp_size),
                                               rle_tmp_size,
                                               d_keys.Current(),
                                               unique_keys.data(),
                                               counts.data(),
                                               num_runs.data(),
                                               nnz_total,
                                               rt.stream.get());

            // ---------------------------------------------------------------------------
            // Histogram + ExclusiveSum
            // This step count non-zero blocks per rows to compute row_ptr.
            // ---------------------------------------------------------------------------

            int nnz_blocks = device2host(num_runs.data(), rt);
            auto block_rows = cu::make_buffer<int>(rt.stream, rt.mr, nnz_blocks, cu::no_init);
            // Extract row index from packed key.
            extract_block_rows<<<div_round_up(nnz_blocks, 128), 128, 0, rt.stream.get()>>>(
                ctd::span<const uint64_t>(unique_keys.data(), nnz_blocks),
                block_rows);
            // Histogram count nnz block per row.
            auto hist = cu::make_buffer<int>(rt.stream, rt.mr, dim_blocks + 1, 0);
            size_t hist_tmp_size = 0;
            cub::DeviceHistogram::HistogramEven(nullptr,
                                                hist_tmp_size,
                                                block_rows.data(),
                                                hist.data(),
                                                dim_blocks + 1,
                                                0,
                                                dim_blocks,
                                                nnz_blocks,
                                                rt.stream.get());
            cub::DeviceHistogram::HistogramEven(make_cub_tmp(hist_tmp_size),
                                                hist_tmp_size,
                                                block_rows.data(),
                                                hist.data(),
                                                dim_blocks + 1,
                                                0,
                                                dim_blocks,
                                                nnz_blocks,
                                                rt.stream.get());
            // Exclusive scan compute CSR row ptr.
            out_rows = cu::make_buffer<int>(rt.stream, rt.mr, dim_blocks + 1, cu::no_init);
            size_t scan_tmp_size = 0;
            cub::DeviceScan::ExclusiveSum(nullptr,
                                          scan_tmp_size,
                                          hist.data(),
                                          out_rows->data(),
                                          dim_blocks + 1,
                                          rt.stream.get());
            cub::DeviceScan::ExclusiveSum(make_cub_tmp(scan_tmp_size),
                                          scan_tmp_size,
                                          hist.data(),
                                          out_rows->data(),
                                          dim_blocks + 1,
                                          rt.stream.get());

            // ---------------------------------------------------------------------------
            // Fill cols and vals non-zero blocks.
            // ---------------------------------------------------------------------------

            // Compute Payload offsets for each block.
            auto payload_offsets = cu::make_buffer<int>(rt.stream, rt.mr, nnz_blocks + 1, cu::no_init);
            cudaMemsetAsync(counts.data() + nnz_blocks, 0, sizeof(int), rt.stream.get());
            size_t off_tmp_size = 0;
            cub::DeviceScan::ExclusiveSum(nullptr,
                                          off_tmp_size,
                                          counts.data(),
                                          payload_offsets.data(),
                                          nnz_blocks + 1,
                                          rt.stream.get());
            cub::DeviceScan::ExclusiveSum(make_cub_tmp(off_tmp_size),
                                          off_tmp_size,
                                          counts.data(),
                                          payload_offsets.data(),
                                          nnz_blocks + 1,
                                          rt.stream.get());

            // Fill cols and vals.
            out_cols = cu::make_buffer<int>(rt.stream, rt.mr, nnz_blocks, cu::no_init);
            out_vals = cu::make_buffer<double>(rt.stream, rt.mr, nnz_blocks * block_dim * block_dim, 0.0);
            fill_bsr_cols_vals<<<div_round_up(nnz_blocks, 128), 128, 0, rt.stream.get()>>>(
                block_dim,
                ctd::span<const uint64_t>(unique_keys.data(), nnz_blocks),
                payload_offsets,
                ctd::span<const uint64_t>(d_payloads.Current(), nnz_total),
                d_vals,
                *out_cols,
                *out_vals);

            return nnz_blocks;
        }
    } // namespace

    BSRMatrix::BSRMatrix(const StiffnessMatrix &A,
                         int block_dim,
                         ctd::span<const int> permutation,
                         CudaRuntime rt)
    {
        static_assert(std::is_same_v<StiffnessMatrix::StorageIndex, int>, "MAS only support int32 index type.");
        if (A.nonZeros() > std::numeric_limits<int>::max())
        {
            throw std::runtime_error("[CudaPcg] A is too large. Non-zero number exceeding int32 max.");
        }

        // if A is not compressed. Copy and compress.
        const StiffnessMatrix *A_csc = &A;
        StiffnessMatrix compressed_A;
        if (!A.isCompressed())
        {
            compressed_A = A;
            compressed_A.makeCompressed();
            A_csc = &compressed_A;
        }

        block_dim_ = block_dim;
        dim_ = div_round_up(A_csc->rows(), block_dim_);

        int rows_num = A_csc->rows();
        int cols_num = A_csc->cols();
        int nnz = A_csc->nonZeros();

        // Upload input CSC to device.
        auto d_col_ptr = cu::make_buffer<int>(rt.stream, rt.mr, cols_num + 1, cu::no_init);
        auto d_row_idx = cu::make_buffer<int>(rt.stream, rt.mr, nnz, cu::no_init);
        auto d_vals = cu::make_buffer<double>(rt.stream, rt.mr, nnz, cu::no_init);
        cudaMemcpyAsync(
            d_col_ptr.data(),
            A_csc->outerIndexPtr(),
            static_cast<size_t>(cols_num + 1) * sizeof(int),
            cudaMemcpyHostToDevice,
            rt.stream.get());
        cudaMemcpyAsync(
            d_row_idx.data(),
            A_csc->innerIndexPtr(),
            static_cast<size_t>(nnz) * sizeof(int),
            cudaMemcpyHostToDevice,
            rt.stream.get());
        cudaMemcpyAsync(
            d_vals.data(),
            A_csc->valuePtr(),
            static_cast<size_t>(nnz) * sizeof(double),
            cudaMemcpyHostToDevice,
            rt.stream.get());

        // Upload permutation if non empty.
        const int *d_perm_ptr = nullptr;
        Buf<int> d_perm;
        if (!permutation.empty())
        {
            d_perm = cu::make_buffer<int>(rt.stream, rt.mr, permutation.size(), cu::no_init);
            cu::copy_bytes(rt.stream, permutation, *d_perm);
            d_perm_ptr = d_perm->data();
        }

        non_zeros_ = build_device_bsr_from_csc(
            d_col_ptr,
            d_row_idx,
            d_vals,
            rows_num,
            cols_num,
            block_dim_,
            d_perm_ptr,
            rows_,
            cols_,
            vals_,
            rt);

        rt.stream.sync();
    }

} // namespace polysolve::linear::mas

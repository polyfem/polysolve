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

        __global__ void histogram(ctd::span<const int> samples,
                                  ctd::span<int> hist,
                                  int num_bins

        )
        {
            for (int i = blockDim.x * blockIdx.x + threadIdx.x;
                 i < samples.size();
                 i += blockDim.x * gridDim.x)
            {
                int sample = samples[i];
                assert(sample >= 0 && sample < num_bins);
                atomicAdd(&hist[sample], 1);
            }
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

        int build_bsr_from_csc(const StiffnessMatrix &A_csc,
                               int block_dim,
                               ctd::span<const int> permutation,
                               Buf<int> &out_rows,
                               Buf<int> &out_cols,
                               Buf<double> &out_vals,
                               CudaRuntime rt)
        {
            const int rows_num = A_csc.rows();
            const int cols_num = A_csc.cols();
            const int nnz = A_csc.nonZeros();
            int bsr_dim = div_round_up(rows_num, block_dim);
            int padded = block_dim * bsr_dim - rows_num;
            int nnz_total = nnz + padded;

            // Upload input CSC to device.
            Buf<int> d_col_ptr = safe_alloc<int>(cols_num + 1, rt, "MAS permuted_bsr input_csc");
            Buf<int> d_row_idx = safe_alloc<int>(nnz, rt, "MAS permuted_bsr input_csc");
            Buf<double> d_vals = safe_alloc<double>(nnz, rt, "MAS permuted_bsr input_csc");
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

            const int *d_perm_ptr = nullptr;
            Buf<int> d_perm;
            if (!permutation.empty())
            {
                d_perm = safe_alloc<int>(permutation.size(), rt, "MAS permuted_bsr permutation");
                cu::copy_bytes(rt.stream, permutation, *d_perm);
                d_perm_ptr = d_perm->data();
            }

            // ---------------------------------------------------------------------------
            // Map each nnz index to input col.
            // ---------------------------------------------------------------------------

            Buf<int> col_of_nnz = safe_alloc<int>(nnz, rt, "MAS permuted_bsr col_of_nnz");
            build_col_of_nnz<<<div_round_up(cols_num, 128), 128, 0, rt.stream.get()>>>(
                cols_num, *d_col_ptr, *col_of_nnz);
            d_col_ptr->destroy();

            // ---------------------------------------------------------------------------
            // Convert each non-zero to [key, payload] pair.
            // ---------------------------------------------------------------------------

            Buf<uint64_t> keys_in = safe_alloc<uint64_t>(nnz_total, rt, "MAS permuted_bsr staging");
            Buf<uint64_t> payloads_in =
                safe_alloc<uint64_t>(nnz_total, rt, "MAS permuted_bsr staging");
            build_keys_payloads<<<div_round_up(nnz_total, 128), 128, 0, rt.stream.get()>>>(
                nnz,
                padded,
                block_dim,
                bsr_dim,
                d_perm_ptr,
                *col_of_nnz,
                *d_row_idx,
                *keys_in,
                *payloads_in);
            col_of_nnz->destroy();
            d_row_idx->destroy();
            if (d_perm)
            {
                d_perm->destroy();
            }

            // ---------------------------------------------------------------------------
            // Radix sort by key. Result should be in row major order.
            // ---------------------------------------------------------------------------

            Buf<uint64_t> keys_alt = safe_alloc<uint64_t>(nnz_total, rt, "MAS permuted_bsr staging");
            Buf<uint64_t> payloads_alt =
                safe_alloc<uint64_t>(nnz_total, rt, "MAS permuted_bsr staging");
            cub::DoubleBuffer<uint64_t> d_keys(keys_in->data(), keys_alt->data());
            cub::DoubleBuffer<uint64_t> d_payloads(payloads_in->data(), payloads_alt->data());
            Buf<char> cub_tmp;
            auto make_cub_tmp = [&cub_tmp, rt](size_t required_size) {
                if (!cub_tmp || cub_tmp->size() < required_size)
                {
                    cub_tmp = safe_alloc<char>(required_size, rt, "MAS permuted_bsr cub_tmp");
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

            Buf<uint64_t> &sorted_keys_buf = (d_keys.Current() == keys_in->data()) ? keys_in : keys_alt;
            Buf<uint64_t> &sorted_payloads_buf =
                (d_payloads.Current() == payloads_in->data()) ? payloads_in : payloads_alt;
            Buf<uint64_t> &stale_keys_buf = (d_keys.Current() == keys_in->data()) ? keys_alt : keys_in;
            Buf<uint64_t> &stale_payloads_buf =
                (d_payloads.Current() == payloads_in->data()) ? payloads_alt : payloads_in;
            stale_keys_buf->destroy();
            stale_payloads_buf->destroy();

            // ---------------------------------------------------------------------------
            // Run-length encode.
            // This step computes non-zero block num + payload count for each block.
            // ---------------------------------------------------------------------------

            Buf<uint64_t> unique_keys = safe_alloc<uint64_t>(nnz_total, rt, "MAS permuted_bsr rle");

            // We later use this buffer to do exlcusive sum for CSR offsets.
            // Thus the size is nnz_total + 1.
            Buf<int> counts = safe_alloc<int>(nnz_total + 1, rt, "MAS permuted_bsr rle");
            Buf<int> num_runs = safe_alloc<int>(1, rt, "MAS permuted_bsr rle");

            size_t rle_tmp_size = 0;
            cub::DeviceRunLengthEncode::Encode(nullptr,
                                               rle_tmp_size,
                                               d_keys.Current(),
                                               unique_keys->data(),
                                               counts->data(),
                                               num_runs->data(),
                                               nnz_total,
                                               rt.stream.get());
            cub::DeviceRunLengthEncode::Encode(make_cub_tmp(rle_tmp_size),
                                               rle_tmp_size,
                                               d_keys.Current(),
                                               unique_keys->data(),
                                               counts->data(),
                                               num_runs->data(),
                                               nnz_total,
                                               rt.stream.get());
            sorted_keys_buf->destroy();

            // ---------------------------------------------------------------------------
            // Histogram + ExclusiveSum
            // This step count non-zero blocks per rows to compute row_ptr.
            // ---------------------------------------------------------------------------

            int nnz_blocks = device2host(num_runs->data(), rt);
            num_runs->destroy();
            Buf<int> block_rows = safe_alloc<int>(nnz_blocks, rt, "MAS permuted_bsr histogram");
            // Extract row index from packed key.
            extract_block_rows<<<div_round_up(nnz_blocks, 128), 128, 0, rt.stream.get()>>>(
                ctd::span<const uint64_t>(unique_keys->data(), nnz_blocks),
                *block_rows);
            // Histogram count nnz block per row.
            Buf<int> hist = safe_alloc<int>(bsr_dim + 1, 0, rt, "MAS permuted_bsr histogram");
            // As of 20260507, CCCL v3.0.0 histogram has a out-of-bound memory write bug.
            // If in the bug is fixed in the future you shall replace the homebrew histogram.
            histogram<<<div_round_up(nnz_blocks, 128), 128, 0, rt.stream.get()>>>(
                *block_rows,
                *hist,
                bsr_dim);
            // Exclusive scan compute CSR row ptr.
            out_rows = safe_alloc<int>(bsr_dim + 1, rt, "MAS permuted_bsr histogram");
            size_t scan_tmp_size = 0;
            cub::DeviceScan::ExclusiveSum(nullptr,
                                          scan_tmp_size,
                                          hist->data(),
                                          out_rows->data(),
                                          bsr_dim + 1,
                                          rt.stream.get());
            cub::DeviceScan::ExclusiveSum(make_cub_tmp(scan_tmp_size),
                                          scan_tmp_size,
                                          hist->data(),
                                          out_rows->data(),
                                          bsr_dim + 1,
                                          rt.stream.get());
            block_rows->destroy();
            hist->destroy();

            // ---------------------------------------------------------------------------
            // Fill cols and vals non-zero blocks.
            // ---------------------------------------------------------------------------

            // Compute Payload offsets for each block.
            Buf<int> payload_offsets = safe_alloc<int>(nnz_blocks + 1, rt, "MAS permuted_bsr outputs");
            cudaMemsetAsync(counts->data() + nnz_blocks, 0, sizeof(int), rt.stream.get());
            size_t off_tmp_size = 0;
            cub::DeviceScan::ExclusiveSum(nullptr,
                                          off_tmp_size,
                                          counts->data(),
                                          payload_offsets->data(),
                                          nnz_blocks + 1,
                                          rt.stream.get());
            cub::DeviceScan::ExclusiveSum(make_cub_tmp(off_tmp_size),
                                          off_tmp_size,
                                          counts->data(),
                                          payload_offsets->data(),
                                          nnz_blocks + 1,
                                          rt.stream.get());
            counts->destroy();
            cub_tmp->destroy();

            // Fill cols and vals.
            out_cols = safe_alloc<int>(nnz_blocks, rt, "MAS permuted_bsr outputs");
            out_vals =
                safe_alloc<double>(nnz_blocks * block_dim * block_dim, 0.0, rt, "MAS permuted_bsr outputs");
            fill_bsr_cols_vals<<<div_round_up(nnz_blocks, 128), 128, 0, rt.stream.get()>>>(
                block_dim,
                ctd::span<const uint64_t>(unique_keys->data(), nnz_blocks),
                *payload_offsets,
                ctd::span<const uint64_t>(d_payloads.Current(), nnz_total),
                *d_vals,
                *out_cols,
                *out_vals);
            unique_keys->destroy();
            payload_offsets->destroy();
            sorted_payloads_buf->destroy();
            d_vals->destroy();

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
            throw std::runtime_error("[MAS] A is too large. Non-zero number exceeding int32 max.");
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
        padded_scalar_num_ = block_dim_ * dim_ - A_csc->rows();
        padded_block_ = -1;
        if (padded_scalar_num_ > 0)
        {
            int tail = dim_ - 1;
            padded_block_ = permutation.empty() ? tail : permutation[tail];
        }

        non_zeros_ = build_bsr_from_csc(
            *A_csc,
            block_dim_,
            permutation,
            rows_,
            cols_,
            vals_,
            rt);

        rt.stream.sync();
    }

} // namespace polysolve::linear::mas

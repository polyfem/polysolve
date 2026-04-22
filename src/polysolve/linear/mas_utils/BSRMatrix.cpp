#include <polysolve/linear/mas_utils/BSRMatrix.hpp>

#include <cuda/buffer>
#include <cuda/std/span>
#include <cuda/algorithm>
#include <Eigen/SparseCore>
#include <algorithm>
#include <limits>
#include <unordered_map>
#include <vector>
#include <cassert>

#include <polysolve/linear/mas_utils/CudaUtils.cuh>

namespace polysolve::linear::mas
{
    namespace
    {
        template <int BLOCK_DIM>
        using Block = Eigen::Matrix<double, BLOCK_DIM, BLOCK_DIM, Eigen::RowMajor>;

        template <int BLOCK_DIM>
        struct ColBlock
        {
            int col = -1;
            Block<BLOCK_DIM> block = Block<BLOCK_DIM>::Zero();
        };

        /// @brief Block CSR matrix builder.
        /// @param[in] A Input matrix.
        /// @param[in] permutation Index permutation. Empty implies identity.
        /// @param[in] dim A dimension.
        /// @param[out] non_zeros BSR non-zeros.
        /// @param[out] h_rows Host BSR data.
        /// @param[out] h_cols Host BSR data.
        /// @param[out] h_vals Host BSR data
        template <int BLOCK_DIM>
        void build_host_bsr(const StiffnessMatrix &A,
                            ctd::span<const int> permutation,
                            int dim,
                            int &non_zeros,
                            std::vector<int> &h_rows,
                            std::vector<int> &h_cols,
                            std::vector<double> &h_vals)
        {
            using Iter = Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator;

            // Stiffness matrix is col major. Convert it to row major CSR layout.
            Eigen::SparseMatrix<double, Eigen::RowMajor> Arow = A;
            Arow.makeCompressed();

            std::vector<int> inverse_permutation;
            if (!permutation.empty())
            {
                inverse_permutation.resize(dim);
                for (int old_bi = 0; old_bi < dim; ++old_bi)
                {
                    inverse_permutation[permutation[old_bi]] = old_bi;
                }
            }

            int block_size = BLOCK_DIM * BLOCK_DIM;
            int estimated_blocks = (A.nonZeros() + block_size - 1) / block_size + 1;
            int padded = BLOCK_DIM * dim - A.rows();
            int old_tail_block = dim - 1;

            h_rows.resize(dim + 1, 0);
            h_cols.clear();
            h_cols.reserve(estimated_blocks);
            h_vals.clear();
            h_vals.reserve(estimated_blocks * block_size);

            // Build one block row at a time.
            for (int bi = 0; bi < dim; ++bi)
            {
                // Count block row scalar non-zeros.
                int old_bi = permutation.empty() ? bi : inverse_permutation[bi];
                int scalar_nnz = 0;
                for (int li = 0; li < BLOCK_DIM; ++li)
                {
                    int scalar_row = old_bi * BLOCK_DIM + li;
                    if (scalar_row >= Arow.rows())
                    {
                        break;
                    }
                    scalar_nnz += Arow.outerIndexPtr()[scalar_row + 1] - Arow.outerIndexPtr()[scalar_row];
                }

                // col2entry maps block col id -> block storage id.
                // row_entries is the block storage.
                std::unordered_map<int, int> col2entry;
                col2entry.reserve((scalar_nnz + BLOCK_DIM - 1) / BLOCK_DIM + 1);
                std::vector<ColBlock<BLOCK_DIM>> block_storage;
                block_storage.reserve((scalar_nnz + BLOCK_DIM - 1) / BLOCK_DIM + 1);

                // Gather col blocks from scalar.
                for (int li = 0; li < BLOCK_DIM; ++li)
                {
                    int scalar_row = old_bi * BLOCK_DIM + li;
                    if (scalar_row >= Arow.rows())
                    {
                        break;
                    }

                    for (Iter it(Arow, scalar_row); it; ++it)
                    {
                        int old_bj = it.col() / BLOCK_DIM;
                        int bj = permutation.empty() ? old_bj : permutation[old_bj];
                        int lj = it.col() - BLOCK_DIM * old_bj;

                        auto map_it = col2entry.find(bj);
                        // No block yet. Create new one.
                        if (map_it == col2entry.end())
                        {
                            col2entry.emplace(bj, block_storage.size());
                            block_storage.emplace_back();
                            block_storage.back().col = bj;
                            block_storage.back().block(li, lj) = it.value();
                        }
                        // Add scalar to existing block storage.
                        else
                        {
                            block_storage[map_it->second].block(li, lj) = it.value();
                        }
                    }
                }

                // Pad trailing diagonal block.
                if (padded > 0 && old_bi == old_tail_block)
                {
                    auto map_it = col2entry.find(bi);
                    if (map_it == col2entry.end())
                    {
                        col2entry.emplace(bi, block_storage.size());
                        block_storage.emplace_back();
                        block_storage.back().col = bi;
                        map_it = col2entry.find(bi);
                    }

                    for (int i = BLOCK_DIM - padded; i < BLOCK_DIM; ++i)
                    {
                        block_storage[map_it->second].block(i, i) = 1.0;
                    }
                }

                std::sort(block_storage.begin(),
                          block_storage.end(),
                          [](ColBlock<BLOCK_DIM> const &a, ColBlock<BLOCK_DIM> const &b) {
                              return a.col < b.col;
                          });

                // Fill BSR layout and BSR topology.
                h_rows[bi + 1] = block_storage.size();
                for (auto &entry : block_storage)
                {
                    h_cols.push_back(entry.col);
                    h_vals.insert(h_vals.end(), entry.block.data(), entry.block.data() + BLOCK_DIM * BLOCK_DIM);
                }
            }

            for (int i = 0; i < dim; ++i)
            {
                h_rows[i + 1] += h_rows[i];
            }

            non_zeros = h_cols.size();
        }
    } // namespace

    BSRMatrix::BSRMatrix(const StiffnessMatrix &A,
                         int block_dim,
                         ctd::span<const int> permutation,
                         CudaRuntime rt)
    {
        if (A.cols() != A.rows() || A.cols() == 0 || A.rows() == 0 || A.nonZeros() == 0)
        {
            throw std::runtime_error("[CudaPcg] Factorization failed due to invalid A");
        }
        if (A.cols() > std::numeric_limits<int>::max() || A.rows() > std::numeric_limits<int>::max())
        {
            throw std::runtime_error("[CudaPcg] A is too large. Row/Col number exceeding int32 max.");
        }

        block_dim_ = block_dim;
        // Pad dimension if neccessary.
        dim_ = (A.rows() + block_dim_ - 1) / block_dim_;
        h_rows_.assign(dim_ + 1, 0);
        h_cols_.clear();
        std::vector<double> h_vals;
        if (block_dim_ == 1)
        {
            build_host_bsr<1>(A,
                              permutation,
                              dim_,
                              non_zeros_,
                              h_rows_,
                              h_cols_,
                              h_vals);
        }
        else if (block_dim_ == 2)
        {
            build_host_bsr<2>(A,
                              permutation,
                              dim_,
                              non_zeros_,
                              h_rows_,
                              h_cols_,
                              h_vals);
        }
        else if (block_dim_ == 3)
        {
            build_host_bsr<3>(A,
                              permutation,
                              dim_,
                              non_zeros_,
                              h_rows_,
                              h_cols_,
                              h_vals);
        }

        // Copy host BSR data to device.
        rows_ = cu::make_buffer<int>(rt.stream, rt.mr, h_rows_.size(), cu::no_init);
        cu::copy_bytes(rt.stream, h_rows_, *rows_);
        cols_ = cu::make_buffer<int>(rt.stream, rt.mr, h_cols_.size(), cu::no_init);
        cu::copy_bytes(rt.stream, h_cols_, *cols_);
        vals_ = cu::make_buffer<double>(rt.stream, rt.mr, h_vals.size(), cu::no_init);
        cu::copy_bytes(rt.stream, h_vals, *vals_);

        rt.stream.sync();
    }

} // namespace polysolve::linear::mas

#include <polysolve/linear/mas_utils/BSRMatrix.hpp>

#include <cuda/buffer>
#include <cuda/std/span>
#include <cuda/algorithm>
#include <Eigen/SparseCore>
#include <algorithm>
#include <cstdint>
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

        std::vector<int> quantize_weight(ctd::span<const double> weight)
        {
            if (weight.empty())
            {
                return {};
            }

            auto [min_it, max_it] = std::minmax_element(weight.begin(), weight.end());
            double min_w = *min_it;
            double max_w = *max_it;
            double w_range = max_w - min_w;
            if (w_range <= 0.0)
            {
                return std::vector<int>(weight.size(), 1);
            }

            constexpr int MAX_QUANT = 1000000;
            std::vector<int> quant(weight.size());
            for (int i = 0; i < weight.size(); ++i)
            {
                double scaled = (weight[i] - min_w) / w_range * MAX_QUANT;
                quant[i] = std::clamp(static_cast<int>(scaled), 1, MAX_QUANT);
            }
            return quant;
        }

        /// Average upper and lower part of topology connection weight.
        void avg_sym_weight(ctd::span<const int> row_ptr,
                            ctd::span<const int> cols,
                            ctd::span<double> weights)
        {
            struct Sum
            {
                double val = 0.0;
                int count = 0;
            };

            std::unordered_map<uint64_t, Sum> pair_weight;

            auto ij_to_key = [](int i, int j) -> uint64_t {
                int u = std::min(i, j);
                int v = std::max(i, j);
                return (static_cast<uint64_t>(u) << 32) | static_cast<uint32_t>(v);
            };

            for (int i = 0; i < row_ptr.size() - 1; ++i)
            {
                for (int n = row_ptr[i]; n < row_ptr[i + 1]; ++n)
                {
                    int j = cols[n];
                    Sum &stat = pair_weight[ij_to_key(i, j)];
                    stat.val += weights[n];
                    stat.count += 1;
                }
            }

            for (int i = 0; i < row_ptr.size() - 1; ++i)
            {
                for (int n = row_ptr[i]; n < row_ptr[i + 1]; ++n)
                {
                    int j = cols[n];
                    auto it = pair_weight.find(ij_to_key(i, j));
                    weights[n] = it->second.val / it->second.count;
                }
            }
        }

        /// @brief Block CSR matrix builder.
        /// @param[in] A Input matrix.
        /// @param[in] permutation Index permutation. Empty implies identity.
        /// @param[in] dim A dimension.
        /// @param[out] non_zeros BSR non-zeros.
        /// @param[out] topology_non_zeros BSR topology non-zeros.
        /// @param[out] h_rows Host BSR data.
        /// @param[out] h_cols Host BSR data.
        /// @param[out] h_vals Host BSR data
        /// @param[out] h_topo_rows Host BSR topology data.
        /// @param[out] h_topo_cols Host BSR topology data.
        /// @param[out] h_topo_weights Host BSR topology data.
        template <int BLOCK_DIM>
        void build_host_bsr(const StiffnessMatrix &A,
                            ctd::span<const int> permutation,
                            int dim,
                            int &non_zeros,
                            int &topology_non_zeros,
                            std::vector<int> &h_rows,
                            std::vector<int> &h_cols,
                            std::vector<double> &h_vals,
                            std::vector<int> &h_topo_rows,
                            std::vector<int> &h_topo_cols,
                            std::vector<int> &h_topo_weights)
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
            h_topo_rows.resize(dim + 1, 0);
            h_topo_cols.clear();
            h_topo_cols.reserve(estimated_blocks);
            h_topo_weights.clear();

            std::vector<double> h_topo_norms;
            h_topo_norms.reserve(estimated_blocks);

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
                // BSR topology is self-excluding BSR with block norm as conenction weight.
                h_rows[bi + 1] = block_storage.size();
                for (auto &entry : block_storage)
                {
                    h_cols.push_back(entry.col);
                    h_vals.insert(h_vals.end(), entry.block.data(), entry.block.data() + BLOCK_DIM * BLOCK_DIM);

                    if (bi != entry.col)
                    {
                        h_topo_rows[bi + 1] += 1;
                        h_topo_cols.push_back(entry.col);
                        h_topo_norms.push_back(entry.block.norm());
                    }
                }
            }

            for (int i = 0; i < dim; ++i)
            {
                h_rows[i + 1] += h_rows[i];
                h_topo_rows[i + 1] += h_topo_rows[i];
            }

            // Metis expect >0 int connection weight.
            // Symmetrize the two directed weights of each undirected edge first,
            // then quantize the averaged value for METIS.
            //
            // In theory input matrix should be symmetric but in practice small floating point
            // error accumulate through assembly will result in different quantized weight.
            // So to prevent metis error we must compute average here.
            avg_sym_weight(h_topo_rows, h_topo_cols, h_topo_norms);
            h_topo_weights = quantize_weight(h_topo_norms);

            non_zeros = h_cols.size();
            topology_non_zeros = h_topo_cols.size();
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
        std::vector<int> h_rows(dim_ + 1, 0);
        std::vector<int> h_cols;
        std::vector<double> h_vals;
        if (block_dim_ == 1)
        {
            build_host_bsr<1>(A,
                              permutation,
                              dim_,
                              non_zeros_,
                              topology_non_zeros_,
                              h_rows,
                              h_cols,
                              h_vals,
                              h_topo_rows_,
                              h_topo_cols_,
                              h_topo_weights_);
        }
        else if (block_dim_ == 2)
        {
            build_host_bsr<2>(A,
                              permutation,
                              dim_,
                              non_zeros_,
                              topology_non_zeros_,
                              h_rows,
                              h_cols,
                              h_vals,
                              h_topo_rows_,
                              h_topo_cols_,
                              h_topo_weights_);
        }
        else if (block_dim_ == 3)
        {
            build_host_bsr<3>(A,
                              permutation,
                              dim_,
                              non_zeros_,
                              topology_non_zeros_,
                              h_rows,
                              h_cols,
                              h_vals,
                              h_topo_rows_,
                              h_topo_cols_,
                              h_topo_weights_);
        }

        // Copy host BSR data to device.
        rows_ = cu::make_buffer<int>(rt.stream, rt.mr, h_rows.size(), cu::no_init);
        cu::copy_bytes(rt.stream, h_rows, *rows_);
        cols_ = cu::make_buffer<int>(rt.stream, rt.mr, h_cols.size(), cu::no_init);
        cu::copy_bytes(rt.stream, h_cols, *cols_);
        vals_ = cu::make_buffer<double>(rt.stream, rt.mr, h_vals.size(), cu::no_init);
        cu::copy_bytes(rt.stream, h_vals, *vals_);

        rt.stream.sync();
    }

} // namespace polysolve::linear::mas

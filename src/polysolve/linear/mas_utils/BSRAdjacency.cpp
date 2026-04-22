#include <polysolve/linear/mas_utils/BSRAdjacency.hpp>

#include <cuda/std/span>
#include <Eigen/SparseCore>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace polysolve::linear::mas
{
    namespace
    {
        std::vector<int> quantize_weight(cuda::std::span<const double> weight)
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
            for (int i = 0; i < static_cast<int>(weight.size()); ++i)
            {
                double scaled = (weight[i] - min_w) / w_range * MAX_QUANT;
                quant[i] = std::clamp(static_cast<int>(scaled), 1, MAX_QUANT);
            }
            return quant;
        }

        /// Average upper and lower part of topology connection weight.
        void avg_sym_weight(cuda::std::span<const int> row_ptr,
                            cuda::std::span<const int> cols,
                            cuda::std::span<double> weights)
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

            for (int i = 0; i < static_cast<int>(row_ptr.size()) - 1; ++i)
            {
                for (int n = row_ptr[i]; n < row_ptr[i + 1]; ++n)
                {
                    int j = cols[n];
                    Sum &stat = pair_weight[ij_to_key(i, j)];
                    stat.val += weights[n];
                    stat.count += 1;
                }
            }

            for (int i = 0; i < static_cast<int>(row_ptr.size()) - 1; ++i)
            {
                for (int n = row_ptr[i]; n < row_ptr[i + 1]; ++n)
                {
                    int j = cols[n];
                    auto it = pair_weight.find(ij_to_key(i, j));
                    if (it == pair_weight.end() || it->second.count == 0)
                    {
                        continue;
                    }
                    weights[n] = it->second.val / it->second.count;
                }
            }
        }

        template <int BLOCK_DIM>
        void build_host_adjacency(const StiffnessMatrix &A,
                                  int dim,
                                  std::vector<int> &row_ptr,
                                  std::vector<int> &cols,
                                  std::vector<int> &weights)
        {
            using Iter = Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator;

            Eigen::SparseMatrix<double, Eigen::RowMajor> Arow = A;
            Arow.makeCompressed();

            row_ptr.assign(dim + 1, 0);
            cols.clear();
            weights.clear();

            std::vector<double> norms;

            // One block row at a time.
            for (int bi = 0; bi < dim; ++bi)
            {
                // Accumulate block Frobenius norm squared per block col.
                std::unordered_map<int, double> col2norm2;

                for (int li = 0; li < BLOCK_DIM; ++li)
                {
                    int scalar_row = bi * BLOCK_DIM + li;
                    if (scalar_row >= Arow.rows())
                    {
                        break;
                    }

                    for (Iter it(Arow, scalar_row); it; ++it)
                    {
                        int bj = it.col() / BLOCK_DIM;
                        if (bj == bi)
                        {
                            continue; // self-excluding
                        }

                        double v = it.value();
                        col2norm2[bj] += v * v;
                    }
                }

                std::vector<int> row_cols;
                row_cols.reserve(col2norm2.size());
                for (auto &kv : col2norm2)
                {
                    row_cols.push_back(kv.first);
                }
                std::sort(row_cols.begin(), row_cols.end());

                row_ptr[bi + 1] = static_cast<int>(row_cols.size());
                for (int bj : row_cols)
                {
                    cols.push_back(bj);
                    norms.push_back(std::sqrt(col2norm2[bj]));
                }
            }

            for (int i = 0; i < dim; ++i)
            {
                row_ptr[i + 1] += row_ptr[i];
            }

            // Stabilize weights for METIS: symmetrize, then quantize.
            avg_sym_weight(row_ptr, cols, norms);
            weights = quantize_weight(norms);
        }
    } // namespace

    BSRAdjacency::BSRAdjacency(const StiffnessMatrix &A, int block_dim)
    {
        if (A.cols() != A.rows() || A.cols() == 0 || A.rows() == 0 || A.nonZeros() == 0)
        {
            throw std::runtime_error("[CudaPcg] Factorization failed due to invalid A");
        }
        if (A.cols() > std::numeric_limits<int>::max() || A.rows() > std::numeric_limits<int>::max())
        {
            throw std::runtime_error("[CudaPcg] A is too large. Row/Col number exceeding int32 max.");
        }
        if (block_dim < 1 || block_dim > 3)
        {
            throw std::runtime_error("[CudaPcg] MAS only supports block size 1, 2, or 3.");
        }

        int dim = (A.rows() + block_dim - 1) / block_dim;
        if (block_dim == 1)
        {
            build_host_adjacency<1>(A, dim, row_ptr, cols, weights);
        }
        else if (block_dim == 2)
        {
            build_host_adjacency<2>(A, dim, row_ptr, cols, weights);
        }
        else
        {
            build_host_adjacency<3>(A, dim, row_ptr, cols, weights);
        }
    }

} // namespace polysolve::linear::mas

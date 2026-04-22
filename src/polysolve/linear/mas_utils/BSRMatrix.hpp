#pragma once

#include <polysolve/linear/mas_utils/CudaUtils.cuh>
#include <polysolve/Types.hpp>

#include <cuda/std/span>

#include <vector>

namespace polysolve::linear::mas
{
    struct BSRView
    {
        int dim;
        int block_dim;
        int non_zeros;
        ctd::span<const int> rows;
        ctd::span<const int> cols;
        ctd::span<const double> vals;
    };

    // Host CSR-style topology. May include self (diagonal).
    struct TopologyView
    {
        ctd::span<const int> row_ptr;
        ctd::span<const int> cols;
    };

    /// @brief Block CSR matrix.
    class BSRMatrix
    {
    public:
        BSRMatrix() = default;

        /// @brief Build from host CSR matrix. Empty permutation implies identity.
        /// Sync stream before return.
        BSRMatrix(const StiffnessMatrix &A,
                  int block_dim,
                  ctd::span<const int> permutation,
                  CudaRuntime rt);

        BSRView view() const
        {
            return BSRView{dim_, block_dim_, non_zeros_, *rows_, *cols_, *vals_};
        }

        TopologyView host_topology_view() const
        {
            return TopologyView{
                h_rows_,
                h_cols_,
            };
        }

    private:
        int dim_ = 0;
        int block_dim_ = 0;
        int non_zeros_ = 0;
        Buf<int> rows_;
        Buf<int> cols_;
        Buf<double> vals_;

        // Host copy of matrix sparsity pattern. Used by MAS topology build.
        std::vector<int> h_rows_;
        std::vector<int> h_cols_;
    };

} // namespace polysolve::linear::mas

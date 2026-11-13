#pragma once

#include <polysolve/Types.hpp>

#include <Eigen/Core>

#include <functional>

/// Thread-parallel kernels of NonsmoothNewton. The work is split over the threads allowed by the caller's
/// tbb::global_control; in builds without TBB (POLYSOLVE_WITH_TBB off) everything runs in the caller. The sparse kernels
/// give the results of the Eigen expressions they replace bit for bit, with any number of threads.
namespace polysolve::nonlinear::parallel
{
    /// @brief Threads allowed by the caller's tbb::global_control (1 in builds without TBB)
    int max_threads();

    /// @brief Number of chunks for n items: one per allowed thread, with at least grain items each, and at least one
    int chunks(Eigen::Index n, Eigen::Index grain);

    /// @brief body(i, begin, end) for the c contiguous chunks [n i / c, n (i + 1) / c) of [0, n), in parallel; a single
    /// chunk runs in the caller
    void for_chunks(Eigen::Index n, int c, const std::function<void(int, Eigen::Index, Eigen::Index)> &body);

    /// @brief y += H (alpha q), with the rows split over the threads. Every entry adds its terms in column order with the
    /// operations of Eigen's sparse * dense product, so y is Eigen's y += H q (alpha = 1) or y -= H q (alpha = -1) bit
    /// for bit. The row indices of each column must be sorted (as Eigen keeps them); y must not alias q.
    void spmv_add(const StiffnessMatrix &H, const Eigen::VectorXd &q, double alpha, Eigen::Ref<Eigen::VectorXd> y);

    /// @brief y = H q as Eigen evaluates the product: zero, then spmv_add with alpha = 1
    void spmv(const StiffnessMatrix &H, const Eigen::VectorXd &q, Eigen::Ref<Eigen::VectorXd> y);

    /// @brief C = A + B as Eigen's sparse sum followed by makeCompressed(): the union of the patterns, with a + b where
    /// both have an entry and a + 0 or 0 + b where one of them has it; the columns are split over the threads.
    /// Operands that are not compressed go to Eigen's sum. C must not be A or B.
    void spadd(const StiffnessMatrix &A, const StiffnessMatrix &B, StiffnessMatrix &C);
} // namespace polysolve::nonlinear::parallel

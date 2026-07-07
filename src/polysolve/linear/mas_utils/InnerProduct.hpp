#pragma once

#include <cuda/buffer>
#include <cuda/std/span>
#include <polysolve/linear/mas_utils/CudaUtils.cuh>

namespace polysolve::linear::mas
{
    /// @brief Compute inner product a dot b. Does not sync implicitly.
    void inner_product(ctd::span<const double> a, ctd::span<const double> b,
                       ctd::span<double> out, CudaRuntime rt);
} // namespace polysolve::linear::mas

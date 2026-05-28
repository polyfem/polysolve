#include <polysolve/linear/mas_utils/InnerProduct.hpp>

#include <cuda/buffer>
#include <cuda/std/span>
#include <cuda/atomic>
#include <cuda/algorithm>
#include <polysolve/linear/mas_utils/CudaUtils.cuh>
#include <cub/cub.cuh>
#include <cassert>

namespace polysolve::linear::mas
{

    namespace
    {
        __global__ void inner_product_kernel(
            ctd::span<const double> a, ctd::span<const double> b, double *out)
        {
            int tid = blockDim.x * blockIdx.x + threadIdx.x;

            double c = (tid < a.size()) ? a[tid] * b[tid] : 0.0;

            using BlockReduce = cub::BlockReduce<double, 128>;
            __shared__ typename BlockReduce::TempStorage tmp;

            double reduced = BlockReduce(tmp).Sum(c);

            if (threadIdx.x == 0)
            {
                cu::atomic_ref<double> a_out{*out};
                a_out.fetch_add(reduced, ctd::memory_order_relaxed);
            }
        }
    } // namespace

    void inner_product(ctd::span<const double> a, ctd::span<const double> b,
                       ctd::span<double> out, CudaRuntime rt)
    {
        assert(a.size() == b.size());
        assert(out.size() == 1);

        cu::fill_bytes(rt.stream, out, 0);
        int grid_num = div_round_up(a.size(), 128);
        inner_product_kernel<<<grid_num, 128, 0, rt.stream.get()>>>(a, b, out.data());
    }
} // namespace polysolve::linear::mas

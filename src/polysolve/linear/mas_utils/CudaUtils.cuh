#pragma once

#include <cuda/stream>
#include <cuda/memory_pool>
#include <cuda/buffer>
#include <cuda/std/__exception/cuda_error.h>
#include <cuda/std/optional>

#include <new>
#include <stdexcept>
#include <string>

#define __both__ __host__ __device__

// Convenient namespace alias for libcudacxx.
namespace cu = ::cuda;
namespace ctd = ::cuda::std;

namespace polysolve::linear::mas
{
    /// @brief ceil(num/denom)
    constexpr int div_round_up(int num, int denom)
    {
        return (num + denom - 1) / denom;
    }

    struct CudaRuntime
    {
        cu::stream_ref stream;
        cu::device_memory_pool_ref mr;
    };

    class CudaOutOfMemoryError : public std::runtime_error
    {
    public:
        using std::runtime_error::runtime_error;
    };

    // Actually, there's nothing safe about this wrapper.
    // It just throw meaningful error indicating exactly where allocation fails.
    template <typename T>
    cu::device_buffer<T> safe_alloc(
        size_t n,
        CudaRuntime rt,
        const char *ctx = nullptr)
    {
        const std::string context = ctx ? ctx : "cuda allocation";
        const size_t requested_bytes = n * sizeof(T);
        try
        {
            return cu::make_buffer<T>(rt.stream, rt.mr, n, cu::no_init);
        }
        catch (const CudaOutOfMemoryError &)
        {
            throw;
        }
        catch (const std::bad_alloc &)
        {
        }
        catch (const cuda::cuda_error &err)
        {
            if (err.status() != cudaErrorMemoryAllocation)
            {
                throw;
            }
        }

        size_t free_bytes = 0;
        size_t total_bytes = 0;
        cudaMemGetInfo(&free_bytes, &total_bytes);
        constexpr size_t mb = 1024ull * 1024ull;
        throw CudaOutOfMemoryError(
            std::string("[MAS] Failed to allocate ")
            + std::to_string((requested_bytes + mb - 1) / mb)
            + " MB for "
            + context
            + "; only "
            + std::to_string(free_bytes / mb)
            + " MB free.");
    }

    // Actually, there's nothing safe about this wrapper.
    // It just throw meaningful error indicating exactly where allocation fails.
    template <typename T>
    cu::device_buffer<T> safe_alloc(
        size_t n,
        T init_val,
        CudaRuntime rt,
        const char *ctx = nullptr)
    {
        const std::string context = ctx ? ctx : "cuda allocation";
        const size_t requested_bytes = n * sizeof(T);
        try
        {
            return cu::make_buffer<T>(rt.stream, rt.mr, n, init_val);
        }
        catch (const CudaOutOfMemoryError &)
        {
            throw;
        }
        catch (const std::bad_alloc &)
        {
        }
        catch (const cuda::cuda_error &err)
        {
            if (err.status() != cudaErrorMemoryAllocation)
            {
                throw;
            }
        }

        size_t free_bytes = 0;
        size_t total_bytes = 0;
        cudaMemGetInfo(&free_bytes, &total_bytes);
        constexpr size_t mb = 1024ull * 1024ull;
        throw CudaOutOfMemoryError(
            std::string("[MAS] Failed to allocate ")
            + std::to_string((requested_bytes + mb - 1) / mb)
            + " MB for "
            + context
            + "; only "
            + std::to_string(free_bytes / mb)
            + " MB free.");
    }

    /// @brief Transfer device scalar src to host.
    template <typename T>
    T device2host(const T *src, CudaRuntime rt)
    {
        T result;
        cudaMemcpyAsync(&result, src, sizeof(T), cudaMemcpyDeviceToHost,
                        rt.stream.get());
        rt.stream.sync();
        return result;
    }

    /// @brief Transfer host scalar val to device dst. Does not sync stream.
    template <typename T>
    void host2device(T *dst, T val, CudaRuntime rt)
    {
        cudaMemcpyAsync(dst, &val, sizeof(T), cudaMemcpyHostToDevice,
                        rt.stream.get());
    }

    /// @brief Nullable device buffer.
    /// It's very annoying device_buffer does not have default ctor for empty buffer.
    template <typename T>
    using Buf = ctd::optional<cu::device_buffer<T>>;

} // namespace polysolve::linear::mas

#pragma once

#include <polysolve/linear/mas_utils/BSRMatrix.hpp>

#include <cusparse.h>

// --------------------------------------------------------------
// RAII wrapper for varous cuSparse handle
// --------------------------------------------------------------

namespace polysolve::linear::mas
{
    class CuSparseHandle
    {
    public:
        cusparseHandle_t raw = nullptr;

        CuSparseHandle()
        {
            cusparseCreate(&raw);
        }

        ~CuSparseHandle()
        {
            if (raw != nullptr)
            {
                cusparseDestroy(raw);
            }
        }

        CuSparseHandle(const CuSparseHandle &) = delete;
        CuSparseHandle(CuSparseHandle &&other) { swap(other); }
        CuSparseHandle &operator=(const CuSparseHandle &) = delete;
        CuSparseHandle &operator=(CuSparseHandle &&other)
        {
            swap(other);
            return *this;
        }

    private:
        void swap(CuSparseHandle &other)
        {
            std::swap(raw, other.raw);
        }
    };

    class CuSparseConstVec
    {
    public:
        cusparseConstDnVecDescr_t raw = nullptr;

        CuSparseConstVec(ctd::span<const double> vec)
        {
            cusparseCreateConstDnVec(&raw, vec.size(), vec.data(), CUDA_R_64F);
        }

        ~CuSparseConstVec()
        {
            if (raw != nullptr)
            {
                cusparseDestroyDnVec(raw);
            }
        }

        CuSparseConstVec(const CuSparseConstVec &) = delete;
        CuSparseConstVec(CuSparseConstVec &&other) { swap(other); }
        CuSparseConstVec &operator=(const CuSparseConstVec &) = delete;
        CuSparseConstVec &operator=(CuSparseConstVec &&other)
        {
            swap(other);
            return *this;
        }

    private:
        void swap(CuSparseConstVec &other)
        {
            std::swap(raw, other.raw);
        }
    };

    /// @brief Minimal RAII wrapper for CuSparse dense vector.
    class CuSparseVec
    {
    public:
        cusparseDnVecDescr_t raw = nullptr;

        CuSparseVec(ctd::span<double> vec)
        {
            cusparseCreateDnVec(&raw, vec.size(), vec.data(), CUDA_R_64F);
        }

        ~CuSparseVec()
        {
            if (raw != nullptr)
            {
                cusparseDestroyDnVec(raw);
            }
        }

        CuSparseVec(const CuSparseVec &) = delete;
        CuSparseVec(CuSparseVec &&other) { swap(other); }
        CuSparseVec &operator=(const CuSparseVec &) = delete;
        CuSparseVec &operator=(CuSparseVec &&other)
        {
            swap(other);
            return *this;
        }

    private:
        void swap(CuSparseVec &other)
        {
            std::swap(raw, other.raw);
        }
    };

    class CuSparseBSR
    {
    public:
        cusparseConstSpMatDescr_t raw = nullptr;

        CuSparseBSR() = default;

        CuSparseBSR(BSRView mat)
        {
            if (mat.block_dim == 1)
            {
                cusparseCreateConstCsr(&raw,
                                       mat.dim,
                                       mat.dim,
                                       mat.non_zeros,
                                       mat.rows.data(),
                                       mat.cols.data(),
                                       mat.vals.data(),
                                       CUSPARSE_INDEX_32I,
                                       CUSPARSE_INDEX_32I,
                                       CUSPARSE_INDEX_BASE_ZERO,
                                       CUDA_R_64F);
            }
            else
            {
                cusparseCreateConstBsr(&raw,
                                       mat.dim,
                                       mat.dim,
                                       mat.non_zeros,
                                       mat.block_dim,
                                       mat.block_dim,
                                       mat.rows.data(),
                                       mat.cols.data(),
                                       mat.vals.data(),
                                       CUSPARSE_INDEX_32I,
                                       CUSPARSE_INDEX_32I,
                                       CUSPARSE_INDEX_BASE_ZERO,
                                       CUDA_R_64F,
                                       CUSPARSE_ORDER_ROW);
            }
        }

        ~CuSparseBSR()
        {
            if (raw != nullptr)
            {
                cusparseDestroySpMat(raw);
            }
        }

        CuSparseBSR(const CuSparseBSR &) = delete;
        CuSparseBSR(CuSparseBSR &&other) { swap(other); }
        CuSparseBSR &operator=(const CuSparseBSR &) = delete;
        CuSparseBSR &operator=(CuSparseBSR &&other)
        {
            swap(other);
            return *this;
        }

    private:
        void swap(CuSparseBSR &other)
        {
            std::swap(raw, other.raw);
        }
    };

} // namespace polysolve::linear::mas

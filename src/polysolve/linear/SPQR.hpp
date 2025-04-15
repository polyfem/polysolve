#pragma once
#ifdef POLYSOLVE_WITH_SPQR
#include <Eigen/Sparse>
#include <Eigen/SPQRSupport>
#include "EigenSolver.hpp"
#include "Solver.hpp"
namespace polysolve::linear
{
    template <>
    void EigenDirect<Eigen::SPQR<StiffnessMatrix>>::analyze_pattern(const StiffnessMatrix &A, const int precond_num);
    template <>
    void EigenDirect<Eigen::SPQR<StiffnessMatrix>>::factorize(const StiffnessMatrix &A);

    class SPQRSolver : public EigenDirect<Eigen::SPQR<StiffnessMatrix>>
    {

        StiffnessMatrix matrixQ() const;
    };
} // namespace polysolve::linear
namespace Eigen
{
    template <typename SPQRType, typename Derived>
    struct SPQR_QSparseProduct;
    namespace internal
    {
        template <typename SPQRType, typename Derived>
        struct traits<SPQR_QSparseProduct<SPQRType, Derived>>
        {
            typedef typename Derived::PlainObject ReturnType;
        };
    } // namespace internal
    template <>
    struct SPQRMatrixQReturnType<SPQR<polysolve::StiffnessMatrix>>
    {

        using SPQRType = SPQR<polysolve::StiffnessMatrix>;
        SPQRMatrixQReturnType(const SPQRType &spqr) : m_spqr(spqr) {}
        template <typename Derived>
        SPQR_QProduct<SPQRType, Derived> operator*(const MatrixBase<Derived> &other)
        {
            return SPQR_QProduct<SPQRType, Derived>(m_spqr, other.derived(), false);
        }
        template <typename Derived>
        SPQR_QSparseProduct<SPQRType, Derived> operator*(const SparseMatrixBase<Derived> &other)
        {
            return SPQR_QSparseProduct<SPQRType, Derived>(m_spqr, other.derived(), false);
        }
        SPQRMatrixQTransposeReturnType<SPQRType> adjoint() const
        {
            return SPQRMatrixQTransposeReturnType<SPQRType>(m_spqr);
        }
        // To use for operations with the transpose of Q
        SPQRMatrixQTransposeReturnType<SPQRType> transpose() const
        {
            return SPQRMatrixQTransposeReturnType<SPQRType>(m_spqr);
        }
        const SPQRType &m_spqr;
    };

    template <typename SPQRType, typename Derived>
    struct SPQR_QSparseProduct : ReturnByValue<SPQR_QSparseProduct<SPQRType, Derived>>
    {
        struct SPQRTypeWrap : public SPQRType
        {
            using SPQRType::m_H;
            using SPQRType::m_HPinv;
            using SPQRType::m_HTau;
        };
        typedef typename SPQRType::Scalar Scalar;
        typedef typename SPQRType::StorageIndex StorageIndex;
        // Define the constructor to get reference to argument types
        SPQR_QSparseProduct(const SPQRType &spqr, const Derived &other, bool transpose) : m_spqr(spqr), m_other(other), m_transpose(transpose) {}

        const SPQRTypeWrap &spqr_w() const { return reinterpret_cast<const SPQRTypeWrap &>(m_spqr); }

        inline Index rows() const { return m_transpose ? m_spqr.rows() : m_spqr.cols(); }
        inline Index cols() const { return m_other.cols(); }
        // Assign to a vector
        template <typename ResType>
        void evalTo(ResType &res) const
        {
            cholmod_sparse y_cd;
            cholmod_sparse *x_cd;
            int method = m_transpose ? SPQR_QTX : SPQR_QX;
            cholmod_common *cc = m_spqr.cholmodCommon();
            y_cd = viewAsCholmod(m_other.const_cast_derived());
            x_cd = SuiteSparseQR_qmult<Scalar>(method, spqr_w().m_H, spqr_w().m_HTau, spqr_w().m_HPinv, &y_cd, cc);
            res = viewAsEigen<Scalar, ColMajor, StorageIndex>(*x_cd);
            cholmod_l_free_sparse(&x_cd, cc);
        }
        const SPQRType &m_spqr;
        const Derived &m_other;
        bool m_transpose;
    };
} // namespace Eigen
#endif

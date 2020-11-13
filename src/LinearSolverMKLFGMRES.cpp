#ifdef POLYSOLVE_WITH_MKL

////////////////////////////////////////////////////////////////////////////////
#include "LinearSolverMKLFGMRES.hpp"
#include <thread>
#include "math.h"
#include "mkl_blas.h"
#include "mkl_spblas.h"
#include "mkl_rci.h"
////////////////////////////////////////////////////////////////////////////////
// extern "C"
// {
//     void dfgmres_init (const MKL_INT *n , const double *x , const double *b , MKL_INT *RCI_request , MKL_INT *ipar , double *dpar , double *tmp );
//     void dfgmres_check (const MKL_INT *n, const double *x, const double *b, MKL_INT *RCI_request, MKL_INT *ipar, double *dpar, double *tmp );
//     void dfgmres (const MKL_INT *n, double *x, double *b, MKL_INT *RCI_request, MKL_INT *ipar, double *dpar, double *tmp );
//     void dfgmres_get (const MKL_INT *n, double *x, double *b, MKL_INT *RCI_request, const MKL_INT *ipar, const double *dpar, double *tmp, MKL_INT *itercount );
// }
////////////////////////////////////////////////////////////////////////////////

namespace polysolve
{
    LinearSolverMKLFGMRES::LinearSolverMKLFGMRES()
    {
    }

    void LinearSolverMKLFGMRES::setType(int _mtype)
    {
    }

    // Set solver parameters
    void LinearSolverMKLFGMRES::setParameters(const json &params)
    {
        if (params.count("conv_tol"))
        {
            conv_tol_ = params["conv_tol"];
        }
        else if (params.count("tolerance"))
        {
            conv_tol_ = params["tolerance"];
        }
    }

    void LinearSolverMKLFGMRES::getInfo(json &params) const
    {
    }

    ////////////////////////////////////////////////////////////////////////////////

    void LinearSolverMKLFGMRES::init()
    {
    }

    ////////////////////////////////////////////////////////////////////////////////

    // - For symmetric matrices the solver needs only the upper triangular part of the system
    // - Make sure diagonal terms are included, even as zeros (Pardiso claims this is
    //   necessary for best performance).

    namespace
    {

        // Count number of non-zeros
        int countNonZeros(const StiffnessMatrix &K)
        {
            int count = 0;
            for (int r = 0; r < K.rows(); ++r)
            {
                for (int j = K.outerIndexPtr()[r]; j < K.outerIndexPtr()[r + 1]; ++j)
                {
                    ++count;
                }
            }
            return count;
        }

        // Compute indices of matrix coeffs in CRS format
        void computeIndices(const StiffnessMatrix &K, Eigen::VectorXi &ia, Eigen::VectorXi &ja)
        {
            int count = 0;
            for (int r = 0; r < K.rows(); ++r)
            {
                ia(r) = count;
                for (int j = K.outerIndexPtr()[r]; j < K.outerIndexPtr()[r + 1]; ++j)
                {
                    ja(count++) = K.innerIndexPtr()[j];
                }
            }
            ia.tail<1>()[0] = count;
        }

        // Compute non-zero coefficients and put them in 'a'
        void computeCoeffs(const StiffnessMatrix &K, Eigen::VectorXd &a)
        {
            int count = 0;
            for (int r = 0; r < K.rows(); ++r)
            {
                for (int j = K.outerIndexPtr()[r]; j < K.outerIndexPtr()[r + 1]; ++j)
                {
                    a(count++) = K.valuePtr()[j];
                }
            }
        }

    } // anonymous namespace

    ////////////////////////////////////////////////////////////////////////////////

    void LinearSolverMKLFGMRES::analyzePattern(const StiffnessMatrix &A, const int precond_num)
    {
        assert(A.isCompressed());

        numRows = (int)A.rows();
        int nnz = countNonZeros(A);
        ia.resize(numRows + 1);
        ja.resize(nnz);
        a.resize(nnz);
        computeIndices(A, ia, ja);
        computeCoeffs(A, a);

        // Convert matrix from 0-based C-notation to Fortran 1-based notation.
        ia = ia.array() + 1;
        ja = ja.array() + 1;
    }

    // -----------------------------------------------------------------------------

    void LinearSolverMKLFGMRES::factorize(const StiffnessMatrix &A)
    {
    }

    ////////////////////////////////////////////////////////////////////////////////

    void LinearSolverMKLFGMRES::solve(const Eigen::Ref<const VectorXd> rhs, Eigen::Ref<VectorXd> result)
    {
        MKL_INT RCI_request;
        MKL_INT itercount = 0, ierr = 0;
        MKL_INT matsize = a.size(), incx = 1, ref_nit = 2;
        char cvar = 'N', cvar1, cvar2;

        Eigen::VectorXd b(numRows);
        Eigen::VectorXd residual(numRows);
        Eigen::VectorXd tmp(numRows * (2 * numRows + 1) + (numRows * (numRows + 9)) / 2 + 1);
        Eigen::VectorXd rhs_(rhs);

        result = VectorXd(rhs_.size());
        result.resize(numRows, 1);

        dfgmres_init(&numRows, result.data(), rhs_.data(), &RCI_request, ipar, dpar, tmp.data());
        if(RCI_request != 0)
        {
            throw std::runtime_error("[MKL] error in dfgmres_init!");
        }

        ipar[7] = 0;
	    ipar[10] = 0;       // no preconditioner
        dpar[0] = 1e-2;     // relative tolerance
	    dpar[1] = conv_tol_;// absolute tolerance

        // Eigen::VectorXd ilu = a, trvec = Eigen::VectorXd::Zero(numRows);
        // ipar[30] = 1;
        // dcsrilu0(&numRows, a.data(), ia.data(), ja.data(), ilu.data(), ipar, dpar, &ierr);
        // if(ierr != 0)
        // {
        //     throw std::runtime_error("[MKL] error in dcsrilu0!");
        // }

        dfgmres_check(&numRows, result.data(), rhs_.data(), &RCI_request, ipar, dpar, tmp.data());
        if(RCI_request != 0)
        {
            throw std::runtime_error("[MKL] error in dfgmres_check!");
        }

        // If RCI_request=0, then the solution was found with the required precision
        while(1)
        {
            
            dfgmres(&numRows, result.data(), rhs_.data(), &RCI_request, ipar, dpar, tmp.data());
            if (RCI_request == 0)
            {
                break;
            }
            // If RCI_request=1, then compute the vector A*tmp[ipar[21]-1]
	        // and put the result in vector tmp[ipar[22]-1]
            else if (RCI_request == 1)
            {
                mkl_dcsrgemv(&cvar, &numRows, a.data(), ia.data(), ja.data(), &tmp[ipar[21] - 1], &tmp[ipar[22] - 1]);
            }
            // If RCI_request=2, then do the user-defined stopping test
	        // The residual stopping test for the computed solution is performed here
            else if (RCI_request == 2 || RCI_request == 4)
            {
                ipar[12] = 1;
                /* Get the current FGMRES solution in the vector b[N] */
                dfgmres_get(&numRows, result.data(), b.data(), &RCI_request, ipar, dpar, tmp.data(), &itercount);
                /* Compute the current true residual via MKL (Sparse) BLAS routines */
                mkl_dcsrgemv(&cvar, &numRows, a.data(), ia.data(), ja.data(), b.data(), residual.data());
                double dvar = -1.0E0;
                MKL_INT i = 1;
                daxpy(&numRows, &dvar, rhs_.data(), &i, residual.data(), &i);
                dvar = dnrm2(&numRows, residual.data(), &i);
                if (dvar < dpar[1]) break;
            }
            // If RCI_request=3, apply the preconditioner to tmp[ipar[21] - 1:ipar[21] + n - 2]
            // put the result in tmp[ipar[22] - 1:ipar[22] + n - 2]
            else if (RCI_request == 3)
            {
                // cvar1='L';
                // cvar='N';
                // cvar2='U';
                // mkl_dcsrtrsv(&cvar1,&cvar,&cvar2,&numRows,ilu.data(),ia.data(),ja.data(),&tmp(ipar[21]-1),trvec.data());
                // cvar1='U';
                // cvar='N';
                // cvar2='N';
                // mkl_dcsrtrsv(&cvar1,&cvar,&cvar2,&numRows,ilu.data(),ia.data(),ja.data(),trvec.data(),&tmp(ipar[22]-1));
                throw std::runtime_error("[MKL] error in dfgmres, no preconditioner!");
            }
            // If RCI_request=4, check if the solution is zero
            // else if (RCI_request == 4)
            // {
            //     /* Get the current FGMRES solution in the vector b[N] */
            //     dfgmres_get(&numRows, result.data(), b.data(), &RCI_request, ipar, dpar, tmp.data(), &itercount);
            //     MKL_INT i = 1;
            //     if (dnrm2(&numRows, b.data(), &i) < 1e-15) break;
            // }
            else
            {
                throw std::runtime_error("[MKL] error in dfgmres!");
            }
        }

        ipar[12] = 0;
	    dfgmres_get(&numRows, result.data(), rhs_.data(), &RCI_request, ipar, dpar, tmp.data(), &itercount);

        // printf("\n[MKL] Number of iterations: %d\n", itercount);
    }

    ////////////////////////////////////////////////////////////////////////////////

    void LinearSolverMKLFGMRES::freeNumericalFactorizationMemory()
    {
    }

    ////////////////////////////////////////////////////////////////////////////////

    LinearSolverMKLFGMRES::~LinearSolverMKLFGMRES()
    {
    }

} // namespace polysolve

#endif

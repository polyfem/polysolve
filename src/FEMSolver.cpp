////////////////////////////////////////////////////////////////////////////////
#include <polysolve/FEMSolver.hpp>

#ifdef POLYSOLVE_WITH_SPECTRA
#include <MatOp/SparseSymMatProd.h>
#include <MatOp/SparseSymShiftSolve.h>
#include <SymEigsSolver.h>
#include <SymEigsShiftSolver.h>
#endif

#include <unsupported/Eigen/SparseExtra>

#ifdef POLYSOLVE_WITH_MATIO
#include "matio.h"
#endif
////////////////////////////////////////////////////////////////////////////////

namespace polysolve
{
    namespace
    {
        Eigen::Vector4d compute_specturm(const StiffnessMatrix &mat)
        {
#ifdef POLYSOLVE_WITH_SPECTRA
            typedef Spectra::SparseSymMatProd<double> MatOp;
            typedef Spectra::SparseSymShiftSolve<double> InvMatOp;
            Eigen::Vector4d res;
            res.setConstant(NAN);

            InvMatOp invOpt(mat);
            Spectra::SymEigsShiftSolver<double, Spectra::LARGEST_MAGN, InvMatOp> small_eig(&invOpt, 2, 4, 0);

            small_eig.init();
            const int n_small = small_eig.compute(100000); //, 1e-8, Spectra::SMALLEST_MAGN);
            if (small_eig.info() == Spectra::SUCCESSFUL)
            {
                res(0) = small_eig.eigenvalues()(1);
                res(1) = small_eig.eigenvalues()(0);
            }

            MatOp op(mat);
            Spectra::SymEigsSolver<double, Spectra::LARGEST_MAGN, MatOp> large_eig(&op, 2, 4);

            large_eig.init();
            const int n_large = large_eig.compute(100000); //, 1e-8, Spectra::LARGEST_MAGN);
            // std::cout<<n_large<<" asdasd "<<large_eig.info()<<std::endl;
            if (large_eig.info() == Spectra::SUCCESSFUL)
            {
                res(2) = large_eig.eigenvalues()(1);
                res(3) = large_eig.eigenvalues()(0);
            }

            return res;
#else
            return Eigen::Vector4d();
#endif
        }
    } // namespace

#ifdef POLYSOLVE_WITH_MATIO
    bool write_matlab(const StiffnessMatrix& mat, Eigen::VectorXd& b, Eigen::VectorXd& x, const std::string &filePath)
    {
        Eigen::VectorXd a(mat.nonZeros());
        Eigen::VectorXi ia(mat.outerSize() + 1), ja(mat.nonZeros());
        int nnz = 0;
        for (int k = 0; k<mat.outerSize(); ++k)
        {
            ia[k] = nnz + 1;
            for (typename StiffnessMatrix::InnerIterator it(mat,k); it; ++it)
            {
                a[nnz] = it.value();
                ja[nnz] = it.row();
                nnz++;
            }
        }
        ia[mat.outerSize()] = nnz + 1;

        mat_sparse_t pa = {0,};
        pa.data = a.data();
        pa.nzmax = a.size();
        pa.ndata = a.size();
        pa.ir = ja.data();
        pa.nir = ja.size();
        pa.njc = mat.outerSize() +1;
        pa.jc = ia.data();
        size_t dims[2];

        dims[0] = mat.outerSize(); dims[1] = mat.outerSize();
        matvar_t *matvar = Mat_VarCreate("A", MAT_C_SPARSE, MAT_T_DOUBLE, 2, dims, &pa, 0);
        if(!matvar)
        {
            printf("error creating variable triplets");
            return 0;
        }

        dims[0] = b.rows(); dims[1] = b.cols();
        matvar_t *bvar = Mat_VarCreate("b", MAT_C_DOUBLE, MAT_T_DOUBLE, 2, dims, b.data(), 0);
        if(!bvar)
        {
            printf("error creating vector b");
            return 0;
        }

        dims[0] = x.rows(); dims[1] = x.cols();
        matvar_t *xvar = Mat_VarCreate("x", MAT_C_DOUBLE, MAT_T_DOUBLE, 2, dims, x.data(), 0);
        if(!xvar)
        {
            printf("error creating vector x");
            return 0;
        }

        mat_t *matfp = NULL;
        matfp = Mat_CreateVer(filePath.c_str(), NULL, MAT_FT_MAT5);
        if (!matfp)
        {
            printf("unable to create mat file");
            return 0;
        }

        Mat_VarWrite(matfp, matvar, MAT_COMPRESSION_ZLIB);
        Mat_VarWrite(matfp, bvar, MAT_COMPRESSION_ZLIB);
        Mat_VarWrite(matfp, xvar, MAT_COMPRESSION_ZLIB);
        Mat_VarFree(matvar);
        
        Mat_Close(matfp);
        return 1;
    }
#endif
} // namespace polysolve

Eigen::Vector4d polysolve::dirichlet_solve(
    LinearSolver &solver, StiffnessMatrix &A, Eigen::VectorXd &f,
    const std::vector<int> &dirichlet_nodes, Eigen::VectorXd &u,
    const int precond_num,
    const std::string &save_path,
    bool compute_spectrum)
{
    // Let Γ be the set of Dirichlet dofs.
    // To implement nonzero Dirichlet boundary conditions, we seek to replace
    // the linear system Au = f with a new system Ãx = g, where
    // - Ã is the matrix A with rows and cols of i ∈ Γ set to identity
    // - g[i] = f[i] for i ∈ Γ
    // - g[i] = f[i] - Σ_{j ∈ Γ} a_ij f[j] for i ∉ Γ
    // In matrix terms, if we call N = diag({1 iff i ∈ Γ}), then we have that
    // g = f - (I-N)*A*N*f

    int n = A.outerSize();
    Eigen::VectorXd N(n);
    N.setZero();
    for (int i : dirichlet_nodes)
    {
        N(i) = 1;
    }

    Eigen::VectorXd g = f - ((1.0 - N.array()).matrix()).asDiagonal() * (A * (N.asDiagonal() * f));

    // if (0) {
    // 	Eigen::MatrixXd rhs(g.size(), 6);
    // 	rhs.col(0) = N;
    // 	rhs.col(1) = f;
    // 	rhs.col(2) = N.asDiagonal() * f;
    // 	rhs.col(3) = A * (N.asDiagonal() * f);
    // 	rhs.col(4) = ((1.0 - N.array()).matrix()).asDiagonal() * (A * (N.asDiagonal() * f));
    // 	rhs.col(5) = g;
    // 	std::cout << rhs << std::endl;
    // }

    std::vector<Eigen::Triplet<double>> coeffs;
    coeffs.reserve(A.nonZeros());
    assert(A.rows() == A.cols());
    for (int k = 0; k < A.outerSize(); ++k)
    {
        for (StiffnessMatrix::InnerIterator it(A, k); it; ++it)
        {
            // it.value();
            // it.row();   // row index
            // it.col();   // col index (here it is equal to k)
            // it.index(); // inner index, here it is equal to it.row()
            if (N(it.row()) != 1 && N(it.col()) != 1)
            {
                coeffs.emplace_back(it.row(), it.col(), it.value());
            }
        }
    }
    // TODO: For numerical stability, we should set diagonal values of the same
    // magnitude than the other entries in the matrix
    for (int k = 0; k < n; ++k)
    {
        coeffs.emplace_back(k, k, N(k));
    }
    // Eigen::saveMarket(A, "A_before.mat");
    A.setFromTriplets(coeffs.begin(), coeffs.end());
    A.makeCompressed();

    // std::cout << A << std::endl;

    Eigen::saveMarket(A, "A.mat");
    Eigen::saveMarketVector(g, "b.mat");

    if (u.size() != n)
    {
        u.resize(n);
        u.setZero();
    }

    solver.analyzePattern(A, precond_num);
    solver.factorize(A);
    solver.solve(g, u);
    f = g;

    // static int save_idx = 0;
    // std::string mat_save_path = std::to_string(save_idx) + ".mat";
    // write_matlab(A, f, u, mat_save_path);
    // save_idx++;

    if (!save_path.empty())
    {
        Eigen::saveMarket(A, save_path);
    }

    if (compute_spectrum)
    {
        return polysolve::compute_specturm(A);
    }
    else
    {
        return Eigen::Vector4d::Zero();
    }
}

void polysolve::prefactorize(
    LinearSolver &solver, StiffnessMatrix &A,
    const std::vector<int> &dirichlet_nodes, const int precond_num,
    const std::string &save_path)
{
    int n = A.outerSize();
    Eigen::VectorXd N(n);
    N.setZero();
    for (int i : dirichlet_nodes)
    {
        N(i) = 1;
    }

    std::vector<Eigen::Triplet<double>> coeffs;
    coeffs.reserve(A.nonZeros());
    assert(A.rows() == A.cols());
    for (int k = 0; k < A.outerSize(); ++k)
    {
        for (StiffnessMatrix::InnerIterator it(A, k); it; ++it)
        {
            // it.value();
            // it.row();   // row index
            // it.col();   // col index (here it is equal to k)
            // it.index(); // inner index, here it is equal to it.row()
            if (N(it.row()) != 1 && N(it.col()) != 1)
            {
                coeffs.emplace_back(it.row(), it.col(), it.value());
            }
        }
    }
    // TODO: For numerical stability, we should set diagonal values of the same
    // magnitude than the other entries in the matrix
    for (int k = 0; k < n; ++k)
    {
        coeffs.emplace_back(k, k, N(k));
    }
    // Eigen::saveMarket(A, "A_before.mat");
    A.setFromTriplets(coeffs.begin(), coeffs.end());
    A.makeCompressed();

    solver.analyzePattern(A, precond_num);
    solver.factorize(A);
    
    if (!save_path.empty())
    {
        Eigen::saveMarket(A, save_path);
    }
}

void polysolve::dirichlet_solve_prefactorized(
    LinearSolver &solver, const StiffnessMatrix &A, Eigen::VectorXd &f,
    const std::vector<int> &dirichlet_nodes, Eigen::VectorXd &u)
{
    // pre-factorized version of dirichlet_solve

    int n = A.outerSize();
    Eigen::VectorXd N(n);
    N.setZero();
    for (int i : dirichlet_nodes)
    {
        N(i) = 1;
    }

    Eigen::VectorXd g = f - ((1.0 - N.array()).matrix()).asDiagonal() * (A * (N.asDiagonal() * f));

    if (u.size() != n)
    {
        u.resize(n);
        u.setZero();
    }

    solver.solve(g, u);
    f = g;
}

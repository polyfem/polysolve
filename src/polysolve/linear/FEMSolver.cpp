////////////////////////////////////////////////////////////////////////////////
#include "FEMSolver.hpp"

#ifdef POLYSOLVE_WITH_SPECTRA
#include <MatOp/SparseSymMatProd.h>
#include <MatOp/SparseSymShiftSolve.h>
#include <SymEigsSolver.h>
#include <SymEigsShiftSolver.h>
#endif

#include <unsupported/Eigen/SparseExtra>

////////////////////////////////////////////////////////////////////////////////

namespace polysolve
{
    namespace
    {
        Eigen::Vector4d compute_spectrum(const StiffnessMatrix &mat)
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

        void slice(const StiffnessMatrix &A, const std::vector<int> &S, StiffnessMatrix &out)
        {
            const int am = A.rows();
            const int sm = S.size();

            // assert(S.minCoeff() >= 0);
            // assert(S.maxCoeff() < sm);

            // Build reindexing maps for columns and rows
            std::vector<std::vector<int>> SI(am);
            for (int i = 0; i < sm; i++)
            {
                SI[S[i]].push_back(i);
            }

            // Take a guess at the number of nonzeros (this assumes uniform distribution
            // not banded or heavily diagonal)
            std::vector<Eigen::Triplet<double>> entries;
            entries.reserve((A.nonZeros() / (am * am)) * (sm * sm));

            // Iterate over outside
            for (int k = 0; k < A.outerSize(); ++k)
            {
                // Iterate over inside
                for (StiffnessMatrix::InnerIterator it(A, k); it; ++it)
                {
                    for (auto rit = SI[it.row()].begin(); rit != SI[it.row()].end(); rit++)
                    {
                        for (auto cit = SI[it.col()].begin(); cit != SI[it.col()].end(); cit++)
                        {
                            entries.emplace_back(*rit, *cit, it.value());
                        }
                    }
                }
            }
            out.resize(sm, sm);
            out.setFromTriplets(entries.begin(), entries.end());
            out.makeCompressed();
        }
    } // namespace
} // namespace polysolve

Eigen::Vector4d polysolve::dirichlet_solve(
    Solver &solver, StiffnessMatrix &A, Eigen::VectorXd &f,
    const std::vector<int> &dirichlet_nodes, Eigen::VectorXd &u,
    const int precond_num,
    const std::string &save_path,
    bool compute_spectrum,
    const bool remove_zero_cols,
    const bool skip_last_cols)
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

    // remove zero cols
    if (remove_zero_cols)
    {
        std::vector<bool> zero_col(A.cols(), true);
        zero_col.back() = !skip_last_cols;
        for (int k = 0; k < A.outerSize(); ++k)
        {
            for (StiffnessMatrix::InnerIterator it(A, k); it; ++it)
            {
                if (skip_last_cols)
                {
                    if (it.row() != A.rows() - 1 && it.col() != A.cols() - 1 && fabs(it.value()) > 1e-12)
                        zero_col[it.col()] = false;
                }
                else
                {
                    if (fabs(it.value()) > 1e-12)
                        zero_col[it.col()] = false;
                }
            }
        }

        std::vector<int> valid;
        for (int i = 0; i < A.rows(); ++i)
        {
            if (!zero_col[i])
                valid.push_back(i);
            else if (skip_last_cols)
            {
                A.coeffRef(A.rows() - 1, i) = 0;
                A.coeffRef(i, A.cols() - 1) = 0;
            }
        }

        StiffnessMatrix As;
        slice(A, valid, As);

        Eigen::VectorXd gs(As.rows());
        int index = 0;
        for (int i = 0; i < zero_col.size(); ++i)
        {
            if (!zero_col[i])
            {
                gs[index++] = g[i];
            }
        }

        if (u.size() != gs.size())
        {
            u.resize(gs.size());
            u.setZero();
        }

        Eigen::VectorXd us = u;
        solver.analyzePattern(As, precond_num);
        solver.factorize(As);
        solver.solve(gs, us);
        f = g;

        u.resize(A.rows());
        index = 0;
        for (int i = 0; i < zero_col.size(); ++i)
        {
            if (zero_col[i])
            {
                u[i] = 0;
                f[i] = 0;
            }
            else
                u[i] = us[index++];
        }
    }
    else
    {
        // Eigen::saveMarket(A, "A.mat");
        // Eigen::saveMarketVector(g, "b.mat");

        if (u.size() != n)
        {
            u.resize(n);
            u.setZero();
        }

        solver.analyzePattern(A, precond_num);
        solver.factorize(A);
        solver.solve(g, u);
        f = g;
    }

    if (!save_path.empty())
    {
        Eigen::saveMarket(A, save_path);
    }

    if (compute_spectrum)
    {
        return polysolve::compute_spectrum(A);
    }
    else
    {
        return Eigen::Vector4d::Zero();
    }
}

void polysolve::prefactorize(
    Solver &solver, StiffnessMatrix &A,
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
    Solver &solver, const StiffnessMatrix &A, Eigen::VectorXd &f,
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

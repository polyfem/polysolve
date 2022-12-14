#pragma once

#ifdef POLYSOLVE_WITH_PETSC

////////////////////////////////////////////////////////////////////////////////

#include <petsc.h>
#include <polysolve/LinearSolver.hpp>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <vector>

////////////////////////////////////////////////////////////////////////////////
//
// https://docs.nvidia.com/cuda/cusolver/index.html#cuSolverDN-function-reference
//

namespace polysolve
{
    class LinearSolverPETSC : public LinearSolver
    {

    public:
        LinearSolverPETSC();
        ~LinearSolverPETSC();

    private:
        POLYSOLVE_DELETE_MOVE_COPY(LinearSolverPETSC)

    public:
        //////////////////////
        // Public interface //
        //////////////////////

        // Set solver parameters
        virtual void setParameters(const json &params) override;

        // Retrieve memory information from cuSolverDN
        virtual void getInfo(json &params) const override;

        // Analyze sparsity pattern (sparse)
        virtual void analyzePattern(const StiffnessMatrix &A, const int precond_num) override;

        // Factorize system matrix for PETSC(AIJ_CUSPARSE: TRUE OR FALSE)
        /*SOLVER INDEX
        0 = PARDISO
        1 = SUPERLU_DIST
        2 = CHOLMOD
        3 = MUMPS
        4 = CUSPARSE
        5 = STRUMPACK
        6 = HYPRE // NOT FULLY IMPLEMENTED YET
        */
        virtual void factorize(StiffnessMatrix &A, int AIJ_CUSPARSE, int SOLVER_INDEX) override;

        // Analyze sparsity pattern (dense, preferred)
        virtual void analyzePattern(const Eigen::MatrixXd &A, const int precond_num) override;

        // Solve the linear system Ax = b
        virtual void solve(const Ref<const VectorXd> b, Ref<VectorXd> x) override;

        // Name of the solver type (for debugging purposes)
        virtual std::string name() const override { return "PETSC_Solver"; }

    protected:
        int init();

    protected:
        // PETSC variables
        Vec b_petsc, x_petsc, y_petsc;
        Mat A_petsc, F;
        KSP ksp;
        PC pc;
        PetscReal norm;
        PetscInt its, GPU_vec;

        // Eigen variables
        Eigen::SparseMatrix<double> A;
    };

} // namespace polysolve

#endif

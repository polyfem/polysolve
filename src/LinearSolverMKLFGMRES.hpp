#pragma once

#ifdef POLYSOLVE_WITH_MKL

////////////////////////////////////////////////////////////////////////////////
#include <polysolve/LinearSolver.hpp>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Sparse>
////////////////////////////////////////////////////////////////////////////////

namespace polysolve
{

    class LinearSolverMKLFGMRES : public LinearSolver
    {

    public:
        LinearSolverMKLFGMRES();
        ~LinearSolverMKLFGMRES();

    private:
        POLYSOLVE_DELETE_MOVE_COPY(LinearSolverMKLFGMRES)

    protected:
        void setType(int _mtype);
        void init();
        void freeNumericalFactorizationMemory();

    public:
        //////////////////////
        // Public interface //
        //////////////////////

        // Set solver parameters
        virtual void setParameters(const json &params) override;

        // Retrieve memory information from Pardiso
        virtual void getInfo(json &params) const override;

        // Analyze sparsity pattern
        virtual void analyzePattern(const StiffnessMatrix &A, const int precond_num) override;

        // Factorize system matrix
        virtual void factorize(const StiffnessMatrix &A) override;

        // Solve the linear system Ax = b
        virtual void solve(const Ref<const VectorXd> b, Ref<VectorXd> x) override;

        // Name of the solver type (for debugging purposes)
        virtual std::string name() const override { return "MKLFGMRES"; }

    protected:
        Eigen::VectorXi ia, ja;
        VectorXd a;

    protected:
        int numRows;
        int nrhs = 1; // Number of right hand sides.

        // Internal solver memory pointer pt,
        // 32-bit: int pt[64]; 64-bit: long int pt[64]
        // or void *pt[64] should be OK on both architectures
        void *pt[64];

        // Pardiso control parameters.
        int ipar[128];
        double dpar[128];

        int numUniqueElements;
    };

} // namespace polysolve

#endif

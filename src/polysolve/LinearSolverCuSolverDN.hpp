#pragma once

#ifdef POLYSOLVE_WITH_CUSOLVER

////////////////////////////////////////////////////////////////////////////////
#include <polysolve/LinearSolver.hpp>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <vector>

//find cuda libraries?

#include <cusolverDn.h>

////////////////////////////////////////////////////////////////////////////////
//
// https://docs.nvidia.com/cuda/cusolver/index.html#cuSolverDN-function-reference
//

namespace polysolve
{
    class LinearSolverCuSolverDN : public LinearSolver
    {

    public:
        LinearSolverCuSolverDN();
        ~LinearSolverCuSolverDN();

    private:
        POLYSOLVE_DELETE_MOVE_COPY(LinearSolverCuSolverDN)

    public:
        //////////////////////
        // Public interface //
        //////////////////////

        // Set solver parameters
        virtual void setParameters(const json &params) override;

        // Retrieve memory information from Pardiso
        virtual void getInfo(json &params) const override;

        // Analyze sparsity pattern
        virtual void analyzePattern(const StiffnessMatrix &A, const int precond_num) override { precond_num_ = precond_num; }

        // Factorize system matrix
        virtual void factorize(const StiffnessMatrix &A) override;

        // Solve the linear system Ax = b
        virtual void solve(const Ref<const VectorXd> b, Ref<VectorXd> x) override;

        // Name of the solver type (for debugging purposes)
        virtual std::string name() const override { return "cuSolverDN"; }

    protected:
        

    private:
        
    };

//TODO: fill this in -MP
}//namespace polysolve


#endif

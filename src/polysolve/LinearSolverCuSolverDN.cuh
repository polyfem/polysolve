#pragma once

#ifdef POLYSOLVE_WITH_CUSOLVER

////////////////////////////////////////////////////////////////////////////////
#include <polysolve/LinearSolver.hpp>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <vector>

//find cuda libraries?

#include <cuda_runtime.h>
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

        // Retrieve memory information from cuSolverDN
        virtual void getInfo(json &params) const override;

        // Analyze sparsity pattern
        virtual void analyzePattern(const StiffnessMatrix &A, const int precond_num) override;

        // Factorize system matrix
        virtual void factorize(const StiffnessMatrix &A) override;

        // Solve the linear system Ax = b
        virtual void solve(const Ref<const VectorXd> b, Ref<VectorXd> x) override;

        // Name of the solver type (for debugging purposes)
        virtual std::string name() const override { return "cuSolverDN"; }

    protected:
        void init();

    protected:

        cusolverDnHandle_t cuHandle;
        cusolverDnParams_t cuParams;
        cudaStream_t stream;

        //device copies
        double *d_A;
        double *d_b;

        //device work buffers
        size_t d_lwork = 0;     /* size of workspace */
        void *d_work = nullptr; /* device workspace for getrf */
        size_t h_lwork = 0;     /* size of workspace */
        void *h_work = nullptr; /* host workspace for getrf */

        int numrows;

        Eigen::MatrixXd Adense;
        
        
    };

//TODO: complete this -MP
}//namespace polysolve


#endif

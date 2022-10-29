#pragma once

#ifdef POLYSOLVE_WITH_AMGCL
#ifdef POLYSOLVE_WITH_CUDA

////////////////////////////////////////////////////////////////////////////////
#include <polysolve/LinearSolver.hpp>

#include <Eigen/Core>
#include <amgcl/backend/cuda.hpp>
#include <amgcl/adapter/crs_tuple.hpp>
#include <amgcl/make_solver.hpp>
#include <amgcl/amg.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/coarsening/plain_aggregates.hpp>
#include <amgcl/coarsening/aggregation.hpp>
#include <amgcl/coarsening/ruge_stuben.hpp>
#include <amgcl/relaxation/spai0.hpp>
#include <amgcl/relaxation/gauss_seidel.hpp>
#include <amgcl/solver/cg.hpp>
#include <amgcl/solver/bicgstab.hpp>
#include <amgcl/solver/gmres.hpp>
#include <amgcl/solver/runtime.hpp>
#include <amgcl/profiler.hpp>
#include <amgcl/io/mm.hpp>

// SET THIS AS AN OPTIONAL HEADER
#ifdef POLYSOLVE_WITH_CUSPARSEILU0
#include <amgcl/relaxation/cusparse_ilu0.hpp>
#endif

#include <amgcl/relaxation/chebyshev.hpp>
#include <amgcl/coarsening/runtime.hpp>
#include <amgcl/relaxation/runtime.hpp>
#include <amgcl/preconditioner/runtime.hpp>
#include <amgcl/value_type/static_matrix.hpp>
// #include <amgcl/backend/vexcl.hpp>
#include <amgcl/adapter/crs_tuple.hpp>
#include <amgcl/adapter/reorder.hpp>
#include <amgcl/adapter/eigen.hpp>
#include <amgcl/adapter/block_matrix.hpp>
#include <amgcl/profiler.hpp>
#include <memory>
#include <type_traits>

////////////////////////////////////////////////////////////////////////////////
//
// WARNING:
// The matrix is assumed to be in row-major format, since AMGCL assumes that the
// outer index is for row. If the matrix is symmetric, you are fine, because CSR
// and CSC are the same. If the matrix is not symmetric and you pass in a
// column-major matrix, the solver will actually solve A^T x = b.
//

namespace polysolve
{
    class LinearSolverAMGCL_cuda : public LinearSolver
    {

    public:
        LinearSolverAMGCL_cuda();
        ~LinearSolverAMGCL_cuda();

    private:
        POLYSOLVE_DELETE_MOVE_COPY(LinearSolverAMGCL_cuda)

    public:
        //////////////////////
        // Public interface //
        //////////////////////

        // Set solver parameters
        virtual void setParameters(const json &params) override;

        // Retrieve information
        virtual void getInfo(json &params) const override;

        // Analyze sparsity pattern
        virtual void analyzePattern(const StiffnessMatrix &A, const int precond_num) override
        {
            precond_num_ = precond_num;
        }

        // Factorize system matrix
        virtual void factorize(const StiffnessMatrix &A) override;

        // Solve the linear system Ax = b
        virtual void solve(const Ref<const VectorXd> b, Ref<VectorXd> x) override;

        // Name of the solver type (for debugging purposes)
        virtual std::string name() const override { return "AMGCL_cuda"; }

    private:
        using Backend = amgcl::backend::cuda<double>;
        using Solver = amgcl::make_solver<
            amgcl::runtime::preconditioner<Backend>,
            amgcl::runtime::solver::wrapper<Backend>>;
        std::unique_ptr<Solver> solver_;
        json params_;
        typename Backend::params backend_params_;

        int precond_num_;
        int block_size_ = 1;

        // Output info
        size_t iterations_;
        double residual_error_;
    };

} // namespace polysolve

#endif
#endif

#pragma once

////////////////////////////////////////////////////////////////////////////////
#include "Solver.hpp"
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <vector>

////////////////////////////////////////////////////////////////////////////////
//

namespace polysolve::linear
{

    class SaddlePointSolver : public Solver
    {

    public:
        SaddlePointSolver();
        ~SaddlePointSolver();

    private:
        POLYSOLVE_DELETE_MOVE_COPY(SaddlePointSolver)

    public:
        //////////////////////
        // Public interface //
        //////////////////////

        // Set solver parameters
        virtual void set_parameters(const json &params) override;

        // Retrieve memory information from Pardiso
        virtual void get_info(json &params) const override;

        // Analyze sparsity pattern
        virtual void analyze_pattern(const StiffnessMatrix &A, const int precond_num) override { precond_num_ = precond_num; }

        // Factorize system matrix
        virtual void factorize(const StiffnessMatrix &A) override;

        // Solve the linear system Ax = b
        virtual void solve(const Ref<const VectorXd> b, Ref<VectorXd> x) override;

        // Name of the solver type (for debugging purposes)
        virtual std::string name() const override { return "SaddlePointSolver"; }

    private:
        StiffnessMatrix Ain_;

        StiffnessMatrix As;
        StiffnessMatrix Bs;
        StiffnessMatrix BsT;
        StiffnessMatrix Cs;
        StiffnessMatrix Ss;

        StiffnessMatrix Wm;
        StiffnessMatrix Wc;

        int max_iter_;
        double conv_tol_;

        int precond_num_;

        std::string asymmetric_solver_name_;
        std::string symmetric_solver_name_;

        json asymmetric_solver_params_;
        json symmetric_solver_params_;

        double final_res_norm_;
        int num_iterations_;
    };

} // namespace polysolve::linear

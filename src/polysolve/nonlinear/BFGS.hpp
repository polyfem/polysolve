// L-BFGS solver (Using the LBFGSpp under MIT License).

#pragma once

#include "Solver.hpp"
#include <polysolve/Utils.hpp>

#include <polysolve/linear/Solver.hpp>

#include <LBFGSpp/BFGSMat.h>

namespace polysolve::nonlinear
{
    class BFGS : public Solver
    {
    public:
        using Superclass = Solver;
        using typename Superclass::Scalar;
        using typename Superclass::TVector;

        BFGS(const json &solver_params,
             const json &linear_solver_params,
             const double dt, const double characteristic_length,
             spdlog::logger &logger);

        std::string name() const override { return "BFGS"; }

    protected:
        virtual void set_default_descent_strategy() override { descent_strategy = 1; }

        using Superclass::descent_strategy_name;
        std::string descent_strategy_name(int descent_strategy) const override;

        void increase_descent_strategy() override;

    protected:
        TVector m_prev_x;    // Previous x
        TVector m_prev_grad; // Previous gradient

        Eigen::MatrixXd hess;

        void reset(const int ndof) override;

        void reset_history(const int ndof);

        virtual bool compute_update_direction(
            Problem &objFunc,
            const TVector &x,
            const TVector &grad,
            TVector &direction) override;

        std::unique_ptr<polysolve::linear::Solver> linear_solver; ///< Linear solver used to solve the linear system
    };
} // namespace polysolve::nonlinear

// L-BFGS solver (Using the LBFGSpp under MIT License).

#pragma once

#include "DescentStrategy.hpp"
#include <polysolve/Utils.hpp>

#include <polysolve/linear/Solver.hpp>

#include <LBFGSpp/BFGSMat.h>

namespace polysolve::nonlinear
{
    class BFGS : public DescentStrategy
    {
    public:
        using Superclass = DescentStrategy;

        BFGS(const json &solver_params,
             const json &linear_solver_params,
             const double characteristic_length,
             spdlog::logger &logger);

        std::string name() const override { return "BFGS"; }

        void reset(const int ndof) override;

        virtual bool compute_update_direction(
            Problem &objFunc,
            const TVector &x,
            const TVector &grad,
            TVector &direction) override;

    private:
        TVector m_prev_x;    // Previous x
        TVector m_prev_grad; // Previous gradient

        Eigen::MatrixXd hess;

        void reset_history(const int ndof);

        std::unique_ptr<polysolve::linear::Solver> linear_solver; ///< Linear solver used to solve the linear system
    };
} // namespace polysolve::nonlinear

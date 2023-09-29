// L-BFGS solver (Using the LBFGSpp under MIT License).

#pragma once

#include "Utils.hpp"
#include "BoxConstraintSolver.hpp"

#include <LBFGSpp/BFGSMat.h>

namespace polysolve::nonlinear
{
    class LBFGSB : public BoxConstraintSolver
    {
    public:
        using Superclass = BoxConstraintSolver;
        using typename Superclass::Scalar;
        using typename Superclass::TVector;

        LBFGSB(const json &solver_params,
               const json &linear_solver_params,
               const double dt,
               const double characteristic_length,
               spdlog::logger &logger);

        std::string name() const override { return "L-BFGS-B"; }

    protected:
        virtual int default_descent_strategy() override { return 1; }

        using Superclass::descent_strategy_name;
        std::string descent_strategy_name(int descent_strategy) const override;

        void increase_descent_strategy() override;

    protected:
        LBFGSpp::BFGSMat<Scalar, true> m_bfgs; // Approximation to the Hessian matrix

        /// The number of corrections to approximate the inverse Hessian matrix.
        /// The L-BFGS routine stores the computation results of previous \ref m
        /// iterations to approximate the inverse Hessian matrix of the current
        /// iteration. This parameter controls the size of the limited memories
        /// (corrections). The default value is \c 6. Values less than \c 3 are
        /// not recommended. Large values will result in excessive computing time.
        int m_history_size = 6;
        int max_submin = 10;

        TVector m_prev_x;    // Previous x
        TVector m_prev_grad; // Previous gradient

        void reset(const int ndof) override;

        void reset_history(const int ndof);

        virtual bool compute_update_direction(
            Problem &objFunc,
            const TVector &x,
            const TVector &grad,
            TVector &direction) override;
    };
} // namespace polysolve::nonlinear

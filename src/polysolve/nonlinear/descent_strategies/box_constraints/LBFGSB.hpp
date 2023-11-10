// L-BFGS solver (Using the LBFGSpp under MIT License).

#pragma once

#include "BoxedDescentStrategy.hpp"
#include <polysolve/Utils.hpp>

#include <LBFGSpp/BFGSMat.h>

namespace polysolve::nonlinear
{
    class LBFGSB : public BoxedDescentStrategy
    {
    public:
        using Superclass = BoxedDescentStrategy;

        LBFGSB(const json &solver_params,
               const double characteristic_length,
               spdlog::logger &logger);

        std::string name() const override { return "L-BFGS-B"; }

    private:
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

        void reset_history(const int ndof);

    public:
        void reset(const int ndof) override;

        bool compute_update_direction(
            Problem &objFunc,
            const TVector &x,
            const TVector &grad,
            const TVector &lower_bound,
            const TVector &upper_bound,
            TVector &direction) override;
    };
} // namespace polysolve::nonlinear

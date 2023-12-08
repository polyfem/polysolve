#pragma once

#include "DescentStrategy.hpp"
#include <polysolve/Utils.hpp>

#include <polysolve/linear/Solver.hpp>

namespace polysolve::nonlinear
{
    class ADAM : public DescentStrategy
    {
    public:
        using Superclass = DescentStrategy;

        ADAM(const json &solver_params,
             const bool is_stochastic,
             const double characteristic_length,
             spdlog::logger &logger);

        std::string name() const override { return is_stochastic_ ? "StochasticADAM" : "ADAM"; }

        void reset(const int ndof) override;

        virtual bool compute_update_direction(
            Problem &objFunc,
            const TVector &x,
            const TVector &grad,
            TVector &direction) override;

        bool is_direction_descent() override { return false; }

    private:
        TVector m_prev_;
        TVector v_prev_;

        double beta_1_, beta_2_;
        double alpha_;

        int t_ = 0;
        double epsilon_;

        bool is_stochastic_;
        double erase_component_probability_ = 0;
    };
} // namespace polysolve::nonlinear

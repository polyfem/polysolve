#pragma once

#include "DescentStrategy.hpp"
#include <polysolve/Utils.hpp>

#include <polysolve/linear/Solver.hpp>

namespace polysolve::nonlinear
{
    class GradientDescent : public DescentStrategy
    {
    public:
        using Superclass = DescentStrategy;

        GradientDescent(const json &solver_params_,
                        const bool is_stochastic,
                        const double characteristic_length,
                        spdlog::logger &logger);

        std::string name() const override { return is_stochastic_ ? "StochasticGradientDescent" : "GradientDescent"; }

        bool compute_update_direction(
            Problem &objFunc,
            const TVector &x,
            const TVector &grad,
            TVector &direction) override;

    private:
        bool is_stochastic_ = false;
        double erase_component_probability_ = 0;
    };
} // namespace polysolve::nonlinear

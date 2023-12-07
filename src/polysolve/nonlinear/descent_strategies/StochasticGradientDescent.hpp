#pragma once

#include "DescentStrategy.hpp"
#include <polysolve/Utils.hpp>

#include <polysolve/linear/Solver.hpp>

namespace polysolve::nonlinear
{
    class StochasticGradientDescent : public DescentStrategy
    {
    public:
        using Superclass = DescentStrategy;

        StochasticGradientDescent(const json &solver_params_,
                                  const double characteristic_length,
                                  spdlog::logger &logger);

        std::string name() const override { return "StochasticGradientDescent"; }

        bool compute_update_direction(
            Problem &objFunc,
            const TVector &x,
            const TVector &grad,
            TVector &direction) override;

    private:
        double erase_component_probability_ = 0;
    };
} // namespace polysolve::nonlinear

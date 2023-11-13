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
                        const double characteristic_length,
                        spdlog::logger &logger);

        std::string name() const override { return "GradientDescent"; }

        bool compute_update_direction(
            Problem &objFunc,
            const TVector &x,
            const TVector &grad,
            TVector &direction) override;
    };
} // namespace polysolve::nonlinear

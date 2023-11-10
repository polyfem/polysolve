#pragma once

#include <polysolve/nonlinear/descent_strategies/DescentStrategy.hpp>

namespace polysolve::nonlinear
{
    class BoxedDescentStrategy : public DescentStrategy
    {
    public:
        BoxedDescentStrategy(const json &solver_params,
                             const double characteristic_length,
                             spdlog::logger &logger)
            : DescentStrategy(solver_params, characteristic_length, logger)
        {
        }

        virtual ~BoxedDescentStrategy() {}

        virtual bool compute_update_direction(
            Problem &objFunc,
            const TVector &x,
            const TVector &grad,
            const TVector &lower_bound,
            const TVector &upper_bound,
            TVector &direction) = 0;

        bool compute_update_direction(
            Problem &objFunc,
            const TVector &x,
            const TVector &grad,
            TVector &direction) override
        {
            assert(false);
            return false;
        };
    };
} // namespace polysolve::nonlinear

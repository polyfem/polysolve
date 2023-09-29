#pragma once

#include "Solver.hpp"
#include "Utils.hpp"

#include <polysolve/linear/Solver.hpp>

namespace polysolve::nonlinear
{
    class GradientDescent : public Solver
    {
    public:
        using Superclass = Solver;
        using typename Superclass::Scalar;
        using typename Superclass::TVector;

        GradientDescent(const json &solver_params_,
                        const double dt,
                        const double characteristic_length,
                        spdlog::logger &logger);

        std::string name() const override { return "GradientDescent"; }

    protected:
        virtual int default_descent_strategy() override { return 1; }

        using Superclass::descent_strategy_name;
        std::string descent_strategy_name(int descent_strategy_) const override;
        void increase_descent_strategy() override;

    protected:
        void reset(const int ndof) override;

        virtual bool compute_update_direction(
            Problem &objFunc,
            const TVector &x,
            const TVector &grad,
            TVector &direction) override;
    };
} // namespace polysolve::nonlinear

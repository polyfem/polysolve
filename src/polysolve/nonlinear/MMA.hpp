#pragma once

#include "MMAAux.hpp"
#include "BoxConstraintSolver.hpp"

namespace polysolve::nonlinear
{
    class MMA : public BoxConstraintSolver
    {
    public:
        using Superclass = BoxConstraintSolver;
        using typename Superclass::Scalar;
        using typename Superclass::TVector;

        MMA(const json &solver_params,
            const double characteristic_length,
            spdlog::logger &logger);

        void set_constraints(const std::vector<std::shared_ptr<Problem>> &constraints) { constraints_ = constraints; }

        std::string name() const override { return "MMA"; }

    protected:
        virtual void set_default_descent_strategy() override { descent_strategy = Solver::MMA_STRATEGY; }

        using Superclass::descent_strategy_name;
        std::string descent_strategy_name(int descent_strategy) const override;
        void increase_descent_strategy() override;

        bool is_direction_descent() override { return false; }

    protected:
        std::shared_ptr<MMAAux> mma;

        std::vector<std::shared_ptr<Problem>> constraints_;

        void reset(const int ndof) override;

        virtual void compute_update_direction(
            Problem &objFunc,
            const TVector &x,
            const TVector &grad,
            TVector &direction) override;
    };

} // namespace polysolve::nonlinear

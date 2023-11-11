#pragma once

#include "MMAAux.hpp"
#include "BoxedDescentStrategy.hpp"

namespace polysolve::nonlinear
{
    class MMA : public BoxedDescentStrategy
    {
    public:
        using Superclass = BoxedDescentStrategy;

        MMA(const json &solver_params,
            const double characteristic_length,
            spdlog::logger &logger);

        void set_constraints(const std::vector<std::shared_ptr<Problem>> &constraints) { constraints_ = constraints; }

        std::string name() const override { return "MMA"; }

    public:
        bool is_direction_descent() override { return false; }

    private:
        std::shared_ptr<MMAAux> mma;

        std::vector<std::shared_ptr<Problem>> constraints_;

    public:
        void reset(const int ndof) override;

        bool compute_boxed_update_direction(
            Problem &objFunc,
            const TVector &x,
            const TVector &grad,
            const TVector &lower_bound,
            const TVector &upper_bound,
            TVector &direction) override;
    };

} // namespace polysolve::nonlinear

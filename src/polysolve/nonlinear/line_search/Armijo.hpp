#pragma once

#include "Backtracking.hpp"

namespace polysolve::nonlinear::line_search
{
    class Armijo : public Backtracking
    {
    public:
        using Superclass = Backtracking;
        using typename Superclass::Scalar;
        using typename Superclass::TVector;

        Armijo(const json &params, spdlog::logger &logger);

        virtual std::string name() override { return "Armijo"; }

    protected:
        virtual void init_compute_descent_step_size(
            const TVector &delta_x,
            const TVector &old_grad) override;

        virtual bool criteria(
            Problem &objFunc,
            const TVector &delta_x,
            const TVector &new_x,
            const double old_energy,
            const TVector &old_grad,
            const double new_energy,
            const double step_size) const override;

        double c;
        double armijo_criteria; ///< cached value: c * delta_x.dot(old_grad)
    };
} // namespace polysolve::nonlinear::line_search

#pragma once

#include "LineSearch.hpp"

namespace polysolve::nonlinear::line_search
{
    class Backtracking : public LineSearch
    {
    public:
        using Superclass = LineSearch;
        using typename Superclass::Scalar;
        using typename Superclass::TVector;

        Backtracking(const json &params, spdlog::logger &logger);

        virtual std::string name() const override { return "Backtracking"; }

        double compute_descent_step_size(
            const TVector &x,
            const TVector &delta_x,
            Problem &objFunc,
            const bool use_grad_norm,
            const double old_energy,
            const TVector &old_grad,
            const double starting_step_size) override;

    protected:
        virtual void init_compute_descent_step_size(
            const TVector &delta_x,
            const TVector &old_grad) {}

        virtual bool criteria(
            const TVector &delta_x,
            Problem &objFunc,
            const bool use_grad_norm,
            const double old_energy,
            const TVector &old_grad,
            const TVector &new_x,
            const double new_energy,
            const double step_size) const;
    };
} // namespace polysolve::nonlinear::line_search

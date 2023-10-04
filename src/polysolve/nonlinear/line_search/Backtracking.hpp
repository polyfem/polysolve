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

        Backtracking(spdlog::logger &logger);

    protected:
        double compute_descent_step_size(
            const TVector &x,
            const TVector &delta_x,
            Problem &objFunc,
            const bool use_grad_norm,
            const double old_energy_in,
            const double starting_step_size) override;
    };
} // namespace polysolve::nonlinear::line_search

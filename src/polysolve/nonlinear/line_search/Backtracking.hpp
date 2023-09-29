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

        Backtracking(const std::shared_ptr<Logger> &logger);

        double line_search(
            const TVector &x,
            const TVector &delta_x,
            Problem &objFunc) override;

    protected:
        double compute_descent_step_size(
            const TVector &x,
            const TVector &delta_x,
            Problem &objFunc,
            const double old_energy_in,
            const double starting_step_size = 1);
    };
} // namespace polysolve::nonlinear::line_search

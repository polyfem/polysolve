#pragma once
#include "LineSearch.hpp"

namespace polysolve::nonlinear::line_search
{
    class NoLineSearch : public LineSearch
    {
    public:
        using Superclass = LineSearch;
        using typename Superclass::Scalar;
        using typename Superclass::TVector;

        NoLineSearch(const json &params, spdlog::logger &logger);

        virtual std::string name() const override { return "None"; }

    protected:
        double compute_descent_step_size(
            const TVector &x,
            const TVector &delta_x,
            Problem &objFunc,
            const bool,
            const double,
            const TVector &,
            const double starting_step_size) override;
    };
} // namespace polysolve::nonlinear::line_search
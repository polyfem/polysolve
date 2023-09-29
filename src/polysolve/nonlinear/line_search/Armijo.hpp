#pragma once

#include "LineSearch.hpp"

namespace polysolve::nonlinear::line_search
{
    class Armijo : public LineSearch
    {
    public:
        using Superclass = LineSearch;
        using typename Superclass::Scalar;
        using typename Superclass::TVector;

        Armijo(spdlog::logger &logger);

        double line_search(
            const TVector &x,
            const TVector &searchDir,
            Problem &objFunc) override;

    protected:
        const double default_alpha_init = 1.0;
        const double c = 0.5;
        const double tau = 0.5;
    };
} // namespace polysolve::nonlinear::line_search

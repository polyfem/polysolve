#pragma once

#include "LineSearch.hpp"

#include <cppoptlib/linesearch/armijo.h>

namespace polysolve::nonlinear::line_search
{
    class CppOptArmijo : public LineSearch
    {
    public:
        using Superclass = LineSearch;
        using typename Superclass::Scalar;
        using typename Superclass::TVector;

        CppOptArmijo(spdlog::logger &logger)
            : Superclass(logger)
        {
        }

    protected:
        double compute_descent_step_size(
            const TVector &x,
            const TVector &delta_x,
            Problem &objFunc,
            const bool,
            const double,
            const double starting_step_size) override
        {
            const double tmp = cppoptlib::Armijo<Problem, 1>::linesearch(x, delta_x, objFunc);
            return std::min(starting_step_size, tmp);
        }
    };
} // namespace polysolve::nonlinear::line_search

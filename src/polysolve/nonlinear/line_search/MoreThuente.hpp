#pragma once

#include "LineSearch.hpp"

#include <cppoptlib/linesearch/morethuente.h>

namespace polysolve::nonlinear::line_search
{
    class MoreThuente : public LineSearch
    {
    public:
        using Superclass = LineSearch;
        using typename Superclass::Scalar;
        using typename Superclass::TVector;

        MoreThuente(spdlog::logger &logger)
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
            return std::min(starting_step_size, cppoptlib::MoreThuente<Problem, 1>::linesearch(x, delta_x, objFunc));
        }
    };
} // namespace polysolve::nonlinear::line_search

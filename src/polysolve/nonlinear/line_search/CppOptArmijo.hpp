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

        CppOptArmijo(const std::shared_ptr<Logger> &logger)
            : Superclass(logger)
        {
        }

        double line_search(
            const TVector &x,
            const TVector &searchDir,
            Problem &objFunc) override
        {
            return cppoptlib::Armijo<Problem, 1>::linesearch(x, searchDir, objFunc);
        }
    };
} // namespace polysolve::nonlinear::line_search

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

        MoreThuente(const std::shared_ptr<Logger> &logger)
            : Superclass(logger)
        {
        }

        double line_search(
            const TVector &x,
            const TVector &searchDir,
            Problem &objFunc) override
        {
            return cppoptlib::MoreThuente<Problem, 1>::linesearch(x, searchDir, objFunc);
        }
    };
} // namespace polysolve::nonlinear::line_search

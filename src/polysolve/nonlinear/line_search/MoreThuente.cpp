#include "MoreThuente.hpp"

#include <cppoptlib/linesearch/morethuente.h>

namespace polysolve::nonlinear::line_search
{
    MoreThuente::MoreThuente(spdlog::logger &logger)
        : Superclass(logger)
    {
    }

    double MoreThuente::compute_descent_step_size(
        const TVector &x,
        const TVector &delta_x,
        Problem &objFunc,
        const bool,
        const double,
        const double starting_step_size)
    {
        const double tmp = cppoptlib::MoreThuente<Problem, 1>::linesearch(x, delta_x, objFunc, starting_step_size);

        return std::min(tmp, starting_step_size);
    }
} // namespace polysolve::nonlinear::line_search

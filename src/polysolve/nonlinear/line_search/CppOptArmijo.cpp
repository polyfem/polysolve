#include "CppOptArmijo.hpp"

#include <cppoptlib/linesearch/armijo.h>

namespace polysolve::nonlinear::line_search
{
    CppOptArmijo::CppOptArmijo(spdlog::logger &logger)
        : Superclass(logger)
    {
    }

    double CppOptArmijo::compute_descent_step_size(
        const TVector &x,
        const TVector &delta_x,
        Problem &objFunc,
        const bool,
        const double,
        const double starting_step_size)
    {
        const double tmp = cppoptlib::Armijo<Problem, 1>::linesearch(x, delta_x, objFunc, starting_step_size);
        assert(tmp <= starting_step_size);
        return tmp;
    }

} // namespace polysolve::nonlinear::line_search

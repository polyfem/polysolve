#include "CppOptArmijo.hpp"

#include <cppoptlib/linesearch/armijo.h>

namespace polysolve::nonlinear::line_search
{
    CppOptArmijo::CppOptArmijo(const json &params, spdlog::logger &logger)
        : Superclass(params, logger), after_check(params, logger)
    {
    }

    double CppOptArmijo::compute_descent_step_size(
        const TVector &x,
        const TVector &delta_x,
        Problem &objFunc,
        const bool use_grad_norm,
        const double old_energy,
        const double starting_step_size)
    {
        double step_size = cppoptlib::Armijo<Problem, 1>::linesearch(x, delta_x, objFunc, starting_step_size);
        // this ensures no collisions and decrease in energy
        return after_check.compute_descent_step_size(x, delta_x, objFunc, use_grad_norm, old_energy, step_size);
    }

} // namespace polysolve::nonlinear::line_search

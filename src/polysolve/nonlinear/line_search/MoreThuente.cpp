#include "MoreThuente.hpp"

#include <cppoptlib/linesearch/morethuente.h>

namespace polysolve::nonlinear::line_search
{
    MoreThuente::MoreThuente(const json &params, spdlog::logger &logger)
        : Superclass(params, logger), after_check(params, logger)
    {
    }

    double MoreThuente::compute_descent_step_size(
        const TVector &x,
        const TVector &delta_x,
        Problem &objFunc,
        const bool use_grad_norm,
        const double old_energy,
        const TVector &old_grad,
        const double starting_step_size)
    {
        const double tmp = cppoptlib::MoreThuente<Problem, 1>::linesearch(x, delta_x, objFunc, starting_step_size);
        // this ensures no collisions and decrease in energy
        return after_check.compute_descent_step_size(x, delta_x, objFunc, use_grad_norm, old_energy, old_grad, std::min(tmp, starting_step_size));
    }
} // namespace polysolve::nonlinear::line_search

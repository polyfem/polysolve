#include "GradientDescent.hpp"

namespace polysolve::nonlinear
{

    GradientDescent::GradientDescent(const json &solver_params_,
                                     const double characteristic_length,
                                     spdlog::logger &logger)
        : Superclass(solver_params_, characteristic_length, logger)
    {
    }

    bool GradientDescent::compute_update_direction(
        Problem &objFunc,
        const TVector &x,
        const TVector &grad,
        TVector &direction)
    {
        direction = -grad;

        return true;
    }

} // namespace polysolve::nonlinear

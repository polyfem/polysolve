#include "StochasticGradientDescent.hpp"

namespace polysolve::nonlinear
{

    StochasticGradientDescent::StochasticGradientDescent(const json &solver_params_,
                                                         const double characteristic_length,
                                                         spdlog::logger &logger)
        : Superclass(solver_params_, characteristic_length, logger)
    {
        erase_component_probability_ = solver_params_["SGD"]["erase_component_probability"];
    }

    bool StochasticGradientDescent::compute_update_direction(
        Problem &objFunc,
        const TVector &x,
        const TVector &grad,
        TVector &direction)
    {
        direction = -grad;

        for (int i = 0; i < direction.size(); ++i)
            direction(i) *= (((rand() % 1000) / 1000.) < erase_component_probability_) ? 0. : 1.;

        return true;
    }

} // namespace polysolve::nonlinear

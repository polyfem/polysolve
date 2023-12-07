#include "GradientDescent.hpp"

namespace polysolve::nonlinear
{

    GradientDescent::GradientDescent(const json &solver_params_,
                                     const bool is_stochastic,
                                     const double characteristic_length,
                                     spdlog::logger &logger)
        : Superclass(solver_params_, characteristic_length, logger), is_stochastic_(is_stochastic)
    {
        if (is_stochastic_)
            erase_component_probability_ = solver_params_["SGD"]["erase_component_probability"];
    }

    bool GradientDescent::compute_update_direction(
        Problem &objFunc,
        const TVector &x,
        const TVector &grad,
        TVector &direction)
    {
        direction = -grad;

        if (is_stochastic_)
        {
            Eigen::VectorXd mask = (Eigen::VectorXd::Random(direction.size()).array() + 1.) / 2.;
            for (int i = 0; i < direction.size(); ++i)
                direction(i) *= (mask(i) < erase_component_probability_) ? 0. : 1.;
        }

        return true;
    }

} // namespace polysolve::nonlinear

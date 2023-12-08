#include "RobustArmijo.hpp"

#include <polysolve/Utils.hpp>

#include <spdlog/spdlog.h>

namespace polysolve::nonlinear::line_search
{
    RobustArmijo::RobustArmijo(const json &params, spdlog::logger &logger)
        : Superclass(params, logger)
    {
        delta_relative_tolerance = params.at("delta_relative_tolerance");
    }

    bool RobustArmijo::criteria(
        const TVector &delta_x,
        const double old_energy,
        const TVector &old_grad,
        const double new_energy,
        const TVector &new_grad,
        const double step_size) const
    {
        if (new_energy <= old_energy + step_size * this->armijo_criteria) // Try Armijo first
            return true;

        if (std::abs(new_energy - old_energy) <= delta_relative_tolerance * std::abs(old_energy))
        {
            // TODO: Compute new_grad here as needed
            // objFunc.gradient(new_x, new_grad);
            assert(old_grad.size() == new_grad.size());

            const double deltaE_approx = step_size / 2 * delta_x.dot(new_grad + old_grad);
            const double abs_eps_est = step_size / 2 * std::abs(delta_x.dot(new_grad - old_grad));

            if (deltaE_approx + abs_eps_est <= step_size * this->armijo_criteria)
                return true;
        }

        return false;
    }

}; // namespace polysolve::nonlinear::line_search

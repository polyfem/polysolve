

#include "Armijo.hpp"

namespace polysolve::nonlinear::line_search
{
    Armijo::Armijo(const json &params, spdlog::logger &logger)
        : Superclass(params, logger)
    {
        c = params["line_search"]["Armijo"]["c"];
    }

    void Armijo::init_compute_descent_step_size(
        const TVector &delta_x,
        const TVector &old_grad)
    {
        armijo_criteria = c * delta_x.dot(old_grad);
        assert(armijo_criteria <= 0);
    }

    bool Armijo::criteria(
        Problem &objFunc,
        const TVector &delta_x,
        const TVector &new_x,
        const double old_energy,
        const TVector &old_grad,
        const double new_energy,
        const double step_size) const
    {
        // TODO: Use use_grad_norm
        return new_energy <= old_energy + step_size * armijo_criteria;
    }

}; // namespace polysolve::nonlinear::line_search

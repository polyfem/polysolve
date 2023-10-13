#include "GradientDescent.hpp"

namespace polysolve::nonlinear
{

    GradientDescent::GradientDescent(const json &solver_params_,
                                     const double characteristic_length,
                                     spdlog::logger &logger)
        : Superclass(solver_params_, characteristic_length, logger)
    {
    }

    std::string GradientDescent::descent_strategy_name(int descent_strategy_) const
    {
        switch (descent_strategy_)
        {
        case 1:
            return "gradient descent";
        default:
            throw std::invalid_argument("invalid descent strategy");
        }
    }

    void GradientDescent::increase_descent_strategy()
    {
        assert(this->descent_strategy <= 1);
    }

    void GradientDescent::reset(const int ndof)
    {
        Superclass::reset(ndof);
        this->descent_strategy = 1;
    }

    void GradientDescent::compute_update_direction(
        Problem &objFunc,
        const TVector &x,
        const TVector &grad,
        TVector &direction)
    {
        direction = -grad;
    }

} // namespace polysolve::nonlinear

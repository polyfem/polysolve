#include "Newton.hpp"

namespace polysolve::nonlinear
{

    Newton::Newton(
        const json &solver_params,
        const json &linear_solver_params,
        const double dt,
        const double characteristic_length,
        spdlog::logger &logger)
        : Superclass(solver_params, dt, characteristic_length, logger)
    {
        force_psd_projection = solver_params["force_psd_projection"];
    }

    // =======================================================================

    std::string Newton::descent_strategy_name(int descent_strategy) const
    {
        switch (descent_strategy)
        {
        case 0:
            return "Newton";
        case 1:
            if (reg_weight == 0)
                return "projected Newton";
            return fmt::format("projected Newton w/ regularization weight={}", reg_weight);
        case 2:
            return "gradient descent";
        default:
            throw std::invalid_argument("invalid descent strategy");
        }
    }

    // =======================================================================

    void Newton::increase_descent_strategy()
    {
        if (this->descent_strategy == 1 && reg_weight < reg_weight_max)
            reg_weight = std::clamp(reg_weight_inc * reg_weight, reg_weight_min, reg_weight_max);
        else
            this->descent_strategy++;
        assert(this->descent_strategy <= 2);
    }

    // =======================================================================

    void Newton::reset(const int ndof)
    {
        Superclass::reset(ndof);
        reg_weight = 0;
        this->descent_strategy = 0;
        internal_solver_info = json::array();
    }

    // =======================================================================

    bool Newton::compute_update_direction(
        Problem &objFunc,
        const TVector &x,
        const TVector &grad,
        TVector &direction)
    {
        if (this->descent_strategy == 2)
        {
            direction = -grad;
            return true;
        }

        if (this->descent_strategy == 1)
            objFunc.set_project_to_psd(true);
        else if (this->descent_strategy == 0)
            objFunc.set_project_to_psd(false);
        else
            assert(false);

        const double residual = solve_linear_system(objFunc, x, grad, direction);

        if (std::isnan(residual))
            // solve_linear_system will increase descent_strategy if needed
            return compute_update_direction(objFunc, x, grad, direction);

        if (!check_direction(residual, grad, direction))
            // check_direction will increase descent_strategy if needed
            return compute_update_direction(objFunc, x, grad, direction);

        reg_weight /= reg_weight_dec;
        if (reg_weight < reg_weight_min)
            reg_weight = 0;

        return true;
    }

    // =======================================================================

    bool Newton::check_direction(
        const double residual, const TVector &grad, const TVector &direction)
    {
        // gradient descent, check descent direction
        if (std::isnan(residual) || residual > std::max(1e-8 * grad.norm(), 1e-5) * characteristic_length)
        {
            increase_descent_strategy();

            m_logger.log(
                log_level(),
                "[{}] large (or nan) linear solve residual {} (||∇f||={}); reverting to {}",
                name(), residual, grad.norm(), this->descent_strategy_name());

            return false;
        }
        else
        {
            m_logger.trace("linear solve residual {}", residual);
        }

        // do this check here because we need to repeat the solve without resetting reg_weight
        if (grad.norm() != 0 && grad.dot(direction) >= 0)
        {
            increase_descent_strategy();
            m_logger.log(
                log_level(), "[{}] direction is not a descent direction (‖g‖={:g}; ‖Δx‖={:g}; Δx⋅g={:g}≥0); reverting to {}",
                name(), grad.norm(), direction.norm(), direction.dot(grad), descent_strategy_name());
            return false;
        }

        return true;
    }

    // =======================================================================

    void Newton::update_solver_info(const double energy)
    {
        Superclass::update_solver_info(energy);
        this->solver_info["internal_solver"] = internal_solver_info;
    }

    // =======================================================================

} // namespace polysolve::nonlinear

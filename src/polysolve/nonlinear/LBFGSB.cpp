#include "LBFGSB.hpp"

#include <LBFGSpp/Cauchy.h>
#include <LBFGSpp/SubspaceMin.h>

namespace polysolve::nonlinear
{
    LBFGSB::LBFGSB(const json &solver_params,
                   const double characteristic_length,
                   spdlog::logger &logger)
        : Superclass(solver_params, characteristic_length, logger)
    {
        m_history_size = solver_params.value("history_size", 6);
    }

    std::string LBFGSB::descent_strategy_name(int descent_strategy) const
    {
        switch (descent_strategy)
        {
        case 1:
            return "L-BFGS-B";
        case 2:
            return "gradient descent";
        default:
            throw std::invalid_argument("invalid descent strategy");
        }
    }

    void LBFGSB::increase_descent_strategy()
    {
        this->descent_strategy++;

        m_bfgs.reset(m_prev_x.size(), m_history_size);
    }

    void LBFGSB::reset(const int ndof)
    {
        Superclass::reset(ndof);

        reset_history(ndof);
    }

    void LBFGSB::reset_history(const int ndof)
    {
        m_bfgs.reset(ndof, m_history_size);
        m_prev_x.resize(ndof);
        m_prev_grad.resize(ndof);

        // Use gradient descent for first iteration
        this->descent_strategy = 1;
    }

    void LBFGSB::compute_update_direction(
        Problem &objFunc,
        const TVector &x,
        const TVector &grad,
        TVector &direction)
    {
        TVector lower_bound = Superclass::get_lower_bound(x);
        TVector upper_bound = Superclass::get_upper_bound(x);

        // for (int i = 0; i < x.size(); i++)
        // 	if (lower_bound(i) > x(i) || upper_bound(i) < x(i))
        // 	{
        // 		m_logger.error("Entry {} value {} exceeds bound [{}, {}]!", i, x(i), lower_bound(i), upper_bound(i));
        // 		log_and_throw_error(m_logger, "Variable bound exceeded!");
        // 	}

        TVector cauchy_point(x.size()), vecc;
        std::vector<int> newact_set, fv_set;
        if (this->descent_strategy == 2)
        {
            // Use gradient descent in the first iteration or if the previous iteration failed
            // direction = -grad;

            LBFGSpp::Cauchy<Scalar>::get_cauchy_point(m_bfgs, x, grad, lower_bound, upper_bound, cauchy_point, vecc, newact_set, fv_set);

            direction = cauchy_point - x;
        }
        else
        {
            // Update s and y
            // s_{i+1} = x_{i+1} - x_i
            // y_{i+1} = g_{i+1} - g_i
            if ((x - m_prev_x).dot(grad - m_prev_grad) > 1e-9 * (grad - m_prev_grad).squaredNorm())
                m_bfgs.add_correction(x - m_prev_x, grad - m_prev_grad);

            // Recursive formula to compute d = -H * g
            // m_bfgs.apply_Hv(grad, -Scalar(1), direction);

            LBFGSpp::Cauchy<Scalar>::get_cauchy_point(m_bfgs, x, grad, lower_bound, upper_bound, cauchy_point, vecc, newact_set, fv_set);

            LBFGSpp::SubspaceMin<Scalar>::subspace_minimize(m_bfgs, x, cauchy_point, grad, lower_bound, upper_bound,
                                                            vecc, newact_set, fv_set, /*Maximum number of iterations*/ max_submin, direction);
        }

        // if (x.size() < 100)
        // {
        //     m_logger.debug("x: {}", x.transpose());
        //     m_logger.debug("grad: {}", grad.transpose());
        //     m_logger.debug("direc: {}", direction.transpose());
        // }

        // maybe remove me?
        if (std::isnan(direction.squaredNorm()))
        {
            reset_history(x.size());
            increase_descent_strategy();
            m_logger.log(
                this->descent_strategy == 2 ? spdlog::level::warn : spdlog::level::debug,
                "nan in direction {} (||∇f||={}); reverting to {}",
                direction.dot(grad), this->descent_strategy_name());
            return compute_update_direction(objFunc, x, grad, direction);
        }
        else if (grad.squaredNorm() != 0 && this->descent_strategy == 1 && direction.dot(grad) > -grad.norm() * direction.norm() * 1e-6)
        {
            reset_history(x.size());
            increase_descent_strategy();
            m_logger.log(
                this->descent_strategy == 2 ? spdlog::level::warn : spdlog::level::debug,
                "L-BFGS direction is not a descent direction (Δx⋅g={}); reverting to {}",
                direction.dot(grad), this->descent_strategy_name());
            return compute_update_direction(objFunc, x, grad, direction);
        }

        m_prev_x = x;
        m_prev_grad = grad;
    }
} // namespace polysolve::nonlinear

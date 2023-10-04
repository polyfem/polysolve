// L-BFGS solver (Using the LBFGSpp under MIT License).

#include "BFGS.hpp"

namespace polysolve::nonlinear
{

    BFGS::BFGS(const json &solver_params,
               const json &linear_solver_params,
               const double dt, const double characteristic_length,
               spdlog::logger &logger)
        : Superclass(solver_params, dt, characteristic_length, logger)
    {
        linear_solver = polysolve::linear::Solver::create(
            linear_solver_params["solver"], linear_solver_params["precond"]);
        linear_solver->setParameters(linear_solver_params);
    }

    std::string BFGS::descent_strategy_name(int descent_strategy) const
    {
        switch (descent_strategy)
        {
        case 1:
            return "BFGS";
        case 2:
            return "gradient descent";
        default:
            throw std::invalid_argument("invalid descent strategy");
        }
    }

    void BFGS::increase_descent_strategy()
    {
        if (this->descent_strategy == 1)
            this->descent_strategy++;

        assert(this->descent_strategy <= 2);
    }

    void BFGS::reset(const int ndof)
    {
        Superclass::reset(ndof);
        reset_history(ndof);
    }

    void BFGS::reset_history(const int ndof)
    {
        m_prev_x.resize(ndof);
        m_prev_grad.resize(ndof);

        hess.setIdentity(ndof, ndof);

        // Use gradient descent for first iteration
        this->descent_strategy = 2;
    }

    bool BFGS::compute_update_direction(
        Problem &objFunc,
        const TVector &x,
        const TVector &grad,
        TVector &direction)
    {
        if (this->descent_strategy == 2)
        {
            direction = -grad;
        }
        else
        {
            try
            {
                linear_solver->analyzePattern_dense(hess, hess.rows());
                linear_solver->factorize_dense(hess);
                linear_solver->solve(-grad, direction);
            }
            catch (const std::runtime_error &err)
            {
                increase_descent_strategy();

                // warn if using gradient descent
                m_logger.warn("Unable to factorize Hessian: \"{}\"; reverting to {}",
                              err.what(), this->descent_strategy_name());
                return compute_update_direction(objFunc, x, grad, direction);
            }

            TVector y = grad - m_prev_grad;
            TVector s = x - m_prev_x;

            double y_s = y.dot(s);
            TVector Bs = hess * s;
            double sBs = s.transpose() * Bs;

            hess += (y * y.transpose()) / y_s - (Bs * Bs.transpose()) / sBs;
        }

        m_prev_x = x;
        m_prev_grad = grad;

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
        else if (grad.squaredNorm() != 0 && direction.dot(grad) >= 0)
        {
            reset_history(x.size());
            increase_descent_strategy();
            m_logger.log(
                this->descent_strategy == 2 ? spdlog::level::warn : spdlog::level::debug,
                "BFGS direction is not a descent direction (Δx⋅g={}≥0); reverting to {}",
                direction.dot(grad), this->descent_strategy_name());
            return compute_update_direction(objFunc, x, grad, direction);
        }

        return true;
    }
} // namespace polysolve::nonlinear

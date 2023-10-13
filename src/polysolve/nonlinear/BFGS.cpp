// L-BFGS solver (Using the LBFGSpp under MIT License).

#include "BFGS.hpp"

namespace polysolve::nonlinear
{

    BFGS::BFGS(const json &solver_params,
               const json &linear_solver_params,
               const double characteristic_length,
               spdlog::logger &logger)
        : Superclass(solver_params, characteristic_length, logger)
    {
        linear_solver = polysolve::linear::Solver::create(
            linear_solver_params["solver"], linear_solver_params["precond"]);
        linear_solver->set_parameters(linear_solver_params);
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

        assert(m_prev_x.size() > 0);
        reset_history(m_prev_x.size());
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

    void BFGS::compute_update_direction(
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
                linear_solver->analyze_pattern_dense(hess, hess.rows());
                linear_solver->factorize_dense(hess);
                linear_solver->solve(-grad, direction);
            }
            catch (const std::runtime_error &err)
            {
                increase_descent_strategy();

                // warn if using gradient descent
                m_logger.warn("Unable to factorize Hessian: \"{}\"; reverting to {}",
                              err.what(), this->descent_strategy_name());
                compute_update_direction(objFunc, x, grad, direction);
                return;
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
    }
} // namespace polysolve::nonlinear

// L-BFGS solver (Using the LBFGSpp under MIT License).

#include "LBFGS.hpp"

namespace polysolve::nonlinear
{
    LBFGS::LBFGS(const json &solver_params,
                 const double characteristic_length,
                 spdlog::logger &logger)
        : Superclass(solver_params,
                     characteristic_length,
                     logger)
    {
        m_history_size = solver_params["LBFGS"]["history_size"];
    }

    std::string LBFGS::descent_strategy_name(int descent_strategy) const
    {
        switch (descent_strategy)
        {
        case Solver::LBFGS_STRATEGY:
            return "L-BFGS";
        case Solver::GRADIENT_DESCENT_STRATEGY:
            return "gradient descent";
        default:
            throw std::invalid_argument("invalid descent strategy");
        }
    }

    void LBFGS::increase_descent_strategy()
    {
        if (this->descent_strategy == Solver::LBFGS_STRATEGY)
            this->descent_strategy++;

        m_bfgs.reset(m_prev_x.size(), m_history_size);

        assert(this->descent_strategy <= Solver::MAX_STRATEGY);
    }

    void LBFGS::reset(const int ndof)
    {
        Superclass::reset(ndof);

        m_bfgs.reset(ndof, m_history_size);

        // Use gradient descent for first iteration
        this->descent_strategy = Solver::GRADIENT_DESCENT_STRATEGY;
    }

    void LBFGS::compute_update_direction(
        Problem &objFunc,
        const TVector &x,
        const TVector &grad,
        TVector &direction)
    {
        if (this->descent_strategy == Solver::GRADIENT_DESCENT_STRATEGY)
        {
            // Use gradient descent in the first iteration or if the previous iteration failed
            direction = -grad;
        }
        else
        {
            // Update s and y
            // s_{i+1} = x_{i+1} - x_i
            // y_{i+1} = g_{i+1} - g_i
            assert(m_prev_x.size() == x.size());
            assert(m_prev_grad.size() == grad.size());
            m_bfgs.add_correction(x - m_prev_x, grad - m_prev_grad);

            // Recursive formula to compute d = -H * g
            m_bfgs.apply_Hv(grad, -Scalar(1), direction);
        }

        m_prev_x = x;
        m_prev_grad = grad;
    }
} // namespace polysolve::nonlinear

// L-BFGS solver (Using the LBFGSpp under MIT License).

#include "LBFGS.hpp"

#include <polysolve/Utils.hpp>

namespace polysolve::nonlinear
{
    LBFGS::LBFGS(const json &solver_params,
                 const double characteristic_length,
                 spdlog::logger &logger)
        : Superclass(solver_params,
                     characteristic_length,
                     logger)
    {
        m_history_size = extract_param("L-BFGS", "history_size", solver_params);
        if (m_history_size <= 0)
            log_and_throw_error(logger, "L-BFGS history_size must be >=1, instead got {}", m_history_size);
    }

    void LBFGS::reset(const int ndof)
    {
        Superclass::reset(ndof);

        m_bfgs.reset(ndof, m_history_size);
        m_prev_x.resize(0);
    }

    bool LBFGS::compute_update_direction(
        Problem &objFunc,
        const TVector &x,
        const TVector &grad,
        TVector &direction)
    {
        if (m_prev_x.size() == 0)
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

        return true;
    }
} // namespace polysolve::nonlinear

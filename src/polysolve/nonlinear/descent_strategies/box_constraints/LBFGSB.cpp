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
        m_history_size = solver_params["LBFGSB"]["history_size"];

        if (m_history_size <= 0)
            log_and_throw_error(logger, "LBFGSB history_size must be >=1, instead got {}", m_history_size);
    }

    void LBFGSB::reset(const int ndof)
    {
        Superclass::reset(ndof);

        reset_history(ndof);
    }

    void LBFGSB::reset_history(const int ndof)
    {
        m_bfgs.reset(ndof, m_history_size);
        m_prev_x.resize(0);
        m_prev_grad.resize(ndof);
    }

    bool LBFGSB::compute_boxed_update_direction(
        Problem &objFunc,
        const TVector &x,
        const TVector &grad,
        const TVector &lower_bound,
        const TVector &upper_bound,
        TVector &direction)
    {

        // for (int i = 0; i < x.size(); i++)
        // 	if (lower_bound(i) > x(i) || upper_bound(i) < x(i))
        // 	{
        // 		m_logger.error("Entry {} value {} exceeds bound [{}, {}]!", i, x(i), lower_bound(i), upper_bound(i));
        // 		log_and_throw_error(m_logger, "Variable bound exceeded!");
        // 	}

        TVector cauchy_point(x.size()), vecc;
        std::vector<int> newact_set, fv_set;
        if (m_prev_x.size() == 0)
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

        m_prev_x = x;
        m_prev_grad = grad;

        return true;
    }
} // namespace polysolve::nonlinear

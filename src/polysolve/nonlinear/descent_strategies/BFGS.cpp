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
        linear_solver = polysolve::linear::Solver::create(linear_solver_params, logger);
        if (!linear_solver->is_dense())
            log_and_throw_error(logger, "BFGS linear solver must be dense, instead got {}", linear_solver->name());
    }

    void BFGS::reset(const int ndof)
    {
        Superclass::reset(ndof);
        reset_history(ndof);
    }

    void BFGS::reset_history(const int ndof)
    {
        m_prev_x.resize(0);
        m_prev_grad.resize(ndof);

        hess.setIdentity(ndof, ndof);
    }

    bool BFGS::compute_update_direction(
        Problem &objFunc,
        const TVector &x,
        const TVector &grad,
        TVector &direction)
    {
        if (m_prev_x.size() == 0)
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
                m_logger.debug("Unable to factorize Hessian: \"{}\";", err.what());
                return false;
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

        return true;
    }
} // namespace polysolve::nonlinear

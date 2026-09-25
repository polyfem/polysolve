#include "Backtracking.hpp"

#include <polysolve/Utils.hpp>

#include <spdlog/spdlog.h>

#include <cmath>

namespace polysolve::nonlinear::line_search
{

    Backtracking::Backtracking(const json &params, spdlog::logger &logger)
        : Superclass(params, logger)
    {
    }

    double Backtracking::compute_descent_step_size(
        const TVector &x,
        const TVector &delta_x,
        Problem &objFunc,
        const bool use_grad_norm,
        const double old_energy,
        const TVector &old_grad,
        const double starting_step_size)
    {
        double step_size = starting_step_size;

        // Local copies: a rejection by is_step_acceptable() may change the
        // problem, after which these are re-evaluated at x.
        double energy0 = old_energy;
        TVector grad0 = old_grad;

        init_compute_descent_step_size(delta_x, grad0);

        for (; step_size > current_min_step_size() && cur_iter < current_max_step_size_iter(); step_size *= step_ratio, ++cur_iter)
        {
            const TVector new_x = step_along(x, step_size, delta_x);

            try
            {
                POLYSOLVE_SCOPED_STOPWATCH("solution changed - constraint set update in LS", constraint_set_update_time, m_logger);
                objFunc.solution_changed(new_x);
            }
            catch (const std::runtime_error &e)
            {
                m_logger.warn("Failed to take step due to \"{}\", reduce step size...", e.what());
                continue;
            }

            if (!objFunc.is_step_valid(x, new_x))
            {
                continue;
            }

            const double new_energy = objFunc(new_x);

            if (!std::isfinite(new_energy))
            {
                continue;
            }

            m_logger.trace("ls it: {} {}: {}, {}: {}", cur_iter, log::delta("E"), new_energy - energy0, log::delta("x"), objFunc.step_norm(delta_x, norm_type));

            if (!criteria(delta_x, objFunc, use_grad_norm, energy0, grad0, new_x, new_energy, step_size))
                continue;

            if (objFunc.is_step_acceptable(x, new_x))
                break; // found a good step size

            // Rejected. The problem may have changed its discretization, so
            // re-evaluate x before comparing further trial points against it.
            m_logger.debug("Step of size {} rejected by the problem; re-evaluating and backtracking", step_size);
            problem_changed = true;
            objFunc.solution_changed(x);
            energy0 = objFunc(x);
            objFunc.gradient(x, grad0);
            if (!std::isfinite(energy0) || !grad0.array().isFinite().all() || delta_x.dot(grad0) >= 0)
            {
                // delta_x was computed for the previous problem and need not
                // descend on the new one. Fail through the caller's usual path
                // (which restores x and ends the line search) so the solver
                // picks a new direction.
                m_logger.debug("Search direction is no longer a descent direction after the problem changed");
                return 0;
            }
            init_compute_descent_step_size(delta_x, grad0);
        }

        return step_size;
    }

    bool Backtracking::criteria(
        const TVector &delta_x,
        Problem &objFunc,
        const bool use_grad_norm,
        const double old_energy,
        const TVector &old_grad,
        const TVector &new_x,
        const double new_energy,
        const double step_size) const
    {
        if (use_grad_norm)
        {
            TVector new_grad;
            objFunc.gradient(new_x, new_grad);
            return objFunc.grad_norm(new_grad, norm_type) < objFunc.grad_norm(old_grad, norm_type); // TODO cache old grad norm
        }
        return new_energy < old_energy;
    }

} // namespace polysolve::nonlinear::line_search

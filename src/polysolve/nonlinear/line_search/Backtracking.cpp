#include "Backtracking.hpp"

#include <polysolve/Utils.hpp>

#include <spdlog/spdlog.h>

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

        init_compute_descent_step_size(delta_x, old_grad);

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

            m_logger.trace("ls it: {} {}: {}, {}: {}", cur_iter, log::delta("E"), new_energy - old_energy, log::delta("x"), objFunc.step_norm(delta_x, norm_type));

            if (!criteria(delta_x, objFunc, use_grad_norm, old_energy, old_grad, new_x, new_energy, step_size))
                continue;

            if (objFunc.is_step_acceptable(x, new_x))
                break; // found a good step size

            // Rejected. The problem may have changed its discretization, and
            // delta_x was computed for the old one, so do not keep searching
            // along it: fail through the caller's usual path (which restores
            // x and ends the line search), and the solver computes a new
            // direction on the problem as it is now. Each line search thus
            // works on a single, fixed problem.
            m_logger.debug("Step of size {} rejected by the problem; ending the line search", step_size);
            problem_changed = true;
            return 0;
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
            // TODO cache old grad norm
            if (objFunc.grad_norm(new_grad, norm_type)
                < objFunc.grad_norm(old_grad, norm_type))
                return true;
            // Optionally also accept a step that decreases the energy: the
            // gradient-only test can stall near convergence, where the
            // gradient plateaus but the energy is still descending.
            return accept_energy_decrease_in_grad_mode && new_energy < old_energy;
        }
        return new_energy < old_energy;
    }

} // namespace polysolve::nonlinear::line_search

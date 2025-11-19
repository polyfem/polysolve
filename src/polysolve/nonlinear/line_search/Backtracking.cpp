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
            const TVector new_x = x + step_size * delta_x;

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

            m_logger.trace("ls it: {} {}: {}, {}: {}", cur_iter, log::delta("E"), new_energy - old_energy, log::delta("x"), delta_x.norm());

            if (criteria(delta_x, objFunc, use_grad_norm, old_energy, old_grad, new_x, new_energy, step_size))
            {
                break; // found a good step size
            }

            if (try_interpolating_step)
            {
                double next_step_size = step_ratio * step_size;
                double half_step_size = 0.5 * (step_ratio * step_size + step_size);

                TVector curr_step_grad, next_step_grad, half_step_grad;
                objFunc.gradient(new_x, curr_step_grad);
                objFunc.gradient(x + next_step_size * delta_x, next_step_grad);
                objFunc.gradient(x + half_step_size * delta_x, half_step_grad);
                
                double curr_step_grad_norm = objFunc.grad_norm(curr_step_grad, "L2");
                double next_step_grad_norm = objFunc.grad_norm(next_step_grad, "L2");
                double half_step_grad_norm = objFunc.grad_norm(half_step_grad, "L2");
                
                double m = (curr_step_grad_norm - next_step_grad_norm) / (step_size - next_step_size);
                double b = curr_step_grad_norm - m * step_size;

                double interpolated_half_step_grad_norm = m * half_step_size + b;

                if (abs((half_step_grad_norm - interpolated_half_step_grad_norm) / half_step_grad_norm) < 1e-5)
                {
                    step_size = -1 * b / m;
                    break; 
                }

            }
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
            return new_grad.norm() < old_grad.norm(); // TODO: cache old_grad.norm()
        }
        return new_energy < old_energy;
    }

} // namespace polysolve::nonlinear::line_search

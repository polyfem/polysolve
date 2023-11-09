#include "Backtracking.hpp"

#include <polysolve/Utils.hpp>

#include <spdlog/spdlog.h>

#include <cfenv>

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
        const double starting_step_size)
    {
        double step_size = starting_step_size;

        TVector grad(x.rows());

        // Find step that reduces the energy
        double cur_energy = std::nan("");
        bool is_step_valid = false;
        while (step_size > current_min_step_size() && cur_iter < current_max_step_size_iter())
        {
            TVector new_x = x + step_size * delta_x;

            try
            {
                POLYSOLVE_SCOPED_STOPWATCH("solution changed - constraint set update in LS", this->constraint_set_update_time, m_logger);
                objFunc.solution_changed(new_x);
            }
            catch (const std::runtime_error &e)
            {
                m_logger.warn("Failed to take step due to \"{}\", reduce step size...", e.what());

                step_size *= step_ratio;
                this->cur_iter++;
                continue;
            }

            if (use_grad_norm)
            {
                objFunc.gradient(new_x, grad);
                cur_energy = grad.squaredNorm();
            }
            else
                cur_energy = objFunc.value(new_x);

            is_step_valid = objFunc.is_step_valid(x, new_x);

            m_logger.trace("ls it: {} delta: {} invalid: {} ", this->cur_iter, (cur_energy - old_energy), !is_step_valid);

            if (!std::isfinite(cur_energy) || cur_energy >= old_energy || !is_step_valid)
            {
                step_size *= step_ratio;
                // max_step_size should return a collision free step
                // assert(objFunc.is_step_collision_free(x, new_x));
            }
            else
            {
                break;
            }
            this->cur_iter++;
        }

        return step_size;
    }
} // namespace polysolve::nonlinear::line_search

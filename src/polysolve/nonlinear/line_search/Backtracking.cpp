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
        assert(!use_grad_norm);
        double step_size = starting_step_size;

        init_compute_descent_step_size(delta_x, old_grad);

        for (; step_size > current_min_step_size() && cur_iter < current_max_step_size_iter(); step_size *= step_ratio, ++cur_iter)
        {
            const TVector new_x = x + step_size * delta_x;

            try
            {
                POLYSOLVE_SCOPED_STOPWATCH("solution changed - constraint set update in LS", constraint_set_update_time, m_logger);
                objFunc.solution_changed(x);
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

            const double new_energy = objFunc.value(new_x);

            TVector new_grad; // TODO: only compute gradient if needed
            objFunc.gradient(new_x, new_grad);

            if (!std::isfinite(new_energy) || !std::isfinite(new_grad.norm()))
            {
                continue;
            }

            m_logger.trace("ls it: {} ΔE: {}", cur_iter, new_energy - old_energy);

            if (criteria(delta_x, old_energy, old_grad, new_energy, new_grad, step_size))
            {
                break; // found a good step size
            }
        }

        return step_size;
    }

    bool Backtracking::criteria(
        const TVector &delta_x,
        const double old_energy,
        const TVector &old_grad,
        const double new_energy,
        const TVector &new_grad,
        const double step_size) const
    {
        return new_energy < old_energy;
    }

} // namespace polysolve::nonlinear::line_search

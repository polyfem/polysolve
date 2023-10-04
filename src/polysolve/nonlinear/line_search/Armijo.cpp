

#include "Armijo.hpp"

#include <polysolve/nonlinear/Utils.hpp>

#include <spdlog/spdlog.h>

namespace polysolve::nonlinear::line_search
{
    Armijo::Armijo(spdlog::logger &logger)
        : Superclass(logger)
    {
        min_step_size = 1e-7;
        max_step_size_iter = 20; // std::numeric_limits<int>::max();

        default_init_step_size = 1.0;
        step_ratio = 0.5;
    }

    double Armijo::compute_descent_step_size(
        const TVector &x,
        const TVector &searchDir,
        Problem &objFunc,
        const bool use_grad_norm,
        const double old_energy_in,
        const double starting_step_size)
    {
        TVector grad(x.rows());
        double f_in = old_energy_in;
        double alpha = starting_step_size;

        bool valid;

        TVector x1 = x + alpha * searchDir;
        {
            POLYSOLVE_SCOPED_STOPWATCH("constraint set update in LS", constraint_set_update_time, m_logger);
            objFunc.solution_changed(x1);
        }

        objFunc.gradient(x, grad);

        double f = use_grad_norm ? grad.squaredNorm() : objFunc.value(x1);
        const double cache = c * grad.dot(searchDir);
        valid = objFunc.is_step_valid(x, x1);

        // max_step_size should return a collision free step
        // assert(objFunc.is_step_collision_free(x, x1));

        while ((std::isinf(f) || std::isnan(f) || f > f_in + alpha * cache || !valid) && alpha > min_step_size && cur_iter <= max_step_size_iter)
        {
            alpha *= step_ratio;
            x1 = x + alpha * searchDir;

            {
                POLYSOLVE_SCOPED_STOPWATCH("constraint set update in LS", constraint_set_update_time, m_logger);
                objFunc.solution_changed(x1);
            }

            if (use_grad_norm)
            {
                objFunc.gradient(x1, grad);
                f = grad.squaredNorm();
            }
            else
                f = objFunc.value(x1);

            valid = objFunc.is_step_valid(x, x1);

            m_logger.trace("ls it: {} f: {} (f_in + alpha * Cache): {} invalid: {} ", cur_iter, f, f_in + alpha * cache, !valid);

            cur_iter++;
        }

        return alpha;
    }

}; // namespace polysolve::nonlinear::line_search

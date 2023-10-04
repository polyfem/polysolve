#include "LineSearch.hpp"

#include "Armijo.hpp"
#include "Backtracking.hpp"
#include "CppOptArmijo.hpp"
#include "MoreThuente.hpp"

#include <polysolve/nonlinear/Utils.hpp>

#include <polysolve/Types.hpp>

#include <spdlog/spdlog.h>

#include <cfenv>

namespace polysolve::nonlinear::line_search
{
    std::shared_ptr<LineSearch> LineSearch::create(const std::string &name, spdlog::logger &logger)
    {
        if (name == "armijo" || name == "Armijo")
        {
            return std::make_shared<Armijo>(logger);
        }
        else if (name == "armijo_alt" || name == "ArmijoAlt")
        {
            return std::make_shared<CppOptArmijo>(logger);
        }
        else if (name == "bisection" || name == "Bisection")
        {
            logger.warn("{} linesearch was renamed to \"backtracking\"; using backtracking line-search", name);
            return std::make_shared<Backtracking>(logger);
        }
        else if (name == "backtracking" || name == "Backtracking")
        {
            return std::make_shared<Backtracking>(logger);
        }
        else if (name == "more_thuente" || name == "MoreThuente")
        {
            return std::make_shared<MoreThuente>(logger);
        }
        else if (name == "none")
        {
            return nullptr;
        }
        else
        {
            log_and_throw_error(logger, "Unknown line search {}!", name);
            return nullptr;
        }
    }

    std::vector<std::string> LineSearch::available_methods()
    {
        return {{"armijo",
                 "armijo_alt",
                 "backtracking",
                 "more_thuente",
                 "none"}};
    }

    LineSearch::LineSearch(spdlog::logger &logger)
        : m_logger(logger)
    {
    }

    double LineSearch::line_search(
        const TVector &x,
        const TVector &delta_x,
        Problem &objFunc)
    {
        // ----------------
        // Begin linesearch
        // ----------------
        double old_energy, step_size;
        {
            POLYSOLVE_SCOPED_STOPWATCH("LS begin", m_logger);

            cur_iter = 0;

            old_energy = objFunc.value(x);
            if (std::isnan(old_energy))
            {
                m_logger.error("Original energy in line search is nan!");
                return std::nan("");
            }

            step_size = default_init_step_size;

            // TODO: removed feature
            // objFunc.heuristic_max_step(delta_x);
        }
        const double initial_energy = old_energy;

        // ----------------------------
        // Find finite energy step size
        // ----------------------------
        {
            POLYSOLVE_SCOPED_STOPWATCH("LS compute finite energy step size", checking_for_nan_inf_time, m_logger);
            step_size = compute_nan_free_step_size(x, delta_x, objFunc, step_size, step_ratio);
            if (std::isnan(step_size))
                return std::nan("");
        }

        const double nan_free_step_size = step_size;
        // -----------------------------
        // Find collision-free step size
        // -----------------------------
        {
            POLYSOLVE_SCOPED_STOPWATCH("Line Search Begin - CCD broad-phase", broad_phase_ccd_time, m_logger);
            TVector new_x = x + step_size * delta_x;
            objFunc.line_search_begin(x, new_x);
        }

        {
            POLYSOLVE_SCOPED_STOPWATCH("CCD narrow-phase", ccd_time, m_logger);
            m_logger.trace("Performing narrow-phase CCD");
            step_size = compute_collision_free_step_size(x, delta_x, objFunc, step_size);
            if (std::isnan(step_size))
                return std::nan("");
        }

        const double collision_free_step_size = step_size;

        TVector grad(x.rows());
        objFunc.gradient(x, grad);

        if (grad.norm() < 1e-30)
            return 1;

        const bool use_grad_norm = grad.norm() < use_grad_norm_tol;
        if (use_grad_norm)
            old_energy = grad.squaredNorm();
        const double starting_step_size = step_size;

        // ----------------------
        // Find descent step size
        // ----------------------
        {
            POLYSOLVE_SCOPED_STOPWATCH("energy min in LS", classical_line_search_time, m_logger);
            step_size = compute_descent_step_size(x, delta_x, objFunc, use_grad_norm, old_energy, step_size);
            if (std::isnan(step_size))
            {
                // Superclass::save_sampled_values("failed-line-search-values.csv", x, delta_x, objFunc);
                return std::nan("");
            }
        }

        const double cur_energy = objFunc.value(x + step_size * delta_x);

        const double descent_step_size = step_size;

        if (cur_iter >= max_step_size_iter || step_size <= min_step_size)
        {
            m_logger.warn(
                "Line search failed to find descent step (f(x)={:g} f(x+αΔx)={:g} α_CCD={:g} α={:g}, ||Δx||={:g}  use_grad_norm={} iter={:d})",
                old_energy, cur_energy, starting_step_size,
                step_size, delta_x.norm(), use_grad_norm, cur_iter);
            objFunc.solution_changed(x);
#ifndef NDEBUG
            // tolerance for rounding error due to multithreading
            assert(abs(initial_energy - objFunc.value(x)) < 1e-15);
#endif
            objFunc.line_search_end();
            return std::nan("");
        }

        {
            POLYSOLVE_SCOPED_STOPWATCH("LS end", m_logger);
            objFunc.line_search_end();
        }

        m_logger.debug(
            "Line search finished (nan_free_step_size={} collision_free_step_size={} descent_step_size={} final_step_size={})",
            nan_free_step_size, collision_free_step_size, descent_step_size, step_size);

        return step_size;
    }

    double LineSearch::compute_nan_free_step_size(
        const TVector &x,
        const TVector &delta_x,
        Problem &objFunc,
        const double starting_step_size,
        const double rate)
    {
        double step_size = starting_step_size;
        TVector new_x = x + step_size * delta_x;

        // Find step that does not result in nan or infinite energy
        while (step_size > min_step_size && cur_iter < max_step_size_iter)
        {
            // Compute the new energy value without contacts
            const double energy = objFunc.value(new_x);
            const bool is_step_valid = objFunc.is_step_valid(x, new_x);

            if (!std::isfinite(energy) || !is_step_valid)
            {
                step_size *= rate;
                new_x = x + step_size * delta_x;
            }
            else
            {
                break;
            }
            cur_iter++;
        }

        if (cur_iter >= max_step_size_iter || step_size <= min_step_size)
        {
            m_logger.error(
                "Line search failed to find a valid finite energy step (cur_iter={:d} step_size={:g})!",
                cur_iter, step_size);
            return std::nan("");
        }

        return step_size;
    }

    double LineSearch::compute_collision_free_step_size(
        const TVector &x,
        const TVector &delta_x,
        Problem &objFunc,
        const double starting_step_size)
    {
        double step_size = starting_step_size;
        TVector new_x = x + step_size * delta_x;

        // Find step that is collision free
        double max_step_size = objFunc.max_step_size(x, new_x);
        if (max_step_size == 0)
        {
            m_logger.error("Line search failed because CCD produced a stepsize of zero!");
            objFunc.line_search_end();
            return std::nan("");
        }

        {
            // #pragma STDC FENV_ACCESS ON
            const int current_round = std::fegetround();
            std::fesetround(FE_DOWNWARD);
            step_size *= max_step_size; // TODO: check me if correct
            std::fesetround(current_round);
        }

        return step_size;
    }
} // namespace polysolve::nonlinear::line_search

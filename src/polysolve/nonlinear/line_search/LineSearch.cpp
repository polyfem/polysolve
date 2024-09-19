#include "LineSearch.hpp"

#include "Armijo.hpp"
#include "Backtracking.hpp"
#include "RobustArmijo.hpp"
#include "NoLineSearch.hpp"

#include <polysolve/Utils.hpp>

#include <polysolve/Types.hpp>

#if defined(SPDLOG_FMT_EXTERNAL)
#include <fmt/color.h>
#else
#include <spdlog/fmt/bundled/color.h>
#endif

#include <cfenv>

namespace polysolve::nonlinear::line_search
{
    static constexpr double NaN = std::numeric_limits<double>::quiet_NaN();

    std::shared_ptr<LineSearch> LineSearch::create(const json &params, spdlog::logger &logger)
    {
        const std::string name = params["line_search"]["method"];
        if (name == "Armijo")
        {
            return std::make_shared<Armijo>(params, logger);
        }
        else if (name == "RobustArmijo")
        {
            return std::make_shared<RobustArmijo>(params, logger);
        }
        else if (name == "Backtracking")
        {
            return std::make_shared<Backtracking>(params, logger);
        }
        else if (name == "None")
        {
            return std::make_shared<NoLineSearch>(params, logger);
        }
        else
        {
            log_and_throw_error(logger, "Unknown line search {}!", name);
            return nullptr;
        }
    }

    std::vector<std::string> LineSearch::available_methods()
    {
        return {{"Armijo", "RobustArmijo", "Backtracking", "None"}};
    }

    LineSearch::LineSearch(const json &params, spdlog::logger &logger)
        : m_logger(logger)
    {
        min_step_size = params["line_search"]["min_step_size"];
        max_step_size_iter = params["line_search"]["max_step_size_iter"];

        min_step_size_final = params["line_search"]["min_step_size_final"];
        max_step_size_iter_final = params["line_search"]["max_step_size_iter_final"];

        default_init_step_size = params["line_search"]["default_init_step_size"];
        step_ratio = params["line_search"]["step_ratio"];
    }

    double LineSearch::line_search(
        const TVector &x,
        const TVector &delta_x,
        Problem &objFunc)
    {
        // ----------------
        // Begin linesearch
        // ----------------
        double initial_energy, step_size;
        TVector initial_grad;
        {
            POLYSOLVE_SCOPED_STOPWATCH("LS begin", m_logger);

            cur_iter = 0;

            initial_energy = objFunc(x);
            if (std::isnan(initial_energy))
            {
                m_logger.error("Original energy in line search is nan!");
                return NaN;
            }

            objFunc.gradient(x, initial_grad);
            if (!initial_grad.array().isFinite().all())
            {
                m_logger.error("Original gradient in line search is nan!");
                return NaN;
            }

            step_size = default_init_step_size;

            // TODO: removed feature
            // objFunc.heuristic_max_step(delta_x);
        }

        // ----------------------------
        // Find finite energy step size
        // ----------------------------
        {
            POLYSOLVE_SCOPED_STOPWATCH("LS compute finite energy step size", checking_for_nan_inf_time, m_logger);
            step_size = compute_nan_free_step_size(x, delta_x, objFunc, step_size, step_ratio);
            if (std::isnan(step_size))
                return NaN;
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
            POLYSOLVE_SCOPED_STOPWATCH("CCD narrow-phase", narrow_phase_ccd_time, m_logger);
            m_logger.trace("Performing narrow-phase CCD");
            step_size = compute_max_step_size(x, delta_x, objFunc, step_size);
            if (std::isnan(step_size))
                return NaN;
        }

        const double collision_free_step_size = step_size;

        if (initial_grad.norm() < 1e-30)
            return step_size;

        // TODO: Fix this
        const bool use_grad_norm = initial_grad.norm() < use_grad_norm_tol;
        const double starting_step_size = step_size;

        // ----------------------
        // Find descent step size
        // ----------------------
        {
            POLYSOLVE_SCOPED_STOPWATCH("energy min in LS", classical_line_search_time, m_logger);
            step_size = compute_descent_step_size(x, delta_x, objFunc, use_grad_norm, initial_energy, initial_grad, step_size);
            if (std::isnan(step_size))
            {
                // Superclass::save_sampled_values("failed-line-search-values.csv", x, delta_x, objFunc);
                return NaN;
            }
        }

        const double cur_energy = objFunc(x + step_size * delta_x);

        const double descent_step_size = step_size;

        if (cur_iter >= current_max_step_size_iter() || step_size <= current_min_step_size())
        {
            m_logger.log(is_final_strategy ? spdlog::level::warn : spdlog::level::debug,
                         "Line search failed to find descent step (f(x)={:g} f(x+αΔx)={:g} α_CCD={:g} α={:g}, ||Δx||={:g}  use_grad_norm={} iter={:d})",
                         initial_energy, cur_energy, starting_step_size,
                         step_size, delta_x.norm(), use_grad_norm, cur_iter);
            objFunc.solution_changed(x);

            // tolerance for rounding error due to multithreading
            assert(abs(initial_energy - objFunc(x)) < 1e-15);

            objFunc.line_search_end();
            return NaN;
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
        while (step_size > current_min_step_size() && cur_iter < current_max_step_size_iter())
        {
            if (!objFunc.is_step_valid(x, new_x) || !std::isfinite(objFunc(new_x)))
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

        if (cur_iter >= current_max_step_size_iter() || step_size <= current_min_step_size())
        {
            m_logger.log(is_final_strategy ? spdlog::level::err : spdlog::level::debug,
                         "Line search failed to find a valid finite energy step (cur_iter={:d} step_size={:g})!",
                         cur_iter, step_size);
            return NaN;
        }

        return step_size;
    }

    // change to max_step_size
    double LineSearch::compute_max_step_size(
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
            m_logger.log(is_final_strategy ? spdlog::level::err : spdlog::level::debug,
                         "Line search failed because CCD produced a stepsize of zero!");
            objFunc.line_search_end();
            return NaN;
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

    void LineSearch::update_solver_info(json &solver_info, const double per_iteration)
    {
        solver_info["line_search_iterations"] = iterations();
        solver_info["time_checking_for_nan_inf"] = checking_for_nan_inf_time / per_iteration;
        solver_info["time_broad_phase_ccd"] = broad_phase_ccd_time / per_iteration;
        solver_info["time_ccd"] = narrow_phase_ccd_time / per_iteration;
        solver_info["time_classical_line_search"] = (classical_line_search_time - constraint_set_update_time) / per_iteration; // Remove double counting
        solver_info["time_line_search_constraint_set_update"] = constraint_set_update_time / per_iteration;
    }

    void LineSearch::reset_times()
    {
        checking_for_nan_inf_time = 0;
        broad_phase_ccd_time = 0;
        narrow_phase_ccd_time = 0;
        constraint_set_update_time = 0;
        classical_line_search_time = 0;
    }

    void LineSearch::log_times() const
    {
        m_logger.debug(
            "[{}][{}] constraint_set_update {:.2e}s, checking_for_nan_inf {:.2e}s, "
            "broad_phase_ccd {:.2e}s, narrow_phase_ccd {:.2e}s, classical_line_search {:.2e}s",
            fmt::format(fmt::fg(fmt::terminal_color::magenta), "timing"),
            name(), constraint_set_update_time, checking_for_nan_inf_time,
            broad_phase_ccd_time, narrow_phase_ccd_time, classical_line_search_time);
    }
} // namespace polysolve::nonlinear::line_search

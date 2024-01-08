#include "LineSearch.hpp"

#include "Armijo.hpp"
#include "Backtracking.hpp"
#include "RobustArmijo.hpp"
#include "CppOptArmijo.hpp"
#include "MoreThuente.hpp"
#include "NoLineSearch.hpp"

#include <polysolve/Utils.hpp>

#include <polysolve/Types.hpp>

#include <spdlog/spdlog.h>

#include <cfenv>

namespace polysolve::nonlinear::line_search
{
    static constexpr double NaN = std::numeric_limits<double>::quiet_NaN();

    std::shared_ptr<LineSearch> LineSearch::create(const json &params, spdlog::logger &logger)
    {
        const std::string name = params["line_search"]["method"];
        if (name == "armijo" || name == "Armijo")
        {
            return std::make_shared<Armijo>(params, logger);
        }
        else if (name == "armijo_alt" || name == "ArmijoAlt")
        {
            return std::make_shared<CppOptArmijo>(params, logger);
        }
        else if (name == "robust_armijo" || name == "RobustArmijo")
        {
            return std::make_shared<RobustArmijo>(params, logger);
        }
        else if (name == "bisection" || name == "Bisection")
        {
            logger.warn("{} linesearch was renamed to \"backtracking\"; using backtracking line-search", name);
            return std::make_shared<Backtracking>(params, logger);
        }
        else if (name == "backtracking" || name == "Backtracking")
        {
            return std::make_shared<Backtracking>(params, logger);
        }
        else if (name == "more_thuente" || name == "MoreThuente")
        {
            return std::make_shared<MoreThuente>(params, logger);
        }
        else if (name == "none" || name == "None")
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
        return {{"Armijo",
                 "ArmijoAlt",
                 "RobustArmijo",
                 "Backtracking",
                 "MoreThuente",
                 "None"}};
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

        use_directional_derivative = params["line_search"]["use_directional_derivative"];
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

            initial_energy = objFunc.value(x);
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
            POLYSOLVE_SCOPED_STOPWATCH("CCD narrow-phase", ccd_time, m_logger);
            m_logger.trace("Performing narrow-phase CCD");
            step_size = compute_collision_free_step_size(x, delta_x, objFunc, step_size);
            if (std::isnan(step_size))
                return NaN;
        }

        const double collision_free_step_size = step_size;

        if (initial_grad.norm() < 1e-30)
            return step_size;

        // TODO: Fix this
        const bool use_grad_norm = initial_grad.norm() < use_grad_norm_tol * abs(initial_energy);
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

        const double cur_energy = objFunc.value(x + step_size * delta_x);

        const double descent_step_size = step_size;

        if (cur_iter >= current_max_step_size_iter() || step_size <= current_min_step_size())
        {
            m_logger.log(is_final_strategy ? spdlog::level::warn : spdlog::level::debug,
                         "Line search failed to find descent step (f(x)={:g} f(x+αΔx)={:g} α_CCD={:g} α={:g}, ||Δx||={:g}  use_grad_norm={} iter={:d})",
                         initial_energy, cur_energy, starting_step_size,
                         step_size, delta_x.norm(), use_grad_norm, cur_iter);
            objFunc.solution_changed(x);

            // tolerance for rounding error due to multithreading
            assert(abs(initial_energy - objFunc.value(x)) < 1e-15);

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

        if (cur_iter >= current_max_step_size_iter() || step_size <= current_min_step_size())
        {
            m_logger.log(is_final_strategy ? spdlog::level::err : spdlog::level::debug,
                         "Line search failed to find a valid finite energy step (cur_iter={:d} step_size={:g})!",
                         cur_iter, step_size);
            return NaN;
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
} // namespace polysolve::nonlinear::line_search

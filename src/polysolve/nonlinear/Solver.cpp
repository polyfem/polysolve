
#include "Solver.hpp"

#include "Utils.hpp"

#include <spdlog/spdlog.h>
#include <spdlog/fmt/bundled/color.h>

#include <iomanip>

namespace polysolve::nonlinear
{
    Solver::Solver(const json &solver_params,
                   const double dt,
                   const double characteristic_length,
                   spdlog::logger &logger)
        : dt(dt), m_logger(logger), characteristic_length(characteristic_length)
    {
        TCriteria criteria = TCriteria::defaults();
        criteria.xDelta = solver_params["x_delta"];
        criteria.fDelta = solver_params["f_delta"];
        criteria.gradNorm = solver_params["grad_norm"];

        criteria.xDelta *= characteristic_length;
        criteria.fDelta *= characteristic_length;
        criteria.gradNorm *= characteristic_length;

        criteria.iterations = solver_params["max_iterations"];
        // criteria.condition = solver_params["condition"];
        this->setStopCriteria(criteria);

        normalize_gradient = solver_params["relative_gradient"];
        use_grad_norm_tol = solver_params["line_search"]["use_grad_norm_tol"];

        first_grad_norm_tol = solver_params["first_grad_norm_tol"];

        use_grad_norm_tol *= characteristic_length;
        first_grad_norm_tol *= characteristic_length;

        set_line_search(solver_params["line_search"]["method"]);
    }

    double Solver::compute_grad_norm(const Eigen::VectorXd &x, const Eigen::VectorXd &grad) const
    {
        return grad.norm();
    }

    void Solver::set_line_search(const std::string &line_search_name)
    {
        m_line_search = line_search::LineSearch::construct_line_search(line_search_name, m_logger);
        solver_info["line_search"] = line_search_name;
    }

    void Solver::minimize(Problem &objFunc, TVector &x)
    {
        constexpr double NaN = std::numeric_limits<double>::quiet_NaN();

        // ---------------------------
        // Initialize the minimization
        // ---------------------------

        reset(x.size()); // place for children to initialize their fields

        TVector grad = TVector::Zero(x.rows());
        TVector delta_x = TVector::Zero(x.rows());

        // double factor = 1e-5;

        // Set these to nan to indicate they have not been computed yet
        double old_energy = NaN;

        {
            POLYSOLVE_SCOPED_TIMER("constraint set update", constraint_set_update_time);
            objFunc.solution_changed(x);
        }

        {
            POLYSOLVE_SCOPED_TIMER("compute gradient", grad_time);
            objFunc.gradient(x, grad);
        }
        double first_grad_norm = compute_grad_norm(x, grad);
        if (std::isnan(first_grad_norm))
        {
            this->m_status = cppoptlib::Status::UserDefined;
            m_error_code = ErrorCode::NAN_ENCOUNTERED;
            log_and_throw_error(m_logger, "[{}] Initial gradient is nan; stopping", name());
            return;
        }
        this->m_current.xDelta = NaN; // we don't know the initial step size
        this->m_current.fDelta = old_energy;
        this->m_current.gradNorm = first_grad_norm / (normalize_gradient ? first_grad_norm : 1);

        const auto current_g_norm = this->m_stop.gradNorm;
        this->m_stop.gradNorm = first_grad_norm_tol;
        this->m_status = checkConvergence(this->m_stop, this->m_current);
        if (this->m_status != cppoptlib::Status::Continue)
        {
            POLYSOLVE_SCOPED_TIMER("compute objective function", obj_fun_time);
            this->m_current.fDelta = objFunc.value(x);
            m_logger.info(
                "[{}] Not even starting, {} (f={:g} ‖∇f‖={:g} g={:g} tol={:g})",
                name(), this->m_status, this->m_current.fDelta, first_grad_norm, this->m_current.gradNorm, this->m_stop.gradNorm);
            update_solver_info(this->m_current.fDelta);
            return;
        }
        this->m_stop.gradNorm = current_g_norm;

        // TODO
        // utils::Timer timer("non-linear solver", this->total_time);
        // timer.start();

        if (m_line_search)
            m_line_search->use_grad_norm_tol = use_grad_norm_tol;

        objFunc.save_to_file(x);

        m_logger.debug(
            "Starting {} solve f₀={:g} ‖∇f₀‖={:g} "
            "(stopping criteria: max_iters={:d} Δf={:g} ‖∇f‖={:g} ‖Δx‖={:g})",
            name(), objFunc.value(x), this->m_current.gradNorm, this->m_stop.iterations,
            this->m_stop.fDelta, this->m_stop.gradNorm, this->m_stop.xDelta);

        update_solver_info(objFunc.value(x));

        do
        {
            if (name() == "MMA")
            {
                POLYSOLVE_SCOPED_TIMER("constraint set update", constraint_set_update_time);
                objFunc.solution_changed(x);
            }

            double energy;
            {
                POLYSOLVE_SCOPED_TIMER("compute objective function", obj_fun_time);
                energy = objFunc.value(x);
            }
            if (!std::isfinite(energy))
            {
                this->m_status = cppoptlib::Status::UserDefined;
                m_error_code = ErrorCode::NAN_ENCOUNTERED;
                log_and_throw_error(m_logger, "[{}] f(x) is nan or inf; stopping", name());
                break;
            }

            {
                POLYSOLVE_SCOPED_TIMER("compute gradient", grad_time);
                objFunc.gradient(x, grad);
            }

            const double grad_norm = compute_grad_norm(x, grad);
            if (std::isnan(grad_norm))
            {
                this->m_status = cppoptlib::Status::UserDefined;
                m_error_code = ErrorCode::NAN_ENCOUNTERED;
                log_and_throw_error(m_logger, "[{}] Gradient is nan; stopping", name());
                break;
            }

            // ------------------------
            // Compute update direction
            // ------------------------

            // Compute a Δx to update the variable
            if (!compute_update_direction(objFunc, x, grad, delta_x))
            {
                this->m_status = cppoptlib::Status::Continue;
                continue;
            }

            if (name() != "MMA" && grad_norm != 0 && delta_x.dot(grad) >= 0)
            {
                increase_descent_strategy();
                m_logger.debug(
                    "[{}] direction is not a descent direction (‖Δx‖={:g}; ‖g‖={:g}; Δx⋅g={:g}≥0); reverting to {}",
                    name(), delta_x.norm(), grad.norm(), delta_x.dot(grad), descent_strategy_name());
                this->m_status = cppoptlib::Status::Continue;
                continue;
            }

            const double delta_x_norm = delta_x.norm();
            if (std::isnan(delta_x_norm))
            {
                increase_descent_strategy();
                this->m_status = cppoptlib::Status::UserDefined;
                m_logger.debug("[{}] Δx is nan; reverting to {}", name(), descent_strategy_name());
                this->m_status = cppoptlib::Status::Continue;
                continue;
            }

            // Use the maximum absolute displacement value divided by the timestep,
            // so the units are in velocity units.
            // TODO: Also divide by the world scale to make this criteria scale invariant.
            this->m_current.xDelta = descent_strategy == 2 ? NaN : (delta_x_norm / dt);
            this->m_current.fDelta = std::abs(old_energy - energy); // / std::abs(old_energy);
            // if normalize_gradient, use relative to first norm
            this->m_current.gradNorm = grad_norm / (normalize_gradient ? first_grad_norm : 1);

            this->m_status = checkConvergence(this->m_stop, this->m_current);

            old_energy = energy;

            // ---------------
            // Variable update
            // ---------------

            // Perform a line_search to compute step scale
            double rate = line_search(x, delta_x, objFunc);
            if (std::isnan(rate))
            {
                // descent_strategy set by line_search upon failure
                if (this->m_status == cppoptlib::Status::Continue)
                    continue;
                else
                    break;
            }

            x += rate * delta_x;

            // -----------
            // Post update
            // -----------

            descent_strategy = default_descent_strategy(); // Reset this for the next iterations

            const double step = (rate * delta_x).norm();

            if (objFunc.stop(x))
            {
                this->m_status = cppoptlib::Status::UserDefined;
                m_error_code = ErrorCode::SUCCESS;
                m_logger.debug("[{}] Objective decided to stop", name());
            }

            objFunc.post_step(this->m_current.iterations, x);

            m_logger.debug(
                "[{}] iter={:d} f={:g} Δf={:g} ‖∇f‖={:g} ‖Δx‖={:g} Δx⋅∇f(x)={:g} rate={:g} ‖step‖={:g}",
                name(), this->m_current.iterations, energy, this->m_current.fDelta,
                this->m_current.gradNorm, this->m_current.xDelta, delta_x.dot(grad), rate, step);

            if (++this->m_current.iterations >= this->m_stop.iterations)
                this->m_status = cppoptlib::Status::IterationLimit;

            update_solver_info(energy);

            objFunc.save_to_file(x);

        } while (objFunc.callback(this->m_current, x) && (this->m_status == cppoptlib::Status::Continue));

        // todo
        // timer.stop();

        // -----------
        // Log results
        // -----------

        if (!allow_out_of_iterations && this->m_status == cppoptlib::Status::IterationLimit)
            log_and_throw_error(m_logger, "[{}] Reached iteration limit (limit={})", name(), this->m_stop.iterations);
        if (this->m_current.iterations == 0)
            log_and_throw_error(m_logger, "[{}] Unable to take a step", name());
        if (this->m_status == cppoptlib::Status::UserDefined && m_error_code != ErrorCode::SUCCESS)
            log_and_throw_error(m_logger, "[{}] Failed to find minimizer", name());

        // todo
        // timer.getElapsedTimeInSec()
        double tot_time = 0;
        m_logger.info(
            "[{}] Finished: {} Took {:g}s (niters={:d} f={:g} Δf={:g} ‖∇f‖={:g} ‖Δx‖={:g} fdelta={} ftol={})",
            name(), this->m_status, tot_time, this->m_current.iterations,
            old_energy, this->m_current.fDelta, this->m_current.gradNorm, this->m_current.xDelta, this->m_current.fDelta, this->m_stop.fDelta);

        log_times();
        update_solver_info(objFunc.value(x));
    }

    double Solver::line_search(const TVector &x, const TVector &delta_x, Problem &objFunc)
    {
        POLYSOLVE_SCOPED_TIMER("line search", line_search_time);

        if (!m_line_search)
            return 1; // no linesearch

        double rate = m_line_search->line_search(x, delta_x, objFunc);

        if (std::isnan(rate) && descent_strategy < 2) // 2 is the max, grad descent
        {
            increase_descent_strategy();
            m_logger.warn(
                "[{}] Line search failed; reverting to {}", name(), descent_strategy_name());
            this->m_status = cppoptlib::Status::Continue; // Try the step again with gradient descent
        }
        else if (std::isnan(rate))
        {
            assert(descent_strategy == 2);                   // failed on gradient descent
            this->m_status = cppoptlib::Status::UserDefined; // Line search failed on gradient descent, so quit!
            log_and_throw_error(m_logger, "[{}] Line search failed on gradient descent; stopping", name());
        }

        return rate;
    }

    void Solver::reset(const int ndof)
    {
        this->m_current.reset();
        descent_strategy = default_descent_strategy();
        m_error_code = ErrorCode::SUCCESS;

        const std::string line_search_name = solver_info["line_search"];
        solver_info = json();
        solver_info["line_search"] = line_search_name;
        solver_info["iterations"] = 0;

        reset_times();
    }

    void Solver::reset_times()
    {
        total_time = 0;
        grad_time = 0;
        assembly_time = 0;
        inverting_time = 0;
        line_search_time = 0;
        obj_fun_time = 0;
        constraint_set_update_time = 0;
        if (m_line_search)
        {
            m_line_search->reset_times();
        }
    }

    void Solver::update_solver_info(const double energy)
    {
        solver_info["status"] = this->status();
        solver_info["error_code"] = m_error_code;
        solver_info["energy"] = energy;
        const auto &crit = this->criteria();
        solver_info["iterations"] = crit.iterations;
        solver_info["xDelta"] = crit.xDelta;
        solver_info["fDelta"] = crit.fDelta;
        solver_info["gradNorm"] = crit.gradNorm;
        solver_info["condition"] = crit.condition;
        solver_info["relative_gradient"] = normalize_gradient;

        double per_iteration = crit.iterations ? crit.iterations : 1;

        solver_info["total_time"] = total_time;
        solver_info["time_grad"] = grad_time / per_iteration;
        solver_info["time_assembly"] = assembly_time / per_iteration;
        solver_info["time_inverting"] = inverting_time / per_iteration;
        solver_info["time_line_search"] = line_search_time / per_iteration;
        solver_info["time_constraint_set_update"] = constraint_set_update_time / per_iteration;
        solver_info["time_obj_fun"] = obj_fun_time / per_iteration;

        if (m_line_search)
        {
            solver_info["line_search_iterations"] = m_line_search->iterations;

            solver_info["time_checking_for_nan_inf"] =
                m_line_search->checking_for_nan_inf_time / per_iteration;
            solver_info["time_broad_phase_ccd"] =
                m_line_search->broad_phase_ccd_time / per_iteration;
            solver_info["time_ccd"] = m_line_search->ccd_time / per_iteration;
            // Remove double counting
            solver_info["time_classical_line_search"] =
                (m_line_search->classical_line_search_time
                 - m_line_search->constraint_set_update_time)
                / per_iteration;
            solver_info["time_line_search_constraint_set_update"] =
                m_line_search->constraint_set_update_time / per_iteration;
        }
    }

    void Solver::log_times()
    {
        m_logger.debug(
            "[{}] grad {:.3g}s, assembly {:.3g}s, inverting {:.3g}s, "
            "line_search {:.3g}s, constraint_set_update {:.3g}s, "
            "obj_fun {:.3g}s, checking_for_nan_inf {:.3g}s, "
            "broad_phase_ccd {:.3g}s, ccd {:.3g}s, "
            "classical_line_search {:.3g}s",
            fmt::format(fmt::fg(fmt::terminal_color::magenta), "timing"),
            grad_time, assembly_time, inverting_time, line_search_time,
            constraint_set_update_time + (m_line_search ? m_line_search->constraint_set_update_time : 0),
            obj_fun_time, m_line_search ? m_line_search->checking_for_nan_inf_time : 0,
            m_line_search ? m_line_search->broad_phase_ccd_time : 0, m_line_search ? m_line_search->ccd_time : 0,
            m_line_search ? m_line_search->classical_line_search_time : 0);
    }
} // namespace polysolve::nonlinear

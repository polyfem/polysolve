
#include "Solver.hpp"

#include "PostStepData.hpp"

#include "descent_strategies/BFGS.hpp"
#include "descent_strategies/Newton.hpp"
#include "descent_strategies/ADAM.hpp"
#include "descent_strategies/GradientDescent.hpp"
#include "descent_strategies/LBFGS.hpp"

#include <polysolve/Utils.hpp>

#include <jse/jse.h>

#include <spdlog/spdlog.h>
#include <spdlog/fmt/bundled/color.h>
#include <spdlog/fmt/ostr.h>

#include <finitediff.hpp>

#include <iomanip>
#include <fstream>

namespace polysolve::nonlinear
{
    namespace
    {
        std::shared_ptr<DescentStrategy> create_solver(
            const std::string &solver_name, const json &solver_params,
            const json &linear_solver_params,
            const double characteristic_length,
            spdlog::logger &logger)
        {
            if (solver_name == "BFGS")
            {
                return std::make_shared<BFGS>(
                    solver_params, linear_solver_params,
                    characteristic_length, logger);
            }

            else if (solver_name == "DenseNewton" || solver_name == "dense_newton")
            {
                return std::make_shared<Newton>(false, solver_params, linear_solver_params, characteristic_length, logger);
            }
            else if (solver_name == "DenseProjectedNewton")
            {
                return std::make_shared<ProjectedNewton>(false, solver_params, linear_solver_params, characteristic_length, logger);
            }
            else if (solver_name == "DenseRegularizedNewton")
            {
                return std::make_shared<RegularizedNewton>(false, false, solver_params, linear_solver_params, characteristic_length, logger);
            }
            else if (solver_name == "DenseRegularizedProjectedNewton")
            {
                return std::make_shared<RegularizedNewton>(false, true, solver_params, linear_solver_params, characteristic_length, logger);
            }

            else if (solver_name == "Newton" || solver_name == "SparseNewton" || solver_name == "sparse_newton")
            {
                return std::make_shared<Newton>(true, solver_params, linear_solver_params, characteristic_length, logger);
            }
            else if (solver_name == "ProjectedNewton")
            {
                return std::make_shared<ProjectedNewton>(true, solver_params, linear_solver_params, characteristic_length, logger);
            }
            else if (solver_name == "RegularizedNewton")
            {
                return std::make_shared<RegularizedNewton>(true, false, solver_params, linear_solver_params, characteristic_length, logger);
            }
            else if (solver_name == "RegularizedProjectedNewton")
            {
                return std::make_shared<RegularizedNewton>(true, true, solver_params, linear_solver_params, characteristic_length, logger);
            }

            else if (solver_name == "LBFGS" || solver_name == "L-BFGS")
            {
                return std::make_shared<LBFGS>(solver_params, characteristic_length, logger);
            }

            else if (solver_name == "StochasticGradientDescent" || solver_name == "stochastic_gradient_descent")
            {
                return std::make_shared<GradientDescent>(solver_params, true, characteristic_length, logger);
            }
            else if (solver_name == "GradientDescent" || solver_name == "gradient_descent")
            {
                return std::make_shared<GradientDescent>(solver_params, false, characteristic_length, logger);
            }

            else if (solver_name == "ADAM" || solver_name == "adam")
            {
                return std::make_shared<ADAM>(solver_params, false, characteristic_length, logger);
            }
            else if (solver_name == "StochasticADAM" || solver_name == "stochastic_adam")
            {
                return std::make_shared<ADAM>(solver_params, true, characteristic_length, logger);
            }
            else
                throw std::runtime_error("Unrecognized solver type: " + solver_name);
        }

    } // namespace

    NLOHMANN_JSON_SERIALIZE_ENUM(
        FiniteDiffStrategy,
        {{FiniteDiffStrategy::NONE, "None"},
         {FiniteDiffStrategy::DIRECTIONAL_DERIVATIVE, "DirectionalDerivative"},
         {FiniteDiffStrategy::FULL_FINITE_DIFF, "FullFiniteDiff"}})

    // Static constructor
    std::unique_ptr<Solver> Solver::create(
        const json &solver_params_in,
        const json &linear_solver_params,
        const double characteristic_length,
        spdlog::logger &logger,
        const bool strict_validation)
    {
        json solver_params = solver_params_in; // mutable copy

        json rules;
        jse::JSE jse;

        jse.strict = strict_validation;
        const std::string input_spec = POLYSOLVE_NON_LINEAR_SPEC;
        std::ifstream file(input_spec);

        if (file.is_open())
            file >> rules;
        else
            log_and_throw_error(logger, "unable to open {} rules", input_spec);

        const bool valid_input = jse.verify_json(solver_params, rules);

        if (!valid_input)
            log_and_throw_error(logger, "invalid input json:\n{}", jse.log2str());

        solver_params = jse.inject_defaults(solver_params, rules);

        auto solver = std::make_unique<Solver>(solver_params, characteristic_length, logger);

        if (solver_params["solver"].is_array())
        {
            for (const auto &j : solver_params["solver"])
            {
                const std::string solver_name = j["type"];
                solver->add_strategy(create_solver(solver_name, j, linear_solver_params, characteristic_length, logger));
            }
        }
        else
        {
            const std::string solver_name = solver_params["solver"];

            if (solver_name == "DenseNewton" || solver_name == "dense_newton")
            {
                auto n = Newton::create_solver(false, solver_params, linear_solver_params, characteristic_length, logger);
                for (auto &s : n)
                    solver->add_strategy(s);
            }
            else if (solver_name == "Newton" || solver_name == "SparseNewton" || solver_name == "sparse_newton")
            {
                auto n = Newton::create_solver(true, solver_params, linear_solver_params, characteristic_length, logger);
                for (auto &s : n)
                    solver->add_strategy(s);
            }
            else
            {
                solver->add_strategy(create_solver(solver_name, solver_params, linear_solver_params, characteristic_length, logger));
            }

            if (solver_name != "GradientDescent" && solver_name != "gradient_descent")
            {

                solver->add_strategy(std::make_unique<GradientDescent>(
                    solver_params, false, characteristic_length, logger));
            }
        }

        solver->set_strategies_iterations(solver_params);
        return solver;
    }

    std::vector<std::string> Solver::available_solvers()
    {
        return {"BFGS",
                "DenseNewton",
                "Newton",
                "ADAM",
                "StochasticADAM",
                "GradientDescent",
                "StochasticGradientDescent",
                "L-BFGS"};
    }

    Solver::Solver(const json &solver_params,
                   const double characteristic_length,
                   spdlog::logger &logger)
        : m_logger(logger), characteristic_length(characteristic_length)
    {
        m_current.reset();

        m_stop.xDelta = solver_params["x_delta"];
        m_stop.fDelta = solver_params["advanced"]["f_delta"];
        m_stop.gradNorm = solver_params["grad_norm"];

        m_stop.xDelta *= characteristic_length;
        m_stop.fDelta *= characteristic_length;
        m_stop.gradNorm *= characteristic_length;

        m_stop.iterations = solver_params["max_iterations"];
        // m_stop.condition = solver_params["condition"];

        use_grad_norm_tol = solver_params["line_search"]["use_grad_norm_tol"];
        first_grad_norm_tol = solver_params["first_grad_norm_tol"];
        allow_out_of_iterations = solver_params["allow_out_of_iterations"];

        use_grad_norm_tol *= characteristic_length;
        first_grad_norm_tol *= characteristic_length;

        f_delta_step_tol = solver_params["advanced"]["f_delta_step_tol"];

        m_descent_strategy = 0;

        set_line_search(solver_params);

        gradient_fd_strategy = solver_params["advanced"]["apply_gradient_fd"];
        gradient_fd_eps = solver_params["advanced"]["gradient_fd_eps"];
    }

    void Solver::set_strategies_iterations(const json &solver_params)
    {
        m_iter_per_strategy.assign(m_strategies.size() + 1, 1);
        if (solver_params["iterations_per_strategy"].is_array())
        {
            m_iter_per_strategy.resize(m_strategies.size() + 1);
            if (solver_params["iterations_per_strategy"].size() != m_iter_per_strategy.size())
                log_and_throw_error(m_logger, "Invalit iter_per_strategy size: {}!={}", solver_params["iterations_per_strategy"].size(), m_iter_per_strategy.size());

            m_iter_per_strategy = solver_params["iterations_per_strategy"].get<std::vector<int>>();
        }
        else
            m_iter_per_strategy.assign(m_strategies.size() + 1, solver_params["iterations_per_strategy"].get<int>());
    }

    double Solver::compute_grad_norm(const TVector &x, const TVector &grad) const
    {
        return grad.norm();
    }

    void Solver::set_line_search(const json &params)
    {
        m_line_search = line_search::LineSearch::create(params, m_logger);
        solver_info["line_search"] = params["line_search"]["method"];
    }

    void Solver::minimize(Problem &objFunc, TVector &x)
    {
        constexpr double NaN = std::numeric_limits<double>::quiet_NaN();

        int previous_strategy = m_descent_strategy;
        int current_strategy_iter = 0;
        // ---------------------------
        // Initialize the minimization
        // ---------------------------
        reset(x.size()); // place for children to initialize their fields

        m_line_search->use_grad_norm_tol = use_grad_norm_tol;

        TVector grad = TVector::Zero(x.rows());
        TVector delta_x = TVector::Zero(x.rows());

        // Set these to nan to indicate they have not been computed yet
        double old_energy = NaN;
        {
            POLYSOLVE_SCOPED_STOPWATCH("constraint set update", constraint_set_update_time, m_logger);
            objFunc.solution_changed(x);
        }

        const auto g_norm_tol = m_stop.gradNorm;
        m_stop.gradNorm = first_grad_norm_tol;

        StopWatch stop_watch("nonlinear solver", total_time, m_logger);
        stop_watch.start();

        m_logger.debug(
            "Starting {} with {} solve f₀={:g} "
            "(stopping criteria: max_iters={:d} Δf={:g} ‖∇f‖={:g} ‖Δx‖={:g})",
            descent_strategy_name(), m_line_search->name(),
            objFunc(x), m_stop.iterations,
            m_stop.fDelta, m_stop.gradNorm, m_stop.xDelta);

        update_solver_info(objFunc(x));
        objFunc.post_step(PostStepData(m_current.iterations, solver_info, x, grad));

        int f_delta_step_cnt = 0;
        double f_delta = 0;

        // Used for logging
        double xDelta = 0, gradNorm = 0;

        do
        {
            m_line_search->set_is_final_strategy(m_descent_strategy == m_strategies.size() - 1);

            m_current.xDelta = NaN;
            m_current.fDelta = NaN;
            m_current.gradNorm = NaN;

            //////////// Energy
            double energy;
            {
                POLYSOLVE_SCOPED_STOPWATCH("compute objective function", obj_fun_time, m_logger);
                energy = objFunc(x);
            }

            if (!std::isfinite(energy))
            {
                m_status = Status::NanEncountered;
                log_and_throw_error(m_logger, "[{}][{}] f(x) is nan or inf; stopping", descent_strategy_name(), m_line_search->name());
                break;
            }

            f_delta = std::abs(old_energy - energy);
            // stop based on f_delta only if the solver has taken over f_delta_step_tol steps with small f_delta
            m_current.fDelta = (f_delta_step_cnt >= f_delta_step_tol) ? f_delta : NaN;

            ///////////// gradient
            {
                POLYSOLVE_SCOPED_STOPWATCH("compute gradient", grad_time, m_logger);
                objFunc.gradient(x, grad);
            }

            {
                POLYSOLVE_SCOPED_STOPWATCH("verify gradient", grad_time, m_logger);
                verify_gradient(objFunc, x, grad);
            }

            const double grad_norm = compute_grad_norm(x, grad);
            if (std::isnan(grad_norm))
            {
                m_status = Status::NanEncountered;
                log_and_throw_error(m_logger, "[{}][{}] Gradient is nan; stopping", descent_strategy_name(), m_line_search->name());
                break;
            }

            m_current.gradNorm = grad_norm;
            gradNorm = m_current.gradNorm;

            m_status = checkConvergence(m_stop, m_current);
            if (m_status != Status::Continue)
                break;

            // ------------------------
            // Compute update direction
            // ------------------------
            // Compute a Δx to update the variable
            //
            bool ok = compute_update_direction(objFunc, x, grad, delta_x);

            if (!ok || std::isnan(grad_norm) || (m_strategies[m_descent_strategy]->is_direction_descent() && grad_norm != 0 && delta_x.dot(grad) >= 0))
            {
                const auto current_name = descent_strategy_name();

                if (!m_strategies[m_descent_strategy]->handle_error())
                    ++m_descent_strategy;
                if (m_descent_strategy >= m_strategies.size())
                {
                    m_status = Status::NotDescentDirection;
                    log_and_throw_error(m_logger, "[{}][{}] direction is not a descent direction on last strategy (‖Δx‖={:g}; ‖g‖={:g}; Δx⋅g={:g}≥0); stopping",
                                        current_name, m_line_search->name(),
                                        delta_x.norm(), compute_grad_norm(x, grad), delta_x.dot(grad));
                }

                if (ok) // if ok, then the direction is not a descent direction
                {
                    m_logger.debug(
                        "[{}][{}] direction is not a descent direction (‖Δx‖={:g}; ‖g‖={:g}; Δx⋅g={:g}≥0); reverting to {}",
                        current_name, m_line_search->name(),
                        delta_x.norm(), compute_grad_norm(x, grad), delta_x.dot(grad), descent_strategy_name());
                }
                m_status = Status::Continue;
                continue;
            }

            const double delta_x_norm = delta_x.norm();
            if (std::isnan(delta_x_norm))
            {
                const auto current_name = descent_strategy_name();
                if (!m_strategies[m_descent_strategy]->handle_error())
                    ++m_descent_strategy;

                if (m_descent_strategy >= m_strategies.size())
                {
                    m_status = Status::NanEncountered;
                    log_and_throw_error(m_logger, "[{}][{}] Δx is nan on last strategy; stopping",
                                        current_name, m_line_search->name());
                }

                m_logger.debug("[{}][{}] Δx is nan; reverting to {}", current_name, m_line_search->name(), descent_strategy_name());
                m_status = Status::Continue;
                continue;
            }

            // Use the maximum absolute displacement value divided by the timestep,
            m_current.xDelta = delta_x_norm;
            xDelta = m_current.xDelta;
            m_status = checkConvergence(m_stop, m_current);
            if (m_status != Status::Continue)
                break;

            // ---------------
            // Variable update
            // ---------------

            m_logger.trace(
                "[{}][{}] pre LS iter={:d} f={:g} ‖∇f‖={:g}",
                descent_strategy_name(), m_line_search->name(),
                m_current.iterations, energy, gradNorm);

            // Perform a line_search to compute step scale
            double rate = m_line_search->line_search(x, delta_x, objFunc);
            if (std::isnan(rate))
            {
                const auto current_name = descent_strategy_name();
                assert(m_status == Status::Continue);
                if (!m_strategies[m_descent_strategy]->handle_error())
                    ++m_descent_strategy;
                if (m_descent_strategy >= m_strategies.size())
                {
                    // Line search failed on gradient descent, so quit!
                    m_status = Status::LineSearchFailed;
                    log_and_throw_error(m_logger, "[{}][{}] Line search failed on last strategy; stopping", current_name, m_line_search->name());
                }

                m_logger.debug("[{}] Line search failed; reverting to {}", current_name, descent_strategy_name());
                continue;
            }

            x += rate * delta_x;
            old_energy = energy;

            // Reset this for the next iterations
            // if the strategy got changed, we start counting
            if (m_descent_strategy != previous_strategy)
                current_strategy_iter = 0;
            // if we did enough lower strategy, we revert back to normal
            if (m_descent_strategy != 0 && current_strategy_iter >= m_iter_per_strategy[m_descent_strategy])
            {

                const auto current_name = descent_strategy_name();
                const std::string prev_strategy_name = descent_strategy_name();

                m_descent_strategy = 0;
                for (auto &s : m_strategies)
                    s->reset(x.size());

                m_logger.debug(
                    "[{}][{}] {} was successful for {} iterations; resetting to {}",
                    current_name, m_line_search->name(), prev_strategy_name, current_strategy_iter, descent_strategy_name());
            }

            previous_strategy = m_descent_strategy;
            ++current_strategy_iter;

            // -----------
            // Post update
            // -----------
            const double step = (rate * delta_x).norm();

            update_solver_info(energy);
            objFunc.post_step(PostStepData(m_current.iterations, solver_info, x, grad));

            if (objFunc.stop(x))
            {
                m_status = Status::ObjectiveCustomStop;
                m_logger.debug("[{}][{}] Objective decided to stop", descent_strategy_name(), m_line_search->name());
            }

            if (f_delta < m_stop.fDelta)
                f_delta_step_cnt++;
            else
                f_delta_step_cnt = 0;

            m_logger.debug(
                "[{}][{}] iter={:d} f={:g} Δf={:g} ‖∇f‖={:g} ‖Δx‖={:g} Δx⋅∇f(x)={:g} rate={:g} ‖step‖={:g}"
                " (stopping criteria: max_iters={:d} Δf={:g} ‖∇f‖={:g} ‖Δx‖={:g})",
                descent_strategy_name(), m_line_search->name(),
                m_current.iterations, energy, f_delta,
                gradNorm, xDelta, delta_x.dot(grad), rate, step,
                m_stop.iterations, m_stop.fDelta, m_stop.gradNorm, m_stop.xDelta);

            if (++m_current.iterations >= m_stop.iterations)
                m_status = Status::IterationLimit;

            // reset the tolerance, since in the first iter it might be smaller
            m_stop.gradNorm = g_norm_tol;
        } while (objFunc.callback(m_current, x) && (m_status == Status::Continue));

        stop_watch.stop();

        // -----------
        // Log results
        // -----------

        if (!allow_out_of_iterations && m_status == Status::IterationLimit)
            log_and_throw_error(m_logger, "[{}][{}] Reached iteration limit (limit={})", descent_strategy_name(), m_line_search->name(), m_stop.iterations);
        if (m_status == Status::NanEncountered)
            log_and_throw_error(m_logger, "[{}][{}] Failed to find minimizer", descent_strategy_name(), m_line_search->name());

        double tot_time = stop_watch.getElapsedTimeInSec();
        const bool succeeded = m_status == Status::GradNormTolerance;
        m_logger.log(
            succeeded ? spdlog::level::info : spdlog::level::err,
            "[{}][{}] Finished: {} Took {:g}s (niters={:d} f={:g} Δf={:g} ‖∇f‖={:g} ‖Δx‖={:g})"
            " (stopping criteria: max_iters={:d} Δf={:g} ‖∇f‖={:g} ‖Δx‖={:g})",
            descent_strategy_name(), m_line_search->name(),
            m_status, tot_time, m_current.iterations,
            old_energy, f_delta, gradNorm, xDelta,
            m_stop.iterations, m_stop.fDelta, m_stop.gradNorm, m_stop.xDelta);

        log_times();
        update_solver_info(objFunc(x));
    }

    void Solver::reset(const int ndof)
    {
        m_current.reset();
        m_descent_strategy = 0;
        m_status = Status::NotStarted;

        const std::string line_search_name = solver_info["line_search"];
        solver_info = json();
        solver_info["line_search"] = line_search_name;
        solver_info["iterations"] = 0;

        for (auto &s : m_strategies)
            s->reset(ndof);

        reset_times();
    }

    void Solver::reset_times()
    {
        total_time = 0;
        grad_time = 0;
        line_search_time = 0;
        obj_fun_time = 0;
        constraint_set_update_time = 0;
        if (m_line_search)
        {
            m_line_search->reset_times();
        }
        for (auto &s : m_strategies)
            s->reset_times();
    }

    void Solver::update_solver_info(const double energy)
    {
        solver_info["status"] = status();
        solver_info["energy"] = energy;
        const auto &crit = criteria();
        solver_info["iterations"] = crit.iterations;
        solver_info["xDelta"] = crit.xDelta;
        solver_info["fDelta"] = crit.fDelta;
        solver_info["gradNorm"] = crit.gradNorm;

        double per_iteration = crit.iterations ? crit.iterations : 1;

        solver_info["total_time"] = total_time;
        solver_info["time_grad"] = grad_time / per_iteration;
        solver_info["time_line_search"] = line_search_time / per_iteration;
        solver_info["time_constraint_set_update"] = constraint_set_update_time / per_iteration;
        solver_info["time_obj_fun"] = obj_fun_time / per_iteration;

        for (auto &s : m_strategies)
            s->update_solver_info(solver_info, per_iteration);

        if (m_line_search)
        {
            solver_info["line_search_iterations"] = m_line_search->iterations();

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
            "[{}] grad {:.3g}s, "
            "line_search {:.3g}s, constraint_set_update {:.3g}s, "
            "obj_fun {:.3g}s, checking_for_nan_inf {:.3g}s, "
            "broad_phase_ccd {:.3g}s, ccd {:.3g}s, "
            "classical_line_search {:.3g}s",
            fmt::format(fmt::fg(fmt::terminal_color::magenta), "timing"),
            grad_time, line_search_time,
            constraint_set_update_time + (m_line_search ? m_line_search->constraint_set_update_time : 0),
            obj_fun_time, m_line_search ? m_line_search->checking_for_nan_inf_time : 0,
            m_line_search ? m_line_search->broad_phase_ccd_time : 0, m_line_search ? m_line_search->ccd_time : 0,
            m_line_search ? m_line_search->classical_line_search_time : 0);
    }

    void Solver::verify_gradient(Problem &objFunc, const TVector &x, const TVector &grad)
    {
        bool match = false;

        switch (gradient_fd_strategy)
        {
        case FiniteDiffStrategy::NONE:
            return;
        case FiniteDiffStrategy::DIRECTIONAL_DERIVATIVE:
        {
            TVector direc = grad.normalized();
            TVector x2 = x + direc * gradient_fd_eps;
            TVector x1 = x - direc * gradient_fd_eps;

            objFunc.solution_changed(x2);
            double J2 = objFunc(x2);

            objFunc.solution_changed(x1);
            double J1 = objFunc(x1);

            double fd = (J2 - J1) / 2 / gradient_fd_eps;
            double analytic = direc.dot(grad);

            match = abs(fd - analytic) < 1e-8 || abs(fd - analytic) < 1e-4 * abs(analytic);

            // Log error in either case to make it more visible in the logs.
            if (match)
                m_logger.debug("step size: {}, finite difference: {}, derivative: {}", gradient_fd_eps, fd, analytic);
            else
                m_logger.error("step size: {}, finite difference: {}, derivative: {}", gradient_fd_eps, fd, analytic);
        }
        break;
        case FiniteDiffStrategy::FULL_FINITE_DIFF:
        {
            TVector grad_fd;
            fd::finite_gradient(
                x, [&](const TVector &x_) {
                    objFunc.solution_changed(x_);
                    return objFunc(x_);
                },
                grad_fd, fd::AccuracyOrder::SECOND, gradient_fd_eps);

            match = (grad_fd - grad).norm() < 1e-8 || (grad_fd - grad).norm() < 1e-4 * (grad).norm();

            if (match)
                m_logger.debug("step size: {}, all gradient components match finite difference", gradient_fd_eps);
            else
                m_logger.error("step size: {}, all gradient components do not match finite difference", gradient_fd_eps);
        }
        break;
        }

        objFunc.solution_changed(x);
    }
} // namespace polysolve::nonlinear

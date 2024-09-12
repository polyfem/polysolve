
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
        m_stop.firstGradNorm = solver_params["first_grad_norm_tol"];
        m_stop.xDeltaDotGrad = -solver_params["advanced"]["derivative_along_delta_x_tol"].get<double>();

        // Make these relative to the characteristic length
        m_logger.trace("Using a characteristic length of {:g}", characteristic_length);
        m_stop.xDelta *= characteristic_length;
        m_stop.fDelta *= characteristic_length;
        m_stop.gradNorm *= characteristic_length;
        m_stop.firstGradNorm *= characteristic_length;
        // m_stop.xDeltaDotGrad *= characteristic_length;

        m_stop.iterations = solver_params["max_iterations"];
        allow_out_of_iterations = solver_params["allow_out_of_iterations"];

        m_stop.fDeltaCount = solver_params["advanced"]["f_delta_step_tol"];

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
        m_line_search->use_grad_norm_tol = params["line_search"]["use_grad_norm_tol"];
        m_line_search->use_grad_norm_tol *= characteristic_length;
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

        TVector grad = TVector::Zero(x.rows());
        TVector delta_x = TVector::Zero(x.rows());

        // Set these to nan to indicate they have not been computed yet
        double old_energy = NaN;
        {
            POLYSOLVE_SCOPED_STOPWATCH("constraint set update", constraint_set_update_time, m_logger);
            objFunc.solution_changed(x);
        }

        StopWatch stop_watch("nonlinear solver", total_time, m_logger);
        stop_watch.start();

        m_logger.debug(
            "Starting {} with {} solve f₀={:g} (stopping criteria: {})",
            descent_strategy_name(), m_line_search->name(), objFunc(x), m_stop);

        update_solver_info(objFunc(x));
        objFunc.post_step(PostStepData(m_current.iterations, solver_info, x, grad));

        do
        {
            m_line_search->set_is_final_strategy(m_descent_strategy == m_strategies.size() - 1);

            // --- Energy ------------------------------------------------------

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

            m_current.fDelta = std::abs(old_energy - energy);

            // --- Gradient ----------------------------------------------------
            {
                POLYSOLVE_SCOPED_STOPWATCH("compute gradient", grad_time, m_logger);
                objFunc.gradient(x, grad);
            }

            {
                POLYSOLVE_SCOPED_STOPWATCH("verify gradient", grad_time, m_logger);
                verify_gradient(objFunc, x, grad);
            }

            m_current.gradNorm = compute_grad_norm(x, grad);
            if (std::isnan(m_current.gradNorm))
            {
                m_status = Status::NanEncountered;
                log_and_throw_error(m_logger, "[{}][{}] Gradient is nan; stopping", descent_strategy_name(), m_line_search->name());
            }

            // Check convergence without these values to avoid impossible linear solves.
            m_current.xDelta = NaN;
            m_current.xDeltaDotGrad = NaN;
            m_status = checkConvergence(m_stop, m_current);
            if (m_status != Status::Continue)
                break;

            // --- Update direction --------------------------------------------

            bool update_direction_successful;
            {
                POLYSOLVE_SCOPED_STOPWATCH("compute update direction", update_direction_time, m_logger);
                update_direction_successful = compute_update_direction(objFunc, x, grad, delta_x);
            }

            m_current.xDelta = delta_x.norm();
            if (!update_direction_successful || std::isnan(m_current.xDelta))
            {
                const auto current_name = descent_strategy_name();
                if (!m_strategies[m_descent_strategy]->handle_error())
                    ++m_descent_strategy;

                if (m_descent_strategy >= m_strategies.size())
                {
                    m_status = Status::UpdateDirectionFailed;
                    log_and_throw_error(
                        m_logger, "[{}][{}] {} on last strategy; stopping",
                        current_name, m_line_search->name(), m_status);
                }

                m_logger.debug(
                    "[{}][{}] {}; reverting to {}", current_name, m_line_search->name(),
                    Status::UpdateDirectionFailed, descent_strategy_name());
                m_status = Status::Continue;
                continue;
            }

            m_current.xDeltaDotGrad = delta_x.dot(grad);

            if (m_strategies[m_descent_strategy]->is_direction_descent() && m_current.gradNorm != 0 && m_current.xDeltaDotGrad >= 0)
            {
                const std::string current_name = descent_strategy_name();

                if (!m_strategies[m_descent_strategy]->handle_error())
                    ++m_descent_strategy;

                if (m_descent_strategy >= m_strategies.size())
                {
                    m_status = Status::NotDescentDirection;
                    log_and_throw_error(
                        m_logger, "[{}][{}] {} on last strategy (‖Δx‖={:g}; ‖g‖={:g}; Δx⋅g={:g}≥0); stopping",
                        current_name, m_line_search->name(), m_status, delta_x.norm(), compute_grad_norm(x, grad),
                        m_current.xDeltaDotGrad);
                }
                else
                {
                    m_status = Status::Continue;
                    m_logger.debug(
                        "[{}][{}] {} (‖Δx‖={:g}; ‖g‖={:g}; Δx⋅g={:g}≥0); reverting to {}",
                        current_name, m_line_search->name(), Status::NotDescentDirection,
                        delta_x.norm(), compute_grad_norm(x, grad), m_current.xDeltaDotGrad,
                        descent_strategy_name());
                }
                continue;
            }

            // --- Check convergence -------------------------------------------

            m_status = checkConvergence(m_stop, m_current);

            if (m_status != Status::Continue)
                break;

            // --- Variable update ---------------------------------------------

            m_logger.trace(
                "[{}][{}] pre LS iter={:d} f={:g} ‖∇f‖={:g}",
                descent_strategy_name(), m_line_search->name(),
                m_current.iterations, energy, m_current.gradNorm);

            // Perform a line_search to compute step scale
            double rate;
            {
                POLYSOLVE_SCOPED_STOPWATCH("line search", line_search_time, m_logger);
                rate = m_line_search->line_search(x, delta_x, objFunc);
            }

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

            {
                TVector x1 = x + rate * delta_x;
                if (objFunc.after_line_search_custom_operation(x, x1))
                    objFunc.solution_changed(x1);
                x = x1;
            }

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

            // m_logger.debug("[{}][{}] rate={:g} ‖step‖={:g}",
            //                descent_strategy_name(), m_line_search->name(), rate, step);

            update_solver_info(energy);
            objFunc.post_step(PostStepData(m_current.iterations, solver_info, x, grad));

            if (objFunc.stop(x))
            {
                m_status = Status::ObjectiveCustomStop;
                m_logger.debug("[{}][{}] Objective decided to stop", descent_strategy_name(), m_line_search->name());
            }

            m_current.fDeltaCount = (m_current.fDelta < m_stop.fDelta) ? (m_current.fDeltaCount + 1) : 0;

            m_logger.debug(
                "[{}][{}] {} (stopping criteria: {})",
                descent_strategy_name(), m_line_search->name(), m_current, m_stop);

            if (++m_current.iterations >= m_stop.iterations)
                m_status = Status::IterationLimit;
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
            "[{}][{}] Finished: {} took {:g}s ({}) (stopping criteria: {})",
            descent_strategy_name(), m_line_search->name(), m_status, tot_time,
            m_current, m_stop);

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
        obj_fun_time = 0;
        grad_time = 0;
        update_direction_time = 0;
        line_search_time = 0;
        constraint_set_update_time = 0;
        if (m_line_search)
            m_line_search->reset_times();
        for (auto &s : m_strategies)
            s->reset_times();
    }

    void Solver::update_solver_info(const double energy)
    {
        solver_info["status"] = status();
        solver_info["energy"] = energy;
        solver_info["iterations"] = m_current.iterations;
        solver_info["xDelta"] = m_current.xDelta;
        solver_info["fDelta"] = m_current.fDelta;
        solver_info["gradNorm"] = m_current.gradNorm;

        double per_iteration = m_current.iterations ? m_current.iterations : 1;

        solver_info["total_time"] = total_time;
        solver_info["time_obj_fun"] = obj_fun_time / per_iteration;
        solver_info["time_grad"] = grad_time / per_iteration;
        // Do not save update_direction_time as it is redundant with the strategies
        solver_info["time_line_search"] = line_search_time / per_iteration;
        solver_info["time_constraint_set_update"] = constraint_set_update_time / per_iteration;

        for (auto &s : m_strategies)
            s->update_solver_info(solver_info, per_iteration);
        if (m_line_search)
            m_line_search->update_solver_info(solver_info, per_iteration);
    }

    void Solver::log_times() const
    {
        m_logger.debug(
            "[{}] f: {:.2e}s, grad_f: {:.2e}s, update_direction: {:.2e}s, "
            "line_search: {:.2e}s, constraint_set_update: {:.2e}s",
            fmt::format(fmt::fg(fmt::terminal_color::magenta), "timing"),
            obj_fun_time, grad_time, update_direction_time, line_search_time,
            constraint_set_update_time);
        for (auto &s : m_strategies)
            s->log_times();
        if (m_line_search)
            m_line_search->log_times();
    }

    void Solver::verify_gradient(Problem &objFunc, const TVector &x, const TVector &grad)
    {
        bool match = false;
        double J = objFunc(x);

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

            double fd_centered = (J2 - J1) / 2 / gradient_fd_eps;
            double fd_right = (J2 - J) / gradient_fd_eps;
            double fd_left = (J - J1) / gradient_fd_eps;
            double analytic = direc.dot(grad);

            match = abs(fd_centered - analytic) < 1e-8 || abs(fd_centered - analytic) < 1e-4 * abs(analytic);

            // Log error in either case to make it more visible in the logs.
            if (match)
                m_logger.debug("step size: {}, finite difference: {} {} {}, derivative: {}", gradient_fd_eps, fd_centered, fd_left, fd_right, analytic);
            else
                m_logger.error("step size: {}, finite difference: {} {} {}, derivative: {}", gradient_fd_eps, fd_centered, fd_left, fd_right, analytic);
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

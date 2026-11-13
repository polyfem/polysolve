#pragma once

#include <polysolve/Utils.hpp>

#include <polysolve/nonlinear/Problem.hpp>

namespace polysolve::nonlinear
{

    class DescentStrategy
    {
    public:
        using TVector = Problem::TVector;
        using Scalar = Problem::Scalar;

        /// @brief Constructor 
        /// @param solver_params_ JSON of solver parameters
        /// @param characteristic_length 
        /// @param logger 
        DescentStrategy(const json &solver_params_,
                        const double characteristic_length,
                        spdlog::logger &logger)
            : m_logger(logger)
        {
        }
        virtual ~DescentStrategy() {}
        virtual std::string name() const = 0;

        virtual void reset(const int ndof) {}
        virtual void reset_times() {}

        /// @brief Update solver info after finding descent direction
        /// @param solver_info JSON of solver parameters
        /// @param per_iteration Number of iterations (used to normalize timings)
        virtual void update_solver_info(json &solver_info, const double per_iteration) {}
        virtual void log_times() const {}

        virtual bool is_direction_descent() { return true; }
        virtual bool handle_error() { return false; }

        /// @brief Called by Solver::minimize on its first strategy at the start of the iteration that follows a step taken
        /// with the line search by another strategy, after the energy and gradient at the new iterate are computed and
        /// before the convergence check.
        /// @param strategy Index of the strategy that took the step in the solver's list (>= 1)
        /// @param rate Step size of its line search
        /// @param energy_before Energy before the step
        /// @param energy_after Energy after the step
        /// @param grad_norm_before Gradient norm before the step (the norm of the stopping test, not rescaled)
        /// @param grad_norm_after Gradient norm after the step
        /// @param grad_tol Gradient norm tolerance of the stopping test (rescaled, comparable to the norms)
        virtual void chain_step_result(const int strategy, const double rate, const double energy_before, const double energy_after,
                                       const double grad_norm_before, const double grad_norm_after, const double grad_tol) {}

        /// @brief Compute descent direction along which to do line search
        /// @param objFunc Problem to be minimized
        /// @param x Current input (n x 1)
        /// @param grad Current gradient (n x 1)
        /// @param direction Current descent direction (n x 1)
        /// @return True if a descent direction was successfully found
        virtual bool compute_update_direction(
            Problem &objFunc,
            const TVector &x,
            const TVector &grad,
            TVector &direction) = 0;

        /// @brief Optionally take the step of this iteration instead of the line search along `direction`.
        /// Called after `direction` passed the descent and convergence checks.
        /// @param objFunc Problem to be minimized
        /// @param x Current input (n x 1)
        /// @param energy Objective value at x
        /// @param grad Current gradient (n x 1)
        /// @param direction Direction returned by compute_update_direction (n x 1)
        /// @param[out] step Step verified by the strategy (n x 1): the solver sets x += step without a line search and
        /// without CCD on the chord, then continues as after a line search (solution_changed, post_step, convergence)
        /// @return True if step was set; false to line search along direction as usual
        virtual bool take_step(
            Problem &objFunc,
            const TVector &x,
            const double energy,
            const TVector &grad,
            const TVector &direction,
            TVector &step)
        {
            return false;
        }

    protected:
        spdlog::logger &m_logger;
    };
} // namespace polysolve::nonlinear
#pragma once

#include <polysolve/Utils.hpp>

#include <polysolve/nonlinear/Problem.hpp>

#include <functional>
#include <string>

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

        /// @brief Whether the direction most recently returned by compute_update_direction is exact, as opposed to inexact (stopped early).
        ///
        /// Exact means the strategy returned the direction it set out to compute. This says nothing about how good that
        /// direction is: a Newton step from a direct solve and a gradient descent step are both exact. A direction is
        /// inexact when the strategy solves for it iteratively and stops before the solve has converged, as in an
        /// inexact Newton method with truncated conjugate gradients.
        ///
        /// The tradeoff of stopping early is that a small direction no longer tells us we are close to the minimum: it
        /// may be small only because the linear solve was cut short. The solver therefore ignores its step-size
        /// stopping criteria (x_delta_tol and rel_x_delta_tol) until an exact direction has been applied. Otherwise it
        /// would end the solve on the truncated direction and throw away the part of the step the linear solve had not
        /// reached yet.
        ///
        /// The default is true, so existing strategies behave as before.
        virtual bool is_direction_exact() const { return true; }

        /// @brief Give the strategy a way to ask if a direction is small enough to end the solve.
        ///
        /// The solver calls this at the start of every minimize(), so strategies never need to. The test is only valid
        /// during that minimize() call.
        ///
        /// @param test Returns true if a direction passes the solver's step-size stopping criteria.
        void set_step_size_convergence_test(std::function<bool(const TVector &)> test)
        {
            m_step_size_convergence_test = std::move(test);
        }

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

    protected:
        /// @brief Whether the solver's step-size stopping criteria would accept this direction.
        ///
        /// This lets a strategy that stops its linear solve early spend the extra work only where it matters. Far from
        /// the minimum a rough direction is good enough, so we stop early. When the rough direction is small enough to
        /// end the solve, though, the strategy should finish its solve and report the result as exact through
        /// is_direction_exact(), because only an exact small direction is evidence of convergence.
        ///
        /// Outside of Solver::minimize() there are no stopping criteria to test against, so this returns false.
        bool step_size_would_converge(const TVector &direction) const
        {
            return m_step_size_convergence_test && m_step_size_convergence_test(direction);
        }

        spdlog::logger &m_logger;

    private:
        std::function<bool(const TVector &)> m_step_size_convergence_test = nullptr;
    };
} // namespace polysolve::nonlinear
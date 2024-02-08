#pragma once

#include "Criteria.hpp"
#include "descent_strategies/DescentStrategy.hpp"
// Line search methods
#include "line_search/LineSearch.hpp"

namespace spdlog
{
    class logger;
}

namespace polysolve::nonlinear
{
    enum class FiniteDiffStrategy
    {
        NONE = 0,
        DIRECTIONAL_DERIVATIVE = 1,
        FULL_FINITE_DIFF = 2
    };

    class Solver
    {
    public:
        using Scalar = typename Problem::Scalar;
        using TVector = typename Problem::TVector;
        using THessian = typename Problem::THessian;

    public:
        // --- Static methods -------------------------------------------------

        // Static constructor
        //
        // @param[in]  solver   Solver type
        // @param[in]  precond  Preconditioner for iterative solvers
        //
        static std::unique_ptr<Solver> create(
            const json &solver_params,
            const json &linear_solver_params,
            const double characteristic_length,
            spdlog::logger &logger,
            const bool strict_validation = true);

        // List available solvers
        static std::vector<std::string> available_solvers();

    public:
        /// @brief Construct a new Nonlinear Solver object
        /// @param solver_params JSON of solver parameters
        /// @param characteristic_length used to scale tolerances
        /// @param logger
        Solver(const json &solver_params,
               const double characteristic_length,
               spdlog::logger &logger);

        virtual ~Solver() = default;

        Criteria &stop_criteria() { return m_stop; }
        const Criteria &stop_criteria() const { return m_stop; }
        const Criteria &criteria() const { return m_current; }
        const Status &status() const { return m_status; }

        void set_strategies_iterations(const json &solver_params);

        virtual double compute_grad_norm(const TVector &x, const TVector &grad) const;

        void set_line_search(const json &params);

        void minimize(Problem &objFunc, TVector &x);

        const json &get_info() const { return solver_info; }

        bool converged() const
        {
            return m_status == Status::XDeltaTolerance
                   || m_status == Status::FDeltaTolerance
                   || m_status == Status::GradNormTolerance;
        }

        size_t max_iterations() const { return m_stop.iterations; }
        size_t &max_iterations() { return m_stop.iterations; }
        bool allow_out_of_iterations = false;

        void add_strategy(const std::shared_ptr<DescentStrategy> &s) { m_strategies.push_back(s); }

        const std::shared_ptr<line_search::LineSearch> &line_search() const { return m_line_search; };

    protected:
        virtual bool compute_update_direction(
            Problem &objFunc,
            const TVector &x,
            const TVector &grad,
            TVector &direction)
        {
            return m_strategies[m_descent_strategy]->compute_update_direction(objFunc, x, grad, direction);
        }

        Criteria m_stop;
        Criteria m_current;
        Status m_status = Status::NotStarted;

        int m_descent_strategy;

        spdlog::logger &m_logger;

        // ====================================================================
        //                        Finite Difference Utilities
        // ====================================================================

        FiniteDiffStrategy gradient_fd_strategy = FiniteDiffStrategy::NONE;
        double gradient_fd_eps = 1e-7;
        virtual void verify_gradient(Problem &objFunc, const TVector &x, const TVector &grad) final;

    private:
        // ====================================================================
        //                        Solver parameters
        // ====================================================================

        double use_grad_norm_tol;
        double first_grad_norm_tol;

        const double characteristic_length;

        int f_delta_step_tol;
        // ====================================================================
        //                           Solver state
        // ====================================================================

        // Reset the solver at the start of a minimization
        void reset(const int ndof);

        std::string descent_strategy_name() const { return m_strategies[m_descent_strategy]->name(); };

        std::shared_ptr<line_search::LineSearch> m_line_search;
        std::vector<std::shared_ptr<DescentStrategy>> m_strategies;

        std::vector<int> m_iter_per_strategy;

        // ====================================================================
        //                            Solver info
        // ====================================================================

        void update_solver_info(const double energy);
        void reset_times();
        void log_times();

        json solver_info;

        double total_time;
        double grad_time;
        double line_search_time;
        double constraint_set_update_time;
        double obj_fun_time;

        // ====================================================================
        //                                 END
        // ====================================================================
    };
} // namespace polysolve::nonlinear

#pragma once

#include "descent_strategies/DescentStrategy.hpp"
// Line search methods
#include "line_search/LineSearch.hpp"

#include <cppoptlib/solver/isolver.h>

namespace spdlog
{
    class logger;
}

namespace polysolve::nonlinear
{

    enum class ErrorCode
    {
        NAN_ENCOUNTERED = -10,
        STEP_TOO_SMALL = -1,
        SUCCESS = 0,
    };

    class Solver : public cppoptlib::ISolver<Problem, /*Ord=*/-1>
    {
    public:
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

        using Superclass = ISolver<Problem, /*Ord=*/-1>;
        using typename Superclass::Scalar;
        using typename Superclass::TCriteria;
        using typename Superclass::TVector;

        /// @brief Construct a new Nonlinear Solver object
        /// @param solver_params JSON of solver parameters
        /// @param dt time step size (use 1 if not time-dependent) TODO
        /// @param logger
        Solver(const std::string &name,
               const json &solver_params,
               const double characteristic_length,
               spdlog::logger &logger);

        void set_strategies_iterations(const json &solver_params);

        virtual double compute_grad_norm(const Eigen::VectorXd &x, const Eigen::VectorXd &grad) const;

        std::string name() const { return m_name; }

        void set_line_search(const json &params);

        void minimize(Problem &objFunc, TVector &x) override;

        const json &get_info() const { return solver_info; }

        ErrorCode error_code() const { return m_error_code; }

        const typename Superclass::TCriteria &getStopCriteria() { return this->m_stop; }
        // setStopCriteria already in ISolver

        bool converged() const
        {
            return this->m_status == cppoptlib::Status::XDeltaTolerance
                   || this->m_status == cppoptlib::Status::FDeltaTolerance
                   || this->m_status == cppoptlib::Status::GradNormTolerance;
        }

        size_t max_iterations() const { return this->m_stop.iterations; }
        size_t &max_iterations() { return this->m_stop.iterations; }
        bool allow_out_of_iterations = false;

        void add_strategy(const std::shared_ptr<DescentStrategy> &s) { m_strategies.push_back(s); }

    private:
        // ====================================================================
        //                        Solver parameters
        // ====================================================================

        double use_grad_norm_tol;
        double first_grad_norm_tol;

        const double characteristic_length;

        // ====================================================================
        //                           Solver state
        // ====================================================================

        // Reset the solver at the start of a minimization
        void reset(const int ndof);

        std::string descent_strategy_name() const { return m_strategies[m_descent_strategy]->name(); };

        std::shared_ptr<line_search::LineSearch> m_line_search;
        std::vector<std::shared_ptr<DescentStrategy>> m_strategies;

        int m_descent_strategy;
        std::vector<int> m_iter_per_strategy;

        // ====================================================================
        //                            Solver info
        // ====================================================================

        std::string m_name;

        void update_solver_info(const double energy);
        void reset_times();
        void log_times();

        json solver_info;

        double total_time;
        double grad_time;
        double line_search_time;
        double constraint_set_update_time;
        double obj_fun_time;

        ErrorCode m_error_code;

        spdlog::logger &m_logger;

        // ====================================================================
        //                                 END
        // ====================================================================
    };
} // namespace polysolve::nonlinear

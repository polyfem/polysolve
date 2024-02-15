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

    enum class FiniteDiffStrategy
    {
        NONE = 0,
        DIRECTIONAL_DERIVATIVE = 1,
        FULL_FINITE_DIFF = 2
    };

    class Solver : public cppoptlib::ISolver<Problem, /*Ord=*/-1>
    {
    public:
        /// @brief Static constructor
        /// @param solver_params JSON of general solver parameters
        /// @param linear_solver_params JSON of linear solver parameters 
        /// @param logger 
        /// @param strict_validation whether or not to strictly validate input JSON
        static std::unique_ptr<Solver> create(
            const json &solver_params,
            const json &linear_solver_params,
            const double characteristic_length,
            spdlog::logger &logger,
            const bool strict_validation = true);

        /// @brief List available solvers
        static std::vector<std::string> available_solvers();

        using Superclass = ISolver<Problem, /*Ord=*/-1>;
        using typename Superclass::Scalar;
        using typename Superclass::TCriteria;
        using typename Superclass::TVector;

        /// @brief Construct a new Nonlinear Solver object
        /// @param solver_params JSON of solver parameters
        /// @param characteristic_length used to scale tolerances
        /// @param logger
        Solver(const json &solver_params,
               const double characteristic_length,
               spdlog::logger &logger);

        /// @brief Set number of iterations array 
        /// @param solver_params JSON of solver parameters (incuding itereations_per_strategy)
        void set_strategies_iterations(const json &solver_params);

        /// @brief Return norm of gradient
        /// @param x why is this included?
        /// @param grad gradient (n x 1) vector
        /// @return norm of given gradient
        virtual double compute_grad_norm(const Eigen::VectorXd &x, const Eigen::VectorXd &grad) const;

        /// @brief Create LineSearch object to be used by solver
        /// @param params JSON of solver parameters
        void set_line_search(const json &params);

        /// @brief Find argument that minimizes the objective function (loops through all strategies and terminates according to stopping criteria)
        /// @param objFunc Problem to be minimized
        /// @param x initial value for function input (n x 1)
        void minimize(Problem &objFunc, TVector &x) override;

        const json &get_info() const { return solver_info; }

        ErrorCode error_code() const { return m_error_code; }

        const typename Superclass::TCriteria &getStopCriteria() { return this->m_stop; }
        // setStopCriteria already in ISolver

        /// @brief Checks whether or not the solver has converged based on x delta, f delta, and grad norm
        /// @return Boolean representing whether or not the solver has converged.
        bool converged() const
        {
            return this->m_status == cppoptlib::Status::XDeltaTolerance
                   || this->m_status == cppoptlib::Status::FDeltaTolerance
                   || this->m_status == cppoptlib::Status::GradNormTolerance;
        }

        size_t max_iterations() const { return this->m_stop.iterations; }
        size_t &max_iterations() { return this->m_stop.iterations; }
        bool allow_out_of_iterations = false;

        /// @brief Add given descent strategy to solver
        /// @param s strategy to be added
        void add_strategy(const std::shared_ptr<DescentStrategy> &s) { m_strategies.push_back(s); }

        const std::shared_ptr<line_search::LineSearch> &line_search() const { return m_line_search; };

    protected:
        /// @brief Compute direction in which the argument should be updated 
        /// @param objFunc Problem to be minimized
        /// @param x Current input (n x 1)
        /// @param grad Gradient at current step (n x 1)
        /// @param[out] direction Current update direction (n x 1)
        /// @return True if update direction was found, False otherwises
        virtual bool compute_update_direction(
            Problem &objFunc,
            const TVector &x,
            const TVector &grad,
            TVector &direction)
        {
            return m_strategies[m_descent_strategy]->compute_update_direction(objFunc, x, grad, direction);
        }

        int m_descent_strategy;

        spdlog::logger &m_logger;

        // ====================================================================
        //                        Finite Difference Utilities
        // ====================================================================

        FiniteDiffStrategy gradient_fd_strategy = FiniteDiffStrategy::NONE;
        double gradient_fd_eps = 1e-7;
        /// @brief Check gradient versus finite difference results
        /// @param objFunc Problem defining relevant objective function
        /// @param x Current input (n x 1)
        /// @param grad Current gradient (n x 1)
        virtual void verify_gradient(Problem &objFunc, const TVector &x, const TVector &grad) final;

    private:
        // ====================================================================
        //                        Solver parameters
        // ====================================================================

        double use_grad_norm_tol;
        double first_grad_norm_tol;
        double derivative_along_delta_x_tol;

        const double characteristic_length;

        int f_delta_step_tol;
        // ====================================================================
        //                           Solver state
        // ====================================================================
        
        /// @brief Reset the solver at the start of a minimization
        /// @param ndof number of degrees of freedom
        void reset(const int ndof);

        std::string descent_strategy_name() const { return m_strategies[m_descent_strategy]->name(); };

        std::shared_ptr<line_search::LineSearch> m_line_search;
        std::vector<std::shared_ptr<DescentStrategy>> m_strategies;

        std::vector<int> m_iter_per_strategy;

        // ====================================================================
        //                            Solver info
        // ====================================================================

        /// @brief Update solver info JSON object
        /// @param energy 
        void update_solver_info(const double energy);

        /// @brief Reset timing members to 0
        void reset_times();

        /// @brief Log time taken in different phases of the solve
        void log_times();

        json solver_info;

        double total_time;
        double grad_time;
        double line_search_time;
        double constraint_set_update_time;
        double obj_fun_time;

        ErrorCode m_error_code;

        // ====================================================================
        //                                 END
        // ====================================================================
    };
} // namespace polysolve::nonlinear

#pragma once

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
    protected:
        static const int NEWTON_STRATEGY = 0;
        static const int REGULARIZED_NEWTON_STRATEGY = 1;

        static const int BFGS_STRATEGY = 1;
        static const int LBFGS_STRATEGY = 1;

        static const int MMA_STRATEGY = 2;

        static const int GRADIENT_DESCENT_STRATEGY = 2;

        static const int MAX_STRATEGY = GRADIENT_DESCENT_STRATEGY;

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
        Solver(const json &solver_params,
               const double characteristic_length,
               spdlog::logger &logger);

        virtual double compute_grad_norm(const Eigen::VectorXd &x, const Eigen::VectorXd &grad) const;

        virtual std::string name() const = 0;

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

    protected:
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
        virtual void reset(const int ndof);

        // Compute the search/update direction
        virtual void compute_update_direction(Problem &objFunc, const TVector &x_vec, const TVector &grad, TVector &direction) = 0;

        virtual void set_default_descent_strategy() = 0;
        virtual void increase_descent_strategy() = 0;

        virtual bool is_direction_descent() { return true; }

        virtual std::string descent_strategy_name(int descent_strategy) const = 0;
        virtual std::string descent_strategy_name() const { return descent_strategy_name(descent_strategy); };

        std::shared_ptr<line_search::LineSearch> m_line_search;

        int descent_strategy; // 0, newton, 1 spd, 2 gradiant

        // ====================================================================
        //                            Solver info
        // ====================================================================

        virtual void update_solver_info(const double energy);
        void reset_times();
        void log_times();

        json solver_info;

        double total_time;
        double grad_time;
        double assembly_time;
        double inverting_time;
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

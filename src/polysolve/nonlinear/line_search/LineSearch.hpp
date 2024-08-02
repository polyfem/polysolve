#pragma once

#include <polysolve/nonlinear/Problem.hpp>

namespace spdlog
{
    class logger;
}

namespace polysolve::nonlinear::line_search
{
    class LineSearch
    {
    public:
        using Scalar = typename Problem::Scalar;
        using TVector = typename Problem::TVector;
      
    public:
        /// @brief Constructor for creating new LineSearch Object
        /// @param params JSON of solver parameters
        /// @param m_logger 
        LineSearch(const json &params, spdlog::logger &m_logger);
        virtual ~LineSearch() = default;

        /// @brief 
        /// @param x Current input vector (n x 1)
        /// @param x_delta Current descent direction (n x 1)
        /// @param objFunc Objective function
        /// @return 
        double line_search(
            const TVector &x,
            const TVector &x_delta,
            Problem &objFunc);

        /// @brief Dispatch function for creating appropriate subclass
        /// @param params JSON of solver parameters
        /// @param logger 
        /// @return Pointer to object of the specified subclass
        static std::shared_ptr<LineSearch> create(
            const json &params,
            spdlog::logger &logger);

        /// @brief  Get list of available method names
        /// @return Vector of names of available methods
        static std::vector<std::string> available_methods();

        virtual std::string name() const = 0;

        void update_solver_info(json &solver_info, const double per_iteration);
        void reset_times();
        void log_times() const;

        void set_is_final_strategy(const bool val)
        {
            is_final_strategy = val;
        }

        double current_min_step_size() const
        {
            return is_final_strategy ? min_step_size_final : min_step_size;
        }

        int current_max_step_size_iter() const
        {
            return is_final_strategy ? max_step_size_iter_final : max_step_size_iter;
        }

        int iterations() const { return cur_iter; }

        double checking_for_nan_inf_time;
        double broad_phase_ccd_time;
        double narrow_phase_ccd_time;
        double constraint_set_update_time;
        double classical_line_search_time;

        double use_grad_norm_tol = -1;

    protected:
        /// @brief Compute step size to use during line search 
        /// @param x Current input (n x 1)
        /// @param delta_x Current step direction (n x 1)
        /// @param objFunc Problem to be minimized
        /// @param use_grad_norm Whether to compare grad norm or energy norm in stopping criteria
        /// @param old_energy Previous energy (scalar)
        /// @param old_grad Previous gradient (n x 1)
        /// @param starting_step_size Initial step size
        /// @return Step size to use in line search
        virtual double compute_descent_step_size(
            const TVector &x,
            const TVector &delta_x,
            Problem &objFunc,
            const bool use_grad_norm,
            const double old_energy,
            const TVector &old_grad,
            const double starting_step_size) = 0;

        spdlog::logger &m_logger;
        double step_ratio;
        int cur_iter;

    private:
        /// @brief Compute step size that avoids nan/infinite energy
        /// @param x Current input (n x 1)
        /// @param delta_x Current step direction (n x 1)
        /// @param objFunc Problem to be minimized
        /// @param starting_step_size Initial step size
        /// @param rate Rate at which to decrease step size if too large
        /// @return Nan free step size
        double compute_nan_free_step_size(
            const TVector &x,
            const TVector &delta_x,
            Problem &objFunc,
            const double starting_step_size, const double rate);

        /// @brief Compute maximum valid step size
        /// @param x Current input (n x 1)
        /// @param delta_x Current step direction (n x 1)
        /// @param objFunc Problem to be minimized
        /// @param starting_step_size Initial step size
        /// @return Maximum valid step size (NaN if it is 0)
        double compute_max_step_size(
            const TVector &x,
            const TVector &delta_x,
            Problem &objFunc,
            const double starting_step_size);

        double min_step_size;
        int max_step_size_iter;
        double min_step_size_final;
        int max_step_size_iter_final;

        bool is_final_strategy;

        double default_init_step_size;
    };
} // namespace polysolve::nonlinear::line_search

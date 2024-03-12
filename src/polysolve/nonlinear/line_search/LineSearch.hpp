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
        LineSearch(const json &params, spdlog::logger &m_logger);
        virtual ~LineSearch() = default;

        double line_search(
            const TVector &x,
            const TVector &grad,
            Problem &objFunc);

        static std::shared_ptr<LineSearch> create(
            const json &params,
            spdlog::logger &logger);

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
        double compute_nan_free_step_size(
            const TVector &x,
            const TVector &delta_x,
            Problem &objFunc,
            const double starting_step_size, const double rate);

        double compute_collision_free_step_size(
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

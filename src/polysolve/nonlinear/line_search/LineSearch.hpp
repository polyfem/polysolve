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

        void reset_times()
        {
            checking_for_nan_inf_time = 0;
            broad_phase_ccd_time = 0;
            ccd_time = 0;
            constraint_set_update_time = 0;
            classical_line_search_time = 0;
        }

        void set_is_final_strategy(const bool val)
        {
            is_final_strategy = val;
        }

        inline double current_min_step_size() const
        {
            return is_final_strategy ? min_step_size_final : min_step_size;
        }

        inline int current_max_step_size_iter() const
        {
            return is_final_strategy ? max_step_size_iter_final : max_step_size_iter;
        }

        double checking_for_nan_inf_time;
        double broad_phase_ccd_time;
        double ccd_time;
        double constraint_set_update_time;
        double classical_line_search_time;

        bool use_directional_derivative = false;
        double use_grad_norm_tol = -1;

        virtual std::string name() = 0;

        inline int iterations() const { return cur_iter; }

    private:
        double min_step_size;
        int max_step_size_iter;
        double min_step_size_final;
        int max_step_size_iter_final;

        bool is_final_strategy;

        double default_init_step_size;

    protected:
        int cur_iter;
        spdlog::logger &m_logger;

        double step_ratio;

        virtual double compute_descent_step_size(
            const TVector &x,
            const TVector &delta_x,
            Problem &objFunc,
            const bool use_grad_norm,
            const double old_energy,
            const TVector &old_grad,
            const double starting_step_size) = 0;

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
    };
} // namespace polysolve::nonlinear::line_search

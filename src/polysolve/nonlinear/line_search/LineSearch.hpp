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

        LineSearch(spdlog::logger &m_logger);
        virtual ~LineSearch() = default;

        double line_search(
            const TVector &x,
            const TVector &grad,
            Problem &objFunc);

        static std::shared_ptr<LineSearch> create(
            const std::string &name,
            spdlog::logger &logger);

        static std::vector<std::string> available_methods();

        void set_min_step_size(const double min_step_size_) { min_step_size = min_step_size_; };

        void reset_times()
        {
            iterations = 0;
            checking_for_nan_inf_time = 0;
            broad_phase_ccd_time = 0;
            ccd_time = 0;
            constraint_set_update_time = 0;
            classical_line_search_time = 0;
        }

        int iterations; ///< total number of backtracking iterations done
        double checking_for_nan_inf_time;
        double broad_phase_ccd_time;
        double ccd_time;
        double constraint_set_update_time;
        double classical_line_search_time;

        double use_grad_norm_tol = -1;

    protected:
        double min_step_size = 0;
        int max_step_size_iter = 100;
        int cur_iter = 0;
        spdlog::logger &m_logger;

        double default_init_step_size = 1;
        double step_ratio = 0.5;

        virtual double compute_descent_step_size(
            const TVector &x,
            const TVector &delta_x,
            Problem &objFunc,
            const bool use_grad_norm,
            const double old_energy_in,
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

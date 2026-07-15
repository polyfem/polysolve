#pragma once

#include "DescentStrategy.hpp"
#include <polysolve/Utils.hpp>

#include <polysolve/linear/Solver.hpp>


namespace polysolve::nonlinear
{
    class Newton : public DescentStrategy
    {
    public:
        using Superclass = DescentStrategy;

        static std::vector<std::shared_ptr<DescentStrategy>> create_solver(
            const bool sparse,
            const json &solver_params,
            const json &linear_solver_params,
            const double characteristic_length,
            spdlog::logger &logger,
            const NormType norm_type);

    protected:
        Newton(const bool sparse,
               const double residual_tolerance,
               const json &solver_params,
               std::shared_ptr<polysolve::linear::Solver> linear_solver,
               const double characteristic_length,
               spdlog::logger &logger,
               const NormType norm_type);

    public:
        Newton(const bool sparse,
               const json &solver_params,
               const json &linear_solver_params,
               const double characteristic_length,
               spdlog::logger &logger,
               const NormType norm_type);

        Newton(const bool sparse,
               const json &solver_params,
               std::shared_ptr<polysolve::linear::Solver> linear_solver,
               const double characteristic_length,
               spdlog::logger &logger,
               const NormType norm_type);

        std::string name() const override { return internal_name() + "Newton"; }

    private:
        double solve_sparse_linear_system(Problem &objFunc,
                                          const TVector &x, const TVector &grad,
                                          TVector &direction);
        double solve_dense_linear_system(Problem &objFunc,
                                         const TVector &x, const TVector &grad,
                                         TVector &direction);

        json internal_solver_info = json::array();

        const bool is_sparse;
        const double characteristic_length;
        double residual_tolerance;
        const NormType norm_type;

        std::shared_ptr<polysolve::linear::Solver> linear_solver; /// Linear solver used to solve the linear system. Note that this can now be shared across compatible `DescentStrategy` instances.

        // Benchmarking variables
        double assembly_time;
        double inverting_time;
        double linear_solve_time;
        double symbolic_factorizer_time;
        double numeric_factorizer_time;
        double solve_time;

    protected:
        std::string internal_name() const { return is_sparse ? "Sparse" : "Dense"; }

        virtual void compute_hessian(Problem &objFunc,
                                     const TVector &x,
                                     Hessian &hessian);


    public:
        bool compute_update_direction(Problem &objFunc, const TVector &x, const TVector &grad, TVector &direction) override;

        void reset(const int ndof) override;
        void update_solver_info(json &solver_info, const double per_iteration) override;
        virtual void update_times(std::vector<double> &linear_times) override;
        void reset_times() override;
        void log_times() const override;
    };

    class ProjectedNewton : public Newton
    {
    public:
        using Superclass = Newton;

        ProjectedNewton(const bool sparse,
                        const json &solver_params,
                        const json &linear_solver_params,
                        const double characteristic_length,
                        spdlog::logger &logger,
                        const NormType norm_type);

        ProjectedNewton(const bool sparse,
                        const json &solver_params,
                        std::shared_ptr<polysolve::linear::Solver> linear_solver,
                        const double characteristic_length,
                        spdlog::logger &logger,
                        const NormType norm_type);

        std::string name() const override { return internal_name() + "ProjectedNewton"; }

    protected:
        void compute_hessian(Problem &objFunc,
                             const TVector &x,
                             Hessian &hessian) override;
    };

    class RegularizedNewton : public Newton
    {
    public:
        using Superclass = Newton;

        RegularizedNewton(const bool sparse, const bool project_to_psd,
                          const json &solver_params,
                          const json &linear_solver_params,
                          const double characteristic_length,
                          spdlog::logger &logger,
                          const NormType norm_type);

        RegularizedNewton(const bool sparse, const bool project_to_psd,
                          const json &solver_params,
                          std::shared_ptr<polysolve::linear::Solver> linear_solver,
                          const double characteristic_length,
                          spdlog::logger &logger,
                          const NormType norm_type);

        std::string name() const override
        {
            return fmt::format("{}RegularizedNewton (reg_weight={:g})", internal_name(), reg_weight);
        }

        void reset(const int ndof) override;
        bool handle_error() override;

    private:
        const bool project_to_psd;
        double reg_weight_min; // needs to be greater than zero
        double reg_weight_max;
        double reg_weight_inc;

        TVector x_cache;
        polysolve::StiffnessMatrix hessian_cache;

        double reg_weight; ///< Regularization Coefficients
    protected:
        void compute_hessian(Problem &objFunc,
                             const TVector &x,
                             Hessian &hessian) override;
    };

} // namespace polysolve::nonlinear

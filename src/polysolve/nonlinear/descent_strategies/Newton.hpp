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
            spdlog::logger &logger);

        Newton(const bool sparse,
               const json &solver_params,
               const json &linear_solver_params,
               const double characteristic_length,
               spdlog::logger &logger);

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
        const double m_characteristic_length;
        double m_residual_tolerance;

        std::unique_ptr<polysolve::linear::Solver> linear_solver; ///< Linear solver used to solve the linear system

        double assembly_time;
        double inverting_time;

    protected:
        std::string internal_name() const { return is_sparse ? "Sparse" : "Dense"; }

        virtual void compute_hessian(Problem &objFunc,
                                     const TVector &x,
                                     polysolve::StiffnessMatrix &hessian);

        virtual void compute_hessian(Problem &objFunc,
                                     const TVector &x,
                                     Eigen::MatrixXd &hessian);

    public:
        bool compute_update_direction(Problem &objFunc, const TVector &x, const TVector &grad, TVector &direction) override;

        void reset(const int ndof) override;
        void update_solver_info(json &solver_info, const double per_iteration) override;

        void reset_times() override
        {
            assembly_time = 0;
            inverting_time = 0;
        }
    };

    class ProjectedNewton : public Newton
    {
    public:
        using Superclass = Newton;

        ProjectedNewton(const bool sparse,
                        const json &solver_params,
                        const json &linear_solver_params,
                        const double characteristic_length,
                        spdlog::logger &logger);

        std::string name() const override { return internal_name() + "ProjectedNewton"; }

    protected:
        void compute_hessian(Problem &objFunc,
                             const TVector &x,
                             polysolve::StiffnessMatrix &hessian) override;

        void compute_hessian(Problem &objFunc,
                             const TVector &x,
                             Eigen::MatrixXd &hessian) override;
    };

    class RegularizedNewton : public Newton
    {
    public:
        using Superclass = Newton;

        RegularizedNewton(const bool sparse,
                          const json &solver_params,
                          const json &linear_solver_params,
                          const double characteristic_length,
                          spdlog::logger &logger);

        std::string name() const override
        {
            return fmt::format("{}RegularizedNewton (reg_weight={:g})", internal_name(), reg_weight);
        }

        void reset(const int ndof) override;
        bool handle_error() override;

    private:
        double reg_weight_min; // needs to be greater than zero
        double reg_weight_max;
        double reg_weight_inc;

        TVector x_cache;
        polysolve::StiffnessMatrix hessian_cache;

        double reg_weight; ///< Regularization Coefficients
    protected:
        void compute_hessian(Problem &objFunc,
                             const TVector &x,
                             polysolve::StiffnessMatrix &hessian) override;

        void compute_hessian(Problem &objFunc,
                             const TVector &x,
                             Eigen::MatrixXd &hessian) override;
    };

} // namespace polysolve::nonlinear

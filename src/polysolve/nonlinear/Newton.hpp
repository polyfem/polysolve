#pragma once

#include "Solver.hpp"
#include <polysolve/Utils.hpp>

namespace polysolve::nonlinear
{
    class Newton : public Solver
    {
    public:
        using Superclass = Solver;
        using typename Superclass::Scalar;
        using typename Superclass::TVector;

        Newton(const json &solver_params,
               const json &linear_solver_params,
               const double characteristic_length,
               spdlog::logger &logger);

        std::string name() const override { return "Newton"; }

    protected:
        virtual double solve_linear_system(Problem &objFunc,
                                           const TVector &x, const TVector &grad,
                                           TVector &direction) = 0;

        void compute_update_direction(Problem &objFunc, const TVector &x, const TVector &grad, TVector &direction) override;
        bool check_direction(const double residual, const TVector &grad, const TVector &direction);

        // ====================================================================
        //                        Solver parameters
        // ====================================================================

        double reg_weight_min; // needs to be greater than zero
        double reg_weight_max;
        double reg_weight_inc;
        double reg_weight_dec;

        // ====================================================================
        //                           Solver state
        // ====================================================================

        void reset(const int ndof) override;

        virtual void set_default_descent_strategy() override
        {
            reg_weight = 0;
            descent_strategy = force_psd_projection ? Solver::REGULARIZED_NEWTON_STRATEGY : Solver::NEWTON_STRATEGY;
        }
        void increase_descent_strategy() override;

        using Superclass::descent_strategy_name;
        std::string descent_strategy_name(int descent_strategy) const override;

        bool force_psd_projection = false; ///< Whether to force the Hessian to be positive semi-definite
        double reg_weight = 0;             ///< Regularization Coefficients

        // ====================================================================
        //                            Solver info
        // ====================================================================

        void update_solver_info(const double energy) override;

        json internal_solver_info = json::array();

        spdlog::level::level_enum log_level() const
        {
            return this->descent_strategy == Solver::MAX_STRATEGY ? spdlog::level::warn : spdlog::level::debug;
        }

        // ====================================================================
        //                                END
        // ====================================================================
    };

} // namespace polysolve::nonlinear

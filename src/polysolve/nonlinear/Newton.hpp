#pragma once

#include "Solver.hpp"
#include "Utils.hpp"

#include <polysolve/linear/Solver.hpp>

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
               const double dt, const double characteristic_length,
               spdlog::logger &logger);

        std::string name() const override { return "Newton"; }

    protected:
        virtual double solve_linear_system(Problem &objFunc,
                                           const TVector &x, const TVector &grad,
                                           TVector &direction) = 0;

        bool compute_update_direction(Problem &objFunc, const TVector &x, const TVector &grad, TVector &direction) override;
        bool check_direction(const double residual, const TVector &grad, const TVector &direction);

        // ====================================================================
        //                        Solver parameters
        // ====================================================================

        static constexpr double reg_weight_min = 1e-8; // needs to be greater than zero
        static constexpr double reg_weight_max = 1e8;
        static constexpr double reg_weight_inc = 10;
        static constexpr double reg_weight_dec = 2;

        // ====================================================================
        //                           Solver state
        // ====================================================================

        void reset(const int ndof) override;

        virtual int default_descent_strategy() override { return force_psd_projection ? 1 : 0; }
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
            return this->descent_strategy == 2 ? spdlog::level::warn : spdlog::level::debug;
        }

        // ====================================================================
        //                                END
        // ====================================================================
    };

} // namespace polysolve::nonlinear

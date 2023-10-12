#pragma once

#include "Solver.hpp"
#include <polysolve/Utils.hpp>

namespace polysolve::nonlinear
{
    class BoxConstraintSolver : public Solver
    {
    public:
        // Static constructor
        //
        // @param[in]  solver   Solver type
        // @param[in]  precond  Preconditioner for iterative solvers
        //
        static std::unique_ptr<Solver> create(const std::string &solver,
                                              const json &solver_params,
                                              const json &linear_solver_params,
                                              const double dt,
                                              const double characteristic_length,
                                              spdlog::logger &logger);
        // List available solvers
        static std::vector<std::string> available_solvers();

        using Superclass = Solver;
        using typename Superclass::Scalar;
        using typename Superclass::TVector;

        BoxConstraintSolver(const json &solver_params,
                            const double dt,
                            const double characteristic_length,
                            spdlog::logger &logger);

        double compute_grad_norm(const Eigen::VectorXd &x, const Eigen::VectorXd &grad) const override;
        Eigen::VectorXd get_lower_bound(const Eigen::VectorXd &x, bool consider_max_change = true) const;
        Eigen::VectorXd get_upper_bound(const Eigen::VectorXd &x, bool consider_max_change = true) const;
        Eigen::VectorXd get_max_change(const Eigen::VectorXd &x) const;

    private:
        Eigen::MatrixXd bounds_;

        double max_change_val_ = 0;
        Eigen::VectorXd max_change_;
    };

} // namespace polysolve::nonlinear
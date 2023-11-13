#pragma once

#include "Solver.hpp"
#include "descent_strategies/box_constraints/BoxedDescentStrategy.hpp"
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
        static std::unique_ptr<Solver> create(
            const json &solver_params,
            const json &linear_solver_params,
            const double characteristic_length,
            spdlog::logger &logger,
            const bool strict_validation = true);

        // List available solvers
        static std::vector<std::string> available_solvers();

        using Superclass = Solver;

        BoxConstraintSolver(const std::string &name,
                            const json &solver_params,
                            const double characteristic_length,
                            spdlog::logger &logger);

        double compute_grad_norm(const Eigen::VectorXd &x, const Eigen::VectorXd &grad) const override;
        Eigen::VectorXd get_lower_bound(const Eigen::VectorXd &x, bool consider_max_change = true) const;
        Eigen::VectorXd get_upper_bound(const Eigen::VectorXd &x, bool consider_max_change = true) const;
        Eigen::VectorXd get_max_change(const Eigen::VectorXd &x) const;

        void add_strategy(const std::shared_ptr<BoxedDescentStrategy> &s)
        {
            Superclass::add_strategy(s);
            m_strategies.push_back(s);
        }

    protected:
        bool compute_update_direction(
            Problem &objFunc,
            const TVector &x,
            const TVector &grad,
            TVector &direction) override;

    private:
        Eigen::MatrixXd bounds_;
        std::vector<std::shared_ptr<BoxedDescentStrategy>> m_strategies;

        double max_change_val_ = 0;
        Eigen::VectorXd max_change_;
    };

} // namespace polysolve::nonlinear
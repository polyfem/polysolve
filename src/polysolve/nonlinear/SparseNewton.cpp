#include "SparseNewton.hpp"

#include <unsupported/Eigen/SparseExtra>

namespace polysolve::nonlinear
{

    SparseNewton::SparseNewton(
        const json &solver_params,
        const json &linear_solver_params,
        const double dt,
        const double characteristic_length,
        spdlog::logger &logger)
        : Superclass(solver_params,
                     linear_solver_params,
                     dt,
                     characteristic_length,
                     logger)
    {
        linear_solver = polysolve::linear::Solver::create(
            linear_solver_params["solver"], linear_solver_params["precond"]);
        linear_solver->setParameters(linear_solver_params);
    }

    double SparseNewton::solve_linear_system(Problem &objFunc,
                                             const TVector &x,
                                             const TVector &grad,
                                             TVector &direction)
    {
        polysolve::StiffnessMatrix hessian;

        {
            POLYSOLVE_SCOPED_STOPWATCH("assembly time", this->assembly_time, m_logger);

            objFunc.hessian(x, hessian);

            if (reg_weight > 0)
            {
                hessian += reg_weight * sparse_identity(hessian.rows(), hessian.cols());
            }
        }

        {
            POLYSOLVE_SCOPED_STOPWATCH("linear solve", this->inverting_time, m_logger);
            // TODO: get the correct size
            linear_solver->analyzePattern(hessian, hessian.rows());

            try
            {
                linear_solver->factorize(hessian);
            }
            catch (const std::runtime_error &err)
            {
                increase_descent_strategy();

                // warn if using gradient descent
                m_logger.log(
                    log_level(), "Unable to factorize Hessian: \"{}\"; reverting to {}",
                    err.what(), this->descent_strategy_name());

                // Eigen::saveMarket(hessian, "problematic_hessian.mtx");
                return std::nan("");
            }

            linear_solver->solve(-grad, direction); // H Δx = -g
        }

        const double residual = (hessian * direction + grad).norm(); // H Δx + g = 0

        json info;
        linear_solver->getInfo(info);
        internal_solver_info.push_back(info);

        return residual;
    }
} // namespace polysolve::nonlinear

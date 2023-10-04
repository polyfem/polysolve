#include "DenseNewton.hpp"
// #include <unsupported/Eigen/SparseExtra>

namespace polysolve::nonlinear
{
    DenseNewton::DenseNewton(
        const json &solver_params,
        const json &linear_solver_params,
        const double dt, const double characteristic_length,
        spdlog::logger &logger)
        : Superclass(solver_params,
                     linear_solver_params,
                     dt,
                     characteristic_length,
                     logger)
    {
        // TODO linear solver
    }

    double DenseNewton::solve_linear_system(Problem &objFunc,
                                            const TVector &x,
                                            const TVector &grad,
                                            TVector &direction)
    {
        Eigen::MatrixXd hessian;

        {
            POLYSOLVE_SCOPED_STOPWATCH("assembly time", this->assembly_time, m_logger);

            objFunc.hessian(x, hessian);

            if (reg_weight > 0)
            {
                for (int k = 0; k < x.size(); k++)
                    hessian(k, k) += reg_weight;
            }
        }

        TVector b = -grad;
        // b.conservativeResize(hessian.rows());
        // b.segment(grad.size(), b.size() - grad.size()).setZero();

        {
            POLYSOLVE_SCOPED_STOPWATCH("linear solve", this->inverting_time, m_logger);

            try
            {
                direction = hessian.ldlt().solve(b);
            }
            catch (const std::runtime_error &err)
            {
                increase_descent_strategy();

                // warn if using gradient descent
                m_logger.log(
                    log_level(), "Unable to factorize Hessian: \"{}\"; reverting to {}",
                    err.what(), this->descent_strategy_name());

                // polyfem::write_sparse_matrix_csv("problematic_hessian.csv", hessian);
                return std::nan("");
            }
        }

        const double residual = (hessian * direction - b).norm(); // H Δx + g = 0

        // TODO solver info
        return residual;
    }
} // namespace polysolve::nonlinear

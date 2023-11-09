#include "Newton.hpp"

namespace polysolve::nonlinear
{

    std::vector<std::shared_ptr<DescentStrategy>> Newton::create_solver(
        const bool sparse,
        const json &solver_params,
        const json &linear_solver_params,
        const double characteristic_length,
        spdlog::logger &logger)
    {
        std::vector<std::shared_ptr<DescentStrategy>> res;
        const bool force_psd_projection = solver_params["Newton"]["force_psd_projection"];
        if (!force_psd_projection)
            res.push_back(std::make_unique<Newton>(
                sparse,
                solver_params, linear_solver_params,
                characteristic_length, logger));

        // TODO disable regularization?
        res.push_back(std::make_unique<RegularizedNewton>(
            sparse,
            solver_params, linear_solver_params,
            characteristic_length, logger));

        // TODO disable projection?
        res.push_back(std::make_unique<ProjectedNewton>(
            sparse,
            solver_params, linear_solver_params,
            characteristic_length, logger));

        return res;
    }

    Newton::Newton(
        const bool sparse,
        const json &solver_params,
        const json &linear_solver_params,
        const double characteristic_length,
        spdlog::logger &logger)
        : Superclass(solver_params, characteristic_length, logger),
          is_sparse(sparse), m_characteristic_length(characteristic_length)
    {
        linear_solver = polysolve::linear::Solver::create(linear_solver_params, logger);
        assert(linear_solver->is_dense() == !sparse);
    }

    ProjectedNewton::ProjectedNewton(
        const bool sparse,
        const json &solver_params,
        const json &linear_solver_params,
        const double characteristic_length,
        spdlog::logger &logger)
        : Superclass(sparse, solver_params, linear_solver_params, characteristic_length, logger)
    {
    }

    RegularizedNewton::RegularizedNewton(
        const bool sparse,
        const json &solver_params,
        const json &linear_solver_params,
        const double characteristic_length,
        spdlog::logger &logger)
        : Superclass(sparse, solver_params, linear_solver_params, characteristic_length, logger)
    {
        reg_weight_min = solver_params["Newton"]["reg_weight_min"];
        reg_weight_max = solver_params["Newton"]["reg_weight_max"];
        reg_weight_inc = solver_params["Newton"]["reg_weight_inc"];
        reg_weight_dec = solver_params["Newton"]["reg_weight_dec"];
    }

    // =======================================================================

    void Newton::reset(const int ndof)
    {
        Superclass::reset(ndof);
        internal_solver_info = json::array();
    }

    void RegularizedNewton::reset(const int ndof)
    {
        Superclass::reset(ndof);
        reg_weight = 0;
    }

    // =======================================================================

    bool Newton::compute_update_direction(
        Problem &objFunc,
        const TVector &x,
        const TVector &grad,
        TVector &direction)
    {
        // solve_linear_system will increase descent_strategy if needed
        const double residual = solve_linear_system(objFunc, x, grad, direction);

        return check_direction(residual, grad, direction);
    }

    // =======================================================================

    // =======================================================================
    double Newton::solve_linear_system(Problem &objFunc,
                                       const TVector &x,
                                       const TVector &grad,
                                       TVector &direction)
    {
        if (is_sparse)
            return solve_sparse_linear_system(objFunc, x, grad, direction);
        else
            return solve_dense_linear_system(objFunc, x, grad, direction);
    }

    double Newton::solve_sparse_linear_system(Problem &objFunc,
                                              const TVector &x,
                                              const TVector &grad,
                                              TVector &direction)
    {
        polysolve::StiffnessMatrix hessian;

        {
            POLYSOLVE_SCOPED_STOPWATCH("assembly time", this->assembly_time, m_logger);
            compute_hessian(objFunc, x, hessian);
        }

        {
            POLYSOLVE_SCOPED_STOPWATCH("linear solve", this->inverting_time, m_logger);
            // TODO: get the correct size
            linear_solver->analyze_pattern(hessian, hessian.rows());

            try
            {
                linear_solver->factorize(hessian);
            }
            catch (const std::runtime_error &err)
            {
                // warn if using gradient descent
                m_logger.debug("Unable to factorize Hessian: \"{}\"", err.what());

                // Eigen::saveMarket(hessian, "problematic_hessian.mtx");
                return std::nan("");
            }

            linear_solver->solve(-grad, direction); // H Δx = -g
        }

        const double residual = (hessian * direction + grad).norm(); // H Δx + g = 0

        json info;
        linear_solver->get_info(info);
        internal_solver_info.push_back(info);

        return residual;
    }

    double Newton::solve_dense_linear_system(Problem &objFunc,
                                             const TVector &x,
                                             const TVector &grad,
                                             TVector &direction)
    {
        Eigen::MatrixXd hessian;

        {
            POLYSOLVE_SCOPED_STOPWATCH("assembly time", this->assembly_time, m_logger);
            compute_hessian(objFunc, x, hessian);
        }

        {
            POLYSOLVE_SCOPED_STOPWATCH("linear solve", this->inverting_time, m_logger);

            try
            {
                linear_solver->analyze_pattern_dense(hessian, hessian.rows());
                linear_solver->factorize_dense(hessian);
                linear_solver->solve(-grad, direction);
            }
            catch (const std::runtime_error &err)
            {
                // warn if using gradient descent
                m_logger.debug("Unable to factorize Hessian: \"{}\"",
                               err.what());

                return std::nan("");
            }
        }

        const double residual = (hessian * direction + grad).norm(); // H Δx + g = 0

        json info;
        linear_solver->get_info(info);
        internal_solver_info.push_back(info);

        return residual;
    }
    // =======================================================================

    void Newton::compute_hessian(Problem &objFunc,
                                 const TVector &x,
                                 polysolve::StiffnessMatrix &hessian)

    {
        objFunc.set_project_to_psd(false);
        objFunc.hessian(x, hessian);
    }

    void ProjectedNewton::compute_hessian(Problem &objFunc,
                                          const TVector &x,
                                          polysolve::StiffnessMatrix &hessian)

    {
        objFunc.set_project_to_psd(true);
        objFunc.hessian(x, hessian);
    }

    void RegularizedNewton::compute_hessian(Problem &objFunc,
                                            const TVector &x,
                                            polysolve::StiffnessMatrix &hessian)

    {
        objFunc.set_project_to_psd(true);
        objFunc.hessian(x, hessian);
        if (reg_weight > 0)
        {
            hessian += reg_weight * sparse_identity(hessian.rows(), hessian.cols());
        }
    }

    void Newton::compute_hessian(Problem &objFunc,
                                 const TVector &x,
                                 Eigen::MatrixXd &hessian)

    {
        objFunc.set_project_to_psd(false);
        objFunc.hessian(x, hessian);
    }

    void ProjectedNewton::compute_hessian(Problem &objFunc,
                                          const TVector &x,
                                          Eigen::MatrixXd &hessian)

    {
        objFunc.set_project_to_psd(true);
        objFunc.hessian(x, hessian);
    }

    void RegularizedNewton::compute_hessian(Problem &objFunc,
                                            const TVector &x,
                                            Eigen::MatrixXd &hessian)

    {
        objFunc.set_project_to_psd(true);
        objFunc.hessian(x, hessian);
        if (reg_weight > 0)
        {
            for (int k = 0; k < x.size(); k++)
                hessian(k, k) += reg_weight;
        }
    }
    // =======================================================================

    // =======================================================================

    bool Newton::check_direction(
        const double residual, const TVector &grad, const TVector &direction)
    {
        // gradient descent, check descent direction
        if (std::isnan(residual) || residual > std::max(1e-8 * grad.norm(), 1e-5) * m_characteristic_length)
        {
            m_logger.debug("[{}] large (or nan) linear solve residual {} (||∇f||={})",
                           name(), residual, grad.norm());

            return false;
        }
        else
        {
            m_logger.trace("linear solve residual {}", residual);
        }

        // do this check here because we need to repeat the solve without resetting reg_weight
        if (grad.norm() != 0 && grad.dot(direction) >= 0)
        {
            m_logger.debug("[{}] direction is not a descent direction (‖g‖={:g}; ‖Δx‖={:g}; Δx⋅g={:g}≥0)",
                           name(), grad.norm(), direction.norm(), direction.dot(grad));
            return false;
        }

        return true;
    }

    // =======================================================================

    void Newton::update_solver_info(json &solver_info, const double per_iteration)
    {
        Superclass::update_solver_info(solver_info, per_iteration);

        solver_info["internal_solver"] = internal_solver_info;
        solver_info["time_assembly"] = assembly_time / per_iteration;
        solver_info["time_inverting"] = inverting_time / per_iteration;
    }

    // =======================================================================

} // namespace polysolve::nonlinear

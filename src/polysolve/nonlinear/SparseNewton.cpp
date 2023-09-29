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
        : Superclass(solver_params, dt, characteristic_length, logger)
    {
        linear_solver = polysolve::linear::Solver::create(
            linear_solver_params["solver"], linear_solver_params["precond"]);
        linear_solver->setParameters(linear_solver_params);

        force_psd_projection = solver_params["force_psd_projection"];
    }

    // =======================================================================

    std::string SparseNewton::descent_strategy_name(int descent_strategy) const
    {
        switch (descent_strategy)
        {
        case 0:
            return "Newton";
        case 1:
            if (reg_weight == 0)
                return "projected Newton";
            return fmt::format("projected Newton w/ regularization weight={}", reg_weight);
        case 2:
            return "gradient descent";
        default:
            throw std::invalid_argument("invalid descent strategy");
        }
    }

    // =======================================================================

    void SparseNewton::increase_descent_strategy()
    {
        if (this->descent_strategy == 1 && reg_weight < reg_weight_max)
            reg_weight = std::clamp(reg_weight_inc * reg_weight, reg_weight_min, reg_weight_max);
        else
            this->descent_strategy++;
        assert(this->descent_strategy <= 2);
    }

    // =======================================================================

    void SparseNewton::reset(const int ndof)
    {
        Superclass::reset(ndof);
        assert(linear_solver != nullptr);
        reg_weight = 0;
        this->descent_strategy = 0;
        internal_solver_info = json::array();
    }

    // =======================================================================

    bool SparseNewton::compute_update_direction(
        Problem &objFunc,
        const TVector &x,
        const TVector &grad,
        TVector &direction)
    {
        if (this->descent_strategy == 2)
        {
            direction = -grad;
            return true;
        }

        polysolve::StiffnessMatrix hessian;

        assemble_hessian(objFunc, x, hessian);

        if (!solve_linear_system(hessian, grad, direction))
            // solve_linear_system will increase descent_strategy if needed
            return compute_update_direction(objFunc, x, grad, direction);

        if (!check_direction(hessian, grad, direction))
            // check_direction will increase descent_strategy if needed
            return compute_update_direction(objFunc, x, grad, direction);

        json info;
        linear_solver->getInfo(info);
        internal_solver_info.push_back(info);

        reg_weight /= reg_weight_dec;
        if (reg_weight < reg_weight_min)
            reg_weight = 0;

        return true;
    }

    // =======================================================================

    void SparseNewton::assemble_hessian(
        Problem &objFunc, const TVector &x, polysolve::StiffnessMatrix &hessian)
    {
        POLYSOLVE_SCOPED_TIMER("assembly time", this->assembly_time);

        if (this->descent_strategy == 1)
            objFunc.set_project_to_psd(true);
        else if (this->descent_strategy == 0)
            objFunc.set_project_to_psd(false);
        else
            assert(false);

        objFunc.hessian(x, hessian);

        if (reg_weight > 0)
        {
            hessian += reg_weight * sparse_identity(hessian.rows(), hessian.cols());
        }
    }

    // =======================================================================

    bool SparseNewton::solve_linear_system(
        const polysolve::StiffnessMatrix &hessian, const TVector &grad, TVector &direction)
    {
        POLYSOLVE_SCOPED_TIMER("linear solve", this->inverting_time);
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
            return false;
        }

        linear_solver->solve(-grad, direction); // H Δx = -g

        return true;
    }

    // =======================================================================

    bool SparseNewton::check_direction(
        const polysolve::StiffnessMatrix &hessian, const TVector &grad, const TVector &direction)
    {
        // gradient descent, check descent direction
        const double residual = (hessian * direction + grad).norm(); // H Δx + g = 0
        if (std::isnan(residual) || residual > std::max(1e-8 * grad.norm(), 1e-5) * characteristic_length)
        {
            increase_descent_strategy();

            m_logger.log(
                log_level(),
                "[{}] large (or nan) linear solve residual {} (||∇f||={}); reverting to {}",
                name(), residual, grad.norm(), this->descent_strategy_name());

            return false;
        }
        else
        {
            m_logger.trace("linear solve residual {}", residual);
        }

        // do this check here because we need to repeat the solve without resetting reg_weight
        if (grad.norm() != 0 && grad.dot(direction) >= 0)
        {
            increase_descent_strategy();
            m_logger.log(
                log_level(), "[{}] direction is not a descent direction (‖g‖={:g}; ‖Δx‖={:g}; Δx⋅g={:g}≥0); reverting to {}",
                name(), grad.norm(), direction.norm(), direction.dot(grad), descent_strategy_name());
            return false;
        }

        return true;
    }

    // =======================================================================

    void SparseNewton::update_solver_info(const double energy)
    {
        Superclass::update_solver_info(energy);
        this->solver_info["internal_solver"] = internal_solver_info;
    }

    // =======================================================================

    static bool has_hessian_nans(const polysolve::StiffnessMatrix &hessian)
    {
        for (int k = 0; k < hessian.outerSize(); ++k)
        {
            for (polysolve::StiffnessMatrix::InnerIterator it(hessian, k); it; ++it)
            {
                if (std::isnan(it.value()))
                    return true;
            }
        }

        return false;
    }
} // namespace polysolve::nonlinear

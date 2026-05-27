#include "Newton.hpp"

#include <polysolve/Utils.hpp>

#if defined(SPDLOG_FMT_EXTERNAL)
#include <fmt/color.h>
#else
#include <spdlog/fmt/bundled/color.h>
#endif

namespace polysolve::nonlinear
{

    std::vector<std::shared_ptr<DescentStrategy>> Newton::create_solver(
        const bool sparse,
        const json &solver_params,
        const json &linear_solver_params,
        const double characteristic_length,
        spdlog::logger &logger,
        const NormType norm_type)
    {
        // Copies stuff from main newton
        json proj_solver_params = R"({"ProjectedNewton": {}})"_json;
        proj_solver_params["ProjectedNewton"]["residual_tolerance"] = solver_params["Newton"]["residual_tolerance"];

        json reg_solver_params = R"({"RegularizedNewton": {}})"_json;
        reg_solver_params["RegularizedNewton"]["residual_tolerance"] = solver_params["Newton"]["residual_tolerance"];
        reg_solver_params["RegularizedNewton"]["reg_weight_min"] = solver_params["Newton"]["reg_weight_min"];
        reg_solver_params["RegularizedNewton"]["reg_weight_max"] = solver_params["Newton"]["reg_weight_max"];
        reg_solver_params["RegularizedNewton"]["reg_weight_inc"] = solver_params["Newton"]["reg_weight_inc"];

        std::vector<std::shared_ptr<DescentStrategy>> res;
        const bool force_psd_projection = solver_params["Newton"]["force_psd_projection"];
        if (!force_psd_projection)
            res.push_back(std::make_unique<Newton>(
                sparse,
                solver_params, linear_solver_params,
                characteristic_length, logger, norm_type));

        const bool use_psd_projection = solver_params["Newton"]["use_psd_projection"];
        if (use_psd_projection)
            res.push_back(std::make_unique<ProjectedNewton>(
                sparse,
                proj_solver_params, linear_solver_params,
                characteristic_length, logger, norm_type));

        const double reg_weight_min = solver_params["Newton"]["reg_weight_min"];
        if (reg_weight_min > 0)
            res.push_back(std::make_unique<RegularizedNewton>(
                sparse, solver_params["Newton"]["use_psd_projection_in_regularized"],
                reg_solver_params, linear_solver_params,
                characteristic_length, logger, norm_type));

        if (res.empty())
            log_and_throw_error(logger, "Newton needs to have at least one of force_psd_projection=false, reg_weight_min>0, or use_psd_projection=true");

        return res;
    }

    Newton::Newton(const bool sparse,
                   const double residual_tolerance,
                   const json &solver_params,
                   const json &linear_solver_params,
                   const double characteristic_length,
                   spdlog::logger &logger,
                   const NormType norm_type)
        : Superclass(solver_params, characteristic_length, logger),
          is_sparse(sparse), characteristic_length(characteristic_length), residual_tolerance(residual_tolerance), norm_type(norm_type)
    {
        linear_solver = polysolve::linear::Solver::create(linear_solver_params, logger);

        if (linear_solver->is_dense() == sparse)
            log_and_throw_error(logger, "Newton linear solver must be {}, instead got {}", sparse ? "sparse" : "dense", linear_solver->name());

        if (residual_tolerance <= 0)
            log_and_throw_error(logger, "Newton residual_tolerance must be > 0, instead got {}", residual_tolerance);
    }

    Newton::Newton(
        const bool sparse,
        const json &solver_params,
        const json &linear_solver_params,
        const double characteristic_length,
        spdlog::logger &logger,
        const NormType norm_type)
        : Newton(sparse, extract_param("Newton", "residual_tolerance", solver_params), solver_params, linear_solver_params, characteristic_length, logger, norm_type)
    {
    }

    ProjectedNewton::ProjectedNewton(
        const bool sparse,
        const json &solver_params,
        const json &linear_solver_params,
        const double characteristic_length,
        spdlog::logger &logger,
        const NormType norm_type)
        : Superclass(sparse, extract_param("ProjectedNewton", "residual_tolerance", solver_params), solver_params, linear_solver_params, characteristic_length, logger, norm_type)
    {
    }

    RegularizedNewton::RegularizedNewton(
        const bool sparse,
        const bool project_to_psd,
        const json &solver_params,
        const json &linear_solver_params,
        const double characteristic_length,
        spdlog::logger &logger,
        const NormType norm_type)
        : Superclass(sparse, extract_param("RegularizedNewton", "residual_tolerance", solver_params), solver_params, linear_solver_params, characteristic_length, logger, norm_type),
          project_to_psd(project_to_psd)
    {
        reg_weight_min = extract_param("RegularizedNewton", "reg_weight_min", solver_params);
        reg_weight_max = extract_param("RegularizedNewton", "reg_weight_max", solver_params);
        reg_weight_inc = extract_param("RegularizedNewton", "reg_weight_inc", solver_params);

        reg_weight = reg_weight_min;

        if (reg_weight_min <= 0)
            log_and_throw_error(logger, "Newton reg_weight_min must be  > 0, instead got {}", reg_weight_min);

        if (reg_weight_inc <= 1)
            log_and_throw_error(logger, "Newton reg_weight_inc must be  > 1, instead got {}", reg_weight_inc);

        if (reg_weight_max <= reg_weight_min)
            log_and_throw_error(logger, "Newton reg_weight_max must be  > {}, instead got {}", reg_weight_min, reg_weight_max);
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
        reg_weight = reg_weight_min;
    }

    // =======================================================================

    bool Newton::compute_update_direction(
        Problem &objFunc,
        const TVector &x,
        const TVector &grad,
        TVector &direction)
    {
        const double residual =
            is_sparse ? solve_sparse_linear_system(objFunc, x, grad, direction)
                      : solve_dense_linear_system(objFunc, x, grad, direction);

        double current_residual_tolerance = residual_tolerance;

        if (std::isnan(residual) || residual > current_residual_tolerance)
        {
            m_logger.debug("[{}] large (or nan) linear solve residual {}>{} (‖∇f‖={})",
                           name(), residual, current_residual_tolerance, objFunc.grad_norm(grad, norm_type));

            return false;
        }
        else
        {
            m_logger.trace("linear solve residual {}", residual);
        }

        return true;
    }

    // =======================================================================

    double Newton::solve_sparse_linear_system(Problem &objFunc,
                                              const TVector &x,
                                              const TVector &grad,
                                              TVector &direction)
    {
        Hessian hessian(std::in_place_type<NewtonHessian>);
        {
            POLYSOLVE_SCOPED_STOPWATCH("assembly time", this->assembly_time, m_logger);
            compute_hessian(objFunc, x, hessian);
        }
        {
            // TODO: get the correct size
            if (objFunc.getSparsityPatternID() == -1)
            {
                POLYSOLVE_SCOPED_STOPWATCH("symbolic factorize", this->symbolic_factorizer_time, m_logger);
                linear_solver->analyze_pattern(hessian, hessian.rows());
            }

            try
            {
                POLYSOLVE_SCOPED_STOPWATCH("numeric factorize", this->numeric_factorizer_time, m_logger);
                linear_solver->factorize(hessian);
            }
            catch (const std::runtime_error &err)
            {
                // warn if using gradient descent
                m_logger.debug("Unable to factorize Hessian: \"{}\"", err.what());

                // Eigen::saveMarket(hessian, "problematic_hessian.mtx");
                return std::nan("");
            }
            {
                POLYSOLVE_SCOPED_STOPWATCH("linear solve", this->linear_solve_time, m_logger);
                linear_solver->solve(-grad, direction); // H Δx = -g
            }
        }

        const double residual = objFunc.grad_norm(hessian * direction + grad, norm_type); // H Δx + g = 0

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
        Hessian hessian_v(std::in_place_type<Eigen::MatrixXd>);
        {
            POLYSOLVE_SCOPED_STOPWATCH("assembly time", this->assembly_time, m_logger);
            compute_hessian(objFunc, x, hessian_v);
        }
        const Eigen::MatrixXd &hessian = hessian_v.get<Eigen::MatrixXd>();
        {
            POLYSOLVE_SCOPED_STOPWATCH("linear solve", this->solve_time, m_logger);

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
                                 Hessian &hessian)
    {
        objFunc.set_project_to_psd(false);
        objFunc.hessian(x, hessian);
    }

    void ProjectedNewton::compute_hessian(Problem &objFunc,
                                          const TVector &x,
                                          Hessian &hessian)
    {
        objFunc.set_project_to_psd(true);
        objFunc.hessian(x, hessian);
    }

    void RegularizedNewton::compute_hessian(Problem &objFunc,
                                            const TVector &x,
                                            Hessian &hessian)
    {
        objFunc.set_project_to_psd(project_to_psd);
        objFunc.hessian(x, hessian);

        // std::visit([&](auto &h) {
        //     using T = std::decay_t<decltype(h)>;
        //     if constexpr (std::is_same_v<T, polysolve::StiffnessMatrix>) {
        //         if (x.size() != x_cache.size() || x != x_cache) {
        //             objFunc.hessian(x, hessian_cache);
        //             x_cache = x;
        //         }
        //         h = hessian_cache;
        //         if (reg_weight > 0)
        //             h += reg_weight * sparse_identity(h.rows(), h.cols());
        //     } else if constexpr (std::is_same_v<T, Eigen::MatrixXd>) {
        //         objFunc.hessian(x, h);
        //         if (reg_weight > 0)
        //             for (int k = 0; k < x.size(); k++)
        //                 h(k, k) += reg_weight;
        //     } else if constexpr (std::is_same_v<T, NewtonHessian>) {
        //         h = objFunc.evalHessian(x);
        //         // TODO: add reg_weight * identity
        //     }
        // }, hessian);
    }

    // =======================================================================

    bool RegularizedNewton::handle_error()
    {
        reg_weight *= reg_weight_inc;
        return reg_weight < reg_weight_max;
    }
    // =======================================================================

    void Newton::update_solver_info(json &solver_info, const double per_iteration)
    {
        Superclass::update_solver_info(solver_info, per_iteration);

        solver_info["internal_solver"] = internal_solver_info;
        solver_info["time_assembly"] = assembly_time / per_iteration;
        solver_info["time_inverting"] = inverting_time / per_iteration;
    }

    void Newton::reset_times()
    {
        assembly_time = 0;
        inverting_time = 0;
        linear_solve_time = 0;
        symbolic_factorizer_time = 0;
        numeric_factorizer_time = 0;
        solve_time = 0;
    }

     void Newton::update_times(std::vector<double> &linear_times)
    {
        linear_times[0] += assembly_time;
        linear_times[1] += symbolic_factorizer_time;
        linear_times[2] += numeric_factorizer_time;
        linear_times[3] += is_sparse ? linear_solve_time : solve_time;
    }

    void Newton::log_times() const
    {
        if (assembly_time <= 0 && inverting_time <= 0)
            return; // nothing to log
        m_logger.debug(
            "[{}][{}] assembly: {:.2e}s; linear_solve: {:.2e}s",
            fmt::format(fmt::fg(fmt::terminal_color::magenta), "timing"),
            name(), assembly_time, inverting_time);
    }

    // =======================================================================

} // namespace polysolve::nonlinear

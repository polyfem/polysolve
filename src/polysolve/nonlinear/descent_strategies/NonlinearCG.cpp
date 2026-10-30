#include "NonlinearCG.hpp"

#include <polysolve/Utils.hpp>

#include <algorithm>

namespace polysolve::nonlinear
{
    NonlinearCG::NonlinearCG(const json &solver_params,
                             const double characteristic_length,
                             spdlog::logger &logger)
        : Superclass(solver_params, characteristic_length, logger)
    {
        const json &params = solver_params.find("NonlinearCG") != solver_params.end()
                                  ? solver_params["NonlinearCG"]
                                  : solver_params;

        const std::string formula = params["formula"];
        if (formula == "FletcherReeves")
            beta_formula_ = BetaFormula::FletcherReeves;
        else if (formula == "PolakRibiere")
            beta_formula_ = BetaFormula::PolakRibiere;
        else if (formula == "HestenesStiefel")
            beta_formula_ = BetaFormula::HestenesStiefel;
        else
            log_and_throw_error(logger, "Unknown NonlinearCG formula: {}", formula);

        restart_frequency_ = params["restart_frequency"];
        if (restart_frequency_ < 0)
            log_and_throw_error(logger, "NonlinearCG restart_frequency must be >= 0, instead got {}", restart_frequency_);
    }

    void NonlinearCG::reset(const int ndof)
    {
        Superclass::reset(ndof);
        ndof_ = ndof;
        prev_direction_.resize(0);
        iters_since_restart_ = 0;
    }

    bool NonlinearCG::compute_update_direction(
        Problem &objFunc,
        const TVector &x,
        const TVector &grad,
        TVector &direction)
    {
        const int effective_restart_frequency = restart_frequency_ > 0 ? restart_frequency_ : ndof_;
        const bool restart = prev_direction_.size() == 0
                              || (effective_restart_frequency > 0 && iters_since_restart_ >= effective_restart_frequency);

        if (restart)
        {
            direction = -grad;
        }
        else
        {
            const double prev_grad_sq_norm = prev_grad_.squaredNorm();
            double beta = 0;

            if (prev_grad_sq_norm > 0)
            {
                switch (beta_formula_)
                {
                case BetaFormula::FletcherReeves:
                    beta = grad.squaredNorm() / prev_grad_sq_norm;
                    break;
                case BetaFormula::PolakRibiere:
                    beta = grad.dot(grad - prev_grad_) / prev_grad_sq_norm;
                    break;
                case BetaFormula::HestenesStiefel:
                {
                    const TVector y = grad - prev_grad_;
                    const double denom = prev_direction_.dot(y);
                    beta = denom != 0 ? grad.dot(y) / denom : 0;
                    break;
                }
                default:
                    log_and_throw_error(m_logger, "Unknown NonlinearCG beta formula");
                }
            }

            // Negative beta triggers an automatic restart to steepest descent.
            beta = std::max(beta, 0.0);

            direction = -grad + beta * prev_direction_;

            // Guard against loss of descent from numerical error.
            if (grad.dot(direction) >= 0)
                direction = -grad;
        }

        prev_grad_ = grad;
        prev_direction_ = direction;
        iters_since_restart_ = restart ? 1 : iters_since_restart_ + 1;

        return true;
    }
} // namespace polysolve::nonlinear

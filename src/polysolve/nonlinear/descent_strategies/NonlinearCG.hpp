#pragma once

#include "DescentStrategy.hpp"
#include <polysolve/Utils.hpp>

namespace polysolve::nonlinear
{
    /// @brief Unpreconditioned nonlinear conjugate gradient descent strategy.
    ///
    /// Computes the search direction as d_k = -g_k + beta_k * d_{k-1}, where
    /// beta_k is computed using the Fletcher-Reeves, Polak-Ribiere, or
    /// Hestenes-Stiefel formula (clipped to be non-negative, so a negative
    /// beta triggers an automatic restart to steepest descent). The direction
    /// is also periodically restarted to steepest descent every
    /// restart_frequency iterations.
    class NonlinearCG : public DescentStrategy
    {
    public:
        using Superclass = DescentStrategy;

        enum class BetaFormula
        {
            FletcherReeves,
            PolakRibiere,
            HestenesStiefel
        };

        NonlinearCG(const json &solver_params,
                    const double characteristic_length,
                    spdlog::logger &logger);

        std::string name() const override { return "NonlinearCG"; }

        void reset(const int ndof) override;

        bool compute_update_direction(
            Problem &objFunc,
            const TVector &x,
            const TVector &grad,
            TVector &direction) override;

    private:
        BetaFormula beta_formula_;
        /// Number of iterations between forced restarts to steepest descent.
        /// A value of 0 defaults to the number of degrees of freedom.
        int restart_frequency_;

        int ndof_ = 0;
        int iters_since_restart_ = 0;

        TVector prev_grad_;
        TVector prev_direction_;
    };
} // namespace polysolve::nonlinear

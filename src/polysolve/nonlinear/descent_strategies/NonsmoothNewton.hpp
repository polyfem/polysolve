#pragma once

#include "DescentStrategy.hpp"
#include <polysolve/Utils.hpp>

#include <polysolve/linear/Solver.hpp>

#include <memory>
#include <string>

namespace polysolve::nonlinear
{
    /// @brief Newton with a protected path on the split f = s + n of the problem (smooth part s, nonsmooth term n; see
    /// Problem::has_nonsmooth_term).
    ///
    /// The direction is p_N = -B^-1 g with B = H_s + proj(H_n) (B = proj(H_s) + proj(H_n) if B is not positive definite
    /// or p_N needs a strong validity shortening). On an iterate with active nonsmooth terms, take_step minimizes the
    /// reduced model m(p) = g_s^T p + 1/2 p^T H_s p + n(x + p) - n(x) with up to max_inner Levenberg-Marquardt steps in a
    /// subspace that starts with p_N and grows with B^-1 (model residual) columns (one factorization of B), limits each
    /// step by a segment guard, a linearized fraction-to-boundary and max_step_size on the segment, and accepts the end
    /// point of the resulting polyline (or a retraction along it) by its true decrease. Otherwise p_N is line searched.
    /// Optional limits on the element volume ratios of the inner points (Options::jacobian_bound, jacobian_floor) keep
    /// the path where the frozen quadratic of s is trusted.
    /// Once the true energy can no longer verify the path (Options::handoff), the rest of the solve is left to the Newton
    /// chain; after a handoff on a small gradient, a slow chain step hands the iterations back (Options::hand_back).
    /// Without a split (has_nonsmooth_term() false) the direction is the plain Newton direction.
    class NonsmoothNewton : public DescentStrategy
    {
    public:
        using Superclass = DescentStrategy;

        /// @brief Parameters (JSON: NonsmoothNewton)
        struct Options
        {
            int max_inner = 40;          ///< inner budget K_max
            double relative_stop = 1e-3; ///< inner stop when a step's model decrease < relative_stop * accumulated decrease (0 = off)
            double ftb_tau = 0.2;        ///< linearized fraction-to-boundary on the active terms: d >= tau d(a) (0 = off)
            bool guard = true;           ///< long-segment guard: validity shortening and scale-not-reject segment cap
            double guard_c = 2.0;        ///< segment cap = min(c * alpha_v * |p_N|_n, abs * h) + activation distance
            double guard_abs = 10.0;
            double guard_check = 0.5; ///< validity shortening of p_N when |p_N|_n > check * h
            double rho_min = 1e-4;    ///< true-energy acceptance ratio
            int max_columns = 100;    ///< subspace column cap
            /// hand the rest of the solve to the Newton chain once the true energy can no longer verify the path: "any"
            /// (an outer iteration whose inner loop stopped at the noise floor or whose path is below the resolution),
            /// "first" (only if no inner step was accepted before) or "off"; both also hand off when the gradient norm is
            /// below line_search/use_grad_norm_tol. Decided before any factorization.
            std::string handoff = "any";
            /// resolution of the true energy E: the inner noise floor also counts 32 eps |E|, and a path whose model
            /// decrease does not exceed 32 eps |E| is not verified by the true energy
            bool resolution = true;
            /// after a handoff on a small gradient, the first step of the Newton chain that ends above the gradient
            /// tolerance, removes more than hand_back_energy times the energy resolution 32 eps |E| and is slow (cut by
            /// its line search, taken by a strategy at index >= 2 of the solver, i.e. behind Newton, or reducing the
            /// gradient norm less than hand_back_contraction times) hands the iterations back to this strategy; the
            /// handoff on a small gradient is then off for the rest of the solve
            bool hand_back = true;
            double hand_back_energy = 100;
            double hand_back_contraction = 10;
            /// Jacobian limits at the end point y of every inner step, from Problem::smooth_min_jacobian(s) (not applied
            /// where they give NaN). Bound: every element keeps J_e(y) >= jacobian_bound * J_e(x), x the outer iterate.
            /// Floor: min_e J_e(y) >= jacobian_floor * J_ref, or >= min_e J_e(x) where x is already below that; J_ref is
            /// min_e J_e at the first iterate of the solve whose direction request this strategy does not decline by the
            /// handoff (and where the problem gives a finite value). A later step that violates one ends the path at the
            /// previous point; a violating first step is halved until it holds and then ends the path. 0 = off.
            double jacobian_bound = 0.6;
            double jacobian_floor = 0.1;

            // internal switches (debug block) used by the strategy
            bool dedupe = true;        ///< true energy at a trial point reuses the nonsmooth state rebuilt there
            bool fast_validity = true; ///< validity check at every segment end

            static Options from_json(const json &solver_params);
        };

        NonsmoothNewton(const json &solver_params,
                        const json &linear_solver_params,
                        const double characteristic_length,
                        spdlog::logger &logger,
                        const NormType norm_type);
        ~NonsmoothNewton() override;

        std::string name() const override { return "NonsmoothNewton"; }

        const Options &options() const { return opts_; }

        bool compute_update_direction(Problem &objFunc, const TVector &x, const TVector &grad, TVector &direction) override;
        bool take_step(Problem &objFunc, const TVector &x, const double energy, const TVector &grad,
                       const TVector &direction, TVector &step) override;
        bool handle_error() override;
        /// @brief The hand-back rule (Options::hand_back)
        void chain_step_result(const int strategy, const double rate, const double energy_before, const double energy_after,
                               const double grad_norm_before, const double grad_norm_after, const double grad_tol) override;

        /// @brief Resets the counters (once per minimize)
        void reset_times() override;
        /// @brief Counters in solver_info["NonsmoothNewton"]
        void update_solver_info(json &solver_info, const double per_iteration) override;

    private:
        struct Impl;
        Options opts_;
        std::unique_ptr<Impl> impl_;
    };
} // namespace polysolve::nonlinear

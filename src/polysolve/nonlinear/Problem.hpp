#pragma once

#include <polysolve/Types.hpp>

#include "Criteria.hpp"
#include "PostStepData.hpp"

#include <memory>
#include <vector>

namespace polysolve::nonlinear
{
    /// @brief Class defining optimization problem to be solved. To be defined by user code
    class Problem
    {
    public:
        static constexpr int Dim = Eigen::Dynamic;
        using Scalar = double;
        using TVector = Eigen::Matrix<Scalar, Dim, 1>;
        using TMatrix = Eigen::Matrix<Scalar, Dim, Dim>;
        using THessian = StiffnessMatrix;

    public:
        Problem() {}
        virtual ~Problem() = default;

        /// @brief Initialize the problem.
        /// @param x0 Initial guess.
        virtual void init(const TVector &x0) {}

        /// @brief Compute the value of the function at x.
        /// @param x Degrees of freedom.
        /// @return The value of the function at x.
        Scalar operator()(const TVector &x) { return value(x); }

        /// @brief Compute the value of the function at x.
        /// @param x Degrees of freedom.
        /// @return The value of the function at x.
        virtual Scalar value(const TVector &x) = 0;

        /// @brief Compute the gradient of the function at x.
        /// @param[in] x Degrees of freedom.
        /// @param[out] grad Gradient of the function at x.
        virtual void gradient(const TVector &x, TVector &grad) = 0;

        /// @brief Compute the Hessian of the function at x.
        /// @param[in] x Degrees of freedom.
        /// @param[out] hessian Hessian of the function at x.
        virtual void hessian(const TVector &x, TMatrix &hessian)
        {
            throw std::runtime_error("Dense Hessian not implemented.");
        }

        /// @brief Compute the Hessian of the function at x.
        /// @param[in] x Degrees of freedom.
        /// @param[out] hessian Hessian of the function at x.
        virtual void hessian(const TVector &x, THessian &hessian) = 0;

        /// @brief Determine if the step from x0 to x1 is valid.
        /// @param x0 Starting point.
        /// @param x1 Ending point.
        /// @return True if the step is valid, false otherwise.
        virtual bool is_step_valid(const TVector &x0, const TVector &x1) { return true; }

        /// @brief Determine a maximum step size from x0 to x1.
        /// @param x0 Starting point.
        /// @param x1 Ending point.
        /// @return Maximum step size.
        virtual double max_step_size(const TVector &x0, const TVector &x1) { return 1; }

        // --- Callbacks ------------------------------------------------------

        /// @brief Callback function for the start of a line search.
        /// @param x0 Starting point.
        /// @param x1 Ending point.
        virtual void line_search_begin(const TVector &x0, const TVector &x1) {}

        /// @brief Callback function for the end of a line search.
        virtual void line_search_end() {}

        /// @brief Callback function for the end of a step.
        /// @param data Post step data.
        virtual void post_step(const PostStepData &data) {}

        /// @brief Set the project to PSD flag.
        /// @param val True if the problem should be projected to PSD, false otherwise.
        virtual void set_project_to_psd(bool val) {}

        /// @brief Callback function for when the solution changes.
        /// @param new_x New solution.
        virtual void solution_changed(const TVector &new_x) {}

        virtual bool after_line_search_custom_operation(const TVector &x0, const TVector &x1) { return false; }

        /// @brief Callback function used to determine if the solver should stop.
        /// @param state Current state of the solver.
        /// @param x Current solution.
        /// @return True if the solver should stop, false otherwise.
        virtual bool callback(const Criteria &state, const TVector &x) { return true; }

        /// @brief Callback function used Determine if the solver should stop.
        /// @param x Current solution.
        /// @return True if the solver should stop, false otherwise.
        virtual bool stop(const TVector &x) { return false; }

        /// --- Misc ----------------------------------------------------------

        /// @brief Sample the function along a direction.
        /// @param[in] x Starting point.
        /// @param[in] direction Direction to sample along.
        /// @param[in] start Starting step size.
        /// @param[in] end Ending step size.
        /// @param[in] num_samples Number of samples to take.
        /// @param[out] alphas Sampled step sizes.
        /// @param[out] fs Sampled function values.
        /// @param[out] valid If each sample is valid.
        void sample_along_direction(
            const Problem::TVector &x,
            const Problem::TVector &direction,
            const double start,
            const double end,
            const int num_samples,
            Eigen::VectorXd &alphas,
            Eigen::VectorXd &fs,
            Eigen::VectorXi &valid);
    };
} // namespace polysolve::nonlinear

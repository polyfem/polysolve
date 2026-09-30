#pragma once

#include <polysolve/Types.hpp>

#include "Criteria.hpp"
#include "PostStepData.hpp"

#include <memory>
#include <set>
#include <vector>

namespace polysolve::nonlinear
{

    enum class NormType
    {
        EUCLIDEAN = 0,
        L2 = 1,
        Linf = 2
    };

    // Defined here (rather than in Solver.cpp, where it previously lived) so
    // that every translation unit that converts a JSON "norm_type" string to
    // a NormType sees this mapping via ADL. NLOHMANN_JSON_SERIALIZE_ENUM
    // expands to template<> to_json/from_json overloads, so defining it in a
    // header is safe (no ODR/duplicate-symbol issues); previously, code
    // outside Solver.cpp's own translation unit silently fell back to
    // nlohmann's generic enum-as-integer conversion, which rejects the
    // string values ("L2", etc.) that json-specs actually declares for this
    // field.
    NLOHMANN_JSON_SERIALIZE_ENUM(
        NormType,
        {{NormType::EUCLIDEAN, "Euclidean"},
         {NormType::L2, "L2"},
         {NormType::Linf, "Linf"}})

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

        virtual bool is_residual() const { return false; }

        /// @brief Number of scalar components per node (e.g., 1 for scalar problems,
        /// 2 or 3 for vector-valued problems). Used as the block size for
        /// multigrid-style linear solvers.
        virtual int dimension() const { return 1; }

        /// @brief Per-row function (block) assignment for multigrid-style linear
        /// solvers, passed through to Solver::set_block_mapping. An empty vector
        /// (the default) means the solver should fall back to its own default
        /// mapping instead of a problem-specific one.
        virtual Eigen::VectorXi block_mapping() const { return Eigen::VectorXi(); }

        /// @brief Report DOFs the linear solver should treat as problematic
        /// (e.g. poorly conditioned or in contact), letting solvers that support
        /// externally-driven subdomain selection (e.g. AMGF) use them instead of
        /// their own row-norm heuristics. Default reports none.
        virtual void get_problematic_dofs(std::set<int> &bad_dofs) {}

        /// @brief Contact patches (one set of DOFs per patch), passed through to
        /// Solver::contact_patches for solvers supporting Schwarz-style handling
        /// of contact DOFs.
        std::vector<std::set<int>> contact_patches;

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

        virtual double grad_norm_rescaling(const NormType norm_type) const { return 1; }
        virtual double step_norm_rescaling(const NormType norm_type) const { return 1; }
        virtual double energy_norm_rescaling(const NormType norm_type) const { return 1; }

        virtual double grad_norm(const TVector &grad, const NormType norm_type) const { return grad.norm(); }
        virtual double step_norm(const TVector &x, const NormType norm_type) const { return x.norm(); }

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

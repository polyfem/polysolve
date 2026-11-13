#pragma once

#include <polysolve/Types.hpp>

#include "Criteria.hpp"
#include "PostStepData.hpp"

#include <array>
#include <limits>
#include <memory>
#include <vector>

namespace polysolve::nonlinear
{

    enum class NormType
    {
        EUCLIDEAN = 0,
        L2 = 1,
        Linf = 2
    };

    /// @brief Dense block of a sparse symmetric matrix: H acts on the dofs indices[0..size) (-1 = no dof, the row and
    /// column are ignored). Used for the optional block form of the nonsmooth term's Hessian.
    /// @note H is unaligned so that the layout does not depend on the Eigen vectorization settings of the translation unit.
    struct HessianBlock
    {
        static constexpr int max_size = 12;
        std::array<int, max_size> indices;
        int size = 0;
        Eigen::Matrix<double, max_size, max_size, Eigen::DontAlign> H;
    };

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

        // --- Nonsmooth term -------------------------------------------------
        // Optional split f = s + n of the objective into a smooth part s and a nonsmooth term n (e.g. a stiff penalty or a log term), used by
        // strategies that model the two parts differently (NonsmoothNewton). The defaults report that no split is
        // available; the other callbacks are only called when has_nonsmooth_term() is true.

        /// @brief True if the callbacks below are implemented for the current problem
        virtual bool has_nonsmooth_term() const { return false; }

        /// @brief Gradient of the smooth part s at x.
        virtual void smooth_gradient(const TVector &x, TVector &grad) { not_supported("smooth_gradient"); }

        /// @brief Hessian of the smooth part s at x (projected to PSD if project_to_psd).
        virtual void smooth_hessian(const TVector &x, const bool project_to_psd, THessian &hessian) { not_supported("smooth_hessian"); }

        /// @brief Largest 2^-i (i < 40) such that a + 2^-i (b - a) is in the domain of the smooth part (is_step_valid and
        /// s finite); 0 if there is none.
        /// @param x Current iterate of the solver (the problem may cache data anchored there).
        virtual double smooth_valid_fraction(const TVector &x, const TVector &a, const TVector &b)
        {
            not_supported("smooth_valid_fraction");
            return 0;
        }

        /// @brief Optional: smallest volume ratio J = det F (deformation gradient relative to the rest configuration) over
        /// the elements of the smooth part at x; NaN if not available (the default). Must not change the state of the
        /// nonsmooth term.
        virtual double smooth_min_jacobian(const TVector &x) { return std::numeric_limits<double>::quiet_NaN(); }

        /// @brief Optional: in one sweep over the elements of the smooth part, the smallest J at y (if want_min; as
        /// smooth_min_jacobian(y)) and the smallest ratio J_e(y) / J_e(x) of an element's volume ratio at y to its value
        /// at x (if want_ratio); NaN where not requested or not available. The default gives smooth_min_jacobian(y) and
        /// no ratio. Must not change the state of the nonsmooth term.
        /// @param x Current iterate of the solver (the problem may cache data anchored there).
        virtual void smooth_min_jacobians(const TVector &x, const TVector &y, const bool want_min, const bool want_ratio,
                                          double &min_j, double &min_ratio)
        {
            min_j = want_min ? smooth_min_jacobian(y) : std::numeric_limits<double>::quiet_NaN();
            min_ratio = std::numeric_limits<double>::quiet_NaN();
        }

        /// @brief Diagonal of the metric used to measure steps (e.g. a lumped mass).
        virtual void metric_diagonal(TVector &diag) { not_supported("metric_diagonal"); }

        /// @brief Rebuild the state of the nonsmooth term (its active set) at y. The value, derivatives and active-set
        /// queries below refer to the state of the last rebuild (or of the last solution_changed).
        virtual void nonsmooth_update(const TVector &y) { not_supported("nonsmooth_update"); }

        /// @brief Value of the nonsmooth term at y.
        virtual double nonsmooth_value(const TVector &y)
        {
            not_supported("nonsmooth_value");
            return 0;
        }

        /// @brief Gradient of the nonsmooth term at y.
        virtual void nonsmooth_gradient(const TVector &y, TVector &grad) { not_supported("nonsmooth_gradient"); }

        /// @brief Hessian of the nonsmooth term at y (projected to PSD if project_to_psd).
        virtual void nonsmooth_hessian(const TVector &y, const bool project_to_psd, THessian &hessian) { not_supported("nonsmooth_hessian"); }

        /// @brief Optional: gradient and unprojected Hessian of the nonsmooth term at y as dense blocks.
        /// @return False if not available (the caller then uses nonsmooth_gradient and nonsmooth_hessian).
        virtual bool nonsmooth_hessian_blocks(const TVector &y, TVector &grad, std::vector<HessianBlock> &blocks) { return false; }

        /// @brief Number of active terms in the current state.
        virtual int nonsmooth_active_size()
        {
            not_supported("nonsmooth_active_size");
            return 0;
        }

        /// @brief For the active terms of the current state with a positive distance at a: the distance at a and its
        /// directional derivative along the segment a -> b.
        virtual void nonsmooth_active_distances(const TVector &a, const TVector &b, TVector &distance, TVector &rate) { not_supported("nonsmooth_active_distances"); }

        /// @brief Length of the step a -> b measured on the geometry of the nonsmooth term (largest point displacement).
        virtual double nonsmooth_step_length(const TVector &a, const TVector &b)
        {
            not_supported("nonsmooth_step_length");
            return 0;
        }

        /// @brief Characteristic length of the geometry of the nonsmooth term (e.g. its mean edge length).
        virtual double nonsmooth_length_scale()
        {
            not_supported("nonsmooth_length_scale");
            return 0;
        }

        /// @brief Distance below which a term becomes active.
        virtual double nonsmooth_activation_distance()
        {
            not_supported("nonsmooth_activation_distance");
            return 0;
        }

        /// @brief Start of one segment a -> b of a piecewise-linear path: until nonsmooth_segment_end, max_step_size and
        /// the nonsmooth state may reuse data of this segment (e.g. its collision candidates).
        virtual void nonsmooth_segment_begin(const TVector &a, const TVector &b) { line_search_begin(a, b); }

        /// @brief End of the segment started by nonsmooth_segment_begin.
        virtual void nonsmooth_segment_end() { line_search_end(); }

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

    private:
        [[noreturn]] static void not_supported(const char *what);
    };
} // namespace polysolve::nonlinear

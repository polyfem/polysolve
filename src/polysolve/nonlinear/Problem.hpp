#pragma once

#include <polysolve/Types.hpp>

#include "PostStepData.hpp"

#include <cppoptlib/problem.h>

#include <memory>
#include <vector>

namespace polysolve::nonlinear
{
    /// @brief Class defining optimization problem to be solved. To be defined by user code
    class Problem : public cppoptlib::Problem<double>
    {
    public:
        using typename cppoptlib::Problem<double>::Scalar;
        using typename cppoptlib::Problem<double>::TVector;
        typedef polysolve::StiffnessMatrix THessian;

        // disable warning for dense hessian
        using cppoptlib::Problem<double>::hessian;

        Problem() {}
        ~Problem() = default;

        virtual void init(const TVector &x0) {}

        /// @brief Defines function to be optimized
        /// @param x input vector (n x 1)
        /// @return value of function
        virtual double value(const TVector &x) override = 0;

        /// @brief Defines gradient of objective function
        /// @param[in] x input vector (n x 1)
        /// @param[out] gradv gradient vector (n x 1)
        virtual void gradient(const TVector &x, TVector &gradv) override = 0;
        
        /// @brief Defines Hessian of objective function
        /// @param[in] x input vector (n x 1)
        /// @param[out] hessian hessian matrix (n x n)
        virtual void hessian(const TVector &x, THessian &hessian) = 0;
		
        /// @brief Determine if a step from solution x0 to solution x1 is allowed
		/// @param x0 Current solution
		/// @param x1 Proposed next solution
		/// @return True if the step is allowed
        virtual bool is_step_valid(const TVector &x0, const TVector &x1) { return true; }
        
        /// @param x0 Current solution (step size = 0)
		/// @param x1 Next solution (step size = 1)
		/// @return Maximum allowable step size
        virtual double max_step_size(const TVector &x0, const TVector &x1) { return 1; }

        /// @brief Initialize variables used during the line search
		/// @param x0 Current solution
		/// @param x1 Next solution
        virtual void line_search_begin(const TVector &x0, const TVector &x1) {}
        
        /// @brief Clear variables used during the line search
        virtual void line_search_end() {}
        
        /// @brief Update fields after a step in the optimization
		/// @param iter_num Optimization iteration number
		/// @param x Current solution
		/// @param data Data containing info about the current iteration
        virtual void post_step(const PostStepData &data) {}

        /// @brief Set project to psd
		/// @param val If true, the form's second derivative is projected to be positive semidefinite
        virtual void set_project_to_psd(bool val) {}

		/// @brief Update cached fields upon a change in the solution
		/// @param new_x New solution
        virtual void solution_changed(const TVector &new_x) {}

        virtual bool stop(const TVector &x) { return false; }

        /// @brief Sample function along a given direction
        /// @param[in] x starting input value (n x 1)
        /// @param[in] direction direction along which to sample (n x 1)
        /// @param[in] start starting step size
        /// @param[in] end ending step size
        /// @param[in] num_samples total number of samples
        /// @param[out] alphas step sizes (num_samples x 1)
        /// @param[out] fs function values along the given direction (num_samples x 1)
        /// @param[out] valid whether or not the given step is valid (num_samples x 1)
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

#pragma once

#include <polysolve/Types.hpp>

#include <cppoptlib/problem.h>

#include <memory>
#include <vector>

namespace polysolve::nonlinear
{
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

        virtual double value(const TVector &x) override = 0;
        virtual void gradient(const TVector &x, TVector &gradv) override = 0;
        virtual void hessian(const TVector &x, THessian &hessian) = 0;

        virtual bool is_step_valid(const TVector &x0, const TVector &x1) const { return true; }
        virtual double max_step_size(const TVector &x0, const TVector &x1) const { return 1; }

        virtual void line_search_begin(const TVector &x0, const TVector &x1) {}
        virtual void line_search_end() {}
        virtual void post_step(const int iter_num, const TVector &x) {}

        virtual void set_project_to_psd(bool val) {}

        virtual void solution_changed(const TVector &new_x) {}

        virtual bool stop(const TVector &x) { return false; }

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

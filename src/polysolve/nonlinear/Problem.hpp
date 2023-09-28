#pragma once

#include <polysolve/types.hpp>

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
        virtual bool is_step_collision_free(const TVector &x0, const TVector &x1) const { return true; }
        virtual double max_step_size(const TVector &x0, const TVector &x1) const { return 1; }

        virtual void line_search_begin(const TVector &x0, const TVector &x1) {}
        virtual void line_search_end() {}
        virtual void post_step(const int iter_num, const TVector &x) {}

        virtual void set_project_to_psd(bool val) {}

        virtual void solution_changed(const TVector &new_x) {}

        virtual void init_lagging(const TVector &x) {}
        virtual void update_lagging(const TVector &x, const int iter_num) {}
        int max_lagging_iterations() const { return -1; }
        bool uses_lagging() const { return false; }

        virtual void save_to_file(const TVector &x0) {}

        virtual bool stop(const TVector &x) { return false; }
    };
} // namespace polysolve::nonlinear

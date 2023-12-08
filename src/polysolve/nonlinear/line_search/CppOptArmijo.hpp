#pragma once

#include "LineSearch.hpp"
#include "Backtracking.hpp"

namespace polysolve::nonlinear::line_search
{
    class CppOptArmijo : public LineSearch
    {
    public:
        using Superclass = LineSearch;
        using typename Superclass::Scalar;
        using typename Superclass::TVector;

        CppOptArmijo(const json &params, spdlog::logger &logger);

        virtual std::string name() override { return "ArmijoAlt"; }

    protected:
        double compute_descent_step_size(
            const TVector &x,
            const TVector &delta_x,
            Problem &objFunc,
            const bool use_grad_norm,
            const double old_energy,
            const TVector &old_grad,
            const double starting_step_size) override;

    private:
        Backtracking after_check;
    };
} // namespace polysolve::nonlinear::line_search

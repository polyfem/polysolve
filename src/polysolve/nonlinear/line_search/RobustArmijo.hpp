#pragma once

#include "Armijo.hpp"

namespace polysolve::nonlinear::line_search
{
    /// @brief Numerically robust Armijo line search algorithm of Longva et al. [2023]
    class RobustArmijo : public Armijo
    {
    public:
        using Superclass = Armijo;
        using typename Superclass::Scalar;
        using typename Superclass::TVector;

        RobustArmijo(const json &params, spdlog::logger &logger);

        virtual std::string name() const override { return "RobustArmijo"; }

    protected:
        bool criteria(
            const TVector &delta_x,
            Problem &objFunc,
            const bool use_grad_norm,
            const double old_energy,
            const TVector &old_grad,
            const TVector &new_x,
            const double new_energy,
            const double step_size) const override;

        double delta_relative_tolerance;
    };

} // namespace polysolve::nonlinear::line_search

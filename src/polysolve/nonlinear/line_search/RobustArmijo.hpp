#pragma once

#include "Armijo.hpp"

namespace polysolve::nonlinear::line_search
{
    class RobustArmijo : public Armijo
    {
    public:
        using Superclass = Armijo;
        using typename Superclass::Scalar;
        using typename Superclass::TVector;

        RobustArmijo(const json &params, spdlog::logger &logger);

        virtual std::string name() override { return "RobustArmijo"; }

    protected:
        bool criteria(
            const TVector &delta_x,
            const double old_energy,
            const TVector &old_grad,
            const double new_energy,
            const TVector &new_grad,
            const double step_size) const override;
    };

} // namespace polysolve::nonlinear::line_search

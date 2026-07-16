#pragma once

#include "Backtracking.hpp"

namespace polysolve::nonlinear::line_search
{
    class ResidualBacktracking : public Backtracking
    {
    public:
        ResidualBacktracking(const json &params, spdlog::logger &logger);

        virtual std::string name() const override { return "ResidualBacktracking"; }

    protected:
        bool criteria(const TVector &delta_x,
                      Problem &objFunc,
                      const bool use_grad_norm,
                      const double old_energy,
                      const TVector &old_grad,
                      const TVector &new_x,
                      const double new_energy,
                      const double step_size) const override;
    };

} // namespace polysolve::nonlinear::line_search
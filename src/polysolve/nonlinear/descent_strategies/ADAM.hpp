#pragma once

#include "DescentStrategy.hpp"
#include <polysolve/Utils.hpp>

#include <polysolve/linear/Solver.hpp>

namespace polysolve::nonlinear
{
    class ADAM : public DescentStrategy
    {
    public:
        using Superclass = DescentStrategy;

        ADAM(const json &solver_params,
             const double characteristic_length,
             spdlog::logger &logger);

        std::string name() const override { return "ADAM"; }

        void reset(const int ndof) override;

        virtual bool compute_update_direction(
            Problem &objFunc,
            const TVector &x,
            const TVector &grad,
            TVector &direction) override;

    private:
        TVector m_prev;
        TVector v_prev;

        double beta_1, beta_2;
        double alpha;

        int t = 0;
        double epsilon;
    };
} // namespace polysolve::nonlinear

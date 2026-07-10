#include "ResidualBacktracking.hpp"

#include <polysolve/Utils.hpp>

#include <spdlog/spdlog.h>

namespace polysolve::nonlinear::line_search
{

    ResidualBacktracking::ResidualBacktracking(const json &params, spdlog::logger &logger)
        : Backtracking(params, logger)
    {
    }

    bool ResidualBacktracking::criteria(const TVector &delta_x,
                                        Problem &objFunc,
                                        const bool use_grad_norm,
                                        const double old_energy,
                                        const TVector &old_grad,
                                        const TVector &new_x,
                                        const double new_energy,
                                        const double step_size) const
    {
        TVector new_grad;
        objFunc.gradient(new_x, new_grad);
        return objFunc.grad_norm(new_grad, norm_type)
               < objFunc.grad_norm(old_grad, norm_type);
    }
} // namespace polysolve::nonlinear::line_search

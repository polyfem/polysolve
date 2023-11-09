#include "NoLineSearch.hpp"

namespace polysolve::nonlinear::line_search
{
    NoLineSearch::NoLineSearch(const json &params, spdlog::logger &logger)
        : Superclass(params, logger)
    {
    }

    double NoLineSearch::compute_descent_step_size(
        const TVector &x,
        const TVector &delta_x,
        Problem &objFunc,
        const bool,
        const double,
        const double starting_step_size)
    {
        return starting_step_size;
    }
} // namespace polysolve::nonlinear::line_search

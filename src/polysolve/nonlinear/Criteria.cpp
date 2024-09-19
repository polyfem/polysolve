// Source: https://github.com/PatWie/CppNumericalSolvers/blob/7eddf28fa5a8872a956d3c8666055cac2f5a535d/include/cppoptlib/meta.h
// License: MIT
#include "Criteria.hpp"

#include <spdlog/fmt/fmt.h>

namespace polysolve::nonlinear
{
    bool is_converged_status(const Status s)
    {
        return s == Status::XDeltaTolerance || s == Status::FDeltaTolerance || s == Status::GradNormTolerance;
    }

    Criteria::Criteria()
    {
        reset();
    }

    void Criteria::reset()
    {
        iterations = 0;
        xDelta = 0;
        fDelta = 0;
        gradNorm = 0;
        firstGradNorm = 0;
        fDeltaCount = 0;
        xDeltaDotGrad = 0;
    }

    void Criteria::print(std::ostream &os) const
    {
        os << print_message();
    }
    std::string Criteria::print_message() const {
        return fmt::format(
            "iters={:d} Δf={:g} ‖∇f‖={:g} ‖Δx‖={:g} Δx⋅∇f(x)={:g}",
            iterations, fDelta, gradNorm, xDelta, xDeltaDotGrad);
    }

    Status checkConvergence(const Criteria &stop, const Criteria &current)
    {
        if (stop.iterations > 0 && current.iterations > stop.iterations)
        {
            return Status::IterationLimit;
        }
        const double stopGradNorm = current.iterations == 0 ? stop.firstGradNorm : stop.gradNorm;
        if (stopGradNorm > 0 && current.gradNorm < stopGradNorm)
        {
            return Status::GradNormTolerance;
        }
        if (stop.xDelta > 0 && current.xDelta < stop.xDelta)
        {
            return Status::XDeltaTolerance;
        }
        if (stop.fDelta > 0 && current.fDelta < stop.fDelta && current.fDeltaCount >= stop.fDeltaCount)
        {
            return Status::FDeltaTolerance;
        }
        // Δx⋅∇f ≥ 0 means that the search direction is not a descent direction
        if (stop.xDeltaDotGrad < 0 && current.xDeltaDotGrad > stop.xDeltaDotGrad)
        {
            return Status::NotDescentDirection;
        }
        return Status::Continue;
    }

    std::string_view status_message(Status s) {
        switch (s)
        {
        case Status::NotStarted:
            return "Solver not started";
        case Status::Continue:
            return "Convergence criteria not reached";
        case Status::IterationLimit:
            return "Iteration limit reached";
        case Status::XDeltaTolerance:
            return "Change in parameter vector too small";
        case Status::FDeltaTolerance:
            return "Change in cost function value too small";
        case Status::GradNormTolerance:
            return "Gradient vector norm too small";
        case Status::ObjectiveCustomStop:
            return "Objective function specified to stop";
        case Status::NanEncountered:
            return "Objective or gradient function returned NaN";
        case Status::NotDescentDirection:
            return "Search direction not a descent direction";
        case Status::LineSearchFailed:
            return "Line search failed";
        case Status::UpdateDirectionFailed:
            return "Update direction could not be computed";
        default:
            return "Unknown status";
        }
    }

    std::ostream &operator<<(std::ostream &os, const Status &s)
    {
        os << status_message(s);
        return os;
    }

    std::ostream &operator<<(std::ostream &os, const Criteria &c)
    {
        c.print(os);
        return os;
    }
} // namespace polysolve::nonlinear

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
        os << fmt::format(
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

    std::ostream &operator<<(std::ostream &os, const Status &s)
    {
        switch (s)
        {
        case Status::NotStarted:
            os << "Solver not started";
            break;
        case Status::Continue:
            os << "Convergence criteria not reached";
            break;
        case Status::IterationLimit:
            os << "Iteration limit reached";
            break;
        case Status::XDeltaTolerance:
            os << "Change in parameter vector too small";
            break;
        case Status::FDeltaTolerance:
            os << "Change in cost function value too small";
            break;
        case Status::GradNormTolerance:
            os << "Gradient vector norm too small";
            break;
        case Status::ObjectiveCustomStop:
            os << "Objective function specified to stop";
            break;
        case Status::NanEncountered:
            os << "Objective or gradient function returned NaN";
            break;
        case Status::NotDescentDirection:
            os << "Search direction not a descent direction";
            break;
        case Status::LineSearchFailed:
            os << "Line search failed";
            break;
        case Status::UpdateDirectionFailed:
            os << "Update direction could not be computed";
            break;
        default:
            os << "Unknown status";
            break;
        }
        return os;
    }

    std::ostream &operator<<(std::ostream &os, const Criteria &c)
    {
        c.print(os);
        return os;
    }
} // namespace polysolve::nonlinear
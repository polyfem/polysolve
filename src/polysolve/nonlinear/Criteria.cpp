// Source: https://github.com/PatWie/CppNumericalSolvers/blob/7eddf28fa5a8872a956d3c8666055cac2f5a535d/include/cppoptlib/meta.h
// License: MIT
#include "Criteria.hpp"

#include <spdlog/fmt/fmt.h>
#include <polysolve/Utils.hpp>

namespace polysolve::nonlinear
{
    bool is_converged_status(const Status s)
    {
        return s == Status::XDeltaTolerance || s == Status::FDeltaTolerance || s == Status::GradNormTolerance || s == Status::NewtonDecrementTolerance || s == Status::RelGradNormTolerance || s == Status::RelXDeltaTolerance;
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
        relGradNorm = 0;
        relXDelta = 0;
        newtonDecrement = 0;
    }

    void Criteria::print(std::ostream &os) const
    {
        os << print_message();
    }
    std::string Criteria::print_message() const {
        return fmt::format(
            "iters={:d} {}={:g} {}={:g} {}_rel={:g} {}={:g} {}_rel={:g} {}={:g} {}={:g}",
            iterations,
            log::delta("f"), fDelta,
            log::norm(log::grad("f")), gradNorm,
            log::norm(log::grad("f")), relGradNorm,
            log::norm(log::delta("x")), xDelta,
            log::norm(log::delta("x")), relXDelta,
            log::delta("x") + log::dot() + log::grad("f(x)"), xDeltaDotGrad,
            "1/2" + log::delta("x") + "^TH" + log::delta("x"), newtonDecrement
        );
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
        if (stop.relXDelta > 0 && current.relXDelta < stop.relXDelta)
        {
            return Status::RelXDeltaTolerance;
        }
        if (stop.relGradNorm > 0 && current.relGradNorm < stop.relGradNorm)
        {
            return Status::RelGradNormTolerance;
        }
        if (stop.newtonDecrement > 0 && current.newtonDecrement < stop.newtonDecrement)
        {
            return Status::NewtonDecrementTolerance;
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
        case Status::RelGradNormTolerance:
            return "Relative gradient vector too small";
        case Status::RelXDeltaTolerance:
            return "Relative change in parameter vector too small";
        case Status::NewtonDecrementTolerance:
            return "Newton decrement too small";
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

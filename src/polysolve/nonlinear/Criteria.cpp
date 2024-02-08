// Source: https://github.com/PatWie/CppNumericalSolvers/blob/7eddf28fa5a8872a956d3c8666055cac2f5a535d/include/cppoptlib/meta.h
// License: MIT
#include "Criteria.hpp"

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
    }

    void Criteria::print(std::ostream &os) const
    {
        os << "Iterations: " << iterations << std::endl;
        os << "xDelta:     " << xDelta << std::endl;
        os << "fDelta:     " << fDelta << std::endl;
        os << "GradNorm:   " << gradNorm << std::endl;
    }

    Status checkConvergence(const Criteria &stop, const Criteria &current)
    {
        if (stop.iterations > 0 && current.iterations > stop.iterations)
        {
            return Status::IterationLimit;
        }
        if (stop.xDelta > 0 && current.xDelta < stop.xDelta)
        {
            return Status::XDeltaTolerance;
        }
        if (stop.fDelta > 0 && current.fDelta < stop.fDelta && current.fDeltaCount >= stop.fDeltaCount)
        {
            return Status::FDeltaTolerance;
        }
        const double stopGradNorm = current.iterations == 0 ? stop.firstGradNorm : stop.gradNorm;
        if (stopGradNorm > 0 && current.gradNorm < stopGradNorm)
        {
            return Status::GradNormTolerance;
        }
        return Status::Continue;
    }

    std::ostream &operator<<(std::ostream &os, const Status &s)
    {
        switch (s)
        {
        case Status::NotStarted:
            os << "Solver not started.";
            break;
        case Status::Continue:
            os << "Convergence criteria not reached.";
            break;
        case Status::IterationLimit:
            os << "Iteration limit reached.";
            break;
        case Status::XDeltaTolerance:
            os << "Change in parameter vector too small.";
            break;
        case Status::FDeltaTolerance:
            os << "Change in cost function value too small.";
            break;
        case Status::GradNormTolerance:
            os << "Gradient vector norm too small.";
            break;
        default:
            os << "Unknown status.";
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
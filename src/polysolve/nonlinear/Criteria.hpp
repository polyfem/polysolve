#pragma once

#include <iostream>
#include <Eigen/Core>

namespace polysolve::nonlinear
{
    enum class Status
    {
        NotStarted = -1,
        Continue = 0,
        IterationLimit,
        XDeltaTolerance,
        FDeltaTolerance,
        GradNormTolerance,
        Condition,
        UserDefined
    };

    template <typename T>
    class Criteria
    {
    public:
        size_t iterations; //!< Maximum number of iterations
        T xDelta;          //!< Minimum change in parameter vector
        T fDelta;          //!< Minimum change in cost function
        T gradNorm;        //!< Minimum norm of gradient vector
        T condition;       //!< Maximum condition number of Hessian

        Criteria()
        {
            reset();
        }

        static Criteria defaults()
        {
            Criteria d;
            d.iterations = 10000;
            d.xDelta = 0;
            d.fDelta = 0;
            d.gradNorm = 1e-4;
            d.condition = 0;
            return d;
        }

        void reset()
        {
            iterations = 0;
            xDelta = 0;
            fDelta = 0;
            gradNorm = 0;
            condition = 0;
        }

        void print(std::ostream &os) const
        {
            os << "Iterations: " << iterations << std::endl;
            os << "xDelta:     " << xDelta << std::endl;
            os << "fDelta:     " << fDelta << std::endl;
            os << "GradNorm:   " << gradNorm << std::endl;
            os << "Condition:  " << condition << std::endl;
        }
    };

    template <typename T>
    Status checkConvergence(const Criteria<T> &stop, const Criteria<T> &current)
    {

        if ((stop.iterations > 0) && (current.iterations > stop.iterations))
        {
            return Status::IterationLimit;
        }
        if ((stop.xDelta > 0) && (current.xDelta < stop.xDelta))
        {
            return Status::XDeltaTolerance;
        }
        if ((stop.fDelta > 0) && (current.fDelta < stop.fDelta))
        {
            return Status::FDeltaTolerance;
        }
        if ((stop.gradNorm > 0) && (current.gradNorm < stop.gradNorm))
        {
            return Status::GradNormTolerance;
        }
        if ((stop.condition > 0) && (current.condition > stop.condition))
        {
            return Status::Condition;
        }
        return Status::Continue;
    }

    inline std::ostream &operator<<(std::ostream &os, const Status &s)
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
        case Status::Condition:
            os << "Condition of Hessian/Covariance matrix too large.";
            break;
        case Status::UserDefined:
            os << "Stop condition defined in the callback.";
            break;
        }
        return os;
    }

    template <typename T>
    std::ostream &operator<<(std::ostream &os, const Criteria<T> &c)
    {
        c.print(os);
        return os;
    }
} // namespace polysolve::nonlinear
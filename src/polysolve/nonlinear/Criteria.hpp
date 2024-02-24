#pragma once

#include <cstddef>
#include <iostream>

namespace polysolve::nonlinear
{
    // Source: https://github.com/PatWie/CppNumericalSolvers/blob/7eddf28fa5a8872a956d3c8666055cac2f5a535d/include/cppoptlib/meta.h
    // License: MIT

    enum class Status
    {
        NotStarted = -1, ///< The solver has not been started
        Continue = 0,    ///< The solver should continue
        // Success cases
        IterationLimit,      ///< The maximum number of iterations has been reached
        XDeltaTolerance,     ///< The change in the parameter vector is below the tolerance
        FDeltaTolerance,     ///< The change in the cost function is below the tolerance
        GradNormTolerance,   ///< The norm of the gradient vector is below the tolerance
        ObjectiveCustomStop, ///< The objective function specified to stop
        // Failure cases
        NanEncountered,        ///< The objective function returned NaN
        NotDescentDirection,   ///< The search direction is not a descent direction
        LineSearchFailed,      ///< The line search failed
        UpdateDirectionFailed, ///< The update direction could not be computed
    };

    bool is_converged_status(const Status s);

    class Criteria
    {
    public:
        size_t iterations;    ///< Maximum number of iterations
        double xDelta;        ///< Minimum change in parameter vector
        double fDelta;        ///< Minimum change in cost function
        double gradNorm;      ///< Minimum norm of gradient vector
        double firstGradNorm; ///< Initial norm of gradient vector
        double xDeltaDotGrad; ///< Dot product of parameter vector and gradient vector
        unsigned fDeltaCount; ///< Number of steps where fDelta is satisfied

        Criteria();

        void reset();

        void print(std::ostream &os) const;
    };

    Status checkConvergence(const Criteria &stop, const Criteria &current);

    std::ostream &operator<<(std::ostream &os, const Status &s);

    std::ostream &operator<<(std::ostream &os, const Criteria &c);
} // namespace polysolve::nonlinear
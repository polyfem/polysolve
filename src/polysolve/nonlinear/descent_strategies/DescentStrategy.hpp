#pragma once

#include <polysolve/Utils.hpp>

#include <polysolve/nonlinear/Problem.hpp>

namespace polysolve::nonlinear
{

    class DescentStrategy
    {
    public:
        using TVector = Problem::TVector;
        using Scalar = Problem::Scalar;

        DescentStrategy(const json &solver_params_,
                        const double characteristic_length,
                        spdlog::logger &logger)
            : m_logger(logger)
        {
        }
        virtual ~DescentStrategy() {}
        virtual std::string name() const = 0;

        virtual void reset(const int ndof) {}
        virtual void reset_times() {}
        virtual void update_solver_info(json &solver_info, const double per_iteration) {}
        virtual void log_times() const {}

        virtual bool is_direction_descent() { return true; }
        virtual bool handle_error() { return false; }

        virtual bool compute_update_direction(
            Problem &objFunc,
            const TVector &x,
            const TVector &grad,
            TVector &direction) = 0;

    protected:
        spdlog::logger &m_logger;
    };
} // namespace polysolve::nonlinear
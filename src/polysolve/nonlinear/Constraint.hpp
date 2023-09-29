#pragma once

namespace polysolve::nonlinear
{
    class Constraint
    {
    public:
        virtual ~Constraint() = default;

        virtual double value(const Eigen::VectorXd &x) = 0;
        virtual double first_derivative(const Eigen::VectorXd &x, Eigen::VectorXd &grad) = 0;
    };
} // namespace polysolve::nonlinear

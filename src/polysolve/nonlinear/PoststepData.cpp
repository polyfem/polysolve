#include "PostStepData.hpp"

namespace polysolve::nonlinear
{
    PostStepData::PostStepData(const int iter_num,
                               const Eigen::VectorXd &x,
                               const Eigen::VectorXd &grad)
        : iter_num(iter_num), x(x), grad(grad)
    {
    }
} // namespace polysolve::nonlinear

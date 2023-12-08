#include "PostStepData.hpp"

namespace polysolve::nonlinear
{
    PostStepData::PostStepData(const int iter_num,
                               const json &solver_info,
                               const Eigen::VectorXd &x,
                               const Eigen::VectorXd &grad)
        : iter_num(iter_num), solver_info(solver_info), x(x), grad(grad)
    {
    }
} // namespace polysolve::nonlinear

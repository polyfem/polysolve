#pragma once

#include <polysolve/Types.hpp>

namespace polysolve::nonlinear
{

    class PoststepData
    {
    public:
        PoststepData(const int iter_num,
                     const Eigen::VectorXd &x,
                     const Eigen::VectorXd &grad);

        const int iter_num;
        const Eigen::VectorXd &x;
        const Eigen::VectorXd &grad;
    };
} // namespace polysolve::nonlinear

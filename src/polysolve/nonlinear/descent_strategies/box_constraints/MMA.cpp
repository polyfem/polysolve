#include "MMA.hpp"

#include <sstream>

namespace polysolve::nonlinear
{
    MMA::MMA(const json &solver_params,
             const double characteristic_length,
             spdlog::logger &logger)
        : Superclass(solver_params, characteristic_length, logger)
    {
    }

    void MMA::reset(const int ndof)
    {
        Superclass::reset(ndof);
        mma.reset();
    }

    bool MMA::compute_boxed_update_direction(
        Problem &objFunc,
        const TVector &x,
        const TVector &grad,
        const TVector &lower_bound,
        const TVector &upper_bound,
        TVector &direction)
    {
        const int m = constraints_.size();

        if (!mma)
            mma = std::make_shared<MMAAux>(x.size(), m);

        Eigen::VectorXd g, gradv, dg;
        g.setZero(m);
        gradv.setZero(x.size());
        dg.setZero(m * x.size());
        for (int i = 0; i < m; i++)
        {
            g(i) = constraints_[i]->value(x);
            constraints_[i]->gradient(x, gradv);
            dg(Eigen::seqN(0, gradv.size(), m)) = gradv;
        }
        std::stringstream ss;
        ss << g.transpose();
        m_logger.trace("Constraint values are {}", ss.str());
        auto y = x;
        mma->Update(y.data(), grad.data(), g.data(), dg.data(), lower_bound.data(), upper_bound.data());
        direction = y - x;

        // maybe remove me
        // else if (grad.squaredNorm() != 0 && direction.dot(grad) > 0)
        // {
        //     polyfem::logger().error("Direction is not a descent direction, stop.");
        //     throw std::runtime_error("Direction is not a descent direction, stop.");
        // }

        return true;
    }
} // namespace polysolve::nonlinear
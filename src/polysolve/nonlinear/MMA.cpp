#include "MMA.hpp"

#include <sstream>

namespace polysolve::nonlinear
{
    MMA::MMA(const json &solver_params,
             const double characteristic_length,
             spdlog::logger &logger)
        : Superclass(solver_params, characteristic_length, logger)
    {
        if (m_line_search->name() != "None")
            log_and_throw_error(m_logger, "Invalid linesearch for MMA; MMA requires 'None' linesearch, instead got {}", m_line_search->name());
    }

    void MMA::reset(const int ndof)
    {
        Superclass::reset(ndof);
        mma.reset();
    }

    std::string MMA::descent_strategy_name(int descent_strategy) const
    {
        switch (descent_strategy)
        {
        case Solver::MMA_STRATEGY:
            return "MMA";
        default:
            throw std::invalid_argument("invalid descent strategy");
        }
    }

    void MMA::increase_descent_strategy()
    {
        assert(this->descent_strategy <= Solver::MAX_STRATEGY);
        this->descent_strategy++;
    }

    void MMA::compute_update_direction(
        Problem &objFunc,
        const TVector &x,
        const TVector &grad,
        TVector &direction)
    {
        TVector lower_bound = Superclass::get_lower_bound(x);
        TVector upper_bound = Superclass::get_upper_bound(x);

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
        m_logger.info("Constraint values are {}", ss.str());
        auto y = x;
        mma->Update(y.data(), grad.data(), g.data(), dg.data(), lower_bound.data(), upper_bound.data());
        direction = y - x;

        // maybe remove me
        if (std::isnan(direction.squaredNorm()))
        {
            log_and_throw_error(m_logger, "nan in direction.");
        }
        // else if (grad.squaredNorm() != 0 && direction.dot(grad) > 0)
        // {
        //     polyfem::logger().error("Direction is not a descent direction, stop.");
        //     throw std::runtime_error("Direction is not a descent direction, stop.");
        // }
    }
} // namespace polysolve::nonlinear
// ADAM from "ADAM: A METHOD FOR STOCHASTIC OPTIMIZATION"

#include "ADAM.hpp"

namespace polysolve::nonlinear
{

    ADAM::ADAM(const json &solver_params,
               const double characteristic_length,
               spdlog::logger &logger)
        : Superclass(solver_params, characteristic_length, logger)
    {
        alpha = solver_params["ADAM"]["alpha"];
        beta_1 = solver_params["ADAM"]["beta_1"];
        beta_2 = solver_params["ADAM"]["beta_2"];
        epsilon = solver_params["ADAM"]["epsilon"];
    }

    void ADAM::reset(const int ndof)
    {
        Superclass::reset(ndof);
        m_prev = Eigen::VectorXd::Zero(ndof);
        v_prev = Eigen::VectorXd::Zero(ndof);
    }

    bool ADAM::compute_update_direction(
        Problem &objFunc,
        const TVector &x,
        const TVector &grad,
        TVector &direction)
    {
        if (m_prev.size() == 0)
            m_prev = Eigen::VectorXd::Zero(x.size());
        if (v_prev.size() == 0)
            v_prev = Eigen::VectorXd::Zero(x.size());

        TVector m = (beta_1 * m_prev) + ((1 - beta_1) * grad);
        TVector v = beta_2 * v_prev;
        for (int i = 0; i < v.size(); ++i)
            v(i) += (1 - beta_2) * grad(i) * grad(i);

        m = m.array() / (1 - pow(beta_1, t));
        v = v.array() / (1 - pow(beta_2, t));

        direction = -alpha * m;
        for (int i = 0; i < v.size(); ++i)
            direction(i) /= sqrt(v(i) + epsilon);

        ++t;

        return true;
    }
} // namespace polysolve::nonlinear

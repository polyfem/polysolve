// ADAM from "ADAM: A METHOD FOR STOCHASTIC OPTIMIZATION"

#include "ADAM.hpp"

namespace polysolve::nonlinear
{

    ADAM::ADAM(const json &solver_params,
               const bool is_stochastic,
               const double characteristic_length,
               spdlog::logger &logger)
        : Superclass(solver_params, characteristic_length, logger), is_stochastic_(is_stochastic)
    {
        std::string param_name = is_stochastic ? "StochasticADAM" : "ADAM";
        alpha_ = solver_params[param_name]["alpha"];
        beta_1_ = solver_params[param_name]["beta_1"];
        beta_2_ = solver_params[param_name]["beta_2"];
        epsilon_ = solver_params[param_name]["epsilon"];
        if (is_stochastic)
            erase_component_probability_ = solver_params["StochasticADAM"]["erase_component_probability"];
    }

    void ADAM::reset(const int ndof)
    {
        Superclass::reset(ndof);
        m_prev_ = Eigen::VectorXd::Zero(ndof);
        v_prev_ = Eigen::VectorXd::Zero(ndof);
        t_ = 0;
    }

    bool ADAM::compute_update_direction(
        Problem &objFunc,
        const TVector &x,
        const TVector &grad,
        TVector &direction)
    {
        if (m_prev_.size() == 0)
            m_prev_ = Eigen::VectorXd::Zero(x.size());
        if (v_prev_.size() == 0)
            v_prev_ = Eigen::VectorXd::Zero(x.size());

        TVector grad_modified = grad;

        if (is_stochastic_)
        {
            Eigen::VectorXd mask = (Eigen::VectorXd::Random(direction.size()).array() + 1.) / 2.;
            for (int i = 0; i < direction.size(); ++i)
                grad_modified(i) *= (mask(i) < erase_component_probability_) ? 0. : 1.;
        }

        TVector m = (beta_1_ * m_prev_) + ((1 - beta_1_) * grad_modified);
        TVector v = beta_2_ * v_prev_;
        for (int i = 0; i < v.size(); ++i)
            v(i) += (1 - beta_2_) * grad_modified(i) * grad_modified(i);

        m = m.array() / (1 - pow(beta_1_, t_));
        v = v.array() / (1 - pow(beta_2_, t_));

        direction = -alpha_ * m;
        for (int i = 0; i < v.size(); ++i)
            direction(i) /= sqrt(v(i) + epsilon_);

        ++t_;

        return true;
    }
} // namespace polysolve::nonlinear

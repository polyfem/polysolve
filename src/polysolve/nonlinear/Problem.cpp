#include "Problem.hpp"

namespace polysolve::nonlinear
{
    void Problem::sample_along_direction(
        const Problem::TVector &x,
        const Problem::TVector &direction,
        const double start,
        const double end,
        const int num_samples,
        Eigen::VectorXd &alphas,
        Eigen::VectorXd &fs,
        Eigen::VectorXi &valid)
    {
        alphas = Eigen::VectorXd::LinSpaced(num_samples, start, end);
        fs.resize(alphas.size());
        valid.resize(alphas.size());

        Problem::TVector new_x;

        for (int i = 0; i < alphas.size(); i++)
        {
            new_x = x + alphas[i] * direction;

            solution_changed(new_x);
            const double fxi = value(new_x);

            fs[i] = fxi;
            valid[i] = is_step_valid(x, new_x);
        }
    }
} // namespace polysolve::nonlinear
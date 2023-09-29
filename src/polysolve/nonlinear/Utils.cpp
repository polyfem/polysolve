#include "Utils.hpp"

#include <fstream>

namespace polysolve::nonlinear
{
    void log_and_throw_error(spdlog::logger &logger, const std::string &msg)
    {
        logger.error(msg);
        throw std::runtime_error(msg);
    }

    Eigen::SparseMatrix<double> sparse_identity(int rows, int cols)
    {
        Eigen::SparseMatrix<double> I(rows, cols);
        I.setIdentity();
        return I;
    }

    bool has_hessian_nans(const polysolve::StiffnessMatrix &hessian)
    {
        for (int k = 0; k < hessian.outerSize(); ++k)
        {
            for (polysolve::StiffnessMatrix::InnerIterator it(hessian, k); it; ++it)
            {
                if (std::isnan(it.value()))
                    return true;
            }
        }

        return false;
    }

    void save_sampled_values(
        const std::string &filename,
        const typename Problem::TVector &x,
        const typename Problem::TVector &delta_x,
        Problem &objFunc,
        spdlog::logger &logger,
        const double starting_step_size,
        const int num_samples)
    {
        std::ofstream samples(filename, std::ios::out);
        if (!samples.is_open())
        {
            log_and_throw_error(logger, "Unable to save sampled values to file \"{}\" !", filename);
        }

        samples << "alpha,f(x + alpha * delta_x),valid,decrease\n";

        objFunc.solution_changed(x);
        double fx = objFunc.value(x);

        Eigen::VectorXd alphas = Eigen::VectorXd::LinSpaced(2 * num_samples - 1, -starting_step_size, starting_step_size);
        for (int i = 0; i < alphas.size(); i++)
        {
            Problem::TVector new_x = x + alphas[i] * delta_x;
            objFunc.solution_changed(new_x);
            double fxi = objFunc.value(new_x);
            samples << alphas[i] << ","
                    << fxi << ","
                    << (objFunc.is_step_valid(x, new_x) ? "true" : "false") << ","
                    << (fxi < fx ? "true" : "false") << "\n";
        }

        samples.close();
    }
} // namespace polysolve::nonlinear
#include "Utils.hpp"

#include <spdlog/fmt/bundled/color.h>

#include <fstream>

namespace polysolve::nonlinear
{

    StopWatch::StopWatch(spdlog::logger &logger)
        : m_logger(logger)
    {
        start();
    }

    StopWatch::StopWatch(const std::string &name, spdlog::logger &logger)
        : m_name(name), m_logger(logger)
    {
        start();
    }

    StopWatch::StopWatch(double &total_time, spdlog::logger &logger)
        : m_total_time(&total_time), m_logger(logger)
    {
        start();
    }

    StopWatch::StopWatch(Timing &timing, spdlog::logger &logger)
        : m_total_time(&timing.time), m_count(&timing.count), m_logger(logger)
    {
        start();
    }

    StopWatch::StopWatch(const std::string &name, double &total_time, spdlog::logger &logger)
        : m_name(name), m_total_time(&total_time), m_logger(logger)
    {
        start();
    }

    StopWatch::StopWatch(const std::string &name, Timing &timing, spdlog::logger &logger)
        : m_name(name), m_total_time(&timing.time), m_count(&timing.count), m_logger(logger)
    {
        start();
    }

    StopWatch::~StopWatch()
    {
        stop();
    }

    inline void StopWatch::start()
    {
        is_running = true;
        m_start = clock::now();
    }

    inline void StopWatch::stop()
    {
        if (!is_running)
            return;
        m_stop = clock::now();

        is_running = false;
        log_msg();
        if (m_total_time)
            *m_total_time += getElapsedTimeInSec();
        if (m_count)
            ++(*m_count);
    }

    inline double StopWatch::getElapsedTimeInSec()
    {
        return std::chrono::duration<double>(m_stop - m_start).count();
    }

    inline void StopWatch::log_msg()
    {
        const static std::string log_fmt_text =
            fmt::format("[{}] {{}} {{:.3g}}s", fmt::format(fmt::fg(fmt::terminal_color::magenta), "timing"));

        if (!m_name.empty())
        {
            m_logger.trace(log_fmt_text, m_name, getElapsedTimeInSec());
        }
    }

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
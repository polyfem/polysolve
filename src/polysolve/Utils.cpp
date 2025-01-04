#include "Utils.hpp"

#if defined(SPDLOG_FMT_EXTERNAL)
#include <fmt/color.h>
#else
#include <spdlog/fmt/bundled/color.h>
#endif

namespace polysolve
{

    StopWatch::StopWatch(const std::string &name, spdlog::logger &logger)
        : m_name(name), m_logger(logger)
    {
        start();
    }

    StopWatch::StopWatch(const std::string &name, double &total_time, spdlog::logger &logger)
        : m_name(name), m_total_time(&total_time), m_logger(logger)
    {
        start();
    }

    StopWatch::~StopWatch()
    {
        stop();
    }

    void StopWatch::start()
    {
        is_running = true;
        m_start = clock::now();
    }

    void StopWatch::stop()
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

    double StopWatch::getElapsedTimeInSec()
    {
        return std::chrono::duration<double>(m_stop - m_start).count();
    }

    void StopWatch::log_msg()
    {
        const static auto log_fmt_text =
            fmt::format("[{}] {{}} {{:.3g}}s", fmt::format(fmt::fg(fmt::terminal_color::magenta), "timing"));

        if (!m_name.empty())
        {
            m_logger.trace(log_fmt_text, m_name, getElapsedTimeInSec());
        }
    }

    void log_and_throw_error(spdlog::logger &logger, const std::string &msg)
    {
        logger.error("{}", msg);
        throw std::runtime_error(msg);
    }

    Eigen::SparseMatrix<double> sparse_identity(int rows, int cols)
    {
        Eigen::SparseMatrix<double> I(rows, cols);
        I.setIdentity();
        return I;
    }

    double extract_param(const std::string &key, const std::string &name, const json &json)
    {
        if (json.find(key) != json.end())
            return json[key][name];

        return json[name];
    }

} // namespace polysolve

#pragma once

#include "Types.hpp"

#include <spdlog/spdlog.h>

#define POLYSOLVE_SCOPED_STOPWATCH(...) polysolve::StopWatch __polysolve_stopwatch(__VA_ARGS__)

namespace polysolve
{

    struct Timing
    {
        operator double() const { return time; }

        void operator+=(const double t)
        {
            time += t;
            ++count;
        }

        double time = 0;
        size_t count = 0;
    };

    class StopWatch
    {
    private:
        using clock = std::chrono::steady_clock;

    public:
        StopWatch(const std::string &name, spdlog::logger &logger);
        StopWatch(const std::string &name, double &total_time, spdlog::logger &logger);

        virtual ~StopWatch();

        void start();
        void stop();

        double getElapsedTimeInSec();
        void log_msg();

    private:
        std::string m_name;
        std::chrono::time_point<clock> m_start, m_stop;
        double *m_total_time = nullptr;
        size_t *m_count = nullptr;
        bool is_running = false;

        spdlog::logger &m_logger;
    };

    [[noreturn]] void log_and_throw_error(spdlog::logger &logger, const std::string &msg);

    template <typename... Args>
    [[noreturn]] void log_and_throw_error(spdlog::logger &logger, const std::string &msg, const Args &...args)
    {
        log_and_throw_error(logger, fmt::format(fmt::runtime(msg), args...));
    }

    Eigen::SparseMatrix<double> sparse_identity(int rows, int cols);

    double extract_param(const std::string &key, const std::string &name, const json &json);

} // namespace polysolve

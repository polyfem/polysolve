#pragma once

#include "Types.hpp"

#include <spdlog/spdlog.h>
#include <limits>
#include <cmath>
#include <cassert>

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

    /// @brief x + s * dx, each coefficient rounded once (fused multiply-add).
    ///
    /// Every place the nonlinear solver forms a trial or accepted step goes
    /// through this, so the configuration it finally takes is bit-for-bit the
    /// one it checked, and differs from the exact point on the ray by at most
    /// half an ulp per coefficient. That is what lets a problem certify a
    /// whole segment and trust any point the line search picks on it.
    /// std::fma rounds once on every platform (in hardware where available).
    inline Eigen::VectorXd step_along(const Eigen::VectorXd &x, const double s, const Eigen::VectorXd &dx)
    {
        assert(x.size() == dx.size());
        Eigen::VectorXd out(x.size());
        for (Eigen::Index i = 0; i < x.size(); ++i)
            out[i] = std::fma(s, dx[i], x[i]);
        return out;
    }

    /// @brief The largest double not above a * b (a, b >= 0), in any rounding
    /// mode: the fused multiply-add gives the exact rounding error of the
    /// product, and when the rounded product overshoots it steps down one
    /// representable value.
    inline double product_rounded_down(const double a, const double b)
    {
        const double p = a * b;
        return std::fma(a, b, -p) < 0 ? std::nextafter(p, -std::numeric_limits<double>::infinity()) : p;
    }

    namespace log
    {
#ifdef POLYSOLVE_WITH_UNICODE
        inline std::string f0() { return "f₀"; }
        inline std::string dot() { return "⋅"; }
        inline std::string ge() { return "≥"; }
        inline std::string delta(const std::string &x) { return std::string("Δ") + x; }
        inline std::string norm(const std::string &x) { return std::string("‖") + x + "‖"; }
        inline std::string grad(const std::string &x) { return std::string("∇") + x; }
#else
        inline std::string f0() { return "f0"; }
        inline std::string dot() { return "*"; }
        inline std::string ge() { return ">="; }
        inline std::string delta(const std::string &x) { return std::string("d") + x; }
        inline std::string norm(const std::string &x) { return std::string("norm(") + x + ")"; }
        inline std::string grad(const std::string &x) { return std::string("grad(") + x + ")"; }
#endif // POLYSOLVE_WITH_UNICODE

    } // namespace log

} // namespace polysolve

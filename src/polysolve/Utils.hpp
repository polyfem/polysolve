#pragma once

#include "Types.hpp"

#include <spdlog/spdlog.h>

#include <functional>
#include <optional>

#define POLYSOLVE_SCOPED_STOPWATCH(...) polysolve::StopWatch __polysolve_stopwatch(__VA_ARGS__)

/// As POLYSOLVE_SCOPED_STOPWATCH, but only when `enabled`: a stopwatch costs a string and two clock
/// reads, which matter in a solve of a few unknowns that runs millions of times.
#define POLYSOLVE_SCOPED_STOPWATCH_IF(enabled, ...)            \
    std::optional<polysolve::StopWatch> __polysolve_stopwatch; \
    if (enabled)                                               \
    __polysolve_stopwatch.emplace(__VA_ARGS__)

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

    /// Validates `params` against jse rules, as jse::JSE::verify_json with `strict`, and returns
    /// them with the rules' defaults injected (jse::JSE::inject_defaults). An invalid input is
    /// logged and thrown, with jse's report, as std::runtime_error.
    ///
    /// The result of every successful validation is remembered, keyed by `spec`, `strict` and the
    /// exact input (types and values, bit for bit), and returned again without running jse: a
    /// validation against the solver specifications takes milliseconds, and applications create
    /// solvers with the same parameters over and over (one per mesh, per thread, ...), each
    /// nonlinear Newton solver creating three linear solvers. Validation has no effect but its
    /// result, so a remembered result is the one a new validation would give. `make_rules` builds
    /// the rules, only when they are needed; `spec` names them, and must always name the same
    /// rules. Failures are not remembered. Thread-safe.
    json validate_and_inject_defaults(
        const json &params,
        const std::string &spec,
        const bool strict,
        const std::function<json()> &make_rules,
        spdlog::logger &logger);

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

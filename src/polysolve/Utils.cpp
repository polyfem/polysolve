#include "Utils.hpp"

#include <jse/jse.h>

#include <cstdint>
#include <cstring>
#include <mutex>
#include <unordered_map>

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

    namespace
    {
        // A key that tells apart any two different json values: the type of every value, and
        // floating point numbers by their bits (dump() prints NaN and infinity alike, and does
        // not tell a signed from an unsigned integer).
        void append_key(const json &j, std::string &key)
        {
            key += char('A' + int(j.type()));
            switch (j.type())
            {
            case json::value_t::object:
                for (const auto &[k, v] : j.items())
                {
                    key += json(k).dump();
                    append_key(v, key);
                }
                key += '}';
                break;
            case json::value_t::array:
                for (const auto &v : j)
                    append_key(v, key);
                key += ']';
                break;
            case json::value_t::number_float:
            {
                const double d = j.get<double>();
                std::uint64_t bits;
                std::memcpy(&bits, &d, sizeof bits);
                key += std::to_string(bits);
                key += ';';
                break;
            }
            default:
                key += j.dump();
                key += ';';
            }
        }
    } // namespace

    json validate_and_inject_defaults(
        const json &params,
        const std::string &spec,
        const bool strict,
        const std::function<json()> &make_rules,
        spdlog::logger &logger)
    {
        // Bounded: an application that validates ever new parameters keeps at most this many.
        constexpr size_t max_entries = 256;
        static std::mutex mutex;
        static std::unordered_map<std::string, json> validated;

        std::string key = spec;
        key += strict ? "|strict|" : "|lenient|";
        append_key(params, key);
        {
            std::lock_guard<std::mutex> lock(mutex);
            const auto it = validated.find(key);
            if (it != validated.end())
                return it->second;
        }

        const json rules = make_rules();
        jse::JSE jse;
        jse.strict = strict;
        if (!jse.verify_json(params, rules))
            log_and_throw_error(logger, "invalid input json:\n{}", jse.log2str());
        json result = jse.inject_defaults(params, rules);

        std::lock_guard<std::mutex> lock(mutex);
        if (validated.size() >= max_entries)
            validated.clear();
        validated.emplace(std::move(key), result);
        return result;
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

#pragma once

#include <string>

namespace polysolve::nonlinear
{
    class Logger
    {
    public:
        template <typename... Args>
        void trace(const std::string &msg, const Args &...args) const {}

        template <typename... Args>
        void debug(const std::string &msg, const Args &...args) const {}

        template <typename... Args>
        void info(const std::string &msg, const Args &...args) const {}

        template <typename... Args>
        void warn(const std::string &msg, const Args &...args) const {}

        template <typename... Args>
        void error(const std::string &msg, const Args &...args) const {}

        template <typename... Args>
        void log_and_throw_error(const std::string &msg, const Args &...args) const {}
    };

} // namespace polysolve::nonlinear
#pragma once

#include <polysolve/types.hpp>

#include "Problem.hpp"

#include <spdlog/spdlog.h>

namespace polysolve::nonlinear
{

// TODO
#define POLYSOLVE_SCOPED_TIMER(...) //

    [[noreturn]] void log_and_throw_error(spdlog::logger &logger, const std::string &msg);

    template <typename... Args>
    [[noreturn]] void log_and_throw_error(spdlog::logger &logger, const std::string &msg, const Args &...args)
    {
        log_and_throw_error(logger, fmt::format(msg, args...));
    }

    static void save_sampled_values(const std::string &filename,
                                    const typename Problem::TVector &x,
                                    const typename Problem::TVector &grad,
                                    Problem &objFunc,
                                    spdlog::logger &logger,
                                    const double starting_step_size = 1e-1,
                                    const int num_samples = 1000);
} // namespace polysolve::nonlinear
#pragma once

////////////////////////////////////////////////////////////////////////////////
#include "EigenSolver.hpp"
#include <chrono>
#include <iostream>
#include <spdlog/spdlog.h>
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// Direct solvers
////////////////////////////////////////////////////////////////////////////////
namespace polysolve::linear
{
    namespace
    {
        using clock = std::chrono::steady_clock;

        double elapsed_seconds(const std::chrono::time_point<clock> &begin)
        {
            return std::chrono::duration<double>(clock::now() - begin).count();
        }
    } // namespace

    // Get info on the last solve step
    template <typename SparseSolver>
    void EigenDirect<SparseSolver>::get_info(json &params) const
    {
        switch (m_Solver.info())
        {
        case Eigen::Success:
            params["solver_info"] = "Success";
            break;
        case Eigen::NumericalIssue:
            params["solver_info"] = "NumericalIssue";
            break;
        case Eigen::NoConvergence:
            params["solver_info"] = "NoConvergence";
            break;
        case Eigen::InvalidInput:
            params["solver_info"] = "InvalidInput";
            break;
        default:
            assert(false);
        }
    }

    // Analyze sparsity pattern
    template <typename SparseSolver>
    void EigenDirect<SparseSolver>::analyze_pattern(const StiffnessMatrix &A, const int precond_num)
    {
        auto phase_begin = clock::now();
        m_Solver.analyzePattern(A);
        SPDLOG_INFO("[{}] [analyze_pattern] [{:.6f}]", name(), elapsed_seconds(phase_begin));
    }

    // Factorize system matrix
    template <typename SparseSolver>
    void EigenDirect<SparseSolver>::factorize(const StiffnessMatrix &A)
    {
        auto phase_begin = clock::now();
        m_Solver.factorize(A);
        if (m_Solver.info() == Eigen::NumericalIssue)
        {
            throw std::runtime_error("[EigenDirect] NumericalIssue encountered.");
        }
        SPDLOG_INFO("[{}] [factorize] [{:.6f}]", name(), elapsed_seconds(phase_begin));
    }

    // Solve the linear system
    template <typename SparseSolver>
    void EigenDirect<SparseSolver>::solve(
        const Ref<const VectorXd> b, Ref<VectorXd> x)
    {
        auto phase_begin = clock::now();
        x = m_Solver.solve(b);
        SPDLOG_INFO("[{}] [solve] [{:.6f}]", name(), elapsed_seconds(phase_begin));
    }

    ////////////////////////////////////////////////////////////////////////////////
    // Iterative solvers
    ////////////////////////////////////////////////////////////////////////////////

    // Set solver parameters
    template <typename SparseSolver>
    void EigenIterative<SparseSolver>::set_parameters(const json &params)
    {
        const std::string solver_name = name();
        if (params.contains(solver_name))
        {
            if (params[solver_name].contains("max_iter"))
            {
                m_Solver.setMaxIterations(params[solver_name]["max_iter"]);
            }
            if (params[solver_name].contains("tolerance"))
            {
                m_Solver.setTolerance(params[solver_name]["tolerance"]);
            }
        }
    }

    // Get info on the last solve step
    template <typename SparseSolver>
    void EigenIterative<SparseSolver>::get_info(json &params) const
    {
        params["solver_iter"] = m_Solver.iterations();
        params["solver_error"] = m_Solver.error();
    }

    // Analyze sparsity pattern
    template <typename SparseSolver>
    void EigenIterative<SparseSolver>::analyze_pattern(const StiffnessMatrix &A, const int precond_num)
    {
        auto phase_begin = clock::now();
        m_Solver.analyzePattern(A);
        SPDLOG_INFO("[{}] [analyze_pattern] [{:.6f}]", name(), elapsed_seconds(phase_begin));
    }

    // Factorize system matrix
    template <typename SparseSolver>
    void EigenIterative<SparseSolver>::factorize(const StiffnessMatrix &A)
    {
        auto phase_begin = clock::now();
        m_A = A;
        m_Solver.factorize(m_A);
        SPDLOG_INFO("[{}] [factorize] [{:.6f}]", name(), elapsed_seconds(phase_begin));
    }

    // Solve the linear system
    template <typename SparseSolver>
    void EigenIterative<SparseSolver>::solve(
        const Ref<const VectorXd> b, Ref<VectorXd> x)
    {
        assert(x.size() == b.size());
        auto phase_begin = clock::now();
        x = m_Solver.solveWithGuess(b, x);
        SPDLOG_INFO("[{}] [solve] [{:.6f}] [iters={}]", name(), elapsed_seconds(phase_begin), m_Solver.iterations());
    }

    ////////////////////////////////////////////////////////////////////////////////
    // Dense solvers
    ////////////////////////////////////////////////////////////////////////////////

    // Get info on the last solve step
    template <typename DenseSolver>
    void EigenDenseSolver<DenseSolver>::get_info(json &params) const
    {
        params["solver_info"] = "Success";
    }

    template <typename DenseSolver>
    void EigenDenseSolver<DenseSolver>::factorize(const StiffnessMatrix &A)
    {
        factorize_dense(Eigen::MatrixXd(A));
    }

    // Factorize system matrix
    template <typename DenseSolver>
    void EigenDenseSolver<DenseSolver>::factorize_dense(const Eigen::MatrixXd &A)
    {
        auto phase_begin = clock::now();
        m_Solver.compute(A);
        SPDLOG_INFO("[{}] [factorize] [{:.6f}]", name(), elapsed_seconds(phase_begin));
    }

    // Solve the linear system
    template <typename DenseSolver>
    void EigenDenseSolver<DenseSolver>::solve(
        const Ref<const VectorXd> b, Ref<VectorXd> x)
    {
        auto phase_begin = clock::now();
        x = m_Solver.solve(b);
        SPDLOG_INFO("[{}] [solve] [{:.6f}]", name(), elapsed_seconds(phase_begin));
    }
} // namespace polysolve::linear

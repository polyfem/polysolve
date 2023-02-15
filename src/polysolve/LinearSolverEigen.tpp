#pragma once

////////////////////////////////////////////////////////////////////////////////
#include <polysolve/LinearSolverEigen.hpp>
#include <iostream>
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// Direct solvers
////////////////////////////////////////////////////////////////////////////////

// Get info on the last solve step
template <typename SparseSolver>
void polysolve::LinearSolverEigenDirect<SparseSolver>::getInfo(json &params) const
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
void polysolve::LinearSolverEigenDirect<SparseSolver>::analyzePattern(const StiffnessMatrix &A, const int precond_num)
{
    m_Solver.analyzePattern(A);
}

// Factorize system matrix
template <typename SparseSolver>
void polysolve::LinearSolverEigenDirect<SparseSolver>::factorize(const StiffnessMatrix &A)
{
    m_Solver.factorize(A);
    if (m_Solver.info() == Eigen::NumericalIssue)
    {
        throw std::runtime_error("[LinearSolverEigenDirect] NumericalIssue encountered.");
    }
}

// Solve the linear system
template <typename SparseSolver>
void polysolve::LinearSolverEigenDirect<SparseSolver>::solve(
    const Ref<const VectorXd> b, Ref<VectorXd> x)
{
    x = m_Solver.solve(b);
}

////////////////////////////////////////////////////////////////////////////////
// Iterative solvers
////////////////////////////////////////////////////////////////////////////////

// Set solver parameters
template <typename SparseSolver>
void polysolve::LinearSolverEigenIterative<SparseSolver>::setParameters(const json &params)
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
void polysolve::LinearSolverEigenIterative<SparseSolver>::getInfo(json &params) const
{
    params["solver_iter"] = m_Solver.iterations();
    params["solver_error"] = m_Solver.error();
}

// Analyze sparsity pattern
template <typename SparseSolver>
void polysolve::LinearSolverEigenIterative<SparseSolver>::analyzePattern(const StiffnessMatrix &A, const int precond_num)
{
    m_Solver.analyzePattern(A);
}

// Factorize system matrix
template <typename SparseSolver>
void polysolve::LinearSolverEigenIterative<SparseSolver>::factorize(const StiffnessMatrix &A)
{
    m_Solver.factorize(A);
}

// Solve the linear system
template <typename SparseSolver>
void polysolve::LinearSolverEigenIterative<SparseSolver>::solve(
    const Ref<const VectorXd> b, Ref<VectorXd> x)
{
    assert(x.size() == b.size());
    x = m_Solver.solveWithGuess(b, x);
}

////////////////////////////////////////////////////////////////////////////////
// Dense solvers
////////////////////////////////////////////////////////////////////////////////

// Get info on the last solve step
template <typename DenseSolver>
void polysolve::LinearSolverEigenDense<DenseSolver>::getInfo(json &params) const
{
    params["solver_info"] = "Success";
}

template <typename DenseSolver>
void polysolve::LinearSolverEigenDense<DenseSolver>::factorize(const StiffnessMatrix &A)
{
    factorize_dense(Eigen::MatrixXd(A));
}

// Factorize system matrix
template <typename DenseSolver>
void polysolve::LinearSolverEigenDense<DenseSolver>::factorize_dense(const Eigen::MatrixXd &A)
{
    m_Solver.compute(A);
}

// Solve the linear system
template <typename DenseSolver>
void polysolve::LinearSolverEigenDense<DenseSolver>::solve(
    const Ref<const VectorXd> b, Ref<VectorXd> x)
{
    x = m_Solver.solve(b);
}

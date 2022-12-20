#pragma once

////////////////////////////////////////////////////////////////////////////////
#include <polysolve/LinearSolver.hpp>
////////////////////////////////////////////////////////////////////////////////

namespace polysolve
{

    // -----------------------------------------------------------------------------

    template <typename SparseSolver>
    class LinearSolverEigenDirect : public LinearSolver
    {
    protected:
        // Solver class
        SparseSolver m_Solver;

        // Name of the solver
        std::string m_Name;

    public:
        // Name of the solver type (for debugging purposes)
        virtual std::string name() const override { return m_Name; }

        // Constructor requires a solver name used for finding parameters in the json file passed to setParameters
        LinearSolverEigenDirect(const std::string &name) { m_Name = name; }

    public:
        // Get info on the last solve step
        virtual void getInfo(json &params) const override;

        // Analyze sparsity pattern
        virtual void analyzePattern(const StiffnessMatrix &K, const int precond_num) override;

        // Factorize system matrix
        virtual void factorize(const StiffnessMatrix &K) override;

        // Solve the linear system
        virtual void solve(const Ref<const VectorXd> b, Ref<VectorXd> x) override;
    };

    // -----------------------------------------------------------------------------

    template <typename SparseSolver>
    class LinearSolverEigenIterative : public LinearSolver
    {
    protected:
        // Solver class
        SparseSolver m_Solver;

        // Name of the solver
        std::string m_Name;

    public:
        // Name of the solver type (for debugging purposes)
        virtual std::string name() const override { return m_Name; }

        // Constructor requires a solver name used for finding parameters in the json file passed to setParameters
        LinearSolverEigenIterative(const std::string &name) { m_Name = name; }

    public:
        // Set solver parameters
        virtual void setParameters(const json &params) override;

        // Get info on the last solve step
        virtual void getInfo(json &params) const override;

        // Analyze sparsity pattern
        virtual void analyzePattern(const StiffnessMatrix &K, const int precond_num) override;

        // Factorize system matrix
        virtual void factorize(const StiffnessMatrix &K) override;

        // Solve the linear system
        virtual void solve(const Ref<const VectorXd> b, Ref<VectorXd> x) override;
    };

    // -----------------------------------------------------------------------------

    // -----------------------------------------------------------------------------

    template <typename DenseSolver>
    class LinearSolverEigenDense : public LinearSolver
    {
    protected:
        // Solver class
        DenseSolver m_Solver;

        // Name of the solver
        std::string m_Name;

    public:
        // Name of the solver type (for debugging purposes)
        virtual std::string name() const override { return m_Name; }

        // Constructor requires a solver name used for finding parameters in the json file passed to setParameters
        LinearSolverEigenDense(const std::string &name) { m_Name = name; }

    public:
        // Get info on the last solve step
        virtual void getInfo(json &params) const override;

        // Factorize system matrix
        virtual void factorize(const StiffnessMatrix &K) override;

        // Factorize system matrix
        virtual void factorize(const Eigen::MatrixXd &K) override;

        // Solve the linear system
        virtual void solve(const Ref<const VectorXd> b, Ref<VectorXd> x) override;
    };

    // -----------------------------------------------------------------------------

} // namespace polysolve

////////////////////////////////////////////////////////////////////////////////

#include <polysolve/LinearSolverEigen.tpp>

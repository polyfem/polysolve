#pragma once

////////////////////////////////////////////////////////////////////////////////
#include "Solver.hpp"
////////////////////////////////////////////////////////////////////////////////

namespace polysolve::linear
{

    // -----------------------------------------------------------------------------

    template <typename SparseSolver>
    class EigenDirect : public Solver
    {
    protected:
        // Solver class
        SparseSolver m_Solver;

        // Name of the solver
        std::string m_Name;

    public:
        // Name of the solver type (for debugging purposes)
        virtual std::string name() const override { return m_Name; }

        // Constructor requires a solver name used for finding parameters in the json file passed to set_parameters
        EigenDirect(const std::string &name) { m_Name = name; }

    public:
        // Get info on the last solve step
        virtual void get_info(json &params) const override;

        // Analyze sparsity pattern
        virtual void analyze_pattern(const StiffnessMatrix &K, const int precond_num) override;

        // Factorize system matrix
        virtual void factorize(const StiffnessMatrix &K) override;

        // Solve the linear system
        virtual void solve(const Ref<const VectorXd> b, Ref<VectorXd> x) override;
    };

    // -----------------------------------------------------------------------------

    template <typename SparseSolver>
    class EigenIterative : public Solver
    {
    protected:
        // Solver class
        SparseSolver m_Solver;

        // Name of the solver
        std::string m_Name;

    public:
        // Name of the solver type (for debugging purposes)
        virtual std::string name() const override { return m_Name; }

        // Constructor requires a solver name used for finding parameters in the json file passed to set_parameters
        EigenIterative(const std::string &name) { m_Name = name; }

    public:
        // Set solver parameters
        virtual void set_parameters(const json &params) override;

        // Get info on the last solve step
        virtual void get_info(json &params) const override;

        // Analyze sparsity pattern
        virtual void analyze_pattern(const StiffnessMatrix &K, const int precond_num) override;

        // Factorize system matrix
        virtual void factorize(const StiffnessMatrix &K) override;

        // Solve the linear system
        virtual void solve(const Ref<const VectorXd> b, Ref<VectorXd> x) override;
    };

    // -----------------------------------------------------------------------------

    // -----------------------------------------------------------------------------

    template <typename DenseSolver>
    class EigenDenseSolver : public Solver
    {
    protected:
        // Solver class
        DenseSolver m_Solver;

        // Name of the solver
        std::string m_Name;

    public:
        // Name of the solver type (for debugging purposes)
        virtual std::string name() const override { return m_Name; }

        // Constructor requires a solver name used for finding parameters in the json file passed to set_parameters
        EigenDenseSolver(const std::string &name) { m_Name = name; }

        bool is_dense() const override { return true; }

    public:
        // Get info on the last solve step
        virtual void get_info(json &params) const override;

        // Factorize system matrix
        virtual void factorize(const StiffnessMatrix &K) override;

        // Factorize system matrix
        virtual void factorize_dense(const Eigen::MatrixXd &K) override;

        // Solve the linear system
        virtual void solve(const Ref<const VectorXd> b, Ref<VectorXd> x) override;
    };

    // -----------------------------------------------------------------------------

} // namespace polysolve::linear

////////////////////////////////////////////////////////////////////////////////

#include "EigenSolver.tpp"

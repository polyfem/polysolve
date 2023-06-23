#ifdef POLYSOLVE_WITH_SYMPILER

#include <polysolve/LinearSolver.hpp>
#include <sympiler/parsy/cholesky_solver.h>

namespace polysolve
{
    class LinearSolverSympiler : public LinearSolver
    {
    public:
        LinearSolverSympiler()
            : m_solver(nullptr), m_A_csc(std::make_unique<sym_lib::parsy::CSC>()) {}
        ~LinearSolverSympiler() = default;

    private:
        POLYSOLVE_DELETE_MOVE_COPY(LinearSolverSympiler)

    protected:
        void setSolverMode(int _solver_mode);
        void updateCSC();

    public:
        //////////////////////
        // Public interface //
        //////////////////////

        // Set solver parameters
        virtual void setParameters(const json &params) override;

        // Retrieve status (success/failure) of factorization
        virtual void getInfo(json &params) const override;

        // Analyze sparsity pattern
        virtual void analyzePattern(const StiffnessMatrix &A, const int precond_num) override;

        // Factorize system matrix
        virtual void factorize(const StiffnessMatrix &A) override;

        // Solve the linear system Ax = b
        virtual void solve(const Ref<const VectorXd> b, Ref<VectorXd> x) override;

        // Name of the solver type (for debugging purposes)
        virtual std::string name() const override { return "Sympiler"; }

    private:
        std::unique_ptr<sym_lib::parsy::SolverSettings> m_solver;
        std::unique_ptr<sym_lib::parsy::CSC> m_A_csc;
        polysolve::StiffnessMatrix m_A_copy;

    protected:
        ////////////////////
        // Sympiler stuff //
        ////////////////////
        int solver_mode = 0; // 0 is normal solve, 1 is row/col addition
        int factorize_status = -1;
    };
} // namespace polysolve

#endif
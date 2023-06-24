#ifdef POLYSOLVE_WITH_SYMPILER

#include <polysolve/LinearSolver.hpp>

namespace polysolve
{
    class LinearSolverSympilerImpl;
    class LinearSolverSympiler : public LinearSolver
    {
    public:
        LinearSolverSympiler();
        ~LinearSolverSympiler();

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
        std::unique_ptr<LinearSolverSympilerImpl> m_pImpl;

    protected:
        ////////////////////
        // Sympiler stuff //
        ////////////////////
        int solver_mode = 0; // 0 is normal solve, 1 is row/col addition
        int factorize_status = -1;
    };
} // namespace polysolve

#endif
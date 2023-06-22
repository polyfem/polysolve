#ifdef POLYSOLVE_WITH_SYMPILER

#include <polysolve/LinearSolver.hpp>
#include <sympiler/parsy/cholesky_solver.h>
#include <Eigen/Sparse>
#include <Eigen/Core>

namespace polysolve
{
    class LinearSolverSympiler : public LinearSolver
    {
    public:
        LinearSolverSympiler() = default;
        ~LinearSolverSympiler() = default;

    private:
        POLYSOLVE_DELETE_MOVE_COPY(LinearSolverSympiler)

    protected:
        void setSolverMode(int _solver_mode);

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
        sym_lib::parsy::SolverSettings *solver_ = nullptr;
        sym_lib::parsy::CSC *A_csc_ = new sym_lib::parsy::CSC;

    protected:
        ////////////////////
        // Sympiler stuff //
        ////////////////////
        int solver_mode = 0; //0 is normal solve, 1 is row/col addition
        int factorize_status;

    };
} // namespace polysolve

#endif
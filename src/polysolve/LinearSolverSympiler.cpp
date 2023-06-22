#ifdef POLYSOLVE_WITH_SYMPILER

#include <polysolve/LinearSolverSympiler.hpp>
#include <Eigen/Sparse>
#include <Eigen/Core>

namespace polysolve
{
    void LinearSolverSympiler::setSolverMode(int _solver_mode)
    {
        solver_mode = _solver_mode;
    }

    void LinearSolverSympiler::setParameters(const json &params)
    {
        if (params.contains("Sympiler"))
        {
            if (params["Sympiler"].contains("solver_mode"))
            {
                setSolverMode(params["Sympiler"]["mtype"].get<int>());
            }
        }
    }

    void LinearSolverSympiler::getInfo(json &params) const
    {
        if (factorize_status == 1)
        {
            params["factorize_info"] = "Success";
        }
        else
        {
            params["factorize_info"] = "Failure";
        }
    }

    void LinearSolverSympiler::analyzePattern(const StiffnessMatrix &A, const int precond_num)
    {
        StiffnessMatrix A_copy = StiffnessMatrix(A);

        A_csc_->nzmax = A_copy.nonZeros();
        A_csc_->ncol = A_csc_->nrow = A_copy.rows();
        A_csc_->p = A_copy.outerIndexPtr();
        A_csc_->i = A_copy.innerIndexPtr();

        A_csc_->x = A_copy.valuePtr();
        A_csc_->stype = -1;
        A_csc_->packed = 1;
        A_csc_->nz = nullptr;

        solver_ = new sym_lib::parsy::SolverSettings(A_csc_);
        solver_->symbolic_analysis();
    }

    // Factorize system matrix
    void LinearSolverSympiler::factorize(const StiffnessMatrix &A)
    {
        // This copy isn't always necessary if matrix doesn't change
        StiffnessMatrix A_copy = StiffnessMatrix(A);

        A_csc_->nzmax = A_copy.nonZeros();
        A_csc_->ncol = A_csc_->nrow = A_copy.rows();
        A_csc_->p = A_copy.outerIndexPtr();
        A_csc_->i = A_copy.innerIndexPtr();

        A_csc_->x = A_copy.valuePtr();
        A_csc_->stype = -1;
        A_csc_->packed = 1;
        A_csc_->nz = nullptr;
        factorize_status = solver_->numerical_factorization(A_csc_);
    }

    // Solve the linear system Ax = b
    void LinearSolverSympiler::solve(const Ref<const Eigen::VectorXd> b, Ref<Eigen::VectorXd> x)
    {
        double *x_ptr = solver_->solve_only(b.data());
        x = Eigen::Map<Eigen::VectorXd>(x_ptr, x.rows(), x.cols());
    }

} // namespace polysolve

#endif
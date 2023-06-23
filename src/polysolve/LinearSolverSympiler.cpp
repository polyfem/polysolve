#ifdef POLYSOLVE_WITH_SYMPILER

#include <polysolve/LinearSolverSympiler.hpp>
#include <Eigen/Core>

namespace polysolve
{
    void LinearSolverSympiler::setSolverMode(int _solver_mode)
    {
        solver_mode = _solver_mode;
    }

    void LinearSolverSympiler::updateCSC()
    {
        m_A_csc->nzmax = m_A_copy.nonZeros();
        m_A_csc->ncol = m_A_csc->nrow = m_A_copy.rows();
        m_A_csc->p = m_A_copy.outerIndexPtr();
        m_A_csc->i = m_A_copy.innerIndexPtr();

        m_A_csc->x = m_A_copy.valuePtr();
        m_A_csc->stype = -1;
        m_A_csc->packed = 1;
        m_A_csc->nz = nullptr;
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
        m_A_copy = StiffnessMatrix(A);
        updateCSC();
        m_solver = std::make_unique<sym_lib::parsy::SolverSettings>(m_A_csc.get());
        m_solver->symbolic_analysis();
    }

    // Factorize system matrix
    void LinearSolverSympiler::factorize(const StiffnessMatrix &A)
    {
        // Only copy when matrix changes
        if (!m_A_copy.isApprox(A))
        {
            m_A_copy = StiffnessMatrix(A);
            updateCSC();
        }
        factorize_status = m_solver->numerical_factorization(m_A_csc.get());
    }

    // Solve the linear system Ax = b
    void LinearSolverSympiler::solve(const Ref<const Eigen::VectorXd> b, Ref<Eigen::VectorXd> x)
    {
        double *x_ptr = m_solver->solve_only(b.data());
        x = Eigen::Map<Eigen::VectorXd>(x_ptr, x.rows(), x.cols());
    }

} // namespace polysolve

#endif
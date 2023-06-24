#ifdef POLYSOLVE_WITH_SYMPILER

#include <polysolve/LinearSolverSympiler.hpp>
#include <sympiler/parsy/cholesky_solver.h>
#include <Eigen/Core>

namespace polysolve
{
class LinearSolverSympilerImpl
{
public:
    LinearSolverSympilerImpl()
        : m_solver(nullptr), m_A_csc(std::make_unique<sym_lib::parsy::CSC>()) {}
    ~LinearSolverSympilerImpl() = default;

    std::unique_ptr<sym_lib::parsy::SolverSettings> m_solver;
    std::unique_ptr<sym_lib::parsy::CSC> m_A_csc;
    polysolve::StiffnessMatrix m_A_copy;
};

    LinearSolverSympiler::LinearSolverSympiler() : m_pImpl(new LinearSolverSympilerImpl) {}
    LinearSolverSympiler::~LinearSolverSympiler() = default;

    void LinearSolverSympiler::setSolverMode(int _solver_mode)
    {
        solver_mode = _solver_mode;
    }

    void LinearSolverSympiler::updateCSC()
    {
        m_pImpl->m_A_csc->nzmax = m_pImpl->m_A_copy.nonZeros();
        m_pImpl->m_A_csc->ncol = m_pImpl->m_A_csc->nrow = m_pImpl->m_A_copy.rows();
        m_pImpl->m_A_csc->p = m_pImpl->m_A_copy.outerIndexPtr();
        m_pImpl->m_A_csc->i = m_pImpl->m_A_copy.innerIndexPtr();

        m_pImpl->m_A_csc->x = m_pImpl->m_A_copy.valuePtr();
        m_pImpl->m_A_csc->stype = -1;
        m_pImpl->m_A_csc->packed = 1;
        m_pImpl->m_A_csc->nz = nullptr;
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
        m_pImpl->m_A_copy = StiffnessMatrix(A);
        updateCSC();
        m_pImpl->m_solver = std::make_unique<sym_lib::parsy::SolverSettings>(m_pImpl->m_A_csc.get());
        m_pImpl->m_solver->symbolic_analysis();
    }

    // Factorize system matrix
    void LinearSolverSympiler::factorize(const StiffnessMatrix &A)
    {
        // Only copy when matrix changes
        if (!m_pImpl->m_A_copy.isApprox(A))
        {
            m_pImpl->m_A_copy = StiffnessMatrix(A);
            updateCSC();
        }
        factorize_status = m_pImpl->m_solver->numerical_factorization(m_pImpl->m_A_csc.get());
    }

    // Solve the linear system Ax = b
    void LinearSolverSympiler::solve(const Ref<const Eigen::VectorXd> b, Ref<Eigen::VectorXd> x)
    {
        double *x_ptr = m_pImpl->m_solver->solve_only(b.data());
        x = Eigen::Map<Eigen::VectorXd>(x_ptr, x.rows(), x.cols());
    }

} // namespace polysolve

#endif
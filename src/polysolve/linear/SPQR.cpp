#include "SPQR.hpp"

namespace polysolve::linear
{
    template <>
    void EigenDirect<Eigen::SPQR<StiffnessMatrix>>::analyze_pattern(const StiffnessMatrix &A, const int precond_num)
    {
        m_Solver.compute(A);
    }
    template <>
    void EigenDirect<Eigen::SPQR<StiffnessMatrix>>::factorize(const StiffnessMatrix &A)
    {
        m_Solver.compute(A);
        if (m_Solver.info() == Eigen::NumericalIssue)
        {
            throw std::runtime_error("[EigenDirect] NumericalIssue encountered.");
        }
    }

} // namespace polysolve::linear

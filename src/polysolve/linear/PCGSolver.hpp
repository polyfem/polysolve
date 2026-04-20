#pragma once

#include "Solver.hpp"

#include <memory>
#include <string>

namespace polysolve::linear
{
    enum class CudaPCGStatus
    {
        Running,
        ReachRelativeTolerance,
        ReachAbsoluteTolerance,
        ReachMaxIterations,
    };

    inline std::string pcg_status_to_string(CudaPCGStatus stat)
    {
        switch (stat)
        {
        case CudaPCGStatus::Running:
            return "Running";
        case CudaPCGStatus::ReachRelativeTolerance:
            return "Reach relative tolerance";
        case CudaPCGStatus::ReachAbsoluteTolerance:
            return "Reach absolute tolerance";
        case CudaPCGStatus::ReachMaxIterations:
            return "Reach max iterations";
        default:
            return "Unknown";
        }
    }

    class CudaPCG : public Solver
    {

    public:
        CudaPCG();
        ~CudaPCG() override;
        POLYSOLVE_DELETE_MOVE_COPY(CudaPCG)

    public:
        //////////////////////
        // Public interface //
        //////////////////////

        // Set solver parameters
        void set_parameters(const json &params) override;

        // Retrieve information
        void get_info(json &params) const override;

        // Analyze sparsity pattern. This is a no-op
        void analyze_pattern(const StiffnessMatrix &A, const int precond_num) override;

        // Factorize system matrix.
        void factorize(const StiffnessMatrix &A) override;

        // Solve the linear system Ax = b
        void solve(const Ref<const VectorXd> b, Ref<VectorXd> x) override;

        // Set solver tolerance
        // void set_tolerance(const double tol) override;

        // Name of the solver type (for debugging purposes)
        std::string name() const override;

    private:
        class CudaPCGImpl;
        std::unique_ptr<CudaPCGImpl> impl_;
    };

} // namespace polysolve::linear

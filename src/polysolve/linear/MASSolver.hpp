#pragma once

#include "Solver.hpp"

#include <memory>
#include <string>

namespace polysolve::linear
{
    enum class MASSolverStatus
    {
        Running,
        ReachRelativeTolerance,
        ReachAbsoluteTolerance,
        ReachMaxIterations,
    };

    inline std::string mas_status_to_string(MASSolverStatus stat)
    {
        switch (stat)
        {
        case MASSolverStatus::Running:
            return "Running";
        case MASSolverStatus::ReachRelativeTolerance:
            return "Reach relative tolerance";
        case MASSolverStatus::ReachAbsoluteTolerance:
            return "Reach absolute tolerance";
        case MASSolverStatus::ReachMaxIterations:
            return "Reach max iterations";
        default:
            return "Unknown";
        }
    }

    class MASSolver : public Solver
    {

    public:
        MASSolver();
        ~MASSolver() override;
        POLYSOLVE_DELETE_MOVE_COPY(MASSolver)

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
        class MASSolverImpl;
        std::unique_ptr<MASSolverImpl> impl_;
    };

} // namespace polysolve::linear

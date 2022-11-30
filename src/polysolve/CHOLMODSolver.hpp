#pragma once

#ifdef POLYSOLVE_WITH_CHOLMOD

////////////////////////////////////////////////////////////////////////////////

#include <polysolve/LinearSolver.hpp>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <cholmod.h>
#include <memory>

#include <nlohmann/json.hpp>
using json = nlohmann::json;

namespace polysolve
{
    /**
     @brief      Base class for cholmod solver.
     */
    class CHOLMODSolver : public LinearSolver
    {

    protected:
        cholmod_common *cm;
        cholmod_sparse A;
        cholmod_dense *x, b;
        cholmod_factor *L;

    public:
        CHOLMODSolver();
        ~CHOLMODSolver();

    private:
        POLYSOLVE_DELETE_MOVE_COPY(CHOLMODSolver)
    public:
        // Set solver parameters
        // virtual void setParameters(const json &params) {}

        // Get info on the last solve step
        virtual void getInfo(json &params) const override;

        // Analyze sparsity pattern
        virtual void analyzePattern(const StiffnessMatrix &A, const int precond_num) override;

        // Factorize system matrix
        void factorize(const StiffnessMatrix &A) override;

        //
        // @brief         { Solve the linear system Ax = b }
        //
        // @param[in]     b     { Right-hand side. }
        // @param[in,out] x     { Unknown to compute. When using an iterative
        //                      solver, the input unknown vector is used as an
        //                      initial guess, and must thus be properly allocated
        //                      and initialized. }
        //
        virtual void solve(const Ref<const VectorXd> b, Ref<VectorXd> x) override;
    };

} // namespace polysolve

#endif

#pragma once

////////////////////////////////////////////////////////////////////////////////

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "cholmod.h"
#include <memory>

#include <nlohmann/json.hpp>
using json = nlohmann::json;


namespace polysolve
{
    typedef Eigen::SparseMatrix<double, Eigen::ColMajor, long int> StiffnessMatrixL;
    /**
     @brief      Base class for cholmod solver.
 */
    class CHOLMODSolver
    {
        
    protected:

        cholmod_common *cm;
        cholmod_sparse A;
        cholmod_dense *x, b;
        cholmod_factor *L;

    public:
        CHOLMODSolver();
        ~CHOLMODSolver();

        // Set solver parameters
        // virtual void setParameters(const json &params) {}

        // Get info on the last solve step
        void getInfo(json &params) const;

        // Analyze sparsity pattern
        void analyzePattern(StiffnessMatrixL &Ain);

        // Factorize system matrix
        void factorize(StiffnessMatrixL &Ain);

        //
        // @brief         { Solve the linear system Ax = b }
        //
        // @param[in]     b     { Right-hand side. }
        // @param[in,out] x     { Unknown to compute. When using an iterative
        //                      solver, the input unknown vector is used as an
        //                      initial guess, and must thus be properly allocated
        //                      and initialized. }
        //
        void solve(Eigen::VectorXd &rhs, Eigen::VectorXd &result);
    };

} // namespace polysolve

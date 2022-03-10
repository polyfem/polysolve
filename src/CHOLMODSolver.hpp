#pragma once

////////////////////////////////////////////////////////////////////////////////

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "cholmod.h"
#include <memory>

#include <nlohmann/json.hpp>
using json = nlohmann::json;

// #define POLYSOLVE_DELETE_MOVE_COPY(Base) \
//     Base(Base &&) = delete;                    \
//     Base &operator=(Base &&) = delete;         \
//     Base(const Base &) = delete;               \
//     Base &operator=(const Base &) = delete;

////////////////////////////////////////////////////////////////////////////////
// TODO:
// - [ ] Support both RowMajor + ColumnMajor sparse matrices
// - [ ] Wrapper around MUMPS
// - [ ] Wrapper around other iterative solvers (AMGCL, ViennaCL, etc.)
// - [ ] Document the json parameters for each
////////////////////////////////////////////////////////////////////////////////

namespace polysolve
{
    typedef Eigen::SparseMatrix<double, Eigen::ColMajor, long int> StiffnessMatrixL;
    /**
     @brief      Base class for cholmod solver.
 */
    class CHOLMODSolver
    {

    // public:
        // Shortcut alias
        // typedef Eigen::VectorXd VectorXd;
        // template <typename T>
        // using Ref = Eigen::Ref<T>;

    // public:
        //////////////////
        // Constructors //
        //////////////////

        // Virtual destructor
       

        // Static constructor
        //
        // @param[in]  solver   Solver type
        // @param[in]  precond  Preconditioner for iterative solvers
        //
        // static std::unique_ptr<LinearSolver> create(const std::string &solver, const std::string &precond);

        // List available solvers
        // static std::vector<std::string> availableSolvers();
        // static std::string defaultSolver();

        // List available preconditioners
        // static std::vector<std::string> availablePrecond();
        // static std::string defaultPrecond();

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

#pragma once

////////////////////////////////////////////////////////////////////////////////
#include <nlohmann/json.hpp>
using json = nlohmann::json;

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <memory>

#define POLYSOLVE_DELETE_MOVE_COPY(Base) \
    Base(Base &&) = delete;                    \
    Base &operator=(Base &&) = delete;         \
    Base(const Base &) = delete;               \
    Base &operator=(const Base &) = delete;

////////////////////////////////////////////////////////////////////////////////
// TODO:
// - [ ] Support both RowMajor + ColumnMajor sparse matrices
// - [ ] Wrapper around MUMPS
// - [ ] Wrapper around other iterative solvers (AMGCL, ViennaCL, etc.)
// - [ ] Document the json parameters for each
////////////////////////////////////////////////////////////////////////////////

namespace polysolve
{
#ifdef POLYSOLVE_LARGE_INDEX
    typedef Eigen::SparseMatrix<double, Eigen::ColMajor, std::ptrdiff_t> StiffnessMatrix;
#else
    typedef Eigen::SparseMatrix<double, Eigen::ColMajor> StiffnessMatrix;
#endif
    /**
 * @brief      Base class for linear solver.
 */
    class LinearSolver
    {

    public:
        // Shortcut alias
        typedef Eigen::VectorXd VectorXd;
        template <typename T>
        using Ref = Eigen::Ref<T>;

    public:
        //////////////////
        // Constructors //
        //////////////////

        // Virtual destructor
        virtual ~LinearSolver() = default;

        // Static constructor
        //
        // @param[in]  solver   Solver type
        // @param[in]  precond  Preconditioner for iterative solvers
        //
        static std::unique_ptr<LinearSolver> create(const std::string &solver, const std::string &precond);

        // List available solvers
        static std::vector<std::string> availableSolvers();
        static std::string defaultSolver();

        // List available preconditioners
        static std::vector<std::string> availablePrecond();
        static std::string defaultPrecond();

    protected:
        // Default constructor
        LinearSolver() = default;

    public:
        //////////////////////
        // Public interface //
        //////////////////////

        // Set solver parameters
        virtual void setParameters(const json &params) {}

        // Get info on the last solve step
        virtual void getInfo(json &params) const {};

        // Analyze sparsity pattern
        virtual void analyzePattern(const StiffnessMatrix &A, const int precond_num) {}

        // Factorize system matrix
        virtual void factorize(const StiffnessMatrix &A) {}

        // Analyze sparsity pattern of a dense matrix
        virtual void analyzePattern(const Eigen::MatrixXd &A, const int precond_num) {}

        // Factorize system matrix of a dense matrix
        virtual void factorize(const Eigen::MatrixXd &A) {}

        //
        // @brief         { Solve the linear system Ax = b }
        //
        // @param[in]     b     { Right-hand side. }
        // @param[in,out] x     { Unknown to compute. When using an iterative
        //                      solver, the input unknown vector is used as an
        //                      initial guess, and must thus be properly allocated
        //                      and initialized. }
        //
        virtual void solve(const Ref<const VectorXd> b, Ref<VectorXd> x) = 0;

    public:
        ///////////
        // Debug //
        ///////////

        // Name of the solver type (for debugging purposes)
        virtual std::string name() const { return ""; }
    };

} // namespace polysolve

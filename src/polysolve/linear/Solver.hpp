#pragma once

#include <polysolve/Types.hpp>

#include <memory>

#define POLYSOLVE_DELETE_MOVE_COPY(Base) \
    Base(Base &&) = delete;              \
    Base &operator=(Base &&) = delete;   \
    Base(const Base &) = delete;         \
    Base &operator=(const Base &) = delete;

////////////////////////////////////////////////////////////////////////////////
// TODO:
// - [ ] Support both RowMajor + ColumnMajor sparse matrices
// - [ ] Wrapper around MUMPS
// - [ ] Wrapper around other iterative solvers (AMGCL, ViennaCL, etc.)
// - [ ] Document the json parameters for each
////////////////////////////////////////////////////////////////////////////////

namespace spdlog
{
    class logger;
}

namespace polysolve::linear
{
    /**
     * @brief      Base class for linear solver.
     */
    class Solver
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
        virtual ~Solver() = default;

        static std::unique_ptr<Solver> create(const json &params,
                                              spdlog::logger &logger,
                                              const bool strict_validation = true);

        // Static constructor
        //
        // @param[in]  solver   Solver type
        // @param[in]  precond  Preconditioner for iterative solvers
        //
        static std::unique_ptr<Solver> create(const std::string &solver, const std::string &precond);

        // List available solvers
        static std::vector<std::string> available_solvers();
        static std::string default_solver();

        // List available preconditioners
        static std::vector<std::string> available_preconds();
        static std::string default_precond();

    protected:
        // Default constructor
        Solver() = default;

    public:
        //////////////////////
        // Public interface //
        //////////////////////

        // Set solver parameters
        virtual void set_parameters(const json &params) {}

        // Get info on the last solve step
        virtual void get_info(json &params) const {};

        // Analyze sparsity pattern
        virtual void analyze_pattern(const StiffnessMatrix &A, const int precond_num) {}

        // Factorize system matrix
        virtual void factorize(const StiffnessMatrix &A) {}

        // Analyze sparsity pattern of a dense matrix
        virtual void analyze_pattern_dense(const Eigen::MatrixXd &A, const int precond_num) {}

        // Factorize system matrix of a dense matrix
        virtual void factorize_dense(const Eigen::MatrixXd &A) {}

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

} // namespace polysolve::linear

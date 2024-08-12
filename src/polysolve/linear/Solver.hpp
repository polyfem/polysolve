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
        typedef Eigen::MatrixXd MatrixXd;
        template <typename T>
        using Ref = Eigen::Ref<T>;

    public:
        //////////////////
        // Constructors //
        //////////////////

        // Virtual destructor
        virtual ~Solver() = default;

        /// Sets the default paramters to the rules (solver and precond are cmake dependent)
        static void apply_default_solver(json &rules, const std::string &prefix = "");

        /// Selects the correct solver based on params using the fallback or the list of solvers if necessary
        static void select_valid_solver(json &params, spdlog::logger &logger);

        /// @brief Static constructor
        ///
        /// @param[in]  params   Parameter of the solver, including name and preconditioner
        /// @param[in]  logger   Logger used for error
        /// @param[in]  strict_validation    strict validation of the input paraams
        /// @return a pointer to a linear solver
        //
        static std::unique_ptr<Solver> create(const json &params,
                                              spdlog::logger &logger,
                                              const bool strict_validation = true);

        /// @brief Static constructor
        ///
        /// @param[in]  solver   Solver type
        /// @param[in]  precond  Preconditioner for iterative solvers
        ///
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

        /// Set solver parameters
        virtual void set_parameters(const json &params) {}

        /// Get info on the last solve step
        virtual void get_info(json &params) const {};

        /// Analyze sparsity pattern
        virtual void analyze_pattern(const StiffnessMatrix &A, const int precond_num) {}

        /// Factorize system matrix
        virtual void factorize(const StiffnessMatrix &A) {}

        /// Analyze sparsity pattern of a dense matrix
        virtual void analyze_pattern_dense(const Eigen::MatrixXd &A, const int precond_num) {}

        /// Factorize system matrix of a dense matrix
        virtual void factorize_dense(const Eigen::MatrixXd &A) {}

        /// If solver uses dense matrices
        virtual bool is_dense() const { return false; }

        /// Set block size for multigrid solvers
        virtual void set_block_size(int block_size) {}

        /// If the problem is nullspace for multigrid solvers
        virtual void set_is_nullspace(const VectorXd &x) {}

        ///
        /// @brief         { Solve the linear system Ax = b }
        ///
        /// @param[in]     b     { Right-hand side. }
        /// @param[in,out] x     { Unknown to compute. When using an iterative
        ///                      solver, the input unknown vector is used as an
        ///                      initial guess, and must thus be properly allocated
        ///                      and initialized. }
        ///
        virtual void solve(const Ref<const VectorXd> b, Ref<VectorXd> x) = 0;
        virtual void solve(const Ref<const VectorXd> b, const Ref<const MatrixXd> nullspace, Ref<VectorXd> x) {}

        /// @brief Name of the solver type (for debugging purposes)
        virtual std::string name() const { return ""; }
    };

} // namespace polysolve::linear

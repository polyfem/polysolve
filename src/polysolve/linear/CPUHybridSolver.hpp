#pragma once

////////////////////////////////////////////////////////////////////////////////
#include "Solver.hpp"

#include <vector>
#include <set>
#include <deque>
#include <unordered_map>
#include <iostream>

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>

#include <chrono>
#include <memory>
#include <string>

#include <HYPRE.h>
#include <HYPRE_parcsr_ls.h>
#include <HYPRE_parcsr_mv.h>

// MPI here is nano-mpi: every rank is a thread of this process, so the solver
// gets hypre's domain-decomposed algorithms without the caller having to
// relaunch itself under mpirun. Everything below is ordinary MPI -- collectives,
// MPI_IN_PLACE, shared-memory windows -- and nanompi.h adds only the three
// entry points that create the ranks, since there is no launcher to do it.
#include <mpi.h>
#include <nanompi.h>

namespace polysolve::linear
{
    class AbstractSolver
    {

    public:
        virtual void compute(const Eigen::SparseMatrix<double> &A) = 0;
        virtual Eigen::VectorXd solve(const Eigen::VectorXd &b) = 0;
        virtual ~AbstractSolver() = default;
    };

    template <typename EigenSolverT>
    class EigenWrapper : public AbstractSolver
    {
        EigenSolverT solver;

    public:
        void compute(const Eigen::SparseMatrix<double> &A) override
        {
            solver.compute(A);
        }

        Eigen::VectorXd solve(const Eigen::VectorXd &b) override
        {
            return solver.solve(b);
        }
    };

    class CPUHybridSolver : public Solver
    {

    public:
        CPUHybridSolver();
        ~CPUHybridSolver();

        typedef Eigen::Map<StiffnessMatrix> SharedSparseMatrix;
        typedef Eigen::Map<Eigen::VectorXd> SharedVector;

        enum SolverCmd
        {
            CMD_CREATE,
            CMD_SET_PARAMETERS,
            CMD_SET_BLOCK_SIZE,
            CMD_SET_BLOCK_MAPPING,
            CMD_FACTORIZE,
            CMD_SOLVE,
            CMD_DESTROY,
            CMD_EXIT
        };

    private:
        POLYSOLVE_DELETE_MOVE_COPY(CPUHybridSolver)

        int solver_id;
        static inline int next_id = 0;
        static inline thread_local bool is_running_worker_loop = false;
        static inline thread_local std::unordered_map<int, std::unique_ptr<CPUHybridSolver>> worker_registry;

    public:
        static void run_worker_loop();
        static inline bool &worker_loop_flag() { return is_running_worker_loop; }

    private:

        // rank 0 is the calling thread; ranks 1..n-1 are worker threads
        static inline nanompi_team *rank_team = nullptr;
        // The ranks exist only while a hybrid solver does. Anything else in the
        // process that uses hypre (the plain HypreSolver, say) runs on the
        // calling thread alone and must still see a one-rank world, so the last
        // solver to be destroyed shuts the team down again.
        static inline int live_solvers = 0;
        static void ensure_ranks();
        static void release_ranks();

    public:
        //////////////////////
        // Public interface //
        //////////////////////

        // Set solver parameters
        virtual void set_parameters(const json &params) override;

        // Set block size for multigrid solvers
        virtual void set_block_size(int block_size) override;

        // Set the function (block) assigned to each row for multigrid solvers
        virtual void set_block_mapping(const Eigen::VectorXi &block_mapping) override;

        // Retrieve solve information
        virtual void get_info(json &params) const override;

        void check_settings() const;

        // Factorize system matrix
        virtual void factorize(const StiffnessMatrix &A) override;

        // Solve the linear system Ax = b
        virtual void solve(const Ref<const VectorXd> b, Ref<VectorXd> x) override;

        // Name of the solver type (for debugging purposes)
        virtual std::string name() const override
        {
            return select_bad_dofs_from_l1_row_norm ? "CPUHybrid" : "CPUAMGF";
        }

        virtual void set_problematic_dofs(const std::set<int> &bad_dofs) override { all_bad_dofs = bad_dofs; }

        // Alternative strategies for picking which DOFs get the dense/direct
        // subdomain treatment, in place of the default GMM row-norm clustering.
        enum class SubdomainSelectionStrategy
        {
            KNEE,
            GMM,
            FD,
            COST,
            APOSTERIORI
        };

        // Low-rank compression of STRUMPACK's frontal matrices, under
        // use_strumpack. NONE is an exact LU factorization.
        enum class StrumpackCompression
        {
            NONE,
            BLR,
            HSS
        };

        // Outer iteration ("krylov"): PCG needs an SPD matrix and a symmetric
        // preconditioner; GMRES (right-preconditioned, restarted) takes any.
        enum class KrylovMethod
        {
            PCG,
            GMRES
        };

        // Preconditioner ("preconditioner"). HYBRID is AMG plus the direct
        // subdomain correction; with gmm_jump_threshold at DBL_MAX no
        // subdomain is ever selected and it is plain BoomerAMG. JACOBI, ILU
        // and IC are block-Jacobi across ranks: each rank works on its own
        // diagonal block, so iteration counts can grow with the rank count.
        //   JACOBI: 1 / diag(A)
        //   ILU:    hypre ILU(k), ilu_type 0 (block-Jacobi ILU), k = ilu_level_of_fill
        //   IC:     Eigen::IncompleteCholesky (Lin-More, AMD ordering, diagonal
        //           scaling, shifted until it succeeds) of each rank's block
        enum class PreconditionerType
        {
            HYBRID,
            JACOBI,
            ILU,
            IC
        };

        // Direct mode ("direct"): skip the Krylov solver and factorize the
        // whole matrix on rank 0 (Pardiso with MKL, otherwise Eigen). The other
        // ranks sit idle, so run direct solves with one rank and set the
        // factorization's threads with MKL_NUM_THREADS.
        enum class DirectFactorization
        {
            NONE,
            LU,
            LLT
        };

    protected:
        // Solver selection (see the enums above)
        KrylovMethod krylov_method = KrylovMethod::PCG;
        PreconditionerType preconditioner_type = PreconditionerType::HYBRID;
        DirectFactorization direct_factorization = DirectFactorization::NONE;
        int gmres_restart = 30;
        int ilu_level_of_fill = 0;
        double ic_initial_shift = 1e-3;
        // Keep the relative residual and elapsed time of every Krylov
        // iteration, returned by get_info as residual_history/time_history.
        bool record_residual_history = true;

        // AMG settings
        double theta = 0.5;

        // Hybrid preconditioner settings
        // When false, bad DOFs are taken only from whatever set_problematic_dofs
        // (and, if enabled, contact_patch_schwarz) supplies instead of being
        // inferred from the row-norm heuristic below (AMGF mode).
        bool select_bad_dofs_from_l1_row_norm = true;
        SubdomainSelectionStrategy subdomain_selection_strategy = SubdomainSelectionStrategy::GMM;
        bool decompose_subdomains = true;
        int min_subdomain_size = 3;
        int max_subdomain_size = 1e9;
        double gmm_jump_threshold = 10.0;
        double gmm_tol = 1e-3;
        int max_gmm_iterations = 20;
        bool expand_subdomains = true;
        bool additive_mode = false;
        // Give each contact patch its own (possibly overlapping) subdomain,
        // solved additively -- an additive Schwarz treatment of contact DOFs.
        bool contact_patch_schwarz = false;
        // Under contact_patch_schwarz, drop any patch that is fully contained
        // in another patch before factorizing -- it would only ever be
        // redundant additive correction, since the larger patch already
        // covers every one of its dofs.
        bool contact_patch_remove_covered = false;
        // Restricted Additive Schwarz: build one subdomain per bad dof (every
        // dof with a nonzero entry in that dof's row/column, since the matrix
        // is assumed structurally symmetric -- see assemble_D), but only let
        // that subdomain correct the single dof it was built around. Uses the
        // same bad-dof set as AMGF/select_bad_dofs. Mutually exclusive with
        // contact_patch_schwarz.
        bool ras_correction = false;
        // Experimental: factorize the large subdomains -- the ones that would
        // otherwise go to Pardiso/Accelerate -- with STRUMPACK's multifrontal
        // LU instead, optionally with low-rank compression of its fronts.
        // Needs polysolve built with POLYSOLVE_WITH_STRUMPACK.
        bool use_strumpack = false;
        StrumpackCompression strumpack_compression = StrumpackCompression::NONE;
        double strumpack_compression_rel_tol = 1e-4;

        // General solver settings
        int dimension_ = 1; // 1 = scalar (Laplace), 2 or 3 = vector (Elasticity)

        // Optional per-row function (block) assignment for HYPRE_BoomerAMGSetDofFunc,
        // indexed globally (i.e. by the row indices factorize() is given). Empty means
        // the default interleaved mapping (row i belongs to function i % dimension_).
        // Kept identically on every rank; each rank slices out its own local rows
        // ([starts[myid], ends[myid]]) when handing HYPRE its local dof_func array.
        Eigen::VectorXi block_mapping_;

        int max_iter_ = 10000;
        double rel_conv_tol_ = 1e-10;
        double abs_conv_tol_ = 0.0;
        double conditioning_threshold = 100.0;

        // When false, skips the per-PCG-iteration timing logs (matmul,
        // amg_v_cycle, subdomain_solve, pcg_iter) so a long solve doesn't
        // flood the log; every other log line is unaffected.
        bool detailed_log = true;

        // solve information
        HYPRE_Int num_iterations = 0;
        HYPRE_Complex final_res_norm = 0;
        // Why the last solve stopped: converged, max_iter, nan, a PCG
        // breakdown (negative_alpha, ...), direct, or a setup failure.
        std::string stop_reason;
        // Relative residual (recursive estimate) after iteration k; entry 0 is
        // the initial residual. time_history[k] is seconds since the Krylov
        // loop started.
        std::vector<double> residual_history;
        std::vector<double> time_history;
        // Phase timings and preconditioner statistics, merged into get_info.
        json info_;

    private:
        bool has_matrix_ = false;

        // Set (identically on every rank) when a preconditioner or direct
        // setup fails; solve() then returns x = 0 with this as stop_reason.
        std::string setup_error_;

        // direct mode, rank 0 only
        std::unique_ptr<AbstractSolver> direct_solver_;

        // rank-local preconditioners (JACOBI, IC)
        Eigen::VectorXd inv_diag_local_;
        std::unique_ptr<Eigen::IncompleteCholesky<double, Eigen::Lower, Eigen::AMDOrdering<int>>> ic_local_;

        // MPI rank distribution
        int myid = 0;
        int num_procs = 1;
        std::vector<int> starts;
        std::vector<int> ends;

        // problem-specific data
        std::unordered_map<int, Eigen::SparseVector<double>> global_to_row;

        // Hypre variables
        HYPRE_IJMatrix A;
        HYPRE_ParCSRMatrix parcsr_A;
        HYPRE_IJVector ij_x;
        HYPRE_IJVector ij_b;

        // hybrid preconditioner data
        std::deque<std::unique_ptr<AbstractSolver>> D_solvers;
        std::set<int> all_bad_dofs;
        std::vector<std::set<int>> bad_indices_sets;
        std::vector<std::vector<int>> bad_indices_arrays;
        std::vector<std::vector<int>> bad_subdomain_assignments;
        std::vector<std::unordered_map<int, int>> index_mappings;

        // Under ras_correction, bad_indices_sets/bad_indices_arrays[i] holds
        // the RAS subdomain built around a single dof; ras_target_dofs[i] is
        // that dof -- the only one whose correction the subdomain is allowed
        // to write back. Shared to every rank in share_bad_subdomains, same
        // as bad_indices_sets.
        std::vector<int> ras_target_dofs;

        // Under contact_patch_schwarz, contact patches (bad_indices_sets) may
        // overlap, so a dof's additive-Schwarz correction is the sum of every
        // patch that covers it. That sum is damped by this single global
        // factor -- 1 / (average number of patches covering a covered dof) --
        // rather than dividing each dof by its own multiplicity, so overlap
        // is corrected for on average without per-dof bookkeeping.
        double overlap_scale_ = 1.0;

        Eigen::VectorXd shared_rhs, shared_result;
        Eigen::VectorXd z1, z2, z3;
        Eigen::VectorXd r, p, buffer;

        // factorization helpers
        void partition_ranks(const int rows);
        void copy_matrix_to_hypre(SharedSparseMatrix &sparse_A);
        void copy_matrix_to_hypre(Eigen::SparseMatrix<double> &sparse_A);

        // solve helpers
        void init_hypre_vectors();

        // hybrid preconditioner helpers
        void assemble_D(int bad_i, int i, Eigen::SparseMatrix<double> &D, SharedSparseMatrix &sparse_A);
        void build_index_mappings();
        void decompose_subdomains_to_disjoint_subsets(SharedSparseMatrix &sparse_A);
        void filter_subdomains(SharedSparseMatrix &sparse_A);
        void expand_subdomains_to_strongly_connected(SharedSparseMatrix &sparse_A);
        void remove_covered_contact_patches();
        void build_ras_subdomains(SharedSparseMatrix &sparse_A);
        void share_bad_subdomains();
        void load_balance_subdomains();
        void select_bad_dofs(SharedSparseMatrix &sparse_A);
        void factorize_submatrix(SharedSparseMatrix &sparse_A);
        void compute_overlap_scale();

        // matrix multiplication
        void matmul(Eigen::VectorXd &x, Eigen::VectorXd &result);
        double dot(Eigen::VectorXd &x, Eigen::VectorXd &y);

        // non-hybrid preconditioners and direct mode
        void setup_local_preconditioner(SharedSparseMatrix &sparse_A);
        void factorize_direct(const StiffnessMatrix &A);
        void apply_precond(const HYPRE_Solver &precond, Eigen::VectorXd &r, Eigen::VectorXd &z, SharedVector &vec, MPI_Win &vec_win);
        void ilu_precond_iter(const HYPRE_Solver &precond, Eigen::VectorXd &b, Eigen::VectorXd &x);
        void record_iteration(double rel_res, const std::chrono::steady_clock::time_point &begin);

        // preconditioning functions
        void custom_mixed_precond_iter(const HYPRE_Solver &precond, Eigen::VectorXd &r, Eigen::VectorXd &z, SharedVector &vec, MPI_Win &vec_win);
        void amg_precond_iter(const HYPRE_Solver &precond, Eigen::VectorXd &b, Eigen::VectorXd &x);
        void dss_precond_iter(Eigen::VectorXd &z, Eigen::VectorXd &r, Eigen::VectorXd &next_z, SharedVector &vec, MPI_Win &vec_win);

        // MPI helpers
        void create_shared_vec(MPI_Win &win, void *&base_ptr, int size);
        int my_size() { return ends[myid] - starts[myid] + 1; };

        // Krylov solve methods
        void pcg_solve(Eigen::VectorXd &rhs, Eigen::VectorXd &result, HYPRE_ParVector &par_b, HYPRE_ParVector &par_x, HYPRE_Solver &precond, SharedVector &vec, MPI_Win &vec_win);
        void gmres_solve(Eigen::VectorXd &rhs, Eigen::VectorXd &result, HYPRE_Solver &precond, SharedVector &vec, MPI_Win &vec_win);
    };

} // namespace polysolve::linear
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

#include <HYPRE.h>
#include <HYPRE_parcsr_ls.h>
#include <HYPRE_parcsr_mv.h>

// MPI here is hypre's thread-MPI backend: ranks are threads of this process.
// hypre's headers already map MPI_Bcast/Allreduce/... onto hypre_MPI_*; the
// shim adds the MPI-3 windows, MPI_IN_PLACE and the init calls it lacks.
#include "tmpi_mpi_compat.hpp"

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
        static inline hypre_tmpi_team *tmpi_team = nullptr;
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
            return "CPUHybrid";
        }

    protected:
        // AMG settings
        double theta = 0.5;

        // Hybrid preconditioner settings
        bool decompose_subdomains = true;
        int min_subdomain_size = 3;
        int max_subdomain_size = 1e9;
        double gmm_jump_threshold = 10.0;
        double gmm_tol = 1e-3;
        int max_gmm_iterations = 20;
        bool expand_subdomains = true;
        bool additive_mode = false;

        // General solver settings
        int dimension_ = 1; // 1 = scalar (Laplace), 2 or 3 = vector (Elasticity)
        int max_iter_ = 10000;
        double rel_conv_tol_ = 1e-10;
        double abs_conv_tol_ = 0.0;
        double conditioning_threshold = 100.0;

        // solve information
        HYPRE_Int num_iterations;
        HYPRE_Complex final_res_norm;

    private:
        bool has_matrix_ = false;

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
        void share_bad_subdomains();
        void load_balance_subdomains();
        void select_bad_dofs(SharedSparseMatrix &sparse_A);
        void factorize_submatrix(SharedSparseMatrix &sparse_A);

        // matrix multiplication
        void matmul(Eigen::VectorXd &x, Eigen::VectorXd &result);
        double dot(Eigen::VectorXd &x, Eigen::VectorXd &y);

        // preconditioning functions
        void custom_mixed_precond_iter(const HYPRE_Solver &precond, Eigen::VectorXd &r, Eigen::VectorXd &z, SharedVector &vec, MPI_Win &vec_win);
        void amg_precond_iter(const HYPRE_Solver &precond, Eigen::VectorXd &b, Eigen::VectorXd &x);
        void dss_precond_iter(Eigen::VectorXd &z, Eigen::VectorXd &r, Eigen::VectorXd &next_z, SharedVector &vec, MPI_Win &vec_win);

        // MPI helpers
        void create_shared_vec(MPI_Win &win, void *&base_ptr, int size);
        int my_size() { return ends[myid] - starts[myid] + 1; };

        // Krylov solve methods
        void pcg_solve(Eigen::VectorXd &rhs, Eigen::VectorXd &result, HYPRE_ParVector &par_b, HYPRE_ParVector &par_x, HYPRE_Solver &precond, SharedVector &vec, MPI_Win &vec_win);
    };

} // namespace polysolve::linear